import sys
import os
import json
import subprocess
import shutil
import warnings
import numpy as np
from scipy.io import wavfile

import tempfile

# FORCE SILENCE
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def convert_to_wav_if_needed(input_path, log_func):
    """
    Tries to read the file. If it fails, converts to WAV using FFmpeg.
    Returns (path_to_read, is_temp)
    """
    try:
        # Check if readable
        try:
            wavfile.read(input_path)
            return input_path, False
        except Exception:
            log_func(f"Direct read failed, attempting conversion for {input_path}")
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp.close()
            output_path = tmp.name
            
            cmd = [
                "ffmpeg", "-y", 
                "-i", input_path, 
                "-ar", "44100", 
                 output_path
            ]
            # Suppress output
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            log_func(f"Converted to {output_path}")
            return output_path, True
    except Exception as e:
        log_func(f"Conversion failed: {str(e)}")
        return input_path, False

def separate_audio(input_path, output_dir, job_id, classification_path=None):
    debug_log = []
    
    def log(msg):
        debug_log.append(str(msg))

    converted_audio_path = None
    is_temp_file = False

    try:
        log(f"Start separation. Input: {input_path}, Job: {job_id}")
        input_path = os.path.abspath(input_path.strip('"'))
        output_dir = os.path.abspath(output_dir.strip('"'))
        
        # 0. Ensure Input is Valid WAV
        # Demucs might handle MP3, but since we had ID3 issues, let's normalize first.
        read_path, is_temp = convert_to_wav_if_needed(input_path, log)
        converted_audio_path = read_path
        is_temp_file = is_temp
        
        if classification_path:
            classification_path = os.path.abspath(classification_path.strip('"'))
        
        input_filename = os.path.basename(converted_audio_path)
        input_no_ext = os.path.splitext(input_filename)[0]
        # Demucs output folder is based on the input filename. 
        # If we converted to a temp file 'tmp123.wav', Demucs will output to 'htdemucs/tmp123'.
        # We need to map this back or rename.
        
        # actually, to preserve the job_id or original name context, we might want to check
        # but let's see what Demucs does.

        # 1. Run Demucs (In-process to bypass torchaudio.save issues)
        print(f"[Demucs] Loading model htdemucs...", file=sys.stderr)
        
        # Imports inside function to avoid heavy load if not needed
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio.transforms as T
        
        # Load Model
        model = get_model("htdemucs")
        model.cpu()
        model.eval()
        
        # Load Audio via Scipy (safe)
        sr, audio_data = wavfile.read(read_path)
        
        # Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
             audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
             audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
             audio_data = (audio_data.astype(np.float32) - 128) / 128.0
             
        # Ensure shape (Channels, Samples)
        if len(audio_data.shape) == 1:
            audio_data = np.expand_dims(audio_data, axis=0) # (1, Samples)
        else:
            audio_data = audio_data.T # (Channels, Samples)
            
        # Demucs expects Stereo (2 channels). Mix/Duplicate if Mono.
        if audio_data.shape[0] == 1:
            audio_data = np.concatenate([audio_data, audio_data], axis=0)
            
        # Convert to Tensor
        wav = torch.tensor(audio_data)
        
        # Resample if needed (Demucs htdemucs is 44100Hz)
        if sr != model.samplerate:
            print(f"[Demucs] Resampling {sr} -> {model.samplerate}Hz", file=sys.stderr)
            resampler = T.Resample(sr, model.samplerate)
            wav = resampler(wav)
            
        # Normalization (Standard Demucs procedure)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        # Separate
        print(f"[Demucs] Separating...", file=sys.stderr)
        # sources shape: (Sources, Channels, Samples)
        sources = apply_model(model, wav[None], device="cpu", shifts=1, split=True, overlap=0.25, progress=True)[0]
        
        # De-normalize
        sources = sources * ref.std() + ref.mean()
        
        print("[Demucs] Separation finished. Saving stems...", file=sys.stderr)
        
        # Save Stems manually using scipy
        stem_names = model.sources # ['drums', 'bass', 'other', 'vocals'] for htdemucs
        
        # Create output structure matching standard Demucs
        # htdemucs/filename_no_ext/
        demucs_folder_name = os.path.splitext(os.path.basename(read_path))[0]
        separated_folder = os.path.join(output_dir, "htdemucs", demucs_folder_name)
        os.makedirs(separated_folder, exist_ok=True)
        
        sources_np = sources.numpy()
        
        for i, name in enumerate(stem_names):
            stem_audio = sources_np[i] # (Channels, Samples)
            stem_audio = stem_audio.T # (Samples, Channels)
            
            out_file = os.path.join(separated_folder, f"{name}.wav")
            wavfile.write(out_file, model.samplerate, stem_audio)
            
        log(f"Demucs output saved to: {separated_folder}")

        # Populate final_stems
        final_stems = {}
        if os.path.exists(os.path.join(separated_folder, "vocals.wav")):
            final_stems["vocals"] = f"/separated_audio/htdemucs/{demucs_folder_name}/vocals.wav"
        if os.path.exists(os.path.join(separated_folder, "other.wav")):
            final_stems["background"] = f"/separated_audio/htdemucs/{demucs_folder_name}/other.wav"


        # 2. Forensic Event Masking (if classification provided)
        if classification_path and os.path.exists(classification_path):
            try:
                log("Starting forensic masking...")
                with open(classification_path, 'r') as f:
                    classification_data = json.load(f)
                
                log(f"Loaded classification data. Keys: {list(classification_data.keys())}")
                if "status" in classification_data and classification_data["status"] == "error":
                     log(f"Classification ERROR: {classification_data.get('message', 'No message')}")

                # Load original Audio (already converted/validated)
                sr, audio_data = wavfile.read(read_path)
                log(f"Loaded audio. Sample rate: {sr}, Shape: {audio_data.shape}")
                
                # =========================================================
                # FREQUENCY-DOMAIN SEPARATION
                # Each category has a specific frequency band.
                # We apply bandpass filters so each stem sounds DIFFERENT.
                # =========================================================
                from scipy.signal import butter, sosfiltfilt

                def bandpass_filter(data, lowcut, highcut, fs, order=4):
                    """Apply a bandpass filter to isolate a frequency range."""
                    nyq = fs / 2.0
                    low = max(lowcut / nyq, 0.001)
                    high = min(highcut / nyq, 0.999)
                    if low >= high:
                        return data
                    sos = butter(order, [low, high], btype='band', output='sos')
                    if len(data) < 30:
                        return data
                    return sosfiltfilt(sos, data.astype(np.float64)).astype(data.dtype)

                def lowpass_filter(data, cutoff, fs, order=4):
                    """Apply a lowpass filter."""
                    nyq = fs / 2.0
                    freq = min(cutoff / nyq, 0.999)
                    sos = butter(order, freq, btype='low', output='sos')
                    if len(data) < 30:
                        return data
                    return sosfiltfilt(sos, data.astype(np.float64)).astype(data.dtype)

                def highpass_filter(data, cutoff, fs, order=4):
                    """Apply a highpass filter."""
                    nyq = fs / 2.0
                    freq = max(cutoff / nyq, 0.001)
                    sos = butter(order, freq, btype='high', output='sos')
                    if len(data) < 30:
                        return data
                    return sosfiltfilt(sos, data.astype(np.float64)).astype(data.dtype)

                # Stem config: categories, frequency band [low_hz, high_hz]
                stems_config = {
                    "vocals":     {"cats": ["Human Voice", "Male Voice", "Female Voice"], "band": [300, 3500]},
                    "background": {"cats": ["Musical Content"], "band": [80, 12000]},
                    "vehicles":   {"cats": ["Vehicle Sound"], "band": [20, 500]},
                    "footsteps":  {"cats": ["Footsteps"], "band": [20, 300]},
                    "animals":    {"cats": ["Animal Signal"], "band": [800, 8000]},
                    "wind":       {"cats": ["Atmospheric Wind"], "band": [20, 800]},
                    "gunshots":   {"cats": ["Gunshot / Explosion"], "band": [30, 15000]},
                    "screams":    {"cats": ["Scream / Aggression"], "band": [500, 5000]},
                    "sirens":     {"cats": ["Siren / Alarm"], "band": [500, 2000]},
                    "impact":     {"cats": ["Impact / Breach"], "band": [30, 4000]},
                    "water":      {"cats": ["Water / Liquid"], "band": [200, 6000]},
                    "electronic": {"cats": ["Electronic Signal"], "band": [1000, 8000]},
                    "tools":      {"cats": ["Tools / Machinery"], "band": [100, 6000]},
                    "domestic":   {"cats": ["Domestic Sound"], "band": [200, 8000]},
                    "crowd":      {"cats": ["Crowd / Public"], "band": [200, 4000]},
                }

                # Check what we already have from Demucs
                has_demucs_vocals = "vocals" in final_stems
                has_demucs_background = "background" in final_stems

                # Convert to float for filtering
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float64) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float64) / 2147483648.0
                else:
                    audio_float = audio_data.astype(np.float64)

                generated_audio = { key: np.zeros_like(audio_float) for key in stems_config }

                events = classification_data.get("soundEvents", [])
                log(f"Found {len(events)} sound events.")

                CLIP_DURATION = 2.0
                count_generated = 0

                for event in events:
                    etype = event.get("type", "")
                    confidence = float(event.get("confidence", 0))
                    target_stem = None

                    etype_lower = etype.lower()
                    for stem_key, cfg in stems_config.items():
                        for trigger in cfg["cats"]:
                            if trigger.lower() in etype_lower or etype_lower in trigger.lower():
                                target_stem = stem_key
                                break
                        if target_stem:
                            break

                    if target_stem:
                        if target_stem == "vocals" and has_demucs_vocals:
                            continue
                        if target_stem == "background" and has_demucs_background:
                            continue

                        start_time = float(event.get("time", 0))
                        end_time = start_time + CLIP_DURATION
                        start_idx = max(0, int(start_time * sr))
                        end_idx = min(len(audio_float), int(end_time * sr))

                        if start_idx < end_idx and (end_idx - start_idx) > 30:
                            segment = audio_float[start_idx:end_idx]
                            band = stems_config[target_stem]["band"]

                            # Apply bandpass filter to isolate the relevant frequencies
                            filtered = bandpass_filter(segment, band[0], band[1], sr)

                            scale = min(1.0, max(0.3, confidence))
                            filtered_scaled = filtered * scale

                            # Accumulate (add, don't overwrite) for overlapping events
                            generated_audio[target_stem][start_idx:end_idx] += filtered_scaled
                            count_generated += 1


                log(f"Processed {count_generated} event segments matches.")

                # Save generated stems
                gen_dir = os.path.join(output_dir, "generated", job_id)
                os.makedirs(gen_dir, exist_ok=True)
                
                for stem_key, audio_arr in generated_audio.items():
                    peak = np.max(np.abs(audio_arr))
                    log(f"Stem {stem_key} peak amplitude: {peak}")
                    
                    if peak > 0.001:
                        # Normalize and convert to int16 for WAV
                        normalized = audio_arr / max(peak, 1e-7)  # Normalize to [-1, 1]
                        normalized = np.clip(normalized, -1.0, 1.0)
                        int16_audio = (normalized * 32767).astype(np.int16)
                        out_file = os.path.join(gen_dir, f"{stem_key}.wav")
                        wavfile.write(out_file, sr, int16_audio)
                        final_stems[stem_key] = f"/separated_audio/generated/{job_id}/{stem_key}.wav"
            
            except Exception as e:
                log(f"Masking Exception: {str(e)}")

        if not final_stems:
             log("No stems were generated.")
             return {"status": "error", "message": "Separation failed, no stems found.", "debug": debug_log}

        return {"status": "success", "stems": final_stems, "debug": debug_log}
    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_log}
    finally:
        if is_temp_file and converted_audio_path and os.path.exists(converted_audio_path):
            try:
                os.unlink(converted_audio_path)
            except:
                pass

if __name__ == "__main__":
    # Ensure no other prints exist in this file!
    if len(sys.argv) > 3:
        # Check for optional 4th arg
        cls_path = sys.argv[4] if len(sys.argv) > 4 else None
        result = separate_audio(sys.argv[1], sys.argv[2], sys.argv[3], cls_path)
        sys.stdout.write(json.dumps(result))
    else:
        sys.stdout.write(json.dumps({"status": "error", "message": "Insufficient arguments"}))
    sys.stdout.flush()