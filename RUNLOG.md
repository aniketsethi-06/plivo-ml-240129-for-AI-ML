# Run Log

**Run 1: Initial Baseline**
* **Score:** ~1600 ms mean response delay
* **Changes:** Ran the provided silence-only baseline. 
* **Reasoning:** Establishing the baseline metric to beat as per the assignment instructions.

**Run 2: Random Forest with Prosodic Features**
* **Score:** 295 ms mean response delay (at 3.0% interrupted turns)
* **Changes:** Replaced silence detection with a Random Forest Classifier trained on prosodic features (RMS energy mean/std, F0 pitch mean/std/slope) extracted from the final 1.5 seconds of audio prior to the pause. 
* **Reasoning:** Silence alone cannot distinguish between a "hold" (thinking) and an "EOT" (end of turn). Extracting pitch slope helps identify the terminal juncture (pitch drop) typical of a completed sentence, while RMS energy detects trailing volume.