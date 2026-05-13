# TODO - Customer Segmentation App Deployment Fix

## Step 1: Fix Render Python runtime
- [x] Update `render.yaml` to pin a supported Python version (recommend 3.11.x) so `pandas` installs via wheels.


## Step 2: Validate dependency install
- [ ] Re-run `pip install -r requirements.txt` using the pinned Python version locally (or via Render logs) to confirm pandas no longer builds from source.

## Step 3: Redeploy
- [ ] Redeploy the app on Render and verify Streamlit starts successfully.

