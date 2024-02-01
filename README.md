# CUDA-Error-Checking
CUDA can fail silently. This code is a crude template of ways to discover device-side errors.

Basically, there are three tricks:

1. Test `cudaSuccess` after every operation involving the device.

2. Check `cudaGetLastError()` after every grid launch.

3. Create your own error flag to test conditions device-side. Then read them host-side to find out what happened.
