int main(int argc, char** argv)
  {
    // ...

    unsigned char cudaFail;                                         //  Host-side flag for error testing.
    unsigned char* cudaFail_d;                                      //  Device-side flag for error testing.
    cudaError_t cudaErr;

    // ...
                                                                    //  Know when device-side allocation fails.
    if(cudaMalloc((void**)&data_d, data_size * sizeof(float)) != cudaSuccess)
      {
        printf("CUDA ERROR: Unable to allocate device-side data.\n");
        exit(1);
      }

    // ...
                                                                    //  Know when copying to device fails.
    if(cudaMemcpy(ctr_d, &ctr, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
      {
        printf("CUDA ERROR: Unable to copy counter to device.\n");
        exit(1);
      }

    // ...
                                                                    //  Set your custom error flag to 1.
                                                                    //  If all goes well device-side, it should remain 1.
    cudaFail = 1;                                                   //  (Innocent until proven guilty.)
    if(cudaMemcpy(cudaFail_d, &cudaFail, sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess)
      {
        printf("CUDA ERROR: Unable to set device-side failure-flag.\n");
        exit(1);
      }

    // ...
                                                                    //  Call kernel with your custom device-side error flag included.
    YourKernelCall<<<ceil(float(len_h) / float(BLOCK_SIZE)), BLOCK_SIZE>>>(/* Blah, blah, blah, arguments */, cudaFail_d);

    cudaDeviceSynchronize();

    cudaErr = cudaGetLastError();                                   //  Retrieve built-in device-side error signal.
    if(cudaErr != cudaSuccess)                                      //  Something went wrong on the device. CUDA reports it here.
      {
        printf("CUDA ERROR: Kernel failed.\n");
        printf("  %s\n", cudaGetErrorString(cudaErr));
        exit(1);
      }
                                                                    //  Retrieve your custom device-side error flag.
    cudaErr = cudaMemcpy(&cudaFail, cudaFail_d, sizeof(char), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)                                      //  Did the retrieval itself fail?
      {
        printf("CUDA ERROR: Unable to copy kernel's error flag to host.\n");
        printf("  %s\n", cudaGetErrorString(cudaErr));
        exit(1);
      }
    if(cudaFail == 0)                                               //  Does your custom error flag indicate failure?
      {
        printf("CUSTOM ERROR: Kernel failed.\n");
        exit(1);
      }

    // ...

    return 0;
  }

/* Your kernel. Whatever else you designed it to do, include the device-side error flag as an argument.
   If one of your routines fails without killing the whole kernel, you can write your own values here
   and test them when control returns to the host. */
__global__ void YourKernelCall(/* Blah, blah, blah, arguments */, unsigned char* cudaFail_g)
  {
    // ...

    if(something_went_wrong)                                        //  Your own test condition:
      *cudaFail_g = 0;                                              //  indicate an outcome that is undesirable, though not terminal.
                                                                    //  Find out about it host-side.
    // ...
  }