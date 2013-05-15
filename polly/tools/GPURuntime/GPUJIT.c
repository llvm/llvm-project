/******************** GPUJIT.cpp - GPUJIT Execution Engine                     \
 * ********************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is dual licensed under the MIT and the University of Illinois    */
/* Open Source License. See LICENSE.TXT for details.                         */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file implements GPUJIT, a ptx string execution engine for GPU.       */
/*                                                                            */
/******************************************************************************/

#include "GPUJIT.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>

/* Define Polly's GPGPU data types. */
struct PollyGPUContextT {
  CUcontext Cuda;
};

struct PollyGPUModuleT {
  CUmodule Cuda;
};

struct PollyGPUFunctionT {
  CUfunction Cuda;
};

struct PollyGPUDeviceT {
  CUdevice Cuda;
};

struct PollyGPUDevicePtrT {
  CUdeviceptr Cuda;
};

struct PollyGPUEventT {
  cudaEvent_t Cuda;
};

/* Dynamic library handles for the CUDA and CUDA runtime library. */
static void *HandleCuda;
static void *HandleCudaRT;

/* Type-defines of function pointer to CUDA driver APIs. */
typedef CUresult CUDAAPI CuMemAllocFcnTy(CUdeviceptr *, size_t);
static CuMemAllocFcnTy *CuMemAllocFcnPtr;

typedef CUresult CUDAAPI CuFuncSetBlockShapeFcnTy(CUfunction, int, int, int);
static CuFuncSetBlockShapeFcnTy *CuFuncSetBlockShapeFcnPtr;

typedef CUresult CUDAAPI
CuParamSetvFcnTy(CUfunction, int, void *, unsigned int);
static CuParamSetvFcnTy *CuParamSetvFcnPtr;

typedef CUresult CUDAAPI CuParamSetSizeFcnTy(CUfunction, unsigned int);
static CuParamSetSizeFcnTy *CuParamSetSizeFcnPtr;

typedef CUresult CUDAAPI CuLaunchGridFcnTy(CUfunction, int, int);
static CuLaunchGridFcnTy *CuLaunchGridFcnPtr;

typedef CUresult CUDAAPI CuMemcpyDtoHFcnTy(void *, CUdeviceptr, size_t);
static CuMemcpyDtoHFcnTy *CuMemcpyDtoHFcnPtr;

typedef CUresult CUDAAPI CuMemcpyHtoDFcnTy(CUdeviceptr, const void *, size_t);
static CuMemcpyHtoDFcnTy *CuMemcpyHtoDFcnPtr;

typedef CUresult CUDAAPI CuMemFreeFcnTy(CUdeviceptr);
static CuMemFreeFcnTy *CuMemFreeFcnPtr;

typedef CUresult CUDAAPI CuModuleUnloadFcnTy(CUmodule);
static CuModuleUnloadFcnTy *CuModuleUnloadFcnPtr;

typedef CUresult CUDAAPI CuCtxDestroyFcnTy(CUcontext);
static CuCtxDestroyFcnTy *CuCtxDestroyFcnPtr;

typedef CUresult CUDAAPI CuInitFcnTy(unsigned int);
static CuInitFcnTy *CuInitFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetCountFcnTy(int *);
static CuDeviceGetCountFcnTy *CuDeviceGetCountFcnPtr;

typedef CUresult CUDAAPI CuCtxCreateFcnTy(CUcontext *, unsigned int, CUdevice);
static CuCtxCreateFcnTy *CuCtxCreateFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetFcnTy(CUdevice *, int);
static CuDeviceGetFcnTy *CuDeviceGetFcnPtr;

typedef CUresult CUDAAPI CuModuleLoadDataExFcnTy(
    CUmodule *, const void *, unsigned int, CUjit_option *, void **);
static CuModuleLoadDataExFcnTy *CuModuleLoadDataExFcnPtr;

typedef CUresult CUDAAPI
CuModuleGetFunctionFcnTy(CUfunction *, CUmodule, const char *);
static CuModuleGetFunctionFcnTy *CuModuleGetFunctionFcnPtr;

typedef CUresult CUDAAPI CuDeviceComputeCapabilityFcnTy(int *, int *, CUdevice);
static CuDeviceComputeCapabilityFcnTy *CuDeviceComputeCapabilityFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetNameFcnTy(char *, int, CUdevice);
static CuDeviceGetNameFcnTy *CuDeviceGetNameFcnPtr;

/* Type-defines of function pointer ot CUDA runtime APIs. */
typedef cudaError_t CUDARTAPI CudaEventCreateFcnTy(cudaEvent_t *);
static CudaEventCreateFcnTy *CudaEventCreateFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventRecordFcnTy(cudaEvent_t, cudaStream_t);
static CudaEventRecordFcnTy *CudaEventRecordFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventSynchronizeFcnTy(cudaEvent_t);
static CudaEventSynchronizeFcnTy *CudaEventSynchronizeFcnPtr;

typedef cudaError_t CUDARTAPI
CudaEventElapsedTimeFcnTy(float *, cudaEvent_t, cudaEvent_t);
static CudaEventElapsedTimeFcnTy *CudaEventElapsedTimeFcnPtr;

typedef cudaError_t CUDARTAPI CudaEventDestroyFcnTy(cudaEvent_t);
static CudaEventDestroyFcnTy *CudaEventDestroyFcnPtr;

typedef cudaError_t CUDARTAPI CudaThreadSynchronizeFcnTy(void);
static CudaThreadSynchronizeFcnTy *CudaThreadSynchronizeFcnPtr;

static void *getAPIHandle(void *Handle, const char *FuncName) {
  char *Err;
  void *FuncPtr;
  dlerror();
  FuncPtr = dlsym(Handle, FuncName);
  if ((Err = dlerror()) != 0) {
    fprintf(stdout, "Load CUDA driver API failed: %s. \n", Err);
    return 0;
  }
  return FuncPtr;
}

static int initialDeviceAPILibraries() {
  HandleCuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!HandleCuda) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  HandleCudaRT = dlopen("libcudart.so", RTLD_LAZY);
  if (!HandleCudaRT) {
    printf("Cannot open library: %s. \n", dlerror());
    return 0;
  }

  return 1;
}

static int initialDeviceAPIs() {
  if (initialDeviceAPILibraries() == 0)
    return 0;

  /* Get function pointer to CUDA Driver APIs.
   *
   * Note that compilers conforming to the ISO C standard are required to
   * generate a warning if a conversion from a void * pointer to a function
   * pointer is attempted as in the following statements. The warning
   * of this kind of cast may not be emitted by clang and new versions of gcc
   * as it is valid on POSIX 2008.
   */
  CuFuncSetBlockShapeFcnPtr = (CuFuncSetBlockShapeFcnTy *)getAPIHandle(
      HandleCuda, "cuFuncSetBlockShape");

  CuParamSetvFcnPtr =
      (CuParamSetvFcnTy *)getAPIHandle(HandleCuda, "cuParamSetv");

  CuParamSetSizeFcnPtr =
      (CuParamSetSizeFcnTy *)getAPIHandle(HandleCuda, "cuParamSetSize");

  CuLaunchGridFcnPtr =
      (CuLaunchGridFcnTy *)getAPIHandle(HandleCuda, "cuLaunchGrid");

  CuMemAllocFcnPtr =
      (CuMemAllocFcnTy *)getAPIHandle(HandleCuda, "cuMemAlloc_v2");

  CuMemFreeFcnPtr = (CuMemFreeFcnTy *)getAPIHandle(HandleCuda, "cuMemFree_v2");

  CuMemcpyDtoHFcnPtr =
      (CuMemcpyDtoHFcnTy *)getAPIHandle(HandleCuda, "cuMemcpyDtoH_v2");

  CuMemcpyHtoDFcnPtr =
      (CuMemcpyHtoDFcnTy *)getAPIHandle(HandleCuda, "cuMemcpyHtoD_v2");

  CuModuleUnloadFcnPtr =
      (CuModuleUnloadFcnTy *)getAPIHandle(HandleCuda, "cuModuleUnload");

  CuCtxDestroyFcnPtr =
      (CuCtxDestroyFcnTy *)getAPIHandle(HandleCuda, "cuCtxDestroy");

  CuInitFcnPtr = (CuInitFcnTy *)getAPIHandle(HandleCuda, "cuInit");

  CuDeviceGetCountFcnPtr =
      (CuDeviceGetCountFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGetCount");

  CuDeviceGetFcnPtr =
      (CuDeviceGetFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGet");

  CuCtxCreateFcnPtr =
      (CuCtxCreateFcnTy *)getAPIHandle(HandleCuda, "cuCtxCreate_v2");

  CuModuleLoadDataExFcnPtr =
      (CuModuleLoadDataExFcnTy *)getAPIHandle(HandleCuda, "cuModuleLoadDataEx");

  CuModuleGetFunctionFcnPtr = (CuModuleGetFunctionFcnTy *)getAPIHandle(
      HandleCuda, "cuModuleGetFunction");

  CuDeviceComputeCapabilityFcnPtr =
      (CuDeviceComputeCapabilityFcnTy *)getAPIHandle(
          HandleCuda, "cuDeviceComputeCapability");

  CuDeviceGetNameFcnPtr =
      (CuDeviceGetNameFcnTy *)getAPIHandle(HandleCuda, "cuDeviceGetName");

  /* Get function pointer to CUDA Runtime APIs. */
  CudaEventCreateFcnPtr =
      (CudaEventCreateFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventCreate");

  CudaEventRecordFcnPtr =
      (CudaEventRecordFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventRecord");

  CudaEventSynchronizeFcnPtr = (CudaEventSynchronizeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaEventSynchronize");

  CudaEventElapsedTimeFcnPtr = (CudaEventElapsedTimeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaEventElapsedTime");

  CudaEventDestroyFcnPtr =
      (CudaEventDestroyFcnTy *)getAPIHandle(HandleCudaRT, "cudaEventDestroy");

  CudaThreadSynchronizeFcnPtr = (CudaThreadSynchronizeFcnTy *)getAPIHandle(
      HandleCudaRT, "cudaThreadSynchronize");

  return 1;
}

void polly_initDevice(PollyGPUContext **Context, PollyGPUDevice **Device) {
  int Major = 0, Minor = 0, DeviceID = 0;
  char DeviceName[256];
  int DeviceCount = 0;

  /* Get API handles. */
  if (initialDeviceAPIs() == 0) {
    fprintf(stdout, "Getting the \"handle\" for the CUDA driver API failed.\n");
    exit(-1);
  }

  if (CuInitFcnPtr(0) != CUDA_SUCCESS) {
    fprintf(stdout, "Initializing the CUDA driver API failed.\n");
    exit(-1);
  }

  /* Get number of devices that supports CUDA. */
  CuDeviceGetCountFcnPtr(&DeviceCount);
  if (DeviceCount == 0) {
    fprintf(stdout, "There is no device supporting CUDA.\n");
    exit(-1);
  }

  /* We select the 1st device as default. */
  *Device = malloc(sizeof(PollyGPUDevice));
  if (*Device == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU device failed.\n");
    exit(-1);
  }
  CuDeviceGetFcnPtr(&((*Device)->Cuda), 0);

  /* Get compute capabilities and the device name. */
  CuDeviceComputeCapabilityFcnPtr(&Major, &Minor, (*Device)->Cuda);
  CuDeviceGetNameFcnPtr(DeviceName, 256, (*Device)->Cuda);
  fprintf(stderr, "> Running on GPU device %d : %s.\n", DeviceID, DeviceName);

  /* Create context on the device. */
  *Context = malloc(sizeof(PollyGPUContext));
  if (*Context == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU context failed.\n");
    exit(-1);
  }
  CuCtxCreateFcnPtr(&((*Context)->Cuda), 0, (*Device)->Cuda);
}

void polly_getPTXModule(void *PTXBuffer, PollyGPUModule **Module) {
  *Module = malloc(sizeof(PollyGPUModule));
  if (*Module == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU module failed.\n");
    exit(-1);
  }

  if (CuModuleLoadDataExFcnPtr(&((*Module)->Cuda), PTXBuffer, 0, 0, 0) !=
          CUDA_SUCCESS) {
    fprintf(stdout, "Loading ptx assembly text failed.\n");
    exit(-1);
  }
}

void polly_getPTXKernelEntry(const char *KernelName, PollyGPUModule *Module,
                             PollyGPUFunction **Kernel) {
  *Kernel = malloc(sizeof(PollyGPUFunction));
  if (*Kernel == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU kernel failed.\n");
    exit(-1);
  }

  /* Locate the kernel entry point. */
  if (CuModuleGetFunctionFcnPtr(&((*Kernel)->Cuda), Module->Cuda, KernelName) !=
          CUDA_SUCCESS) {
    fprintf(stdout, "Loading kernel function failed.\n");
    exit(-1);
  }
}

void polly_startTimerByCudaEvent(PollyGPUEvent **Start, PollyGPUEvent **Stop) {
  *Start = malloc(sizeof(PollyGPUEvent));
  if (*Start == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU start timer failed.\n");
    exit(-1);
  }
  CudaEventCreateFcnPtr(&((*Start)->Cuda));

  *Stop = malloc(sizeof(PollyGPUEvent));
  if (*Stop == 0) {
    fprintf(stdout, "Allocate memory for Polly GPU stop timer failed.\n");
    exit(-1);
  }
  CudaEventCreateFcnPtr(&((*Stop)->Cuda));

  /* Record the start time. */
  CudaEventRecordFcnPtr((*Start)->Cuda, 0);
}

void polly_stopTimerByCudaEvent(PollyGPUEvent *Start, PollyGPUEvent *Stop,
                                float *ElapsedTimes) {
  /* Record the end time. */
  CudaEventRecordFcnPtr(Stop->Cuda, 0);
  CudaEventSynchronizeFcnPtr(Start->Cuda);
  CudaEventSynchronizeFcnPtr(Stop->Cuda);
  CudaEventElapsedTimeFcnPtr(ElapsedTimes, Start->Cuda, Stop->Cuda);
  CudaEventDestroyFcnPtr(Start->Cuda);
  CudaEventDestroyFcnPtr(Stop->Cuda);
  fprintf(stderr, "Processing time: %f (ms).\n", *ElapsedTimes);

  free(Start);
  free(Stop);
}

void polly_allocateMemoryForHostAndDevice(
    void **HostData, PollyGPUDevicePtr **DevData, int MemSize) {
  if ((*HostData = (int *)malloc(MemSize)) == 0) {
    fprintf(stdout, "Could not allocate host memory.\n");
    exit(-1);
  }

  *DevData = malloc(sizeof(PollyGPUDevicePtr));
  if (*DevData == 0) {
    fprintf(stdout, "Allocate memory for GPU device memory pointer failed.\n");
    exit(-1);
  }
  CuMemAllocFcnPtr(&((*DevData)->Cuda), MemSize);
}

void polly_copyFromHostToDevice(PollyGPUDevicePtr *DevData, void *HostData,
                                int MemSize) {
  CUdeviceptr CuDevData = DevData->Cuda;
  CuMemcpyHtoDFcnPtr(CuDevData, HostData, MemSize);
}

void polly_copyFromDeviceToHost(void *HostData, PollyGPUDevicePtr *DevData,
                                int MemSize) {
  if (CuMemcpyDtoHFcnPtr(HostData, DevData->Cuda, MemSize) != CUDA_SUCCESS) {
    fprintf(stdout, "Copying results from device to host memory failed.\n");
    exit(-1);
  }
}

void polly_setKernelParameters(PollyGPUFunction *Kernel, int BlockWidth,
                               int BlockHeight, PollyGPUDevicePtr *DevData) {
  int ParamOffset = 0;

  CuFuncSetBlockShapeFcnPtr(Kernel->Cuda, BlockWidth, BlockHeight, 1);
  CuParamSetvFcnPtr(Kernel->Cuda, ParamOffset, &(DevData->Cuda),
                    sizeof(DevData->Cuda));
  ParamOffset += sizeof(DevData->Cuda);
  CuParamSetSizeFcnPtr(Kernel->Cuda, ParamOffset);
}

void polly_launchKernel(PollyGPUFunction *Kernel, int GridWidth,
                        int GridHeight) {
  if (CuLaunchGridFcnPtr(Kernel->Cuda, GridWidth, GridHeight) != CUDA_SUCCESS) {
    fprintf(stdout, "Launching CUDA kernel failed.\n");
    exit(-1);
  }
  CudaThreadSynchronizeFcnPtr();
  fprintf(stdout, "CUDA kernel launched.\n");
}

void polly_cleanupGPGPUResources(
    void *HostData, PollyGPUDevicePtr *DevData, PollyGPUModule *Module,
    PollyGPUContext *Context, PollyGPUFunction *Kernel) {
  if (HostData) {
    free(HostData);
    HostData = 0;
  }

  if (DevData->Cuda) {
    CuMemFreeFcnPtr(DevData->Cuda);
    free(DevData);
  }

  if (Module->Cuda) {
    CuModuleUnloadFcnPtr(Module->Cuda);
    free(Module);
  }

  if (Context->Cuda) {
    CuCtxDestroyFcnPtr(Context->Cuda);
    free(Context);
  }

  if (Kernel) {
    free(Kernel);
  }

  dlclose(HandleCuda);
  dlclose(HandleCudaRT);
}
