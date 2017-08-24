/******************** GPUJIT.c - GPUJIT Execution Engine **********************/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is dual licensed under the MIT and the University of Illinois    */
/* Open Source License. See LICENSE.TXT for details.                          */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/*  This file implements GPUJIT, a ptx string execution engine for GPU.       */
/*                                                                            */
/******************************************************************************/

#include "GPUJIT.h"

#ifdef HAS_LIBCUDART
#include <cuda.h>
#include <cuda_runtime.h>
#endif /* HAS_LIBCUDART */

#ifdef HAS_LIBOPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif /* __APPLE__ */
#endif /* HAS_LIBOPENCL */

#include <assert.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int DebugMode;
static int CacheMode;
#define max(x, y) ((x) > (y) ? (x) : (y))

static PollyGPURuntime Runtime = RUNTIME_NONE;

static void debug_print(const char *format, ...) {
  if (!DebugMode)
    return;

  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}
#define dump_function() debug_print("-> %s\n", __func__)

#define KERNEL_CACHE_SIZE 10

static void err_runtime() __attribute__((noreturn));
static void err_runtime() {
  fprintf(stderr, "Runtime not correctly initialized.\n");
  exit(-1);
}

struct PollyGPUContextT {
  void *Context;
};

struct PollyGPUFunctionT {
  void *Kernel;
};

struct PollyGPUDevicePtrT {
  void *DevicePtr;
};

/******************************************************************************/
/*                                  OpenCL                                    */
/******************************************************************************/
#ifdef HAS_LIBOPENCL

struct OpenCLContextT {
  cl_context Context;
  cl_command_queue CommandQueue;
};

struct OpenCLKernelT {
  cl_kernel Kernel;
  cl_program Program;
  const char *BinaryString;
};

struct OpenCLDevicePtrT {
  cl_mem MemObj;
};

/* Dynamic library handles for the OpenCL runtime library. */
static void *HandleOpenCL;
static void *HandleOpenCLBeignet;

/* Type-defines of function pointer to OpenCL Runtime API. */
typedef cl_int clGetPlatformIDsFcnTy(cl_uint NumEntries,
                                     cl_platform_id *Platforms,
                                     cl_uint *NumPlatforms);
static clGetPlatformIDsFcnTy *clGetPlatformIDsFcnPtr;

typedef cl_int clGetDeviceIDsFcnTy(cl_platform_id Platform,
                                   cl_device_type DeviceType,
                                   cl_uint NumEntries, cl_device_id *Devices,
                                   cl_uint *NumDevices);
static clGetDeviceIDsFcnTy *clGetDeviceIDsFcnPtr;

typedef cl_int clGetDeviceInfoFcnTy(cl_device_id Device,
                                    cl_device_info ParamName,
                                    size_t ParamValueSize, void *ParamValue,
                                    size_t *ParamValueSizeRet);
static clGetDeviceInfoFcnTy *clGetDeviceInfoFcnPtr;

typedef cl_int clGetKernelInfoFcnTy(cl_kernel Kernel, cl_kernel_info ParamName,
                                    size_t ParamValueSize, void *ParamValue,
                                    size_t *ParamValueSizeRet);
static clGetKernelInfoFcnTy *clGetKernelInfoFcnPtr;

typedef cl_context clCreateContextFcnTy(
    const cl_context_properties *Properties, cl_uint NumDevices,
    const cl_device_id *Devices,
    void CL_CALLBACK *pfn_notify(const char *Errinfo, const void *PrivateInfo,
                                 size_t CB, void *UserData),
    void *UserData, cl_int *ErrcodeRet);
static clCreateContextFcnTy *clCreateContextFcnPtr;

typedef cl_command_queue
clCreateCommandQueueFcnTy(cl_context Context, cl_device_id Device,
                          cl_command_queue_properties Properties,
                          cl_int *ErrcodeRet);
static clCreateCommandQueueFcnTy *clCreateCommandQueueFcnPtr;

typedef cl_mem clCreateBufferFcnTy(cl_context Context, cl_mem_flags Flags,
                                   size_t Size, void *HostPtr,
                                   cl_int *ErrcodeRet);
static clCreateBufferFcnTy *clCreateBufferFcnPtr;

typedef cl_int
clEnqueueWriteBufferFcnTy(cl_command_queue CommandQueue, cl_mem Buffer,
                          cl_bool BlockingWrite, size_t Offset, size_t Size,
                          const void *Ptr, cl_uint NumEventsInWaitList,
                          const cl_event *EventWaitList, cl_event *Event);
static clEnqueueWriteBufferFcnTy *clEnqueueWriteBufferFcnPtr;

typedef cl_program
clCreateProgramWithLLVMIntelFcnTy(cl_context Context, cl_uint NumDevices,
                                  const cl_device_id *DeviceList,
                                  const char *Filename, cl_int *ErrcodeRet);
static clCreateProgramWithLLVMIntelFcnTy *clCreateProgramWithLLVMIntelFcnPtr;

typedef cl_program clCreateProgramWithBinaryFcnTy(
    cl_context Context, cl_uint NumDevices, const cl_device_id *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries, cl_int *BinaryStatus,
    cl_int *ErrcodeRet);
static clCreateProgramWithBinaryFcnTy *clCreateProgramWithBinaryFcnPtr;

typedef cl_int clBuildProgramFcnTy(
    cl_program Program, cl_uint NumDevices, const cl_device_id *DeviceList,
    const char *Options,
    void(CL_CALLBACK *pfn_notify)(cl_program Program, void *UserData),
    void *UserData);
static clBuildProgramFcnTy *clBuildProgramFcnPtr;

typedef cl_kernel clCreateKernelFcnTy(cl_program Program,
                                      const char *KernelName,
                                      cl_int *ErrcodeRet);
static clCreateKernelFcnTy *clCreateKernelFcnPtr;

typedef cl_int clSetKernelArgFcnTy(cl_kernel Kernel, cl_uint ArgIndex,
                                   size_t ArgSize, const void *ArgValue);
static clSetKernelArgFcnTy *clSetKernelArgFcnPtr;

typedef cl_int clEnqueueNDRangeKernelFcnTy(
    cl_command_queue CommandQueue, cl_kernel Kernel, cl_uint WorkDim,
    const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
    const size_t *LocalWorkSize, cl_uint NumEventsInWaitList,
    const cl_event *EventWaitList, cl_event *Event);
static clEnqueueNDRangeKernelFcnTy *clEnqueueNDRangeKernelFcnPtr;

typedef cl_int clEnqueueReadBufferFcnTy(cl_command_queue CommandQueue,
                                        cl_mem Buffer, cl_bool BlockingRead,
                                        size_t Offset, size_t Size, void *Ptr,
                                        cl_uint NumEventsInWaitList,
                                        const cl_event *EventWaitList,
                                        cl_event *Event);
static clEnqueueReadBufferFcnTy *clEnqueueReadBufferFcnPtr;

typedef cl_int clFlushFcnTy(cl_command_queue CommandQueue);
static clFlushFcnTy *clFlushFcnPtr;

typedef cl_int clFinishFcnTy(cl_command_queue CommandQueue);
static clFinishFcnTy *clFinishFcnPtr;

typedef cl_int clReleaseKernelFcnTy(cl_kernel Kernel);
static clReleaseKernelFcnTy *clReleaseKernelFcnPtr;

typedef cl_int clReleaseProgramFcnTy(cl_program Program);
static clReleaseProgramFcnTy *clReleaseProgramFcnPtr;

typedef cl_int clReleaseMemObjectFcnTy(cl_mem Memobject);
static clReleaseMemObjectFcnTy *clReleaseMemObjectFcnPtr;

typedef cl_int clReleaseCommandQueueFcnTy(cl_command_queue CommandQueue);
static clReleaseCommandQueueFcnTy *clReleaseCommandQueueFcnPtr;

typedef cl_int clReleaseContextFcnTy(cl_context Context);
static clReleaseContextFcnTy *clReleaseContextFcnPtr;

static void *getAPIHandleCL(void *Handle, const char *FuncName) {
  char *Err;
  void *FuncPtr;
  dlerror();
  FuncPtr = dlsym(Handle, FuncName);
  if ((Err = dlerror()) != 0) {
    fprintf(stderr, "Load OpenCL Runtime API failed: %s. \n", Err);
    return 0;
  }
  return FuncPtr;
}

static int initialDeviceAPILibrariesCL() {
  HandleOpenCLBeignet = dlopen("/usr/local/lib/beignet/libcl.so", RTLD_LAZY);
  HandleOpenCL = dlopen("libOpenCL.so", RTLD_LAZY);
  if (!HandleOpenCL) {
    fprintf(stderr, "Cannot open library: %s. \n", dlerror());
    return 0;
  }
  return 1;
}

/* Get function pointer to OpenCL Runtime API.
 *
 * Note that compilers conforming to the ISO C standard are required to
 * generate a warning if a conversion from a void * pointer to a function
 * pointer is attempted as in the following statements. The warning
 * of this kind of cast may not be emitted by clang and new versions of gcc
 * as it is valid on POSIX 2008. For compilers required to generate a warning,
 * we temporarily disable -Wpedantic, to avoid bloating the output with
 * unnecessary warnings.
 *
 * Reference:
 * http://pubs.opengroup.org/onlinepubs/9699919799/functions/dlsym.html
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
static int initialDeviceAPIsCL() {
  if (initialDeviceAPILibrariesCL() == 0)
    return 0;

  // FIXME: We are now always selecting the Intel Beignet driver if it is
  // available on the system, instead of a possible NVIDIA or AMD OpenCL
  // API. This selection should occurr based on the target architecture
  // chosen when compiling.
  void *Handle =
      (HandleOpenCLBeignet != NULL ? HandleOpenCLBeignet : HandleOpenCL);

  clGetPlatformIDsFcnPtr =
      (clGetPlatformIDsFcnTy *)getAPIHandleCL(Handle, "clGetPlatformIDs");

  clGetDeviceIDsFcnPtr =
      (clGetDeviceIDsFcnTy *)getAPIHandleCL(Handle, "clGetDeviceIDs");

  clGetDeviceInfoFcnPtr =
      (clGetDeviceInfoFcnTy *)getAPIHandleCL(Handle, "clGetDeviceInfo");

  clGetKernelInfoFcnPtr =
      (clGetKernelInfoFcnTy *)getAPIHandleCL(Handle, "clGetKernelInfo");

  clCreateContextFcnPtr =
      (clCreateContextFcnTy *)getAPIHandleCL(Handle, "clCreateContext");

  clCreateCommandQueueFcnPtr = (clCreateCommandQueueFcnTy *)getAPIHandleCL(
      Handle, "clCreateCommandQueue");

  clCreateBufferFcnPtr =
      (clCreateBufferFcnTy *)getAPIHandleCL(Handle, "clCreateBuffer");

  clEnqueueWriteBufferFcnPtr = (clEnqueueWriteBufferFcnTy *)getAPIHandleCL(
      Handle, "clEnqueueWriteBuffer");

  if (HandleOpenCLBeignet)
    clCreateProgramWithLLVMIntelFcnPtr =
        (clCreateProgramWithLLVMIntelFcnTy *)getAPIHandleCL(
            Handle, "clCreateProgramWithLLVMIntel");

  clCreateProgramWithBinaryFcnPtr =
      (clCreateProgramWithBinaryFcnTy *)getAPIHandleCL(
          Handle, "clCreateProgramWithBinary");

  clBuildProgramFcnPtr =
      (clBuildProgramFcnTy *)getAPIHandleCL(Handle, "clBuildProgram");

  clCreateKernelFcnPtr =
      (clCreateKernelFcnTy *)getAPIHandleCL(Handle, "clCreateKernel");

  clSetKernelArgFcnPtr =
      (clSetKernelArgFcnTy *)getAPIHandleCL(Handle, "clSetKernelArg");

  clEnqueueNDRangeKernelFcnPtr = (clEnqueueNDRangeKernelFcnTy *)getAPIHandleCL(
      Handle, "clEnqueueNDRangeKernel");

  clEnqueueReadBufferFcnPtr =
      (clEnqueueReadBufferFcnTy *)getAPIHandleCL(Handle, "clEnqueueReadBuffer");

  clFlushFcnPtr = (clFlushFcnTy *)getAPIHandleCL(Handle, "clFlush");

  clFinishFcnPtr = (clFinishFcnTy *)getAPIHandleCL(Handle, "clFinish");

  clReleaseKernelFcnPtr =
      (clReleaseKernelFcnTy *)getAPIHandleCL(Handle, "clReleaseKernel");

  clReleaseProgramFcnPtr =
      (clReleaseProgramFcnTy *)getAPIHandleCL(Handle, "clReleaseProgram");

  clReleaseMemObjectFcnPtr =
      (clReleaseMemObjectFcnTy *)getAPIHandleCL(Handle, "clReleaseMemObject");

  clReleaseCommandQueueFcnPtr = (clReleaseCommandQueueFcnTy *)getAPIHandleCL(
      Handle, "clReleaseCommandQueue");

  clReleaseContextFcnPtr =
      (clReleaseContextFcnTy *)getAPIHandleCL(Handle, "clReleaseContext");

  return 1;
}
#pragma GCC diagnostic pop

/* Context and Device. */
static PollyGPUContext *GlobalContext = NULL;
static cl_device_id GlobalDeviceID = NULL;

/* Fd-Decl: Print out OpenCL Error codes to human readable strings. */
static void printOpenCLError(int Error);

static void checkOpenCLError(int Ret, const char *format, ...) {
  if (Ret == CL_SUCCESS)
    return;

  printOpenCLError(Ret);
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  exit(-1);
}

static PollyGPUContext *initContextCL() {
  dump_function();

  PollyGPUContext *Context;

  cl_platform_id PlatformID = NULL;
  cl_device_id DeviceID = NULL;
  cl_uint NumDevicesRet;
  cl_int Ret;

  char DeviceRevision[256];
  char DeviceName[256];
  size_t DeviceRevisionRetSize, DeviceNameRetSize;

  static __thread PollyGPUContext *CurrentContext = NULL;

  if (CurrentContext)
    return CurrentContext;

  /* Get API handles. */
  if (initialDeviceAPIsCL() == 0) {
    fprintf(stderr, "Getting the \"handle\" for the OpenCL Runtime failed.\n");
    exit(-1);
  }

  /* Get number of devices that support OpenCL. */
  static const int NumberOfPlatforms = 1;
  Ret = clGetPlatformIDsFcnPtr(NumberOfPlatforms, &PlatformID, NULL);
  checkOpenCLError(Ret, "Failed to get platform IDs.\n");
  // TODO: Extend to CL_DEVICE_TYPE_ALL?
  static const int NumberOfDevices = 1;
  Ret = clGetDeviceIDsFcnPtr(PlatformID, CL_DEVICE_TYPE_GPU, NumberOfDevices,
                             &DeviceID, &NumDevicesRet);
  checkOpenCLError(Ret, "Failed to get device IDs.\n");

  GlobalDeviceID = DeviceID;
  if (NumDevicesRet == 0) {
    fprintf(stderr, "There is no device supporting OpenCL.\n");
    exit(-1);
  }

  /* Get device revision. */
  Ret =
      clGetDeviceInfoFcnPtr(DeviceID, CL_DEVICE_VERSION, sizeof(DeviceRevision),
                            DeviceRevision, &DeviceRevisionRetSize);
  checkOpenCLError(Ret, "Failed to fetch device revision.\n");

  /* Get device name. */
  Ret = clGetDeviceInfoFcnPtr(DeviceID, CL_DEVICE_NAME, sizeof(DeviceName),
                              DeviceName, &DeviceNameRetSize);
  checkOpenCLError(Ret, "Failed to fetch device name.\n");

  debug_print("> Running on GPU device %d : %s.\n", DeviceID, DeviceName);

  /* Create context on the device. */
  Context = (PollyGPUContext *)malloc(sizeof(PollyGPUContext));
  if (Context == 0) {
    fprintf(stderr, "Allocate memory for Polly GPU context failed.\n");
    exit(-1);
  }
  Context->Context = (OpenCLContext *)malloc(sizeof(OpenCLContext));
  if (Context->Context == 0) {
    fprintf(stderr, "Allocate memory for Polly OpenCL context failed.\n");
    exit(-1);
  }
  ((OpenCLContext *)Context->Context)->Context =
      clCreateContextFcnPtr(NULL, NumDevicesRet, &DeviceID, NULL, NULL, &Ret);
  checkOpenCLError(Ret, "Failed to create context.\n");

  static const int ExtraProperties = 0;
  ((OpenCLContext *)Context->Context)->CommandQueue =
      clCreateCommandQueueFcnPtr(((OpenCLContext *)Context->Context)->Context,
                                 DeviceID, ExtraProperties, &Ret);
  checkOpenCLError(Ret, "Failed to create command queue.\n");

  if (CacheMode)
    CurrentContext = Context;

  GlobalContext = Context;
  return Context;
}

static void freeKernelCL(PollyGPUFunction *Kernel) {
  dump_function();

  if (CacheMode)
    return;

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  cl_int Ret;
  Ret = clFlushFcnPtr(((OpenCLContext *)GlobalContext->Context)->CommandQueue);
  checkOpenCLError(Ret, "Failed to flush command queue.\n");
  Ret = clFinishFcnPtr(((OpenCLContext *)GlobalContext->Context)->CommandQueue);
  checkOpenCLError(Ret, "Failed to finish command queue.\n");

  if (((OpenCLKernel *)Kernel->Kernel)->Kernel) {
    cl_int Ret =
        clReleaseKernelFcnPtr(((OpenCLKernel *)Kernel->Kernel)->Kernel);
    checkOpenCLError(Ret, "Failed to release kernel.\n");
  }

  if (((OpenCLKernel *)Kernel->Kernel)->Program) {
    cl_int Ret =
        clReleaseProgramFcnPtr(((OpenCLKernel *)Kernel->Kernel)->Program);
    checkOpenCLError(Ret, "Failed to release program.\n");
  }

  if (Kernel->Kernel)
    free((OpenCLKernel *)Kernel->Kernel);

  if (Kernel)
    free(Kernel);
}

static PollyGPUFunction *getKernelCL(const char *BinaryBuffer,
                                     const char *KernelName) {
  dump_function();

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  static __thread PollyGPUFunction *KernelCache[KERNEL_CACHE_SIZE];
  static __thread int NextCacheItem = 0;

  for (long i = 0; i < KERNEL_CACHE_SIZE; i++) {
    // We exploit here the property that all Polly-ACC kernels are allocated
    // as global constants, hence a pointer comparision is sufficient to
    // determin equality.
    if (KernelCache[i] &&
        ((OpenCLKernel *)KernelCache[i]->Kernel)->BinaryString ==
            BinaryBuffer) {
      debug_print("  -> using cached kernel\n");
      return KernelCache[i];
    }
  }

  PollyGPUFunction *Function = malloc(sizeof(PollyGPUFunction));
  if (Function == 0) {
    fprintf(stderr, "Allocate memory for Polly GPU function failed.\n");
    exit(-1);
  }
  Function->Kernel = (OpenCLKernel *)malloc(sizeof(OpenCLKernel));
  if (Function->Kernel == 0) {
    fprintf(stderr, "Allocate memory for Polly OpenCL kernel failed.\n");
    exit(-1);
  }

  if (!GlobalDeviceID) {
    fprintf(stderr, "GPGPU-code generation not initialized correctly.\n");
    exit(-1);
  }

  cl_int Ret;

  if (HandleOpenCLBeignet) {
    // TODO: This is a workaround, since clCreateProgramWithLLVMIntel only
    // accepts a filename to a valid llvm-ir file as an argument, instead
    // of accepting the BinaryBuffer directly.
    FILE *fp = fopen("kernel.ll", "wb");
    if (fp != NULL) {
      fputs(BinaryBuffer, fp);
      fclose(fp);
    }

    ((OpenCLKernel *)Function->Kernel)->Program =
        clCreateProgramWithLLVMIntelFcnPtr(
            ((OpenCLContext *)GlobalContext->Context)->Context, 1,
            &GlobalDeviceID, "kernel.ll", &Ret);
    checkOpenCLError(Ret, "Failed to create program from llvm.\n");
    unlink("kernel.ll");
  } else {
    size_t BinarySize = strlen(BinaryBuffer);
    ((OpenCLKernel *)Function->Kernel)->Program =
        clCreateProgramWithBinaryFcnPtr(
            ((OpenCLContext *)GlobalContext->Context)->Context, 1,
            &GlobalDeviceID, (const size_t *)&BinarySize,
            (const unsigned char **)&BinaryBuffer, NULL, &Ret);
    checkOpenCLError(Ret, "Failed to create program from binary.\n");
  }

  Ret = clBuildProgramFcnPtr(((OpenCLKernel *)Function->Kernel)->Program, 1,
                             &GlobalDeviceID, NULL, NULL, NULL);
  checkOpenCLError(Ret, "Failed to build program.\n");

  ((OpenCLKernel *)Function->Kernel)->Kernel = clCreateKernelFcnPtr(
      ((OpenCLKernel *)Function->Kernel)->Program, KernelName, &Ret);
  checkOpenCLError(Ret, "Failed to create kernel.\n");

  ((OpenCLKernel *)Function->Kernel)->BinaryString = BinaryBuffer;

  if (CacheMode) {
    if (KernelCache[NextCacheItem])
      freeKernelCL(KernelCache[NextCacheItem]);

    KernelCache[NextCacheItem] = Function;

    NextCacheItem = (NextCacheItem + 1) % KERNEL_CACHE_SIZE;
  }

  return Function;
}

static void copyFromHostToDeviceCL(void *HostData, PollyGPUDevicePtr *DevData,
                                   long MemSize) {
  dump_function();

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  cl_int Ret;
  Ret = clEnqueueWriteBufferFcnPtr(
      ((OpenCLContext *)GlobalContext->Context)->CommandQueue,
      ((OpenCLDevicePtr *)DevData->DevicePtr)->MemObj, CL_TRUE, 0, MemSize,
      HostData, 0, NULL, NULL);
  checkOpenCLError(Ret, "Copying data from host memory to device failed.\n");
}

static void copyFromDeviceToHostCL(PollyGPUDevicePtr *DevData, void *HostData,
                                   long MemSize) {
  dump_function();

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  cl_int Ret;
  Ret = clEnqueueReadBufferFcnPtr(
      ((OpenCLContext *)GlobalContext->Context)->CommandQueue,
      ((OpenCLDevicePtr *)DevData->DevicePtr)->MemObj, CL_TRUE, 0, MemSize,
      HostData, 0, NULL, NULL);
  checkOpenCLError(Ret, "Copying results from device to host memory failed.\n");
}

static void launchKernelCL(PollyGPUFunction *Kernel, unsigned int GridDimX,
                           unsigned int GridDimY, unsigned int BlockDimX,
                           unsigned int BlockDimY, unsigned int BlockDimZ,
                           void **Parameters) {
  dump_function();

  cl_int Ret;
  cl_uint NumArgs;

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  OpenCLKernel *CLKernel = (OpenCLKernel *)Kernel->Kernel;
  Ret = clGetKernelInfoFcnPtr(CLKernel->Kernel, CL_KERNEL_NUM_ARGS,
                              sizeof(cl_uint), &NumArgs, NULL);
  checkOpenCLError(Ret, "Failed to get number of kernel arguments.\n");

  /* Argument sizes are stored at the end of the Parameters array. */
  for (cl_uint i = 0; i < NumArgs; i++) {
    Ret = clSetKernelArgFcnPtr(CLKernel->Kernel, i,
                               *((int *)Parameters[NumArgs + i]),
                               (void *)Parameters[i]);
    checkOpenCLError(Ret, "Failed to set Kernel argument %d.\n", i);
  }

  unsigned int GridDimZ = 1;
  size_t GlobalWorkSize[3] = {BlockDimX * GridDimX, BlockDimY * GridDimY,
                              BlockDimZ * GridDimZ};
  size_t LocalWorkSize[3] = {BlockDimX, BlockDimY, BlockDimZ};

  static const int WorkDim = 3;
  OpenCLContext *CLContext = (OpenCLContext *)GlobalContext->Context;
  Ret = clEnqueueNDRangeKernelFcnPtr(CLContext->CommandQueue, CLKernel->Kernel,
                                     WorkDim, NULL, GlobalWorkSize,
                                     LocalWorkSize, 0, NULL, NULL);
  checkOpenCLError(Ret, "Launching OpenCL kernel failed.\n");
}

static void freeDeviceMemoryCL(PollyGPUDevicePtr *Allocation) {
  dump_function();

  OpenCLDevicePtr *DevPtr = (OpenCLDevicePtr *)Allocation->DevicePtr;
  cl_int Ret = clReleaseMemObjectFcnPtr((cl_mem)DevPtr->MemObj);
  checkOpenCLError(Ret, "Failed to free device memory.\n");

  free(DevPtr);
  free(Allocation);
}

static PollyGPUDevicePtr *allocateMemoryForDeviceCL(long MemSize) {
  dump_function();

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  PollyGPUDevicePtr *DevData = malloc(sizeof(PollyGPUDevicePtr));
  if (DevData == 0) {
    fprintf(stderr, "Allocate memory for GPU device memory pointer failed.\n");
    exit(-1);
  }
  DevData->DevicePtr = (OpenCLDevicePtr *)malloc(sizeof(OpenCLDevicePtr));
  if (DevData->DevicePtr == 0) {
    fprintf(stderr, "Allocate memory for GPU device memory pointer failed.\n");
    exit(-1);
  }

  cl_int Ret;
  ((OpenCLDevicePtr *)DevData->DevicePtr)->MemObj =
      clCreateBufferFcnPtr(((OpenCLContext *)GlobalContext->Context)->Context,
                           CL_MEM_READ_WRITE, MemSize, NULL, &Ret);
  checkOpenCLError(Ret,
                   "Allocate memory for GPU device memory pointer failed.\n");

  return DevData;
}

static void *getDevicePtrCL(PollyGPUDevicePtr *Allocation) {
  dump_function();

  OpenCLDevicePtr *DevPtr = (OpenCLDevicePtr *)Allocation->DevicePtr;
  return (void *)DevPtr->MemObj;
}

static void synchronizeDeviceCL() {
  dump_function();

  if (!GlobalContext) {
    fprintf(stderr, "GPGPU-code generation not correctly initialized.\n");
    exit(-1);
  }

  if (clFinishFcnPtr(((OpenCLContext *)GlobalContext->Context)->CommandQueue) !=
      CL_SUCCESS) {
    fprintf(stderr, "Synchronizing device and host memory failed.\n");
    exit(-1);
  }
}

static void freeContextCL(PollyGPUContext *Context) {
  dump_function();

  cl_int Ret;

  GlobalContext = NULL;

  OpenCLContext *Ctx = (OpenCLContext *)Context->Context;
  if (Ctx->CommandQueue) {
    Ret = clReleaseCommandQueueFcnPtr(Ctx->CommandQueue);
    checkOpenCLError(Ret, "Could not release command queue.\n");
  }

  if (Ctx->Context) {
    Ret = clReleaseContextFcnPtr(Ctx->Context);
    checkOpenCLError(Ret, "Could not release context.\n");
  }

  free(Ctx);
  free(Context);
}

static void printOpenCLError(int Error) {

  switch (Error) {
  case CL_SUCCESS:
    // Success, don't print an error.
    break;

  // JIT/Runtime errors.
  case CL_DEVICE_NOT_FOUND:
    fprintf(stderr, "Device not found.\n");
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    fprintf(stderr, "Device not available.\n");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    fprintf(stderr, "Compiler not available.\n");
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    fprintf(stderr, "Mem object allocation failure.\n");
    break;
  case CL_OUT_OF_RESOURCES:
    fprintf(stderr, "Out of resources.\n");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "Out of host memory.\n");
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    fprintf(stderr, "Profiling info not available.\n");
    break;
  case CL_MEM_COPY_OVERLAP:
    fprintf(stderr, "Mem copy overlap.\n");
    break;
  case CL_IMAGE_FORMAT_MISMATCH:
    fprintf(stderr, "Image format mismatch.\n");
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    fprintf(stderr, "Image format not supported.\n");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    fprintf(stderr, "Build program failure.\n");
    break;
  case CL_MAP_FAILURE:
    fprintf(stderr, "Map failure.\n");
    break;
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    fprintf(stderr, "Misaligned sub buffer offset.\n");
    break;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    fprintf(stderr, "Exec status error for events in wait list.\n");
    break;
  case CL_COMPILE_PROGRAM_FAILURE:
    fprintf(stderr, "Compile program failure.\n");
    break;
  case CL_LINKER_NOT_AVAILABLE:
    fprintf(stderr, "Linker not available.\n");
    break;
  case CL_LINK_PROGRAM_FAILURE:
    fprintf(stderr, "Link program failure.\n");
    break;
  case CL_DEVICE_PARTITION_FAILED:
    fprintf(stderr, "Device partition failed.\n");
    break;
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    fprintf(stderr, "Kernel arg info not available.\n");
    break;

  // Compiler errors.
  case CL_INVALID_VALUE:
    fprintf(stderr, "Invalid value.\n");
    break;
  case CL_INVALID_DEVICE_TYPE:
    fprintf(stderr, "Invalid device type.\n");
    break;
  case CL_INVALID_PLATFORM:
    fprintf(stderr, "Invalid platform.\n");
    break;
  case CL_INVALID_DEVICE:
    fprintf(stderr, "Invalid device.\n");
    break;
  case CL_INVALID_CONTEXT:
    fprintf(stderr, "Invalid context.\n");
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    fprintf(stderr, "Invalid queue properties.\n");
    break;
  case CL_INVALID_COMMAND_QUEUE:
    fprintf(stderr, "Invalid command queue.\n");
    break;
  case CL_INVALID_HOST_PTR:
    fprintf(stderr, "Invalid host pointer.\n");
    break;
  case CL_INVALID_MEM_OBJECT:
    fprintf(stderr, "Invalid memory object.\n");
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    fprintf(stderr, "Invalid image format descriptor.\n");
    break;
  case CL_INVALID_IMAGE_SIZE:
    fprintf(stderr, "Invalid image size.\n");
    break;
  case CL_INVALID_SAMPLER:
    fprintf(stderr, "Invalid sampler.\n");
    break;
  case CL_INVALID_BINARY:
    fprintf(stderr, "Invalid binary.\n");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    fprintf(stderr, "Invalid build options.\n");
    break;
  case CL_INVALID_PROGRAM:
    fprintf(stderr, "Invalid program.\n");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    fprintf(stderr, "Invalid program executable.\n");
    break;
  case CL_INVALID_KERNEL_NAME:
    fprintf(stderr, "Invalid kernel name.\n");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    fprintf(stderr, "Invalid kernel definition.\n");
    break;
  case CL_INVALID_KERNEL:
    fprintf(stderr, "Invalid kernel.\n");
    break;
  case CL_INVALID_ARG_INDEX:
    fprintf(stderr, "Invalid arg index.\n");
    break;
  case CL_INVALID_ARG_VALUE:
    fprintf(stderr, "Invalid arg value.\n");
    break;
  case CL_INVALID_ARG_SIZE:
    fprintf(stderr, "Invalid arg size.\n");
    break;
  case CL_INVALID_KERNEL_ARGS:
    fprintf(stderr, "Invalid kernel args.\n");
    break;
  case CL_INVALID_WORK_DIMENSION:
    fprintf(stderr, "Invalid work dimension.\n");
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    fprintf(stderr, "Invalid work group size.\n");
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    fprintf(stderr, "Invalid work item size.\n");
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    fprintf(stderr, "Invalid global offset.\n");
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    fprintf(stderr, "Invalid event wait list.\n");
    break;
  case CL_INVALID_EVENT:
    fprintf(stderr, "Invalid event.\n");
    break;
  case CL_INVALID_OPERATION:
    fprintf(stderr, "Invalid operation.\n");
    break;
  case CL_INVALID_GL_OBJECT:
    fprintf(stderr, "Invalid GL object.\n");
    break;
  case CL_INVALID_BUFFER_SIZE:
    fprintf(stderr, "Invalid buffer size.\n");
    break;
  case CL_INVALID_MIP_LEVEL:
    fprintf(stderr, "Invalid mip level.\n");
    break;
  case CL_INVALID_GLOBAL_WORK_SIZE:
    fprintf(stderr, "Invalid global work size.\n");
    break;
  case CL_INVALID_PROPERTY:
    fprintf(stderr, "Invalid property.\n");
    break;
  case CL_INVALID_IMAGE_DESCRIPTOR:
    fprintf(stderr, "Invalid image descriptor.\n");
    break;
  case CL_INVALID_COMPILER_OPTIONS:
    fprintf(stderr, "Invalid compiler options.\n");
    break;
  case CL_INVALID_LINKER_OPTIONS:
    fprintf(stderr, "Invalid linker options.\n");
    break;
  case CL_INVALID_DEVICE_PARTITION_COUNT:
    fprintf(stderr, "Invalid device partition count.\n");
    break;
  case -69: // OpenCL 2.0 Code for CL_INVALID_PIPE_SIZE
    fprintf(stderr, "Invalid pipe size.\n");
    break;
  case -70: // OpenCL 2.0 Code for CL_INVALID_DEVICE_QUEUE
    fprintf(stderr, "Invalid device queue.\n");
    break;

  // NVIDIA specific error.
  case -9999:
    fprintf(stderr, "NVIDIA invalid read or write buffer.\n");
    break;

  default:
    fprintf(stderr, "Unknown error code!\n");
    break;
  }
}

#endif /* HAS_LIBOPENCL */
/******************************************************************************/
/*                                   CUDA                                     */
/******************************************************************************/
#ifdef HAS_LIBCUDART

struct CUDAContextT {
  CUcontext Cuda;
};

struct CUDAKernelT {
  CUfunction Cuda;
  CUmodule CudaModule;
  const char *BinaryString;
};

struct CUDADevicePtrT {
  CUdeviceptr Cuda;
};

/* Dynamic library handles for the CUDA and CUDA runtime library. */
static void *HandleCuda;
static void *HandleCudaRT;

/* Type-defines of function pointer to CUDA driver APIs. */
typedef CUresult CUDAAPI CuMemAllocFcnTy(CUdeviceptr *, size_t);
static CuMemAllocFcnTy *CuMemAllocFcnPtr;

typedef CUresult CUDAAPI CuMemAllocManagedFcnTy(CUdeviceptr *, size_t,
                                                unsigned int);
static CuMemAllocManagedFcnTy *CuMemAllocManagedFcnPtr;

typedef CUresult CUDAAPI CuLaunchKernelFcnTy(
    CUfunction F, unsigned int GridDimX, unsigned int GridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int BlockDimY,
    unsigned int BlockDimZ, unsigned int SharedMemBytes, CUstream HStream,
    void **KernelParams, void **Extra);
static CuLaunchKernelFcnTy *CuLaunchKernelFcnPtr;

typedef CUresult CUDAAPI CuMemcpyDtoHFcnTy(void *, CUdeviceptr, size_t);
static CuMemcpyDtoHFcnTy *CuMemcpyDtoHFcnPtr;

typedef CUresult CUDAAPI CuMemcpyHtoDFcnTy(CUdeviceptr, const void *, size_t);
static CuMemcpyHtoDFcnTy *CuMemcpyHtoDFcnPtr;

typedef CUresult CUDAAPI CuMemFreeFcnTy(CUdeviceptr);
static CuMemFreeFcnTy *CuMemFreeFcnPtr;

typedef CUresult CUDAAPI CuModuleUnloadFcnTy(CUmodule);
static CuModuleUnloadFcnTy *CuModuleUnloadFcnPtr;

typedef CUresult CUDAAPI CuProfilerStopFcnTy();
static CuProfilerStopFcnTy *CuProfilerStopFcnPtr;

typedef CUresult CUDAAPI CuCtxDestroyFcnTy(CUcontext);
static CuCtxDestroyFcnTy *CuCtxDestroyFcnPtr;

typedef CUresult CUDAAPI CuInitFcnTy(unsigned int);
static CuInitFcnTy *CuInitFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetCountFcnTy(int *);
static CuDeviceGetCountFcnTy *CuDeviceGetCountFcnPtr;

typedef CUresult CUDAAPI CuCtxCreateFcnTy(CUcontext *, unsigned int, CUdevice);
static CuCtxCreateFcnTy *CuCtxCreateFcnPtr;

typedef CUresult CUDAAPI CuCtxGetCurrentFcnTy(CUcontext *);
static CuCtxGetCurrentFcnTy *CuCtxGetCurrentFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetFcnTy(CUdevice *, int);
static CuDeviceGetFcnTy *CuDeviceGetFcnPtr;

typedef CUresult CUDAAPI CuModuleLoadDataExFcnTy(CUmodule *, const void *,
                                                 unsigned int, CUjit_option *,
                                                 void **);
static CuModuleLoadDataExFcnTy *CuModuleLoadDataExFcnPtr;

typedef CUresult CUDAAPI CuModuleLoadDataFcnTy(CUmodule *Module,
                                               const void *Image);
static CuModuleLoadDataFcnTy *CuModuleLoadDataFcnPtr;

typedef CUresult CUDAAPI CuModuleGetFunctionFcnTy(CUfunction *, CUmodule,
                                                  const char *);
static CuModuleGetFunctionFcnTy *CuModuleGetFunctionFcnPtr;

typedef CUresult CUDAAPI CuDeviceComputeCapabilityFcnTy(int *, int *, CUdevice);
static CuDeviceComputeCapabilityFcnTy *CuDeviceComputeCapabilityFcnPtr;

typedef CUresult CUDAAPI CuDeviceGetNameFcnTy(char *, int, CUdevice);
static CuDeviceGetNameFcnTy *CuDeviceGetNameFcnPtr;

typedef CUresult CUDAAPI CuLinkAddDataFcnTy(CUlinkState State,
                                            CUjitInputType Type, void *Data,
                                            size_t Size, const char *Name,
                                            unsigned int NumOptions,
                                            CUjit_option *Options,
                                            void **OptionValues);
static CuLinkAddDataFcnTy *CuLinkAddDataFcnPtr;

typedef CUresult CUDAAPI CuLinkCreateFcnTy(unsigned int NumOptions,
                                           CUjit_option *Options,
                                           void **OptionValues,
                                           CUlinkState *StateOut);
static CuLinkCreateFcnTy *CuLinkCreateFcnPtr;

typedef CUresult CUDAAPI CuLinkCompleteFcnTy(CUlinkState State, void **CubinOut,
                                             size_t *SizeOut);
static CuLinkCompleteFcnTy *CuLinkCompleteFcnPtr;

typedef CUresult CUDAAPI CuLinkDestroyFcnTy(CUlinkState State);
static CuLinkDestroyFcnTy *CuLinkDestroyFcnPtr;

typedef CUresult CUDAAPI CuCtxSynchronizeFcnTy();
static CuCtxSynchronizeFcnTy *CuCtxSynchronizeFcnPtr;

/* Type-defines of function pointer ot CUDA runtime APIs. */
typedef cudaError_t CUDARTAPI CudaThreadSynchronizeFcnTy(void);
static CudaThreadSynchronizeFcnTy *CudaThreadSynchronizeFcnPtr;

static void *getAPIHandleCUDA(void *Handle, const char *FuncName) {
  char *Err;
  void *FuncPtr;
  dlerror();
  FuncPtr = dlsym(Handle, FuncName);
  if ((Err = dlerror()) != 0) {
    fprintf(stderr, "Load CUDA driver API failed: %s. \n", Err);
    return 0;
  }
  return FuncPtr;
}

static int initialDeviceAPILibrariesCUDA() {
  HandleCuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!HandleCuda) {
    fprintf(stderr, "Cannot open library: %s. \n", dlerror());
    return 0;
  }

  HandleCudaRT = dlopen("libcudart.so", RTLD_LAZY);
  if (!HandleCudaRT) {
    fprintf(stderr, "Cannot open library: %s. \n", dlerror());
    return 0;
  }

  return 1;
}

/* Get function pointer to CUDA Driver APIs.
 *
 * Note that compilers conforming to the ISO C standard are required to
 * generate a warning if a conversion from a void * pointer to a function
 * pointer is attempted as in the following statements. The warning
 * of this kind of cast may not be emitted by clang and new versions of gcc
 * as it is valid on POSIX 2008. For compilers required to generate a warning,
 * we temporarily disable -Wpedantic, to avoid bloating the output with
 * unnecessary warnings.
 *
 * Reference:
 * http://pubs.opengroup.org/onlinepubs/9699919799/functions/dlsym.html
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
static int initialDeviceAPIsCUDA() {
  if (initialDeviceAPILibrariesCUDA() == 0)
    return 0;

  CuLaunchKernelFcnPtr =
      (CuLaunchKernelFcnTy *)getAPIHandleCUDA(HandleCuda, "cuLaunchKernel");

  CuMemAllocFcnPtr =
      (CuMemAllocFcnTy *)getAPIHandleCUDA(HandleCuda, "cuMemAlloc_v2");

  CuMemAllocManagedFcnPtr = (CuMemAllocManagedFcnTy *)getAPIHandleCUDA(
      HandleCuda, "cuMemAllocManaged");

  CuMemFreeFcnPtr =
      (CuMemFreeFcnTy *)getAPIHandleCUDA(HandleCuda, "cuMemFree_v2");

  CuMemcpyDtoHFcnPtr =
      (CuMemcpyDtoHFcnTy *)getAPIHandleCUDA(HandleCuda, "cuMemcpyDtoH_v2");

  CuMemcpyHtoDFcnPtr =
      (CuMemcpyHtoDFcnTy *)getAPIHandleCUDA(HandleCuda, "cuMemcpyHtoD_v2");

  CuModuleUnloadFcnPtr =
      (CuModuleUnloadFcnTy *)getAPIHandleCUDA(HandleCuda, "cuModuleUnload");

  CuProfilerStopFcnPtr =
      (CuProfilerStopFcnTy *)getAPIHandleCUDA(HandleCuda, "cuProfilerStop");

  CuCtxDestroyFcnPtr =
      (CuCtxDestroyFcnTy *)getAPIHandleCUDA(HandleCuda, "cuCtxDestroy");

  CuInitFcnPtr = (CuInitFcnTy *)getAPIHandleCUDA(HandleCuda, "cuInit");

  CuDeviceGetCountFcnPtr =
      (CuDeviceGetCountFcnTy *)getAPIHandleCUDA(HandleCuda, "cuDeviceGetCount");

  CuDeviceGetFcnPtr =
      (CuDeviceGetFcnTy *)getAPIHandleCUDA(HandleCuda, "cuDeviceGet");

  CuCtxCreateFcnPtr =
      (CuCtxCreateFcnTy *)getAPIHandleCUDA(HandleCuda, "cuCtxCreate_v2");

  CuCtxGetCurrentFcnPtr =
      (CuCtxGetCurrentFcnTy *)getAPIHandleCUDA(HandleCuda, "cuCtxGetCurrent");

  CuModuleLoadDataExFcnPtr = (CuModuleLoadDataExFcnTy *)getAPIHandleCUDA(
      HandleCuda, "cuModuleLoadDataEx");

  CuModuleLoadDataFcnPtr =
      (CuModuleLoadDataFcnTy *)getAPIHandleCUDA(HandleCuda, "cuModuleLoadData");

  CuModuleGetFunctionFcnPtr = (CuModuleGetFunctionFcnTy *)getAPIHandleCUDA(
      HandleCuda, "cuModuleGetFunction");

  CuDeviceComputeCapabilityFcnPtr =
      (CuDeviceComputeCapabilityFcnTy *)getAPIHandleCUDA(
          HandleCuda, "cuDeviceComputeCapability");

  CuDeviceGetNameFcnPtr =
      (CuDeviceGetNameFcnTy *)getAPIHandleCUDA(HandleCuda, "cuDeviceGetName");

  CuLinkAddDataFcnPtr =
      (CuLinkAddDataFcnTy *)getAPIHandleCUDA(HandleCuda, "cuLinkAddData");

  CuLinkCreateFcnPtr =
      (CuLinkCreateFcnTy *)getAPIHandleCUDA(HandleCuda, "cuLinkCreate");

  CuLinkCompleteFcnPtr =
      (CuLinkCompleteFcnTy *)getAPIHandleCUDA(HandleCuda, "cuLinkComplete");

  CuLinkDestroyFcnPtr =
      (CuLinkDestroyFcnTy *)getAPIHandleCUDA(HandleCuda, "cuLinkDestroy");

  CuCtxSynchronizeFcnPtr =
      (CuCtxSynchronizeFcnTy *)getAPIHandleCUDA(HandleCuda, "cuCtxSynchronize");

  /* Get function pointer to CUDA Runtime APIs. */
  CudaThreadSynchronizeFcnPtr = (CudaThreadSynchronizeFcnTy *)getAPIHandleCUDA(
      HandleCudaRT, "cudaThreadSynchronize");

  return 1;
}
#pragma GCC diagnostic pop

static PollyGPUContext *initContextCUDA() {
  dump_function();
  PollyGPUContext *Context;
  CUdevice Device;

  int Major = 0, Minor = 0, DeviceID = 0;
  char DeviceName[256];
  int DeviceCount = 0;

  static __thread PollyGPUContext *CurrentContext = NULL;

  if (CurrentContext)
    return CurrentContext;

  /* Get API handles. */
  if (initialDeviceAPIsCUDA() == 0) {
    fprintf(stderr, "Getting the \"handle\" for the CUDA driver API failed.\n");
    exit(-1);
  }

  if (CuInitFcnPtr(0) != CUDA_SUCCESS) {
    fprintf(stderr, "Initializing the CUDA driver API failed.\n");
    exit(-1);
  }

  /* Get number of devices that supports CUDA. */
  CuDeviceGetCountFcnPtr(&DeviceCount);
  if (DeviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA.\n");
    exit(-1);
  }

  CuDeviceGetFcnPtr(&Device, 0);

  /* Get compute capabilities and the device name. */
  CuDeviceComputeCapabilityFcnPtr(&Major, &Minor, Device);
  CuDeviceGetNameFcnPtr(DeviceName, 256, Device);
  debug_print("> Running on GPU device %d : %s.\n", DeviceID, DeviceName);

  /* Create context on the device. */
  Context = (PollyGPUContext *)malloc(sizeof(PollyGPUContext));
  if (Context == 0) {
    fprintf(stderr, "Allocate memory for Polly GPU context failed.\n");
    exit(-1);
  }
  Context->Context = malloc(sizeof(CUDAContext));
  if (Context->Context == 0) {
    fprintf(stderr, "Allocate memory for Polly CUDA context failed.\n");
    exit(-1);
  }

  // In cases where managed memory is used, it is quite likely that
  // `cudaMallocManaged` / `polly_mallocManaged` was called before
  // `polly_initContext` was called.
  //
  // If `polly_initContext` calls `CuCtxCreate` when there already was a
  // pre-existing context created by the runtime API, this causes code running
  // on P100 to hang. So, we query for a pre-existing context to try and use.
  // If there is no pre-existing context, we create a new context

  // The possible pre-existing context from previous runtime API calls.
  CUcontext MaybeRuntimeAPIContext;
  if (CuCtxGetCurrentFcnPtr(&MaybeRuntimeAPIContext) != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxGetCurrent failed.\n");
    exit(-1);
  }

  // There was no previous context, initialise it.
  if (MaybeRuntimeAPIContext == NULL) {
    if (CuCtxCreateFcnPtr(&(((CUDAContext *)Context->Context)->Cuda), 0,
                          Device) != CUDA_SUCCESS) {
      fprintf(stderr, "cuCtxCreateFcnPtr failed.\n");
      exit(-1);
    }
  } else {
    ((CUDAContext *)Context->Context)->Cuda = MaybeRuntimeAPIContext;
  }

  if (CacheMode)
    CurrentContext = Context;

  return Context;
}

static void freeKernelCUDA(PollyGPUFunction *Kernel) {
  dump_function();

  if (CacheMode)
    return;

  if (((CUDAKernel *)Kernel->Kernel)->CudaModule)
    CuModuleUnloadFcnPtr(((CUDAKernel *)Kernel->Kernel)->CudaModule);

  if (Kernel->Kernel)
    free((CUDAKernel *)Kernel->Kernel);

  if (Kernel)
    free(Kernel);
}

static PollyGPUFunction *getKernelCUDA(const char *BinaryBuffer,
                                       const char *KernelName) {
  dump_function();

  static __thread PollyGPUFunction *KernelCache[KERNEL_CACHE_SIZE];
  static __thread int NextCacheItem = 0;

  for (long i = 0; i < KERNEL_CACHE_SIZE; i++) {
    // We exploit here the property that all Polly-ACC kernels are allocated
    // as global constants, hence a pointer comparision is sufficient to
    // determin equality.
    if (KernelCache[i] &&
        ((CUDAKernel *)KernelCache[i]->Kernel)->BinaryString == BinaryBuffer) {
      debug_print("  -> using cached kernel\n");
      return KernelCache[i];
    }
  }

  PollyGPUFunction *Function = malloc(sizeof(PollyGPUFunction));
  if (Function == 0) {
    fprintf(stderr, "Allocate memory for Polly GPU function failed.\n");
    exit(-1);
  }
  Function->Kernel = (CUDAKernel *)malloc(sizeof(CUDAKernel));
  if (Function->Kernel == 0) {
    fprintf(stderr, "Allocate memory for Polly CUDA function failed.\n");
    exit(-1);
  }

  CUresult Res;
  CUlinkState LState;
  CUjit_option Options[6];
  void *OptionVals[6];
  float Walltime = 0;
  unsigned long LogSize = 8192;
  char ErrorLog[8192], InfoLog[8192];
  void *CuOut;
  size_t OutSize;

  // Setup linker options
  // Return walltime from JIT compilation
  Options[0] = CU_JIT_WALL_TIME;
  OptionVals[0] = (void *)&Walltime;
  // Pass a buffer for info messages
  Options[1] = CU_JIT_INFO_LOG_BUFFER;
  OptionVals[1] = (void *)InfoLog;
  // Pass the size of the info buffer
  Options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  OptionVals[2] = (void *)LogSize;
  // Pass a buffer for error message
  Options[3] = CU_JIT_ERROR_LOG_BUFFER;
  OptionVals[3] = (void *)ErrorLog;
  // Pass the size of the error buffer
  Options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  OptionVals[4] = (void *)LogSize;
  // Make the linker verbose
  Options[5] = CU_JIT_LOG_VERBOSE;
  OptionVals[5] = (void *)1;

  memset(ErrorLog, 0, sizeof(ErrorLog));

  CuLinkCreateFcnPtr(6, Options, OptionVals, &LState);
  Res = CuLinkAddDataFcnPtr(LState, CU_JIT_INPUT_PTX, (void *)BinaryBuffer,
                            strlen(BinaryBuffer) + 1, 0, 0, 0, 0);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "PTX Linker Error:\n%s\n%s", ErrorLog, InfoLog);
    exit(-1);
  }

  Res = CuLinkCompleteFcnPtr(LState, &CuOut, &OutSize);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "Complete ptx linker step failed.\n");
    fprintf(stderr, "\n%s\n", ErrorLog);
    exit(-1);
  }

  debug_print("CUDA Link Completed in %fms. Linker Output:\n%s\n", Walltime,
              InfoLog);

  Res = CuModuleLoadDataFcnPtr(&(((CUDAKernel *)Function->Kernel)->CudaModule),
                               CuOut);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "Loading ptx assembly text failed.\n");
    exit(-1);
  }

  Res = CuModuleGetFunctionFcnPtr(&(((CUDAKernel *)Function->Kernel)->Cuda),
                                  ((CUDAKernel *)Function->Kernel)->CudaModule,
                                  KernelName);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "Loading kernel function failed.\n");
    exit(-1);
  }

  CuLinkDestroyFcnPtr(LState);

  ((CUDAKernel *)Function->Kernel)->BinaryString = BinaryBuffer;

  if (CacheMode) {
    if (KernelCache[NextCacheItem])
      freeKernelCUDA(KernelCache[NextCacheItem]);

    KernelCache[NextCacheItem] = Function;

    NextCacheItem = (NextCacheItem + 1) % KERNEL_CACHE_SIZE;
  }

  return Function;
}

static void synchronizeDeviceCUDA() {
  dump_function();
  if (CuCtxSynchronizeFcnPtr() != CUDA_SUCCESS) {
    fprintf(stderr, "Synchronizing device and host memory failed.\n");
    exit(-1);
  }
}

static void copyFromHostToDeviceCUDA(void *HostData, PollyGPUDevicePtr *DevData,
                                     long MemSize) {
  dump_function();

  CUdeviceptr CuDevData = ((CUDADevicePtr *)DevData->DevicePtr)->Cuda;
  CuMemcpyHtoDFcnPtr(CuDevData, HostData, MemSize);
}

static void copyFromDeviceToHostCUDA(PollyGPUDevicePtr *DevData, void *HostData,
                                     long MemSize) {
  dump_function();

  if (CuMemcpyDtoHFcnPtr(HostData, ((CUDADevicePtr *)DevData->DevicePtr)->Cuda,
                         MemSize) != CUDA_SUCCESS) {
    fprintf(stderr, "Copying results from device to host memory failed.\n");
    exit(-1);
  }
}

static void launchKernelCUDA(PollyGPUFunction *Kernel, unsigned int GridDimX,
                             unsigned int GridDimY, unsigned int BlockDimX,
                             unsigned int BlockDimY, unsigned int BlockDimZ,
                             void **Parameters) {
  dump_function();

  unsigned GridDimZ = 1;
  unsigned int SharedMemBytes = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
  CUstream Stream = 0;
  void **Extra = 0;

  CUresult Res;
  Res =
      CuLaunchKernelFcnPtr(((CUDAKernel *)Kernel->Kernel)->Cuda, GridDimX,
                           GridDimY, GridDimZ, BlockDimX, BlockDimY, BlockDimZ,
                           SharedMemBytes, Stream, Parameters, Extra);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "Launching CUDA kernel failed.\n");
    exit(-1);
  }
}

// Maximum number of managed memory pointers.
#define DEFAULT_MAX_POINTERS 4000
// For the rationale behing a list of free pointers, see `polly_freeManaged`.
void **g_managedptrs;
unsigned long long g_nmanagedptrs = 0;
unsigned long long g_maxmanagedptrs = 0;

__attribute__((constructor)) static void initManagedPtrsBuffer() {
  g_maxmanagedptrs = DEFAULT_MAX_POINTERS;
  const char *maxManagedPointersString = getenv("POLLY_MAX_MANAGED_POINTERS");
  if (maxManagedPointersString)
    g_maxmanagedptrs = atoll(maxManagedPointersString);

  g_managedptrs = (void **)malloc(sizeof(void *) * g_maxmanagedptrs);
}

// Add a pointer as being allocated by cuMallocManaged
void addManagedPtr(void *mem) {
  assert(g_maxmanagedptrs > 0 && "g_maxmanagedptrs was set to 0!");
  assert(g_nmanagedptrs < g_maxmanagedptrs &&
         "We have hit the maximum number of "
         "managed pointers allowed. Set the "
         "POLLY_MAX_MANAGED_POINTERS environment variable. ");
  g_managedptrs[g_nmanagedptrs++] = mem;
}

int isManagedPtr(void *mem) {
  for (unsigned long long i = 0; i < g_nmanagedptrs; i++) {
    if (g_managedptrs[i] == mem)
      return 1;
  }
  return 0;
}

void polly_freeManaged(void *mem) {
  dump_function();

  // In a real-world program this was used (COSMO), there were more `free`
  // calls in the original source than `malloc` calls. Hence, replacing all
  // `free`s with `cudaFree` does not work, since we would try to free
  // 'illegal' memory.
  // As a quick fix, we keep a free list and check if `mem` is a managed memory
  // pointer. If it is, we call `cudaFree`.
  // If not, we pass it along to the underlying allocator.
  // This is a hack, and can be removed if the underlying issue is fixed.
  if (isManagedPtr(mem)) {
    if (CuMemFreeFcnPtr((size_t)mem) != CUDA_SUCCESS) {
      fprintf(stderr, "cudaFree failed.\n");
      exit(-1);
    }
    return;
  } else {
    free(mem);
  }
}

void *polly_mallocManaged(size_t size) {
  // Note: [Size 0 allocations]
  // Sometimes, some runtime computation of size could create a size of 0
  // for an allocation. In these cases, we do not wish to fail.
  // The CUDA API fails on size 0 allocations.
  // So, we allocate size a minimum of size 1.
  if (!size && DebugMode)
    fprintf(stderr, "cudaMallocManaged called with size 0. "
                    "Promoting to size 1");
  size = max(size, 1);
  PollyGPUContext *_ = polly_initContextCUDA();
  assert(_ && "polly_initContextCUDA failed");

  void *newMemPtr;
  const CUresult Res = CuMemAllocManagedFcnPtr((CUdeviceptr *)&newMemPtr, size,
                                               CU_MEM_ATTACH_GLOBAL);
  if (Res != CUDA_SUCCESS) {
    fprintf(stderr, "cudaMallocManaged failed for size: %zu\n", size);
    exit(-1);
  }
  addManagedPtr(newMemPtr);
  return newMemPtr;
}

static void freeDeviceMemoryCUDA(PollyGPUDevicePtr *Allocation) {
  dump_function();
  CUDADevicePtr *DevPtr = (CUDADevicePtr *)Allocation->DevicePtr;
  CuMemFreeFcnPtr((CUdeviceptr)DevPtr->Cuda);
  free(DevPtr);
  free(Allocation);
}

static PollyGPUDevicePtr *allocateMemoryForDeviceCUDA(long MemSize) {
  if (!MemSize && DebugMode)
    fprintf(stderr, "allocateMemoryForDeviceCUDA called with size 0. "
                    "Promoting to size 1");
  // see: [Size 0 allocations]
  MemSize = max(MemSize, 1);
  dump_function();

  PollyGPUDevicePtr *DevData = malloc(sizeof(PollyGPUDevicePtr));
  if (DevData == 0) {
    fprintf(stderr,
            "Allocate memory for GPU device memory pointer failed."
            " Line: %d | Size: %ld\n",
            __LINE__, MemSize);
    exit(-1);
  }
  DevData->DevicePtr = (CUDADevicePtr *)malloc(sizeof(CUDADevicePtr));
  if (DevData->DevicePtr == 0) {
    fprintf(stderr,
            "Allocate memory for GPU device memory pointer failed."
            " Line: %d | Size: %ld\n",
            __LINE__, MemSize);
    exit(-1);
  }

  CUresult Res =
      CuMemAllocFcnPtr(&(((CUDADevicePtr *)DevData->DevicePtr)->Cuda), MemSize);

  if (Res != CUDA_SUCCESS) {
    fprintf(stderr,
            "Allocate memory for GPU device memory pointer failed."
            " Line: %d | Size: %ld\n",
            __LINE__, MemSize);
    exit(-1);
  }

  return DevData;
}

static void *getDevicePtrCUDA(PollyGPUDevicePtr *Allocation) {
  dump_function();

  CUDADevicePtr *DevPtr = (CUDADevicePtr *)Allocation->DevicePtr;
  return (void *)DevPtr->Cuda;
}

static void freeContextCUDA(PollyGPUContext *Context) {
  dump_function();

  CUDAContext *Ctx = (CUDAContext *)Context->Context;
  if (Ctx->Cuda) {
    CuProfilerStopFcnPtr();
    CuCtxDestroyFcnPtr(Ctx->Cuda);
    free(Ctx);
    free(Context);
  }

  dlclose(HandleCuda);
  dlclose(HandleCudaRT);
}

#endif /* HAS_LIBCUDART */
/******************************************************************************/
/*                                    API                                     */
/******************************************************************************/

PollyGPUContext *polly_initContext() {
  DebugMode = getenv("POLLY_DEBUG") != 0;
  CacheMode = getenv("POLLY_NOCACHE") == 0;

  dump_function();

  PollyGPUContext *Context;

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    Context = initContextCUDA();
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    Context = initContextCL();
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }

  return Context;
}

void polly_freeKernel(PollyGPUFunction *Kernel) {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    freeKernelCUDA(Kernel);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    freeKernelCL(Kernel);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

PollyGPUFunction *polly_getKernel(const char *BinaryBuffer,
                                  const char *KernelName) {
  dump_function();

  PollyGPUFunction *Function;

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    Function = getKernelCUDA(BinaryBuffer, KernelName);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    Function = getKernelCL(BinaryBuffer, KernelName);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }

  return Function;
}

void polly_copyFromHostToDevice(void *HostData, PollyGPUDevicePtr *DevData,
                                long MemSize) {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    copyFromHostToDeviceCUDA(HostData, DevData, MemSize);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    copyFromHostToDeviceCL(HostData, DevData, MemSize);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

void polly_copyFromDeviceToHost(PollyGPUDevicePtr *DevData, void *HostData,
                                long MemSize) {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    copyFromDeviceToHostCUDA(DevData, HostData, MemSize);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    copyFromDeviceToHostCL(DevData, HostData, MemSize);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

void polly_launchKernel(PollyGPUFunction *Kernel, unsigned int GridDimX,
                        unsigned int GridDimY, unsigned int BlockDimX,
                        unsigned int BlockDimY, unsigned int BlockDimZ,
                        void **Parameters) {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    launchKernelCUDA(Kernel, GridDimX, GridDimY, BlockDimX, BlockDimY,
                     BlockDimZ, Parameters);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    launchKernelCL(Kernel, GridDimX, GridDimY, BlockDimX, BlockDimY, BlockDimZ,
                   Parameters);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

void polly_freeDeviceMemory(PollyGPUDevicePtr *Allocation) {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    freeDeviceMemoryCUDA(Allocation);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    freeDeviceMemoryCL(Allocation);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

PollyGPUDevicePtr *polly_allocateMemoryForDevice(long MemSize) {
  dump_function();

  PollyGPUDevicePtr *DevData;

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    DevData = allocateMemoryForDeviceCUDA(MemSize);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    DevData = allocateMemoryForDeviceCL(MemSize);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }

  return DevData;
}

void *polly_getDevicePtr(PollyGPUDevicePtr *Allocation) {
  dump_function();

  void *DevPtr;

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    DevPtr = getDevicePtrCUDA(Allocation);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    DevPtr = getDevicePtrCL(Allocation);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }

  return DevPtr;
}

void polly_synchronizeDevice() {
  dump_function();

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    synchronizeDeviceCUDA();
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    synchronizeDeviceCL();
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

void polly_freeContext(PollyGPUContext *Context) {
  dump_function();

  if (CacheMode)
    return;

  switch (Runtime) {
#ifdef HAS_LIBCUDART
  case RUNTIME_CUDA:
    freeContextCUDA(Context);
    break;
#endif /* HAS_LIBCUDART */
#ifdef HAS_LIBOPENCL
  case RUNTIME_CL:
    freeContextCL(Context);
    break;
#endif /* HAS_LIBOPENCL */
  default:
    err_runtime();
  }
}

/* Initialize GPUJIT with CUDA as runtime library. */
PollyGPUContext *polly_initContextCUDA() {
#ifdef HAS_LIBCUDART
  Runtime = RUNTIME_CUDA;
  return polly_initContext();
#else
  fprintf(stderr, "GPU Runtime was built without CUDA support.\n");
  exit(-1);
#endif /* HAS_LIBCUDART */
}

/* Initialize GPUJIT with OpenCL as runtime library. */
PollyGPUContext *polly_initContextCL() {
#ifdef HAS_LIBOPENCL
  Runtime = RUNTIME_CL;
  return polly_initContext();
#else
  fprintf(stderr, "GPU Runtime was built without OpenCL support.\n");
  exit(-1);
#endif /* HAS_LIBOPENCL */
}
