//===- cuda.h - CUDA driver API subset for the AGX runner -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The declarations of the CUDA driver API that CudaRuntimeWrappers.cpp uses,
// so that the wrappers build without a CUDA toolkit against a driver that
// implements the API, such as ZLUDA's zluda_agx on Asahi Linux
// (MLIR_ENABLE_AGX_RUNNER). The values and layouts are those of the driver
// ABI. Functions that the ABI versions are declared under their versioned
// symbol names, as the driver exports them.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AGX_CUDA_COMPAT_CUDA_H
#define MLIR_AGX_CUDA_COMPAT_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> // the wrappers rely on cuda.h including it (getenv, alloca)
#ifndef _WIN32
#include <alloca.h>
#endif

#define CUDA_VERSION 12000

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUresult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_DEINITIALIZED 4

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

typedef enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
} CUdevice_attribute;

typedef enum CUfunction_attribute {
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
} CUfunction_attribute;

typedef enum CUjit_option {
  CU_JIT_ERROR_LOG_BUFFER = 5,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
  CU_JIT_OPTIMIZATION_LEVEL = 7,
} CUjit_option;

enum { CU_EVENT_DISABLE_TIMING = 2 };
enum { CU_STREAM_NON_BLOCKING = 1 };
enum { CU_MEM_ATTACH_GLOBAL = 1 };

typedef enum CUlaunchAttributeID {
  CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
  CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
  CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
} CUlaunchAttributeID;

typedef enum CUclusterSchedulingPolicy {
  CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
} CUclusterSchedulingPolicy;

typedef union CUlaunchAttributeValue {
  char pad[64];
  int cooperative;
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  } clusterDim;
  CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
} CUlaunchAttributeValue;

typedef struct CUlaunchAttribute {
  CUlaunchAttributeID id;
  char pad[4];
  CUlaunchAttributeValue value;
} CUlaunchAttribute;

typedef struct CUlaunchConfig {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute *attrs;
  unsigned int numAttrs;
} CUlaunchConfig;

typedef struct CUtensorMap {
  alignas(64) cuuint64_t opaque[16];
} CUtensorMap;
typedef enum CUtensorMapDataType { CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0 } CUtensorMapDataType;
typedef enum CUtensorMapInterleave { CU_TENSOR_MAP_INTERLEAVE_NONE = 0 } CUtensorMapInterleave;
typedef enum CUtensorMapSwizzle { CU_TENSOR_MAP_SWIZZLE_NONE = 0 } CUtensorMapSwizzle;
typedef enum CUtensorMapL2promotion { CU_TENSOR_MAP_L2_PROMOTION_NONE = 0 } CUtensorMapL2promotion;
typedef enum CUtensorMapFloatOOBfill { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0 } CUtensorMapFloatOOBfill;

#define cuCtxPopCurrent cuCtxPopCurrent_v2
#define cuCtxPushCurrent cuCtxPushCurrent_v2
#define cuEventDestroy cuEventDestroy_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemFree cuMemFree_v2
#define cuMemHostRegister cuMemHostRegister_v2
#define cuStreamDestroy cuStreamDestroy_v2

CUresult cuInit(unsigned int flags);
CUresult cuGetErrorName(CUresult error, const char **name);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attrib, CUdevice device);
CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice device);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext *ctx);
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                            CUjit_option *options, void **optionValues);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
CUresult cuFuncSetAttribute(CUfunction function, CUfunction_attribute attrib, int value);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream stream,
                        void **kernelParams, void **extra);
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams,
                          void **extra);
CUresult cuStreamCreate(CUstream *stream, unsigned int flags);
CUresult cuStreamDestroy(CUstream stream);
CUresult cuStreamSynchronize(CUstream stream);
CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult cuEventCreate(CUevent *event, unsigned int flags);
CUresult cuEventDestroy(CUevent event);
CUresult cuEventRecord(CUevent event, CUstream stream);
CUresult cuEventSynchronize(CUevent event);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytes);
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytes, unsigned int flags);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream stream);
CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream);
CUresult cuMemHostRegister(void *p, size_t bytes, unsigned int flags);
CUresult cuMemHostUnregister(void *p);
CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
                                cuuint32_t tensorRank, void *globalAddress,
                                const cuuint64_t *globalDim, const cuuint64_t *globalStrides,
                                const cuuint32_t *boxDim, const cuuint32_t *elementStrides,
                                CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                                CUtensorMapL2promotion l2Promotion,
                                CUtensorMapFloatOOBfill oobFill);

#ifdef __cplusplus
}
#endif

#endif // MLIR_AGX_CUDA_COMPAT_CUDA_H
