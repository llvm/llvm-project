//===- CudaRuntimeWrappers.cpp - MLIR CUDA API wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the CUDA library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <stdio.h>

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

#ifdef MLIR_ENABLE_CUDA_CUSPARSE
#include "cusparse.h"
#ifdef MLIR_ENABLE_CUDA_CUSPARSELT
#include "cusparseLt.h"
#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE

#ifdef _WIN32
#define MLIR_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MLIR_CUDA_WRAPPERS_EXPORT
#endif // _WIN32

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

#define CUSPARSE_REPORT_IF_ERROR(expr)                                         \
  {                                                                            \
    cusparseStatus_t status = (expr);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuSPARSE '%s' failed with '%s'\n", #expr,               \
              cusparseGetErrorString(status));                                 \
    }                                                                          \
  }

thread_local static int32_t defaultDevice = 0;

const char *kDebugEnvironmentVariable = "MLIR_CUDA_DEBUG";

/// Helper method that checks environment value for debugging.
bool isDebugEnabled() {
  static bool isInitialized = false;
  static bool isEnabled = false;
  if (!isInitialized)
    isEnabled = getenv(kDebugEnvironmentVariable) != nullptr;
  return isEnabled;
}

#define debug_print(fmt, ...)                                                  \
  do {                                                                         \
    if (isDebugEnabled())                                                      \
      fprintf(stderr, "%s:%d:%s(): " fmt, "CudaRuntimeWrappers.cpp", __LINE__, \
              __func__, __VA_ARGS__);                                          \
  } while (0)

// Returns default CUdevice
CUdevice getDefaultCuDevice() {
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  return device;
}

// Make the primary context of the current default device current for the
// duration
//  of the instance and restore the previous context on destruction.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

#ifdef MLIR_ENABLE_CUDA_CUSPARSE
// Note that (1) Nvidia confirms the safety to share handle across multiple
// instances, and streams. (2) Clients are responsible to call the @mgpu
// environment initialization/destruction in a thread-safe manner, e.g.,
// at the beginning of the program before multi-threads are created.
static cusparseHandle_t cusparse_env = nullptr;

#ifdef MLIR_ENABLE_CUDA_CUSPARSELT
// cusparseLtHandle_t is not a pointer type, so we need an additional flag to
// indicate whether it is initialized.
static cusparseLtHandle_t cusparseLt_env;
static bool cusparseLt_initiated = false;

#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoad(void *data) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoadJIT(void *data,
                                                                int optLevel) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  char jitErrorBuffer[4096] = {0};
  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                               CU_JIT_OPTIMIZATION_LEVEL};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                            reinterpret_cast<void *>(optLevel)};

  CUresult result =
      cuModuleLoadDataEx(&module, data, 3, jitOptions, jitOptionsVals);
  if (result) {
    fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
    CUDA_REPORT_IF_ERROR(result);
  }
  return module;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload(CUmodule module) {
  CUDA_REPORT_IF_ERROR(cuModuleUnload(module));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUfunction
mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra) {
  ScopedContext scopedContext;
  int32_t maxShmem = 0;
  CUdevice device = getDefaultCuDevice();
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  CUDA_REPORT_IF_ERROR(cuDeviceGetAttribute(
      &maxShmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device));
  if (maxShmem < smem) {
    fprintf(stderr,
            "Requested shared memory (%dkb) is larger than maximum allowed "
            "shared memory (%dkb) for this device\n",
            smem, maxShmem);
  }
  CUDA_REPORT_IF_ERROR(cuFuncSetAttribute(
      function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  debug_print("Launching kernel, grid=%ld,%ld,%ld, "
              "threads: %ld, %ld, %ld, "
              "smem: %dkb\n",
              gridX, gridY, gridZ, blockX, blockY, blockZ, smem);
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamWaitEvent(CUstream stream,
                                                              CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern MLIR_CUDA_WRAPPERS_EXPORT "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern MLIR_CUDA_WRAPPERS_EXPORT "C" void mgpuEventRecord(CUevent event,
                                                          CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes != 0)
    CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" void mgpuMemFree(void *ptr, CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src),
                                     sizeBytes, stream));
}

extern "C" void mgpuMemset32(void *dst, unsigned int value, size_t count,
                             CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, unsigned short value, size_t count,
                             CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                                        value, count, stream));
}

///
/// Helper functions for writing mlir example code
///

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0));
}

/// Registers a memref with the CUDA runtime. `descriptor` is a pointer to a
/// ranked memref descriptor struct of rank `rank`. Helpful until we have
/// transfer functions implemented.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {
  // Only densely packed tensors are currently supported.
  int64_t *denseStrides = (int64_t *)alloca(rank * sizeof(int64_t));
  int64_t *sizes = descriptor->sizes;
  for (int64_t i = rank - 1, runningStride = 1; i >= 0; i--) {
    denseStrides[i] = runningStride;
    runningStride *= sizes[i];
  }
  uint64_t sizeBytes = sizes[0] * denseStrides[0] * elementSizeBytes;
  int64_t *strides = &sizes[rank];
  (void)strides;
  for (unsigned i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[i] &&
           "Mismatch in computed dense strides");

  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

// Allows to unregister byte array with the CUDA runtime.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuMemHostUnregister(void *ptr) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostUnregister(ptr));
}

/// Unregisters a memref with the CUDA runtime. `descriptor` is a pointer to a
/// ranked memref descriptor struct of rank `rank`
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuMemHostUnregisterMemRef(int64_t rank,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto *ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
}

///
/// Runtime methods using CUDA 12.0+ driver
///

#if (CUDA_VERSION >= 12000)

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuTensorMapEncodeTiled(
    CUtensorMap *tensorMap,             // Tensor map object
    CUtensorMapDataType tensorDataType, // Tensor data type
    cuuint32_t tensorRank,              // Dimensionality of tensor
    void *globalAddress,                // Starting address
    const cuuint64_t *globalDim,        // Tensor size (number of elements)
    const cuuint64_t *globalStrides,    // Stride size (in bytes)
    const cuuint32_t *boxDim,           // Traversal box (number of elments)
    const cuuint32_t *elementStrides,   // Traversal stride
    CUtensorMapInterleave interleave,   // Type of interleaved layout
    CUtensorMapSwizzle swizzle,         // Bank swizzling pattern
    CUtensorMapL2promotion l2Promotion, // L2 promotion size
    CUtensorMapFloatOOBfill oobFill     // Padding zfill or NaN fill
) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuTensorMapEncodeTiled(
      tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
      globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion,
      oobFill));
  debug_print("Created TMA descriptor\n Addr: %p\n"
              "data type : %d\n"
              "rank : %d\n"
              "globalDim[5]: %zu, %zu, %zu, %zu, %zu\n"
              "globalStrides[5]: %zu, %zu, %zu, %zu, %zu\n"
              "boxDim[5]: %u, %u, %u, %u, %u\n"
              "elementStrides[5]: %u, %u, %u, %u, %u\n"
              "interleave: %u \n"
              "swizzle: %u \n"
              "l2Promotion: %u \n"
              "oobFill: %u \n",
              (void *)&tensorMap, tensorDataType, tensorRank, globalDim[0],
              globalDim[1], globalDim[2], globalDim[3], globalDim[4],
              globalStrides[0], globalStrides[1], globalStrides[2],
              globalStrides[3], globalStrides[4], boxDim[0], boxDim[1],
              boxDim[2], boxDim[3], boxDim[4], elementStrides[0],
              elementStrides[1], elementStrides[2], elementStrides[3],
              elementStrides[4], interleave, swizzle, l2Promotion, oobFill);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *mgpuTensorMapEncodeTiledMemref(
    int64_t tensorRank,                       // Dimensionality of tensor
    StridedMemRefType<char, 1> *descriptor,   // Starting address
    const CUtensorMapDataType tensorDataType, // Stride size (in bytes)
    CUtensorMapInterleave interleave,         // Type of interleaved layout
    CUtensorMapSwizzle swizzle,               // Bank swizzling pattern
    CUtensorMapL2promotion l2Promotion,       // L2 promotion size
    CUtensorMapFloatOOBfill oobFill,          // Padding zfill or NaN fill
    int64_t *inputBoxDims // Tensor size (number of elements)
) {
  CUtensorMap tensorMap;

  auto *globalAddress = descriptor->data;
  uint32_t boxDim[5] = {0}, elementStrides[5] = {0};
  uint64_t globalDim[5] = {0}, globalStrides[5] = {0};
  uint32_t tensorRank32 = uint32_t(tensorRank);

  static const int elementSizeInBytes[] = {1, 2, 4, 4, 8, 8, 2,
                                           4, 8, 2, 4, 4, 4};
  for (int64_t r = 0; r < tensorRank; ++r) {
    elementStrides[r] = uint32_t(1);
    boxDim[r] = static_cast<uint32_t>(inputBoxDims[tensorRank - r - 1]);
    globalDim[r] = static_cast<uint64_t>(descriptor->sizes[tensorRank - r - 1]);
  }

  globalStrides[0] = globalDim[0] * elementSizeInBytes[tensorDataType];
  for (int r = 1; r < tensorRank - 1; r++)
    globalStrides[r] = globalStrides[r - 1] * globalDim[1] *
                       elementSizeInBytes[tensorDataType];

  ScopedContext scopedContext;
  mgpuTensorMapEncodeTiled(&tensorMap, tensorDataType, tensorRank32,
                           globalAddress, globalDim, globalStrides, boxDim,
                           elementStrides, interleave, swizzle, l2Promotion,
                           oobFill);
  // Copy created tensor map to device
  CUdeviceptr dTensorMap;
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&dTensorMap, sizeof(CUtensorMap)));
  CUDA_REPORT_IF_ERROR(cuMemcpy(dTensorMap,
                                reinterpret_cast<CUdeviceptr>(&tensorMap),
                                sizeof(CUtensorMap)));
  return reinterpret_cast<void *>(dTensorMap);
}
#endif

#ifdef MLIR_ENABLE_CUDA_CUSPARSE

///
/// Wrapper methods for the cuSparse library.
///

// Some macro magic to get float/double alpha and beta on host.
// TODO: add support to passing alpha and beta as arguments
#define ALPHABETA(dtp, alpha, beta)                                            \
  __nv_bfloat16(alpha##16bf) = 1.0f;                                           \
  __nv_bfloat16(beta##16bf) = 1.0f;                                            \
  __half(alpha##16f) = 1.0f;                                                   \
  __half(beta##16f) = 1.0f;                                                    \
  float(alpha##f) = 1.0f;                                                      \
  float(beta##f) = 1.0f;                                                       \
  double(alpha##d) = 1.0;                                                      \
  double(beta##d) = 1.0;                                                       \
  const void *(alpha##p) = nullptr;                                            \
  const void *(beta##p) = nullptr;                                             \
  if (dtp == CUDA_R_16BF || dtp == CUDA_C_16BF) {                              \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##16bf));                     \
    (beta##p) = reinterpret_cast<void *>(&(beta##16bf));                       \
  } else if (dtp == CUDA_R_16F || dtp == CUDA_C_16F) {                         \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##16f));                      \
    (beta##p) = reinterpret_cast<void *>(&(beta##16f));                        \
  } else if (dtp == CUDA_R_32F || dtp == CUDA_C_32F) {                         \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##f));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##f));                          \
  } else {                                                                     \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##d));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##d));                          \
  }

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCreateSparseEnv() {
  // ScopedContext is for cuda initialization.
  ScopedContext scopedContext;
  assert(!cusparse_env && "client called mgpuCreateSparseEnv() twice");
  CUSPARSE_REPORT_IF_ERROR(cusparseCreate(&cusparse_env));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroySparseEnv() {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroy(cusparse_env));
  cusparse_env = nullptr;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnVec(intptr_t size, void *values, int32_t dtp, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = nullptr;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnVec(&vec, size, values, dTp))
  return reinterpret_cast<void *>(vec);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnVec(void *v, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = reinterpret_cast<cusparseDnVecDescr_t>(v);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnVec(vec))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnMat(intptr_t rows, intptr_t cols, void *values, int32_t dtp,
                CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = nullptr;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnMat(&mat, rows, cols, /*ld=*/cols,
                                               values, dTp, CUSPARSE_ORDER_ROW))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnMat(void *m, CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = reinterpret_cast<cusparseDnMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCoo(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowIdxs,
              void *colIdxs, void *values, int32_t itp, int32_t dtp,
              CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCoo(&mat, rows, cols, nnz, rowIdxs,
                                             colIdxs, values, iTp,
                                             CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}

#ifdef CUSPARSE_COO_AOS // deprecated in cuSPARSE 11.2
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCooAoS(intptr_t rows, intptr_t cols, intptr_t nnz, void *idxs,
                 void *values, int32_t itp, int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCooAoS(
      &mat, rows, cols, nnz, idxs, values, iTp, CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}
#endif // CUSPARSE_COO_AOS

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCsr(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowPos,
              void *colIdxs, void *values, int32_t ptp, int32_t itp,
              int32_t dtp, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  auto pTp = static_cast<cusparseIndexType_t>(ptp);
  auto iTp = static_cast<cusparseIndexType_t>(itp);
  auto dTp = static_cast<cudaDataType_t>(dtp);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCsr(&mat, rows, cols, nnz, rowPos,
                                             colIdxs, values, pTp, iTp,
                                             CUSPARSE_INDEX_BASE_ZERO, dTp))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroySpMat(void *m, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = reinterpret_cast<cusparseSpMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroySpMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpMVBufferSize(
    int32_t ma, void *a, void *x, void *y, int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV_bufferSize(
      cusparse_env, modeA, alphap, matA, vecX, betap, vecY, cTp,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMV(int32_t ma, void *a, void *x,
                                                   void *y, int32_t ctp,
                                                   void *buf,
                                                   CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV(cusparse_env, modeA, alphap, matA, vecX,
                                        betap, vecY, cTp,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpMMBufferSize(int32_t ma, int32_t mb, void *a, void *b, void *c,
                   int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM_bufferSize(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMM(int32_t ma, int32_t mb,
                                                   void *a, void *b, void *c,
                                                   int32_t ctp, void *buf,
                                                   CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM(cusparse_env, modeA, modeB, alphap,
                                        matA, matB, betap, matC, cTp,
                                        CUSPARSE_SPMM_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSDDMMBufferSize(int32_t ma, int32_t mb, void *a, void *b, void *c,
                    int32_t ctp, CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM_bufferSize(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))
  return bufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSDDMM(int32_t ma, int32_t mb,
                                                    void *a, void *b, void *c,
                                                    int32_t ctp, void *buf,
                                                    CUstream /*stream*/) {
  assert(cusparse_env && "client did not call mgpuCreateSparseEnv()");
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM(cusparse_env, modeA, modeB, alphap,
                                         matA, matB, betap, matC, cTp,
                                         CUSPARSE_SDDMM_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuSpGEMMCreateDescr(CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = nullptr;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_createDescr(&spgemmDesc))
  return reinterpret_cast<void *>(spgemmDesc);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpGEMMDestroyDescr(void *s, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_destroyDescr(spgemmDesc))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpGEMMWorkEstimation(
    void *s, int32_t ma, int32_t mb, void *a, void *b, void *c, int32_t ctp,
    intptr_t bs, void *buf, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t newBufferSize = bs;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_workEstimation(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &newBufferSize, buf))
  return newBufferSize;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpGEMMCompute(void *s, int32_t ma, int32_t mb, void *a, void *b, void *c,
                  int32_t ctp, intptr_t bsz2, void *buf2, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t newBufferSize2 = bsz2;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpGEMM_compute(
      cusparse_env, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &newBufferSize2, buf2))
  return newBufferSize2;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpGEMMCopy(void *s, int32_t ma, int32_t mb, void *a, void *b, void *c,
               int32_t ctp, CUstream /*stream*/) {
  cusparseSpGEMMDescr_t spgemmDesc = reinterpret_cast<cusparseSpGEMMDescr_t>(s);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseSpMatDescr_t matB = reinterpret_cast<cusparseSpMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseSpGEMM_copy(cusparse_env, modeA, modeB, alphap, matA, matB, betap,
                          matC, cTp, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpMatGetSize(void *m, void *r, void *c, void *n, CUstream /*stream*/) {
  cusparseConstSpMatDescr_t matDescr =
      reinterpret_cast<cusparseConstSpMatDescr_t>(m);
  int64_t *rows = reinterpret_cast<int64_t *>(r);
  int64_t *cols = reinterpret_cast<int64_t *>(c);
  int64_t *nnz = reinterpret_cast<int64_t *>(n);
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMatGetSize(matDescr, rows, cols, nnz));
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSetCsrPointers(void *m, void *p, void *c, void *v, CUstream /*stream*/) {
  cusparseSpMatDescr_t matDescr = reinterpret_cast<cusparseSpMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseCsrSetPointers(matDescr, p, c, v));
}

#ifdef MLIR_ENABLE_CUDA_CUSPARSELT

///
/// Wrapper methods for the cuSparseLt library.
///

struct cusparseLtSpMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  // TODO: the following three are associated with the SpMM operator rather than
  // the sparse matrix. Create workspace buffers and pass them to the SpMM
  // execution.
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulDescriptor_t matmul;
  void *values{nullptr};
};

struct cusparseLtDnMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  void *values{nullptr};
};

static_assert(sizeof(cusparseLtHandle_t) == 11024,
              "Unexpected cusparseLt handle size");
static_assert(sizeof(cusparseLtSpMatHandleAndData) == 44104,
              "Unexpected cusparseLt sparse matrix handle size");
static_assert(sizeof(cusparseLtDnMatHandleAndData) == 11032,
              "Unexpected cusparseLt dense matrix handle size");

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuCreateSparseLtEnv() {
  // ScopedContext is for cuda initialization.
  ScopedContext scopedContext;
  assert(!cusparseLt_initiated &&
         "client called mgpuCreateSparseLtEnv() twice");
  // Note that cuSparseLt still uses cusparseStatus_t.
  CUSPARSE_REPORT_IF_ERROR(cusparseLtInit(&cusparseLt_env));
  cusparseLt_initiated = true;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuDestroySparseLtEnv() {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDestroy(&cusparseLt_env));
  cusparseLt_initiated = false;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCreateCuSparseLtDnMat(void *dh, intptr_t rows, intptr_t cols, void *values,
                          int32_t dtp, CUstream /*stream*/) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto dnmat_handle = reinterpret_cast<cusparseLtDnMatHandleAndData *>(dh);
  dnmat_handle->values = values;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  // Assume row-major when deciding lda.
  const uint32_t alignment = 16;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDenseDescriptorInit(
      &cusparseLt_env, &(dnmat_handle->mat), rows, cols, /*lda=*/cols,
      alignment, dTp, CUSPARSE_ORDER_ROW))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtDnMat(void *dh, CUstream /*stream*/) {
  auto dnmat_handle = reinterpret_cast<cusparseLtDnMatHandleAndData *>(dh);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(dnmat_handle->mat)))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCusparseLtCreate2To4SpMat(void *sh, intptr_t rows, intptr_t cols,
                              void *values, int32_t dtp, CUstream /*stream*/) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto spmat_handle = reinterpret_cast<cusparseLtSpMatHandleAndData *>(sh);
  spmat_handle->values = values;
  auto dTp = static_cast<cudaDataType_t>(dtp);
  // Assume row-major when deciding lda.
  const uint32_t alignment = 16;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtStructuredDescriptorInit(
      &cusparseLt_env, &(spmat_handle->mat), rows, cols, /*ld=*/cols, alignment,
      dTp, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtSpMat(void *sh, CUstream /*stream*/) {
  auto spmat_handle = reinterpret_cast<cusparseLtSpMatHandleAndData *>(sh);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(spmat_handle->mat)))
}

// Several things are being done in this stage, algorithm selection, planning,
// and returning workspace and compressed matrices data buffer sizes.
// The parameter prune_flag is used to indicate whether pruning and pruning
// check will happen 0 means not prune or prune check, 1 means prune, 2 means
// prune & prune check
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMMBufferSize(void *bs, int32_t ma, int32_t mb, void *a, void *b,
                             void *c, int32_t ctp, int32_t prune_flag,
                             CUstream stream) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  // TODO: support more advanced settings, e.g., the input right operand is a
  // sparse matrix assuming matA is the sparse matrix
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);
  auto matB = reinterpret_cast<cusparseLtDnMatHandleAndData *>(b);
  auto matC = reinterpret_cast<cusparseLtDnMatHandleAndData *>(c);
  auto workspace_size = reinterpret_cast<int64_t *>(bs);
  auto compressed_size = &(reinterpret_cast<int64_t *>(bs)[1]);
  auto compressed_buffer_size = &(reinterpret_cast<int64_t *>(bs)[2]);
  auto cTp = static_cast<cusparseComputeType>(ctp);

  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulDescriptorInit(
      &cusparseLt_env, &(matA->matmul), modeA, modeB, &(matA->mat),
      &(matB->mat), &(matC->mat), &(matC->mat), cTp))
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulAlgSelectionInit(
      &cusparseLt_env, &(matA->alg_sel), &(matA->matmul),
      CUSPARSELT_MATMUL_ALG_DEFAULT))
  int alg = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulAlgSetAttribute(
      &cusparseLt_env, &(matA->alg_sel), CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg,
      sizeof(alg)))

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulPlanInit(
      &cusparseLt_env, &(matA->plan), &(matA->matmul), &(matA->alg_sel)))

  // Pruning step (in-place).
  if (prune_flag > 0)
    CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMAPrune(
        &cusparseLt_env, &(matA->matmul), matA->values, matA->values,
        CUSPARSELT_PRUNE_SPMMA_STRIP, stream))

  // Check structure of A.
  // Note that this adds a synchronization on the stream.
  // TODO: Do we want that?
  if (prune_flag == 2) {
    int *dvalid = (int *)mgpuMemAlloc(sizeof(int), stream);
    CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMAPruneCheck(
        &cusparseLt_env, &(matA->matmul), matA->values, dvalid, stream))
    int valid = 0;
    mgpuMemcpy(&valid, dvalid, sizeof(int), stream);
    mgpuStreamSynchronize(stream);
    mgpuMemFree(dvalid, stream);
    if (valid != 0)
      fprintf(stderr, "CUPARSE-LT: sparse matrix is not 2:4; computed results "
                      "will be invalid\n");
  }

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulGetWorkspace(
      &cusparseLt_env, &(matA->plan), workspace_size))
  CUSPARSE_REPORT_IF_ERROR(cusparseLtSpMMACompressedSize(
      &cusparseLt_env, &(matA->plan), compressed_size, compressed_buffer_size))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMM(void *a, void *b, void *c, void *d_workspace,
                   void *dA_compressed, void *dA_compressedBuffer,
                   CUstream stream) {
  assert(cusparseLt_initiated && "client did not call mgpuCreateSparseLtEnv()");
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);
  auto matB = reinterpret_cast<cusparseLtDnMatHandleAndData *>(b);
  auto matC = reinterpret_cast<cusparseLtDnMatHandleAndData *>(c);

  ALPHABETA(CUDA_R_32F, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseLtSpMMACompress(&cusparseLt_env, &(matA->plan), (matA->values),
                              dA_compressed, dA_compressedBuffer, stream))

  // TODO: add support to multi-stream execution
  // Perform the matrix multiplication. D = A*B+C using C==D for now
  CUSPARSE_REPORT_IF_ERROR(
      cusparseLtMatmul(&cusparseLt_env, &(matA->plan), alphap, dA_compressed,
                       matB->values, betap, matC->values,
                       /*dD*/ matC->values, d_workspace, nullptr, 0))

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(matA->mat)))
  // destroy the plan associated with the sparse matrix
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulPlanDestroy(&(matA->plan)))
}

#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE
