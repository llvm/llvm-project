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
#include "cusparse.h"

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
      CUdevice device;
      CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, device));
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT CUmodule mgpuModuleLoad(void *data) {
  ScopedContext scopedContext;
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
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
  CUdeviceptr ptr;
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
/// Wrapper methods for the cuSparse library.
///

static inline cudaDataType_t dataTp(int32_t width) {
  switch (width) {
  case 32:
    return CUDA_R_32F;
  default:
    return CUDA_R_64F;
  }
}

static inline cusparseIndexType_t idxTp(int32_t width) {
  switch (width) {
  case 32:
    return CUSPARSE_INDEX_32I;
  default:
    return CUSPARSE_INDEX_64I;
  }
}

// Some macro magic to get float/double alpha and beta on host.
#define ALPHABETA(w, alpha, beta)                                              \
  float(alpha##f) = 1.0f;                                                      \
  float(beta##f) = 1.0f;                                                       \
  double(alpha##d) = 1.0;                                                      \
  double(beta##d) = 1.0;                                                       \
  const void *(alpha##p) = nullptr;                                            \
  const void *(beta##p) = nullptr;                                             \
  if ((w) == 32) {                                                             \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##f));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##f));                          \
  } else {                                                                     \
    (alpha##p) = reinterpret_cast<void *>(&(alpha##d));                        \
    (beta##p) = reinterpret_cast<void *>(&(beta##d));                          \
  }

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateSparseEnv(CUstream /*stream*/) {
  cusparseHandle_t handle = nullptr;
  CUSPARSE_REPORT_IF_ERROR(cusparseCreate(&handle))
  return reinterpret_cast<void *>(handle);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroySparseEnv(void *h, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroy(handle))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnVec(intptr_t size, void *values, int32_t dw, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = nullptr;
  cudaDataType_t dtp = dataTp(dw);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnVec(&vec, size, values, dtp))
  return reinterpret_cast<void *>(vec);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnVec(void *v, CUstream /*stream*/) {
  cusparseDnVecDescr_t vec = reinterpret_cast<cusparseDnVecDescr_t>(v);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnVec(vec))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateDnMat(intptr_t rows, intptr_t cols, void *values, int32_t dw,
                CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = nullptr;
  cudaDataType_t dtp = dataTp(dw);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateDnMat(&mat, rows, cols, /*ld=*/cols,
                                               values, dtp, CUSPARSE_ORDER_ROW))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyDnMat(void *m, CUstream /*stream*/) {
  cusparseDnMatDescr_t mat = reinterpret_cast<cusparseDnMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroyDnMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCoo(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowIdxs,
              void *colIdxs, void *values, int32_t iw, int32_t dw,
              CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  cusparseIndexType_t itp = idxTp(iw);
  cudaDataType_t dtp = dataTp(dw);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCoo(&mat, rows, cols, nnz, rowIdxs,
                                             colIdxs, values, itp,
                                             CUSPARSE_INDEX_BASE_ZERO, dtp))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateCsr(intptr_t rows, intptr_t cols, intptr_t nnz, void *rowPos,
              void *colIdxs, void *values, int32_t pw, int32_t iw, int32_t dw,
              CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = nullptr;
  cusparseIndexType_t ptp = idxTp(pw);
  cusparseIndexType_t itp = idxTp(iw);
  cudaDataType_t dtp = dataTp(dw);
  CUSPARSE_REPORT_IF_ERROR(cusparseCreateCsr(&mat, rows, cols, nnz, rowPos,
                                             colIdxs, values, ptp, itp,
                                             CUSPARSE_INDEX_BASE_ZERO, dtp))
  return reinterpret_cast<void *>(mat);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroySpMat(void *m, CUstream /*stream*/) {
  cusparseSpMatDescr_t mat = reinterpret_cast<cusparseSpMatDescr_t>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseDestroySpMat(mat))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpMVBufferSize(
    void *h, void *a, void *x, void *y, int32_t dw, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t dtp = dataTp(dw);
  ALPHABETA(dw, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alphap, matA, vecX, betap, vecY,
      dtp, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
  return bufferSize == 0 ? 1 : bufferSize; // avoid zero-alloc
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMV(void *h, void *a, void *x,
                                                   void *y, int32_t dw,
                                                   void *buf,
                                                   CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t dtp = dataTp(dw);
  ALPHABETA(dw, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alphap, matA, vecX,
                   betap, vecY, dtp, CUSPARSE_SPMV_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t mgpuSpMMBufferSize(
    void *h, void *a, void *b, void *c, int32_t dw, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t dtp = dataTp(dw);
  ALPHABETA(dw, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, alphap, matA, matB, betap, matC, dtp,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  return bufferSize == 0 ? 1 : bufferSize; // avoid zero-alloc
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMM(void *h, void *a, void *b,
                                                   void *c, int32_t dw,
                                                   void *buf,
                                                   CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t dtp = dataTp(dw);
  ALPHABETA(dw, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(
      cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, alphap, matA, matB, betap,
                   matC, dtp, CUSPARSE_SPMM_ALG_DEFAULT, buf))
}
