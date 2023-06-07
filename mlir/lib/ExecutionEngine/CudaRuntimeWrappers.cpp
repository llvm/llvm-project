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

#if MLIR_ENABLE_CUDA_CUSPARSE
#include "cusparse.h"
#if MLIR_ENABLE_CUDA_CUSPARSELT
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

#if MLIR_ENABLE_CUDA_CUSPARSE

///
/// Wrapper methods for the cuSparse library.
///

// Some macro magic to get float/double alpha and beta on host.
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

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpMVBufferSize(void *h, int32_t ma, void *a, void *x, void *y, int32_t ctp,
                   CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(
      cusparseSpMV_bufferSize(handle, modeA, alphap, matA, vecX, betap, vecY,
                              cTp, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
  return bufferSize == 0 ? 1 : bufferSize; // avoid zero-alloc
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSpMV(void *h, int32_t ma, void *a,
                                                   void *x, void *y,
                                                   int32_t ctp, void *buf,
                                                   CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnVecDescr_t vecX = reinterpret_cast<cusparseDnVecDescr_t>(x);
  cusparseDnVecDescr_t vecY = reinterpret_cast<cusparseDnVecDescr_t>(y);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMV(handle, modeA, alphap, matA, vecX,
                                        betap, vecY, cTp,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSpMMBufferSize(void *h, int32_t ma, int32_t mb, void *a, void *b, void *c,
                   int32_t ctp, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM_bufferSize(
      handle, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
  return bufferSize == 0 ? 1 : bufferSize; // avoid zero-alloc
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSpMM(void *h, int32_t ma, int32_t mb, void *a, void *b, void *c,
         int32_t ctp, void *buf, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseSpMatDescr_t matA = reinterpret_cast<cusparseSpMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseDnMatDescr_t matC = reinterpret_cast<cusparseDnMatDescr_t>(c);
  cudaDataType_t cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSpMM(handle, modeA, modeB, alphap, matA,
                                        matB, betap, matC, cTp,
                                        CUSPARSE_SPMM_ALG_DEFAULT, buf))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT intptr_t
mgpuSDDMMBufferSize(void *h, int32_t ma, int32_t mb, void *a, void *b, void *c,
                    int32_t ctp, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  size_t bufferSize = 0;
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM_bufferSize(
      handle, modeA, modeB, alphap, matA, matB, betap, matC, cTp,
      CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))
  return bufferSize == 0 ? 1 : bufferSize; // avoid zero-alloc
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuSDDMM(void *h, int32_t ma, int32_t mb, void *a, void *b, void *c,
          int32_t ctp, void *buf, CUstream /*stream*/) {
  cusparseHandle_t handle = reinterpret_cast<cusparseHandle_t>(h);
  cusparseOperation_t modeA = static_cast<cusparseOperation_t>(ma);
  cusparseOperation_t modeB = static_cast<cusparseOperation_t>(mb);
  cusparseDnMatDescr_t matA = reinterpret_cast<cusparseDnMatDescr_t>(a);
  cusparseDnMatDescr_t matB = reinterpret_cast<cusparseDnMatDescr_t>(b);
  cusparseSpMatDescr_t matC = reinterpret_cast<cusparseSpMatDescr_t>(c);
  auto cTp = static_cast<cudaDataType_t>(ctp);
  ALPHABETA(cTp, alpha, beta)
  CUSPARSE_REPORT_IF_ERROR(cusparseSDDMM(handle, modeA, modeB, alphap, matA,
                                         matB, betap, matC, cTp,
                                         CUSPARSE_SDDMM_ALG_DEFAULT, buf))
}

#if MLIR_ENABLE_CUDA_CUSPARSELT

///
/// Wrapper methods for the cuSparseLt library.
///

struct cusparseLtSpMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  void *values{nullptr};
  // TODO: the following is associated with the SpMM operator rather than the
  // sparse matrix. Create workspace buffers and pass them to the SpMM
  // execution.
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulDescriptor_t matmul;
};

struct cusparseLtDnMatHandleAndData {
  cusparseLtMatDescriptor_t mat;
  void *values{nullptr};
};

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuAssertSparseLTEnvHandleSize() {
  assert(sizeof(cusparseLtHandle_t) == 11024);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuAssertSparseLtSpMatHandleSize() {
  return assert(sizeof(cusparseLtSpMatHandleAndData) == 44104);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void mgpuSparseLtDnMatHandleSize() {
  return assert(sizeof(cusparseLtDnMatHandleAndData) == 11032);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void *
mgpuCreateSparseLtEnv(void *h, CUstream /*stream*/) {
  // note that cuSparseLt still uses cusparseStatus_t
  CUSPARSE_REPORT_IF_ERROR(
      cusparseLtInit(reinterpret_cast<cusparseLtHandle_t *>(h)))
  return;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroySparseLtEnv(void *h, CUstream /*stream*/) {
  auto handle = reinterpret_cast<cusparseLtHandle_t *>(h);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDestroy(handle))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCreateCuSparseLtDnMat(void *dh, void *h, intptr_t rows, intptr_t cols,
                          void *values, int32_t dw, CUstream /*stream*/) {
  cusparseLtMatDescriptor_t mat;
  auto handle = reinterpret_cast<cusparseLtHandle_t *>(h);
  auto dnmat_handle = reinterpret_cast<cusparseLtDnMatHandleAndData *>(dh);
  cudaDataType_t dtp = dataTp(dw);
  // assuming row-major when deciding lda
  CUSPARSE_REPORT_IF_ERROR(cusparseLtDenseDescriptorInit(
      handle, &(dh->mat), rows, cols, /*lda=*/cols,
      /*alignment=*/16, dtp, CUSPARSE_ORDER_ROW))
  dnmat_handle->values = values;
}

// This can be used to destroy both dense matrices and sparse matrices in
// cusparseLt
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtSpMat(void *m, CUstream /*stream*/) {
  auto matAndData = reinterpret_cast<cusparseLtSpMatHandleAndData>(m);
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuDestroyCuSparseLtDnMat(void *m, CUstream /*stream*/) {
  auto matAndData = reinterpret_cast<cusparseLtDnMatHandleAndData>(m);
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(mat->mat)))
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCusparseLtCreate2To4SpMat(void *sh, void *h, intptr_t rows, intptr_t cols,
                              void *values, int32_t dw, CUstream /*stream*/) {
  auto spmat_handle = reinterpret_cast<cusparseLtSpMatHandleAndData *>(sh);
  spmat_handle->values = values;
  auto handle = reinterpret_cast<cusparseLtHandle_t *>(h);
  cudaDataType_t dtp = dataTp_cusparseLt(dw);
  // assuming row-major when deciding lda
  CUSPARSE_REPORT_IF_ERROR(cusparseLtStructuredDescriptorInit(
      handle, &(sh->mat), rows, cols, /*ld=*/cols, /*alignment=*/16, dtp,
      CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
}

// Several things are being done in this stage, algorithm selection, planning,
// and returning workspace and compressed matrices data buffer sizes.
extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMMBufferSize(void *workspace_size, void *compressed_size,
                             void *compressed_buffer_size, void *h, void *a,
                             CUstream /*stream*/) {
  // TODO: support more advanced settings, e.g., the input right operand is a
  // sparse matrix assuming matA is the sparse matrix
  auto handle = reinterpret_cast<cusparseLtHandle_t *>(h);
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);

  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      handle, &(matWithData.alg_sel), &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  int alg = 0;
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      handle, &(matWithData.alg_sel), CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg,
      sizeof(alg)))
  // TODO: add transpose support
  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      handle, &(matA.matmul), c, CUSPARSE_OPERATION_NON_TRANSPOSE, &(matA->mat),
      &matB, &matC, &matC, compute_type))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(handle, &(matWithData.plan), &matmul,
                                          &(matWithData.alg_sel)))

  CHECK_CUSPARSE(
      cusparseLtMatmulGetWorkspace(handle, &(matA.plan), workspace_size))
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
      handle, &(matA.plan), compressed_size, compressed_buffer_size))

  // avoid zero-alloc
  *workspace_size = (*workspace_size == 0 ? 1 : *workspace_size);
  *compressed_size = (*compressed_size == 0 ? 1 : *compressed_size);
  *compressed_buffer_size =
      (*compressed_buffer_size == 0 ? 1 : *compressed_buffer_size);
  return;
}

extern "C" MLIR_CUDA_WRAPPERS_EXPORT void
mgpuCuSparseLtSpMM(void *alg_sel, void *plan, void *matmul, void *h, void *a,
                   void *b, void *c, int32_t dw, void *buf, void *dA_compressed,
                   void *dA_compressedBuffer, CUstream stream) {
  auto handle = reinterpret_cast<cusparseLtHandle_t *>(h);
  auto matA = reinterpret_cast<cusparseLtSpMatHandleAndData *>(a);
  auto matB = reinterpret_cast<cusparseLtDnMatHandleAndData *>(b);
  auto matC = reinterpret_cast<cusparseLtDnMatHandleAndData *>(c);

  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulDescriptor_t matmul;

  ALPHABETA(dw, alpha, beta)

  CHECK_CUSPARSE(cusparseLtSpMMACompress(handle, &(matA->plan), &(matA->values),
                                         dA_compressed, dA_compressedBuffer,
                                         stream))

  // TODO: add support to multi-stream execution
  // Perform the matrix multiplication. D = A*B+C using C==D for now
  CHECK_CUSPARSE(
      cusparseLtMatmul(handle, reinterpret_cast<cusparseLtMatmulPlan_t *>(plan),
                       &alpha, dA_compressed, dB, &beta, matC->values,
                       /*dD*/ matC->values, d_workspace, &stream, 1))

  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatDescriptorDestroy(&(mat->mat)))
  // destroy the plan associated with the sparse matrix
  CUSPARSE_REPORT_IF_ERROR(cusparseLtMatmulPlanDestroy(&(mat->plan)))
}

#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE
