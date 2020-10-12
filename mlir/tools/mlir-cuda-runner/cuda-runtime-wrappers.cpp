//===- cuda-runtime-wrappers.cpp - MLIR CUDA runner wrapper library -------===//
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

#include <cassert>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda.h"

#define CUDA_REPORT_IF_ERROR(expr)                                             \
  [](CUresult result) {                                                        \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = nullptr;                                                \
    cuGetErrorName(result, &name);                                             \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    llvm::errs() << "'" << #expr << "' failed with '" << name << "'\n";        \
  }(expr)

// Static initialization of CUDA context for device ordinal 0.
static auto InitializeCtx = [] {
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/0));
  CUcontext context;
  CUDA_REPORT_IF_ERROR(cuCtxCreate(&context, /*flags=*/0, device));
  return 0;
}();

extern "C" CUmodule mgpuModuleLoad(void *data) {
  CUmodule module = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  return module;
}

extern "C" CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" void mgpuLaunchKernel(CUfunction function, intptr_t gridX,
                                 intptr_t gridY, intptr_t gridZ,
                                 intptr_t blockX, intptr_t blockY,
                                 intptr_t blockZ, int32_t smem, CUstream stream,
                                 void **params, void **extra) {
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, smem, stream, params,
                                      extra));
}

extern "C" CUstream mgpuStreamCreate() {
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" CUevent mgpuEventCreate() {
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" void mgpuEventDestroy(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" void mgpuEventRecord(CUevent event, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {

  llvm::SmallVector<int64_t, 4> denseStrides(rank);
  llvm::ArrayRef<int64_t> sizes(descriptor->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}
