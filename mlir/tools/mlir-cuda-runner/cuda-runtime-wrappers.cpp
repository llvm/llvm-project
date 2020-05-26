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

namespace {
int32_t reportErrorIfAny(CUresult result, const char *where) {
  if (result != CUDA_SUCCESS) {
    llvm::errs() << "CUDA failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mgpuModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      cuModuleLoadData(reinterpret_cast<CUmodule *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mgpuModuleGetFunction(void **function, void *module,
                                         const char *name) {
  return reportErrorIfAny(
      cuModuleGetFunction(reinterpret_cast<CUfunction *>(function),
                          reinterpret_cast<CUmodule>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mgpuLaunchKernel(void *function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem, void *stream,
                                    void **params, void **extra) {
  return reportErrorIfAny(
      cuLaunchKernel(reinterpret_cast<CUfunction>(function), gridX, gridY,
                     gridZ, blockX, blockY, blockZ, smem,
                     reinterpret_cast<CUstream>(stream), params, extra),
      "LaunchKernel");
}

extern "C" void *mgpuGetStreamHelper() {
  CUstream stream;
  reportErrorIfAny(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "StreamCreate");
  return stream;
}

extern "C" int32_t mgpuStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "StreamSync");
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  reportErrorIfAny(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0),
                   "MemHostRegister");
}

// Allows to register a MemRef with the CUDA runtime. Initializes array with
// value. Helpful until we have transfer functions implemented.
template <typename T>
void mcuMemHostRegisterMemRef(const DynamicMemRefType<T> &mem_ref, T value) {
  llvm::SmallVector<int64_t, 4> denseStrides(mem_ref.rank);
  llvm::ArrayRef<int64_t> sizes(mem_ref.sizes, mem_ref.rank);
  llvm::ArrayRef<int64_t> strides(mem_ref.strides, mem_ref.rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto count = denseStrides.front();

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  auto *pointer = mem_ref.data + mem_ref.offset;
  std::fill_n(pointer, count, value);
  mgpuMemHostRegister(pointer, count * sizeof(T));
}

extern "C" void mcuMemHostRegisterFloat(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> mem_ref = {rank, ptr};
  mcuMemHostRegisterMemRef(DynamicMemRefType<float>(mem_ref), 1.23f);
}

extern "C" void mcuMemHostRegisterInt32(int64_t rank, void *ptr) {
  UnrankedMemRefType<int32_t> mem_ref = {rank, ptr};
  mcuMemHostRegisterMemRef(DynamicMemRefType<int32_t>(mem_ref), 123);
}
