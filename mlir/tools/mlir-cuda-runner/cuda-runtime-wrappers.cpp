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

extern "C" int32_t mcuModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      cuModuleLoadData(reinterpret_cast<CUmodule *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mcuModuleGetFunction(void **function, void *module,
                                        const char *name) {
  return reportErrorIfAny(
      cuModuleGetFunction(reinterpret_cast<CUfunction *>(function),
                          reinterpret_cast<CUmodule>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mcuLaunchKernel(void *function, intptr_t gridX,
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

extern "C" void *mcuGetStreamHelper() {
  CUstream stream;
  reportErrorIfAny(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "StreamCreate");
  return stream;
}

extern "C" int32_t mcuStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "StreamSync");
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the CUDA runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mcuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  reportErrorIfAny(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0),
                   "MemHostRegister");
}

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N> struct MemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// Allows to register a MemRef with the CUDA runtime. Initializes array with
// value. Helpful until we have transfer functions implemented.
template <typename T>
void mcuMemHostRegisterMemRef(T *pointer, llvm::ArrayRef<int64_t> sizes,
                              llvm::ArrayRef<int64_t> strides, T value) {
  assert(sizes.size() == strides.size());
  llvm::SmallVector<int64_t, 4> denseStrides(strides.size());

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto count = denseStrides.front();

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  std::fill_n(pointer, count, value);
  mcuMemHostRegister(pointer, count * sizeof(T));
}

extern "C" void mcuMemHostRegisterMemRef1dFloat(float *allocated,
                                                float *aligned, int64_t offset,
                                                int64_t size, int64_t stride) {
  mcuMemHostRegisterMemRef(aligned + offset, {size}, {stride}, 1.23f);
}

extern "C" void mcuMemHostRegisterMemRef2dFloat(float *allocated,
                                                float *aligned, int64_t offset,
                                                int64_t size0, int64_t size1,
                                                int64_t stride0,
                                                int64_t stride1) {
  mcuMemHostRegisterMemRef(aligned + offset, {size0, size1}, {stride0, stride1},
                           1.23f);
}

extern "C" void mcuMemHostRegisterMemRef3dFloat(float *allocated,
                                                float *aligned, int64_t offset,
                                                int64_t size0, int64_t size1,
                                                int64_t size2, int64_t stride0,
                                                int64_t stride1,
                                                int64_t stride2) {
  mcuMemHostRegisterMemRef(aligned + offset, {size0, size1, size2},
                           {stride0, stride1, stride2}, 1.23f);
}

extern "C" void mcuMemHostRegisterMemRef1dInt32(int32_t *allocated,
                                                int32_t *aligned,
                                                int64_t offset, int64_t size,
                                                int64_t stride) {
  mcuMemHostRegisterMemRef(aligned + offset, {size}, {stride}, 123);
}

extern "C" void mcuMemHostRegisterMemRef2dInt32(int32_t *allocated,
                                                int32_t *aligned,
                                                int64_t offset, int64_t size0,
                                                int64_t size1, int64_t stride0,
                                                int64_t stride1) {
  mcuMemHostRegisterMemRef(aligned + offset, {size0, size1}, {stride0, stride1},
                           123);
}

extern "C" void
mcuMemHostRegisterMemRef3dInt32(int32_t *allocated, int32_t *aligned,
                                int64_t offset, int64_t size0, int64_t size1,
                                int64_t size2, int64_t stride0, int64_t stride1,
                                int64_t stride2) {
  mcuMemHostRegisterMemRef(aligned + offset, {size0, size1, size2},
                           {stride0, stride1, stride2}, 123);
}
