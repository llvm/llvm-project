//===- mlir_test_spirv_cpu_runner_c_wrappers.cpp - Runner testing library -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A small library for SPIR-V cpu runner testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RunnerUtils.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// NOLINTBEGIN(*-identifier-naming)

extern "C" EXPORT void
_mlir_ciface_fillI32Buffer(StridedMemRefType<int32_t, 1> *mem_ref,
                           int32_t value) {
  std::fill_n(mem_ref->basePtr, mem_ref->sizes[0], value);
}

extern "C" EXPORT void
_mlir_ciface_fillF32Buffer1D(StridedMemRefType<float, 1> *mem_ref,
                             float value) {
  std::fill_n(mem_ref->basePtr, mem_ref->sizes[0], value);
}

extern "C" EXPORT void
_mlir_ciface_fillF32Buffer2D(StridedMemRefType<float, 2> *mem_ref,
                             float value) {
  std::fill_n(mem_ref->basePtr, mem_ref->sizes[0] * mem_ref->sizes[1], value);
}

extern "C" EXPORT void
_mlir_ciface_fillF32Buffer3D(StridedMemRefType<float, 3> *mem_ref,
                             float value) {
  std::fill_n(mem_ref->basePtr,
              mem_ref->sizes[0] * mem_ref->sizes[1] * mem_ref->sizes[2], value);
}

// NOLINTEND(*-identifier-naming)
