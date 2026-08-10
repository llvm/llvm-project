//===- Runtimes.h - Possible runtimes for AMD GPUs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_RUNTIMES_H
#define MLIR_CONVERSION_GPUTOROCDL_RUNTIMES_H

namespace mlir {
namespace gpu {
namespace amd {
/// Potential runtimes for AMD GPU kernels
enum Runtime {
  Unknown = 0,
  HIP = 1,
  OpenCL = 2,
};
} // end namespace amd
} // end namespace gpu
} // end namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_RUNTIMES_H
