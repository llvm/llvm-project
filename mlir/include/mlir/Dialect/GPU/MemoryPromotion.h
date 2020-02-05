//===- MemoryPromotion.h - Utilities for moving data across GPU -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares the utility functions that generate IR copying
// the data between different levels of memory hierarchy.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_MEMORYPROMOTION_H
#define MLIR_DIALECT_GPU_MEMORYPROMOTION_H

namespace mlir {

namespace gpu {
class GPUFuncOp;
}

/// Promotes a function argument to workgroup memory in the given function. The
/// copies will be inserted in the beginning and in the end of the function.
void promoteToWorkgroupMemory(gpu::GPUFuncOp op, unsigned arg);

} // end namespace mlir

#endif // MLIR_DIALECT_GPU_MEMORYPROMOTION_H
