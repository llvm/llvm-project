//===- GPUToLLVM.h - Convert GPU to LLVM dialect ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares registration functions for converting GPU to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUCOMMON_GPUTOLLVM_H
#define MLIR_CONVERSION_GPUCOMMON_GPUTOLLVM_H

namespace mlir {
class DialectRegistry;
namespace gpu {
/// Registers the `ConvertToLLVMOpInterface` interface on the `gpu::GPUModuleOP`
/// operation.
void registerConvertGpuToLLVMInterface(DialectRegistry &registry);
} // namespace gpu
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUTOLLVM_H
