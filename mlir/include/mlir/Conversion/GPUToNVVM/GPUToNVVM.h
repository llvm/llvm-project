//===- GPUToNVVM.h - Convert GPU to NVVM dialect ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares registration functions for converting GPU to NVVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTONVVM_GPUTONVVM_H
#define MLIR_CONVERSION_GPUTONVVM_GPUTONVVM_H

namespace mlir {
class DialectRegistry;
namespace NVVM {
/// Registers the `ConvertToLLVMAttrInterface` interface on the
/// `NVVM::NVVMTargetAttr` attribute. This interface populates the conversion
/// target, LLVM type converter, and pattern set for converting GPU operations
/// to NVVM.
void registerConvertGpuToNVVMInterface(DialectRegistry &registry);
} // namespace NVVM
} // namespace mlir

#endif // MLIR_CONVERSION_GPUTONVVM_GPUTONVVM_H
