//===- GPUInterfaces.h - OpenACC external models ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_EXTENSIONS_GPUINTERFACES_H
#define MLIR_DIALECT_OPENACC_EXTENSIONS_GPUINTERFACES_H

namespace mlir {
class DialectRegistry;
namespace acc {
/// Register OpenACC interfaces for GPU IR. This does not load dialects.
void registerGPUInterfaces(DialectRegistry &registry);
} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_EXTENSIONS_GPUINTERFACES_H
