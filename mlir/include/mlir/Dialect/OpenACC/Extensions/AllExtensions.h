//===- AllExtensions.h - OpenACC external interfaces ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_EXTENSIONS_ALLEXTENSIONS_H
#define MLIR_DIALECT_OPENACC_EXTENSIONS_ALLEXTENSIONS_H

namespace mlir {
class DialectRegistry;
namespace acc {
/// Register the standard OpenACC models for GPU, LLVM and MemRef IR. Models
/// for MemRef and LLVM also load the dialects used by their IR builders when
/// OpenACC is loaded. The GPU model attaches when GPU and OpenACC are loaded.
void registerAllExtensions(DialectRegistry &registry);
} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_EXTENSIONS_ALLEXTENSIONS_H
