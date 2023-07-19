//===- BufferizationToMemRef.h - Bufferization to MemRef conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
#define MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTBUFFERIZATIONTOMEMREF
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createBufferizationToMemRefPass();
} // namespace mlir

#endif // MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
