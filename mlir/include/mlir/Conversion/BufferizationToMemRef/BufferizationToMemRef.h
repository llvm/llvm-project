//===- BufferizationToMemRef.h - Bufferization to MemRef conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
#define MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

/// Collect a set of patterns to convert memory-related operations from the
/// Bufferization dialect to the MemRef dialect.
void populateBufferizationToMemRefConversionPatterns(
    RewritePatternSet &patterns);

std::unique_ptr<Pass> createBufferizationToMemRefPass();
} // namespace mlir

#endif // MLIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
