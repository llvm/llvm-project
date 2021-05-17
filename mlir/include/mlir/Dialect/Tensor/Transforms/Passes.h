//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Bufferize.h"

namespace mlir {

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateTensorBufferizePatterns(BufferizeTypeConverter &typeConverter,
                                     RewritePatternSet &patterns);

/// Creates an instance of `tensor` dialect bufferization pass.
std::unique_ptr<Pass> createTensorBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace tensor {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace tensor

} // end namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES_H_
