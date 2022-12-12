//===- TransformUtils.h - Transform Dialect Utils ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMUTILS_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMUTILS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace transform {

/// A simple pattern rewriter that can be constructed from a context. This is
/// necessary to apply patterns to a specific op locally.
class TrivialPatternRewriter : public PatternRewriter {
public:
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};

} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMUTILS_H
