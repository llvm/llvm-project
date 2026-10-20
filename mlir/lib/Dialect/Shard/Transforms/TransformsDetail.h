//===- TransformsDetail.h - -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H
#define MLIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace shard {

template <typename Op>
struct OpRewritePatternWithSymbolTableCollection : OpRewritePattern<Op> {
  template <typename... OpRewritePatternArgs>
  OpRewritePatternWithSymbolTableCollection(
      SymbolTableCollection &symbolTableCollection,
      OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern<Op>(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection) {}

protected:
  SymbolTableCollection &symbolTableCollection;
};

} // namespace shard
} // namespace mlir

#endif // MLIR_DIALECT_SHARD_TRANSFORMS_TRANSFORMSDETAIL_H
