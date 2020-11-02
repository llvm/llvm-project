//===- FrozenRewritePatternList.h - FrozenRewritePatternList ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_FROZENREWRITEPATTERNLIST_H
#define MLIR_REWRITE_FROZENREWRITEPATTERNLIST_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
/// This class represents a frozen set of patterns that can be processed by a
/// pattern applicator. This class is designed to enable caching pattern lists
/// such that they need not be continuously recomputed.
class FrozenRewritePatternList {
  using PatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  /// Freeze the patterns held in `patterns`, and take ownership.
  FrozenRewritePatternList(OwningRewritePatternList &&patterns);

  /// Return the patterns held by this list.
  iterator_range<llvm::pointee_iterator<PatternListT::const_iterator>>
  getPatterns() const {
    return llvm::make_pointee_range(patterns);
  }

private:
  /// The patterns held by this list.
  std::vector<std::unique_ptr<RewritePattern>> patterns;
};

} // end namespace mlir

#endif // MLIR_REWRITE_FROZENREWRITEPATTERNLIST_H
