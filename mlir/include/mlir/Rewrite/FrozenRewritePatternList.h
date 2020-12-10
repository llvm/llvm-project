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
namespace detail {
class PDLByteCode;
} // end namespace detail

/// This class represents a frozen set of patterns that can be processed by a
/// pattern applicator. This class is designed to enable caching pattern lists
/// such that they need not be continuously recomputed.
class FrozenRewritePatternList {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  /// Freeze the patterns held in `patterns`, and take ownership.
  FrozenRewritePatternList(OwningRewritePatternList &&patterns);
  FrozenRewritePatternList(FrozenRewritePatternList &&patterns);
  ~FrozenRewritePatternList();

  /// Return the native patterns held by this list.
  iterator_range<llvm::pointee_iterator<NativePatternListT::const_iterator>>
  getNativePatterns() const {
    return llvm::make_pointee_range(nativePatterns);
  }

  /// Return the compiled PDL bytecode held by this list. Returns null if
  /// there are no PDL patterns within the list.
  const detail::PDLByteCode *getPDLByteCode() const {
    return pdlByteCode.get();
  }

private:
  /// The set of.
  std::vector<std::unique_ptr<RewritePattern>> nativePatterns;

  /// The bytecode containing the compiled PDL patterns.
  std::unique_ptr<detail::PDLByteCode> pdlByteCode;
};

} // end namespace mlir

#endif // MLIR_REWRITE_FROZENREWRITEPATTERNLIST_H
