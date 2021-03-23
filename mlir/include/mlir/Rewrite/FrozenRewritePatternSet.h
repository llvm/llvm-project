//===- FrozenRewritePatternSet.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_FROZENREWRITEPATTERNSET_H
#define MLIR_REWRITE_FROZENREWRITEPATTERNSET_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace detail {
class PDLByteCode;
} // end namespace detail

/// This class represents a frozen set of patterns that can be processed by a
/// pattern applicator. This class is designed to enable caching pattern lists
/// such that they need not be continuously recomputed. Note that all copies of
/// this class share the same compiled pattern list, allowing for a reduction in
/// the number of duplicated patterns that need to be created.
class FrozenRewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  /// Freeze the patterns held in `patterns`, and take ownership.
  FrozenRewritePatternSet();
  FrozenRewritePatternSet(RewritePatternSet &&patterns);
  FrozenRewritePatternSet(FrozenRewritePatternSet &&patterns) = default;
  FrozenRewritePatternSet(const FrozenRewritePatternSet &patterns) = default;
  FrozenRewritePatternSet &
  operator=(const FrozenRewritePatternSet &patterns) = default;
  FrozenRewritePatternSet &
  operator=(FrozenRewritePatternSet &&patterns) = default;
  ~FrozenRewritePatternSet();

  /// Return the native patterns held by this list.
  iterator_range<llvm::pointee_iterator<NativePatternListT::const_iterator>>
  getNativePatterns() const {
    const NativePatternListT &nativePatterns = impl->nativePatterns;
    return llvm::make_pointee_range(nativePatterns);
  }

  /// Return the compiled PDL bytecode held by this list. Returns null if
  /// there are no PDL patterns within the list.
  const detail::PDLByteCode *getPDLByteCode() const {
    return impl->pdlByteCode.get();
  }

private:
  /// The internal implementation of the frozen pattern list.
  struct Impl {
    /// The set of native C++ rewrite patterns.
    NativePatternListT nativePatterns;

    /// The bytecode containing the compiled PDL patterns.
    std::unique_ptr<detail::PDLByteCode> pdlByteCode;
  };

  /// A pointer to the internal pattern list. This uses a shared_ptr to avoid
  /// the need to compile the same pattern list multiple times. For example,
  /// during multi-threaded pass execution, all copies of a pass can share the
  /// same pattern list.
  std::shared_ptr<Impl> impl;
};

// TODO: FrozenRewritePatternList is soft-deprecated and will be removed in the
// future.
using FrozenRewritePatternList = FrozenRewritePatternSet;

} // end namespace mlir

#endif // MLIR_REWRITE_FROZENREWRITEPATTERNSET_H
