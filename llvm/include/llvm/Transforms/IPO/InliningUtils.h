//===- InliningUtils.h - Shared inlining utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared utilities used by the inliner passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INLININGUTILS_H
#define LLVM_TRANSFORMS_IPO_INLININGUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

namespace llvm {

/// Check if Function F appears in the inline history chain.
/// InlineHistory is a vector of (Function, ParentHistoryID) pairs.
/// Returns true if F was already inlined in the chain leading to
/// InlineHistoryID.
inline bool inlineHistoryIncludes(
    Function *F, int InlineHistoryID,
    const SmallVectorImpl<std::pair<Function *, int>> &InlineHistory) {
  while (InlineHistoryID != -1) {
    assert(unsigned(InlineHistoryID) < InlineHistory.size() &&
           "Invalid inline history ID");
    if (InlineHistory[InlineHistoryID].first == F)
      return true;
    InlineHistoryID = InlineHistory[InlineHistoryID].second;
  }
  return false;
}

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INLININGUTILS_H
