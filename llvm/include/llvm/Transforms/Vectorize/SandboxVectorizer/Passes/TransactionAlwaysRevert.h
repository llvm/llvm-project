//===- TransactionAlwaysRevert.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a region pass that always reverts the transaction without checking
// its cost. This is mainly used as a final pass in lit tests.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONALWAYSREVERT_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONALWAYSREVERT_H

#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

class TransactionAlwaysRevert : public RegionPass {
public:
  TransactionAlwaysRevert() : RegionPass("tr-revert") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final {
    auto &Tracker = Rgn.getContext().getTracker();
    bool HasChanges = !Tracker.empty();
    Tracker.revert();
    return HasChanges;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TRANSACTIONALWAYSREVERT_H
