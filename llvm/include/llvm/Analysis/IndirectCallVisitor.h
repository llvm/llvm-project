//===-- IndirectCallVisitor.h - indirect call visitor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements defines a visitor class and a helper function that find
// all indirect call-sites in a function.

#ifndef LLVM_ANALYSIS_INDIRECTCALLVISITOR_H
#define LLVM_ANALYSIS_INDIRECTCALLVISITOR_H

#include "llvm/IR/InstVisitor.h"
#include <vector>

namespace llvm {
// Visitor class that finds all indirect call.
struct PGOIndirectCallVisitor : public InstVisitor<PGOIndirectCallVisitor> {
  std::vector<CallBase *> IndirectCalls;
  PGOIndirectCallVisitor() = default;

  void visitCallBase(CallBase &Call) {
    if (Call.isIndirectCall())
      IndirectCalls.push_back(&Call);
  }
};

// Helper function that finds all indirect call sites.
inline std::vector<CallBase *> findIndirectCalls(Function &F) {
  PGOIndirectCallVisitor ICV;
  ICV.visit(F);
  return ICV.IndirectCalls;
}
} // namespace llvm

#endif
