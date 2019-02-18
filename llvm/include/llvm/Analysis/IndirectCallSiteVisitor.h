//===-- IndirectCallSiteVisitor.h - indirect call-sites visitor -----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements defines a visitor class and a helper function that find
// all indirect call-sites in a function.

#ifndef LLVM_ANALYSIS_INDIRECTCALLSITEVISITOR_H
#define LLVM_ANALYSIS_INDIRECTCALLSITEVISITOR_H

#include "llvm/IR/InstVisitor.h"
#include <vector>

namespace llvm {
// Visitor class that finds all indirect call sites.
struct PGOIndirectCallSiteVisitor
    : public InstVisitor<PGOIndirectCallSiteVisitor> {
  std::vector<Instruction *> IndirectCallInsts;
  PGOIndirectCallSiteVisitor() {}

  void visitCallSite(CallSite CS) {
    if (CS.isIndirectCall())
      IndirectCallInsts.push_back(CS.getInstruction());
  }
};

// Helper function that finds all indirect call sites.
inline std::vector<Instruction *> findIndirectCallSites(Function &F) {
  PGOIndirectCallSiteVisitor ICV;
  ICV.visit(F);
  return ICV.IndirectCallInsts;
}
}

#endif
