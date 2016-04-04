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

#include "llvm/IR/InstVisitor.h"
#include <vector>

namespace llvm {
// Visitor class that finds all indirect call sites.
struct PGOIndirectCallSiteVisitor
    : public InstVisitor<PGOIndirectCallSiteVisitor> {
  std::vector<Instruction *> IndirectCallInsts;
  PGOIndirectCallSiteVisitor() {}

  void visitCallSite(CallSite CS) {
    if (CS.getCalledFunction() || !CS.getCalledValue())
      return;
    Instruction *I = CS.getInstruction();
    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      if (CI->isInlineAsm())
        return;
    }
    if (isa<Constant>(CS.getCalledValue()))
      return;
    IndirectCallInsts.push_back(I);
  }
};

// Helper function that finds all indirect call sites.
static inline std::vector<Instruction *> findIndirectCallSites(Function &F) {
  PGOIndirectCallSiteVisitor ICV;
  ICV.visit(F);
  return ICV.IndirectCallInsts;
}
}
