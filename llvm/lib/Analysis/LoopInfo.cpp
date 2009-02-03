//===- LoopInfo.cpp - Natural Loop Calculator -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that the
// loops identified may actually be several natural loops that share the same
// header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <ostream>
using namespace llvm;

char LoopInfo::ID = 0;
static RegisterPass<LoopInfo>
X("loops", "Natural Loop Construction", true, true);

//===----------------------------------------------------------------------===//
// Loop implementation
//

//===----------------------------------------------------------------------===//
// LoopInfo implementation
//
bool LoopInfo::runOnFunction(Function &) {
  releaseMemory();
  LI->Calculate(getAnalysis<DominatorTree>().getBase());    // Update
  return false;
}

void LoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTree>();
}
