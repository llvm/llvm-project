//===- ProfileInfo.cpp - Profile Info Interface ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the abstract ProfileInfo interface, and the default
// "no profile" implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include <set>
using namespace llvm;

// Register the ProfileInfo interface, providing a nice name to refer to.
static RegisterAnalysisGroup<ProfileInfo> Z("Profile Information");
char ProfileInfo::ID = 0;

ProfileInfo::~ProfileInfo() {}

unsigned ProfileInfo::getExecutionCount(BasicBlock *BB) const {
  pred_iterator PI = pred_begin(BB), PE = pred_end(BB);

  // Are there zero predecessors of this block?
  if (PI == PE) {
    // If this is the entry block, look for the Null -> Entry edge.
    if (BB == &BB->getParent()->getEntryBlock())
      return getEdgeWeight(0, BB);
    else
      return 0;   // Otherwise, this is a dead block.
  }

  // Otherwise, if there are predecessors, the execution count of this block is
  // the sum of the edge frequencies from the incoming edges.  Note that if
  // there are multiple edges from a predecessor to this block that we don't
  // want to count its weight multiple times.  For this reason, we keep track of
  // the predecessors we've seen and only count them if we haven't run into them
  // yet.
  //
  // We don't want to create an std::set unless we are dealing with a block that
  // has a LARGE number of in-edges.  Handle the common case of having only a
  // few in-edges with special code.
  //
  BasicBlock *FirstPred = *PI;
  unsigned Count = getEdgeWeight(FirstPred, BB);
  ++PI;
  if (PI == PE) return Count;   // Quick exit for single predecessor blocks

  BasicBlock *SecondPred = *PI;
  if (SecondPred != FirstPred) Count += getEdgeWeight(SecondPred, BB);
  ++PI;
  if (PI == PE) return Count;   // Quick exit for two predecessor blocks

  BasicBlock *ThirdPred = *PI;
  if (ThirdPred != FirstPred && ThirdPred != SecondPred)
    Count += getEdgeWeight(ThirdPred, BB);
  ++PI;
  if (PI == PE) return Count;   // Quick exit for three predecessor blocks

  std::set<BasicBlock*> ProcessedPreds;
  ProcessedPreds.insert(FirstPred);
  ProcessedPreds.insert(SecondPred);
  ProcessedPreds.insert(ThirdPred);
  for (; PI != PE; ++PI)
    if (ProcessedPreds.insert(*PI).second)
      Count += getEdgeWeight(*PI, BB);
  return Count;
}



//===----------------------------------------------------------------------===//
//  NoProfile ProfileInfo implementation
//

namespace {
  struct VISIBILITY_HIDDEN NoProfileInfo 
    : public ImmutablePass, public ProfileInfo {
    static char ID; // Class identification, replacement for typeinfo
    NoProfileInfo() : ImmutablePass(&ID) {}
  };
}  // End of anonymous namespace

char NoProfileInfo::ID = 0;
// Register this pass...
static RegisterPass<NoProfileInfo>
X("no-profile", "No Profile Information", false, true);

// Declare that we implement the ProfileInfo interface
static RegisterAnalysisGroup<ProfileInfo, true> Y(X);

ImmutablePass *llvm::createNoProfileInfoPass() { return new NoProfileInfo(); }
