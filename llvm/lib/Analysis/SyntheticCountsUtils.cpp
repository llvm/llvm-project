//===--- SyntheticCountsUtils.cpp - synthetic counts propagation utils ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for propagating synthetic counts.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SyntheticCountsUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

// Given a set of functions in an SCC, propagate entry counts to functions
// called by the SCC.
static void
propagateFromSCC(const SmallPtrSetImpl<Function *> &SCCFunctions,
                 function_ref<Scaled64(CallSite CS)> GetCallSiteRelFreq,
                 function_ref<uint64_t(Function *F)> GetCount,
                 function_ref<void(Function *F, uint64_t)> AddToCount) {

  SmallVector<CallSite, 16> CallSites;

  // Gather all callsites in the SCC.
  auto GatherCallSites = [&]() {
    for (auto *F : SCCFunctions) {
      assert(F && !F->isDeclaration());
      for (auto &I : instructions(F)) {
        if (auto CS = CallSite(&I)) {
          CallSites.push_back(CS);
        }
      }
    }
  };

  GatherCallSites();

  // Partition callsites so that the callsites that call functions in the same
  // SCC come first.
  auto Mid = partition(CallSites, [&](CallSite &CS) {
    auto *Callee = CS.getCalledFunction();
    if (Callee)
      return SCCFunctions.count(Callee);
    // FIXME: Use the !callees metadata to propagate counts through indirect
    // calls.
    return 0U;
  });

  // For functions in the same SCC, update the counts in two steps:
  // 1. Compute the additional count for each function by propagating the counts
  // along all incoming edges to the function that originate from the same SCC
  // and summing them up.
  // 2. Add the additional counts to the functions in the SCC.
  // This ensures that the order of
  // traversal of functions within the SCC doesn't change the final result.

  DenseMap<Function *, uint64_t> AdditionalCounts;
  for (auto It = CallSites.begin(); It != Mid; It++) {
    auto &CS = *It;
    auto RelFreq = GetCallSiteRelFreq(CS);
    Function *Callee = CS.getCalledFunction();
    Function *Caller = CS.getCaller();
    RelFreq *= Scaled64(GetCount(Caller), 0);
    uint64_t AdditionalCount = RelFreq.toInt<uint64_t>();
    AdditionalCounts[Callee] += AdditionalCount;
  }

  // Update the counts for the functions in the SCC.
  for (auto &Entry : AdditionalCounts)
    AddToCount(Entry.first, Entry.second);

  // Now update the counts for functions not in SCC.
  for (auto It = Mid; It != CallSites.end(); It++) {
    auto &CS = *It;
    auto Weight = GetCallSiteRelFreq(CS);
    Function *Callee = CS.getCalledFunction();
    Function *Caller = CS.getCaller();
    Weight *= Scaled64(GetCount(Caller), 0);
    AddToCount(Callee, Weight.toInt<uint64_t>());
  }
}

/// Propgate synthetic entry counts on a callgraph.
///
/// This performs a reverse post-order traversal of the callgraph SCC. For each
/// SCC, it first propagates the entry counts to the functions within the SCC
/// through call edges and updates them in one shot. Then the entry counts are
/// propagated to functions outside the SCC.
void llvm::propagateSyntheticCounts(
    const CallGraph &CG, function_ref<Scaled64(CallSite CS)> GetCallSiteRelFreq,
    function_ref<uint64_t(Function *F)> GetCount,
    function_ref<void(Function *F, uint64_t)> AddToCount) {

  SmallVector<SmallPtrSet<Function *, 8>, 16> SCCs;
  for (auto I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    auto SCC = *I;

    SmallPtrSet<Function *, 8> SCCFunctions;
    for (auto *Node : SCC) {
      Function *F = Node->getFunction();
      if (F && !F->isDeclaration()) {
        SCCFunctions.insert(F);
      }
    }
    SCCs.push_back(SCCFunctions);
  }

  for (auto &SCCFunctions : reverse(SCCs))
    propagateFromSCC(SCCFunctions, GetCallSiteRelFreq, GetCount, AddToCount);
}
