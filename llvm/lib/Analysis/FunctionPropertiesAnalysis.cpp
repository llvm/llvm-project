//===- FunctionPropertiesAnalysis.cpp - Function Properties Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include <deque>

using namespace llvm;

namespace {
int64_t getNrBlocksFromCond(const BasicBlock &BB) {
  int64_t Ret = 0;
  if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
    if (BI->isConditional())
      Ret += BI->getNumSuccessors();
  } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
    Ret += (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
  }
  return Ret;
}

int64_t getUses(const Function &F) {
  return ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();
}
} // namespace

void FunctionPropertiesInfo::reIncludeBB(const BasicBlock &BB) {
  updateForBB(BB, +1);
}

void FunctionPropertiesInfo::updateForBB(const BasicBlock &BB,
                                         int64_t Direction) {
  assert(Direction == 1 || Direction == -1);
  BasicBlockCount += Direction;
  BlocksReachedFromConditionalInstruction +=
      (Direction * getNrBlocksFromCond(BB));
  for (const auto &I : BB) {
    if (auto *CS = dyn_cast<CallBase>(&I)) {
      const auto *Callee = CS->getCalledFunction();
      if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
        DirectCallsToDefinedFunctions += Direction;
    }
    if (I.getOpcode() == Instruction::Load) {
      LoadInstCount += Direction;
    } else if (I.getOpcode() == Instruction::Store) {
      StoreInstCount += Direction;
    }
  }
  TotalInstructionCount += Direction * BB.sizeWithoutDebug();
}

void FunctionPropertiesInfo::updateAggregateStats(const Function &F,
                                                  const LoopInfo &LI) {

  Uses = getUses(F);
  TopLevelLoopCount = llvm::size(LI);
  MaxLoopDepth = 0;
  std::deque<const Loop *> Worklist;
  llvm::append_range(Worklist, LI);
  while (!Worklist.empty()) {
    const auto *L = Worklist.front();
    MaxLoopDepth =
        std::max(MaxLoopDepth, static_cast<int64_t>(L->getLoopDepth()));
    Worklist.pop_front();
    llvm::append_range(Worklist, L->getSubLoops());
  }
}

FunctionPropertiesInfo FunctionPropertiesInfo::getFunctionPropertiesInfo(
    const Function &F, FunctionAnalysisManager &FAM) {

  FunctionPropertiesInfo FPI;
  // The const casts are due to the getResult API - there's no mutation of F.
  const auto &LI = FAM.getResult<LoopAnalysis>(const_cast<Function &>(F));
  const auto &DT =
      FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(F));
  for (const auto &BB : F)
    if (DT.isReachableFromEntry(&BB))
      FPI.reIncludeBB(BB);
  FPI.updateAggregateStats(F, LI);
  return FPI;
}

void FunctionPropertiesInfo::print(raw_ostream &OS) const {
  OS << "BasicBlockCount: " << BasicBlockCount << "\n"
     << "BlocksReachedFromConditionalInstruction: "
     << BlocksReachedFromConditionalInstruction << "\n"
     << "Uses: " << Uses << "\n"
     << "DirectCallsToDefinedFunctions: " << DirectCallsToDefinedFunctions
     << "\n"
     << "LoadInstCount: " << LoadInstCount << "\n"
     << "StoreInstCount: " << StoreInstCount << "\n"
     << "MaxLoopDepth: " << MaxLoopDepth << "\n"
     << "TopLevelLoopCount: " << TopLevelLoopCount << "\n"
     << "TotalInstructionCount: " << TotalInstructionCount << "\n\n";
}

AnalysisKey FunctionPropertiesAnalysis::Key;

FunctionPropertiesInfo
FunctionPropertiesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPropertiesInfo::getFunctionPropertiesInfo(F, FAM);
}

PreservedAnalyses
FunctionPropertiesPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of CFA for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<FunctionPropertiesAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

FunctionPropertiesUpdater::FunctionPropertiesUpdater(
    FunctionPropertiesInfo &FPI, const CallBase &CB)
    : FPI(FPI), CallSiteBB(*CB.getParent()), Caller(*CallSiteBB.getParent()) {

  // For BBs that are likely to change, we subtract from feature totals their
  // contribution. Some features, like max loop counts or depths, are left
  // invalid, as they will be updated post-inlining.
  SmallPtrSet<const BasicBlock *, 4> LikelyToChangeBBs;
  // The CB BB will change - it'll either be split or the callee's body (single
  // BB) will be pasted in.
  LikelyToChangeBBs.insert(&CallSiteBB);

  // The caller's entry BB may change due to new alloca instructions.
  LikelyToChangeBBs.insert(&*Caller.begin());

  // The successors may become unreachable in the case of `invoke` inlining.
  // We track successors separately, too, because they form a boundary, together
  // with the CB BB ('Entry') between which the inlined callee will be pasted.
  Successors.insert(succ_begin(&CallSiteBB), succ_end(&CallSiteBB));

  // Exclude the CallSiteBB, if it happens to be its own successor (1-BB loop).
  // We are only interested in BBs the graph moves past the callsite BB to
  // define the frontier past which we don't want to re-process BBs. Including
  // the callsite BB in this case would prematurely stop the traversal in
  // finish().
  Successors.erase(&CallSiteBB);

  for (const auto *BB : Successors)
    LikelyToChangeBBs.insert(BB);

  // Commit the change. While some of the BBs accounted for above may play dual
  // role - e.g. caller's entry BB may be the same as the callsite BB - set
  // insertion semantics make sure we account them once. This needs to be
  // followed in `finish`, too.
  for (const auto *BB : LikelyToChangeBBs)
    FPI.updateForBB(*BB, -1);
}

void FunctionPropertiesUpdater::finish(FunctionAnalysisManager &FAM) const {
  SetVector<const BasicBlock *> Worklist;

  if (&CallSiteBB != &*Caller.begin()) {
    FPI.reIncludeBB(*Caller.begin());
    Worklist.insert(&*Caller.begin());
  }

  // Update feature values from the BBs that were copied from the callee, or
  // might have been modified because of inlining. The latter have been
  // subtracted in the FunctionPropertiesUpdater ctor.
  // There could be successors that were reached before but now are only
  // reachable from elsewhere in the CFG.
  // Suppose the following diamond CFG (lines are arrows pointing down):
  //    A
  //  /   \
  // B     C
  //  \   /
  //    D
  // There's a call site in C that is inlined. Upon doing that, it turns out
  // it expands to
  //   call void @llvm.trap()
  //   unreachable
  // D isn't reachable from C anymore, but we did discount it when we set up
  // FunctionPropertiesUpdater, so we need to re-include it here.

  const auto &DT =
      FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(Caller));
  for (const auto *Succ : Successors)
    if (DT.isReachableFromEntry(Succ) && Worklist.insert(Succ))
      FPI.reIncludeBB(*Succ);

  auto I = Worklist.size();
  bool CSInsertion = Worklist.insert(&CallSiteBB);
  (void)CSInsertion;
  assert(CSInsertion);
  for (; I < Worklist.size(); ++I) {
    const auto *BB = Worklist[I];
    FPI.reIncludeBB(*BB);
    for (const auto *Succ : successors(BB))
      Worklist.insert(Succ);
  }

  const auto &LI = FAM.getResult<LoopAnalysis>(const_cast<Function &>(Caller));
  FPI.updateAggregateStats(Caller, LI);
  assert(FPI == FunctionPropertiesInfo::getFunctionPropertiesInfo(Caller, FAM));
}
