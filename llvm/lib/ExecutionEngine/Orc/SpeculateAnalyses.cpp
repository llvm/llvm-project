//===-- SpeculateAnalyses.cpp  --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SpeculateAnalyses.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"

namespace {
using namespace llvm;
std::vector<const BasicBlock *> findBBwithCalls(const Function &F,
                                                bool IndirectCall = false) {
  std::vector<const BasicBlock *> BBs;

  auto findCallInst = [&IndirectCall](const Instruction &I) {
    if (auto Call = dyn_cast<CallBase>(&I)) {
      if (Call->isIndirectCall())
        return IndirectCall;
      else
        return true;
    } else
      return false;
  };
  for (auto &BB : F)
    if (findCallInst(*BB.getTerminator()) ||
        llvm::any_of(BB.instructionsWithoutDebug(), findCallInst))
      BBs.emplace_back(&BB);

  return BBs;
}
} // namespace

// Implementations of Queries shouldn't need to lock the resources
// such as LLVMContext, each argument (function) has a non-shared LLVMContext
namespace llvm {
namespace orc {

// Collect direct calls only
void BlockFreqQuery::findCalles(const BasicBlock *BB,
                                DenseSet<StringRef> &CallesNames) {
  assert(BB != nullptr && "Traversing Null BB to find calls?");

  auto getCalledFunction = [&CallesNames](const CallBase *Call) {
    auto CalledValue = Call->getCalledOperand()->stripPointerCasts();
    if (auto DirectCall = dyn_cast<Function>(CalledValue))
      CallesNames.insert(DirectCall->getName());
  };
  for (auto &I : BB->instructionsWithoutDebug())
    if (auto CI = dyn_cast<CallInst>(&I))
      getCalledFunction(CI);

  if (auto II = dyn_cast<InvokeInst>(BB->getTerminator()))
    getCalledFunction(II);
}

// blind calculation
size_t BlockFreqQuery::numBBToGet(size_t numBB) {
  // small CFG
  if (numBB < 4)
    return numBB;
  // mid-size CFG
  else if (numBB < 20)
    return (numBB / 2);
  else
    return (numBB / 2) + (numBB / 4);
}

BlockFreqQuery::ResultTy BlockFreqQuery::
operator()(Function &F, FunctionAnalysisManager &FAM) {
  DenseMap<StringRef, DenseSet<StringRef>> CallerAndCalles;
  DenseSet<StringRef> Calles;
  SmallVector<std::pair<const BasicBlock *, uint64_t>, 8> BBFreqs;

  auto IBBs = findBBwithCalls(F);

  if (IBBs.empty())
    return None;

  auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);

  for (const auto I : IBBs)
    BBFreqs.push_back({I, BFI.getBlockFreq(I).getFrequency()});

  assert(IBBs.size() == BBFreqs.size() && "BB Count Mismatch");

  llvm::sort(BBFreqs.begin(), BBFreqs.end(),
             [](decltype(BBFreqs)::const_reference BBF,
                decltype(BBFreqs)::const_reference BBS) {
               return BBF.second > BBS.second ? true : false;
             });

  // ignoring number of direct calls in a BB
  auto Topk = numBBToGet(BBFreqs.size());

  for (size_t i = 0; i < Topk; i++)
    findCalles(BBFreqs[i].first, Calles);

  assert(!Calles.empty() && "Running Analysis on Function with no calls?");

  CallerAndCalles.insert({F.getName(), std::move(Calles)});

  return CallerAndCalles;
}
} // namespace orc
} // namespace llvm
