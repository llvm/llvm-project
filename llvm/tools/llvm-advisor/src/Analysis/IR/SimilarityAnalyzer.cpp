//===--- SimilarityAnalyzer.cpp - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/SimilarityAnalyzer.h"
#include "Analysis/IR/IRAnalysisUtils.h"
#include "llvm/IR/Instruction.h"

using namespace llvm;
using namespace llvm::advisor;

static DenseMap<unsigned, int64_t> opcodeHistogram(const Function &F) {
  DenseMap<unsigned, int64_t> Hist;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      ++Hist[I.getOpcode()];
  return Hist;
}

static double jaccardLike(const DenseMap<unsigned, int64_t> &A,
                          const DenseMap<unsigned, int64_t> &B) {
  double MinSum = 0.0;
  double MaxSum = 0.0;
  DenseSet<unsigned> Keys;
  for (const auto &KV : A)
    Keys.insert(KV.first);
  for (const auto &KV : B)
    Keys.insert(KV.first);
  for (unsigned K : Keys) {
    int64_t VA = A.lookup(K);
    int64_t VB = B.lookup(K);
    MinSum += std::min(VA, VB);
    MaxSum += std::max(VA, VB);
  }
  if (MaxSum == 0.0)
    return 1.0;
  return MinSum / MaxSum;
}

Expected<std::unique_ptr<CapabilityResult>>
SimilarityAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withIRModule(Context, CapID, UnitID,
                      [&](LLVMContext &, Module &M) {
    SmallVector<const Function *, 32> Funcs;
    DenseMap<const Function *, DenseMap<unsigned, int64_t>> Histograms;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      Funcs.push_back(&F);
      Histograms[&F] = opcodeHistogram(F);
    }

    json::Array Pairs;
    for (size_t I = 0, E = Funcs.size(); I < E; ++I) {
      for (size_t J = I + 1; J < E; ++J) {
        const Function *A = Funcs[I];
        const Function *B = Funcs[J];
        double Score = jaccardLike(Histograms[A], Histograms[B]);
        if (Score < 0.5)
          continue;
        Pairs.push_back(json::Object{
            {"lhs", A->getName()},
            {"rhs", B->getName()},
            {"score", Score},
        });
      }
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"similar_pairs", std::move(Pairs)},
    });
  });
}
