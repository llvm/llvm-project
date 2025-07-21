//===-- SimplifySwitchVar.cpp - Switch Variable simplification ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements switch variable simplification. It looks for a
/// linear relationship between the case value of a switch and the constant
/// offset of an operation. Knowing this relationship, we can simplify
/// multiple individual operations into a single, more generic one, which
/// can help with further optimizations.
///
/// It is similar to SimplifyIndVar, but instead of looking at an
/// induction variable and modeling its scalar evolution over
/// multiple iterations, it analyzes the switch variable and
/// models how it affects constant offsets.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifySwitchVar.h"
#include "llvm/IR/PatternMatch.h"
#include <random>

using namespace llvm;
using namespace PatternMatch;

/// Return the BB, where (most of) the cases meet.
/// In that BB are phi nodes, that contain the case BBs.
static BasicBlock *findMostCommonSuccessor(SwitchInst *Switch) {
  uint64_t Max = 0;
  BasicBlock *MostCommonSuccessor = nullptr;

  for (auto &Case : Switch->cases()) {
    auto *CaseBB = Case.getCaseSuccessor();
    auto GetNumPredecessors = [](BasicBlock *BB) -> uint64_t {
      return std::distance(predecessors(BB).begin(), predecessors(BB).end());
    };

    auto Length = GetNumPredecessors(CaseBB);

    if (Length > Max) {
      Max = Length;
      MostCommonSuccessor = CaseBB;
    }

    for (auto *Successor : successors(CaseBB)) {
      auto Length = GetNumPredecessors(Successor);
      if (Length > Max) {
        Max = Length;
        MostCommonSuccessor = Successor;
      }
    }
  }

  return MostCommonSuccessor;
}

namespace {
struct PhiCase {
  int PhiIndex;
  Value *IncomingValue;
  ConstantInt *CaseValue;
};
} // namespace

/// Collect the incoming value, index and associated case value from a phi node.
/// Ignores incoming values, which do not come from a case BB from the switch.
static SmallVector<PhiCase> collectPhiCases(PHINode &Phi, SwitchInst *Switch) {
  SmallVector<PhiCase> PhiInputs;

  for (auto *IncomingBlock : Phi.blocks()) {
    auto *CaseVal = Switch->findCaseDest(IncomingBlock);
    if (!CaseVal)
      continue;

    auto PhiIdx = Phi.getBasicBlockIndex(IncomingBlock);
    PhiInputs.push_back({PhiIdx, Phi.getIncomingValue(PhiIdx), CaseVal});
  }
  return PhiInputs;
}

namespace {
enum SupportedOp {
  GetElementPtr,
  IntegerAdd,
  Unsupported,
};

struct NewInstParameters {
  SupportedOp Op;
  Value *BaseValue;
  Type *OffsetTy;
};
} // namespace

/// Find the common Base Value, Operation type and Index Type of the found phi
/// incoming values.
static NewInstParameters findInstParameters(SmallVector<PhiCase> &PhiCases) {
  auto Op = SupportedOp::Unsupported;
  DenseMap<Value *, uint64_t> BaseAddressCounts;
  Type *OffsetTy = nullptr;

  for (auto &Case : PhiCases) {
    auto *GEP = dyn_cast<GetElementPtrInst>(Case.IncomingValue);
    bool IsAdd =
        match(Case.IncomingValue, m_Add(m_Value(), m_AnyIntegralConstant()));

    if (GEP) {
      Op = SupportedOp::GetElementPtr;
      BaseAddressCounts[GEP->getPointerOperand()] += 1;
      OffsetTy = GEP->getOperand(GEP->getNumIndices())->getType();
      continue;
    }

    if (IsAdd) {
      Op = SupportedOp::IntegerAdd;
      auto *Add = cast<Instruction>(Case.IncomingValue);
      BaseAddressCounts[Add->getOperand(Add->getNumOperands() - 2)] += 1;
      OffsetTy = Add->getOperand(Add->getNumOperands() - 1)->getType();
      continue;
    }

    BaseAddressCounts[Case.IncomingValue] += 1;
  }

  unsigned Max = 0;
  Value *BaseValue;
  for (auto &Base : BaseAddressCounts) {
    if (Base.second > Max) {
      BaseValue = Base.first;
      Max = Base.second;
    }
  }

  return {Op, BaseValue, OffsetTy};
}

/// Collect valid cases.
/// A case is valid if it uses the same base value (or is the base value like a
/// pointer from an alloca) and it has a constant offset.
static SmallVector<PhiCase>
collectValidCases(SmallVector<PhiCase> &PhiCases,
                  NewInstParameters NewInstParameters,
                  DenseMap<int64_t, int64_t> &CaseOffsetMap) {
  SmallVector<PhiCase> FilteredCases;
  auto *BaseValue = NewInstParameters.BaseValue;
  auto CurrentOp = NewInstParameters.Op;

  switch (CurrentOp) {
  case SupportedOp::GetElementPtr: {
    for (auto &Case : PhiCases) {
      auto *GEP = dyn_cast<GetElementPtrInst>(Case.IncomingValue);

      if (!GEP) {
        if (Case.IncomingValue != BaseValue) {
          continue;
        }
        CaseOffsetMap[Case.CaseValue->getSExtValue()] = 0;
        FilteredCases.push_back(Case);
        continue;
      }

      if (GEP->getPointerOperand() != BaseValue) {
        continue;
      }

      auto &DL = GEP->getParent()->getDataLayout();
      APInt Offset(DL.getTypeSizeInBits(GEP->getPointerOperandType()), 0);
      if (!GEP->accumulateConstantOffset(GEP->getDataLayout(), Offset)) {
        continue;
      }
      CaseOffsetMap[Case.CaseValue->getSExtValue()] = Offset.getSExtValue();
      FilteredCases.push_back(Case);
    }
    break;
  }
  case SupportedOp::IntegerAdd: {
    for (auto &Case : PhiCases) {
      bool IsAdd =
          match(Case.IncomingValue, m_Add(m_Value(), m_AnyIntegralConstant()));

      if (!IsAdd) {
        if (Case.IncomingValue != BaseValue) {
          continue;
        }
        CaseOffsetMap[Case.CaseValue->getSExtValue()] = 0;
        FilteredCases.push_back(Case);
        continue;
      }

      auto *AddInst = dyn_cast<Instruction>(Case.IncomingValue);
      if (AddInst->getOperand(0) != BaseValue) {
        continue;
      }
      auto *Offset = cast<ConstantInt>(AddInst->getOperand(1));
      if (!Offset) {
        continue;
      }

      CaseOffsetMap[Case.CaseValue->getSExtValue()] = Offset->getSExtValue();
      FilteredCases.push_back(Case);
      continue;
    }
    break;
  }
  case SupportedOp::Unsupported: {
    llvm_unreachable("Unsupported Operation for SimplifySwitchVar.");
  }
  }
  return FilteredCases;
}

namespace {
struct FuncParams {
  int64_t Slope;
  int64_t Bias;
};
} // namespace

using RandomEngine = std::minstd_rand;
using RandomDistribution = std::uniform_int_distribution<int>;
/// Find the linear function that models the switch variable progression.
/// Uses random sampling to find the best fit, even if outliers are present.
/// Abort if there are too many outliers (> 50%)
static std::optional<FuncParams>
findLinearFunction(DenseMap<int64_t, int64_t> &Cases,
                   SmallVector<PhiCase> &PhiCases) {
  RandomEngine Rand(0xdeadbeef);
  RandomDistribution RandDist(0, Cases.size() - 1);

  // Repeat the process at most 5 times, because if at least 80% of the points
  // lie on the line, then we will find the line with 99.4% probability within 5
  // tries. See https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
  for (int I = 0; I < 5; ++I) {
    auto Index0 = RandDist(Rand);
    auto Index1 = RandDist(Rand);
    while (Index0 == Index1) {
      Index1 = RandDist(Rand);
    }

    auto X0 = PhiCases[Index0].CaseValue->getSExtValue();
    auto X1 = PhiCases[Index1].CaseValue->getSExtValue();
    auto Y0 = Cases[X0];
    auto Y1 = Cases[X1];

    int64_t Slope = (Y1 - Y0) / (X1 - X0);
    int64_t Bias = (Y0 - (Slope * X0));

    auto Count = llvm::count_if(Cases, [Bias, Slope](auto Case) {
      return Slope * Case.first + Bias == Case.second;
    });

    float InlierRatio = (float)Count / (float)Cases.size();
    if (InlierRatio > 0.5f) {
      return std::optional<FuncParams>({Slope, Bias});
    }
  }

  return std::nullopt;
}

PreservedAnalyses SimplifySwitchVarPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  bool Changed = false;
  BasicBlock *MostCommonSuccessor;
  // collect switch insts
  for (auto &BB : F) {
    if (auto *Switch = dyn_cast<SwitchInst>(BB.getTerminator())) {
      // get the most common successor for the phi nodes
      MostCommonSuccessor = findMostCommonSuccessor(Switch);

      for (auto &Phi : MostCommonSuccessor->phis()) {
        // filter out the phis, whose incoming blocks do not come from the
        // switch
        if (none_of(Phi.blocks(), [&Switch](BasicBlock *BB) {
              return Switch->findCaseDest(BB) != nullptr;
            }))
          continue;
        SmallVector<PhiCase> PhiCases = collectPhiCases(Phi, Switch);

        auto InstParameters = findInstParameters(PhiCases);
        if (InstParameters.Op == SupportedOp::Unsupported)
          continue;

        DenseMap<int64_t, int64_t> CaseOffsetMap;
        PhiCases = collectValidCases(PhiCases, InstParameters, CaseOffsetMap);
        if (CaseOffsetMap.size() < 2)
          continue;

        auto FuncParams = findLinearFunction(CaseOffsetMap, PhiCases);
        if (!FuncParams.has_value())
          continue;

        auto F = FuncParams.value();
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::allInSet<CFGAnalyses>();
}
