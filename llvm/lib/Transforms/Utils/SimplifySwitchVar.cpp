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
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::allInSet<CFGAnalyses>();
}
