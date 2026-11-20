//===- HoistZeroCostConditionalValues.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// StructurizeCFG serializes the arms of an if/else diamond into conditional
// regions connected by Flow blocks. The predicates keep the arms mutually
// exclusive, but the CFG contains a Then -> Flow -> Else path. Machine liveness
// does not generally track that predicate relationship, so the Then value can
// remain live through Else and interfere with the Else value. Even an Else
// instruction with zero target latency can then lead to extra register copies.
//
// This pass aims to make those values easier to coalesce and reduce copies and
// register pressure. It matches a Flow -> Else -> Merge triangle, where Flow
// can also branch directly to Merge. For example:
//
//   Flow:
//     %v = phi [ %modified, Then ], [ poison, Entry ]
//     %cond = phi [ false, Then ], [ true, Entry ]
//     br %cond, Else, Merge
//   Else:
//     %e = <zero-cost instruction>
//     br Merge
//   Merge:
//     %result = phi [ %v, Flow ], [ %e, Else ]
//
// Hoisting %e to a block that dominates the poison input lets that input carry
// %e. If %v now equals %e whenever Else executes, both Merge inputs can use %v,
// and %result can be removed. Nested value/condition PHIs may require filling
// more than one poison input. Real inputs on paths that skip Else stay intact.
//
// Dominance alone does not prove this conditional equality: a Flow PHI can
// carry a different real value on an Else-taking path. Match corresponding
// value and condition PHIs and validate the entire candidate before changing
// IR. Also require safe speculative execution, available operands, and a hoist
// that precedes every matched PHI, so loops cannot supply a previous
// iteration's Else value. Unfamiliar inputs, cyclic PHI dependencies, and
// chains exceeding the depth limit are conservatively rejected. Keeping this
// optimization after StructurizeCFG lets it check the final CFG without
// changing SSA construction.
//
// Zero latency is a profitability heuristic, not a guarantee of fewer physical
// registers. Hoisting extends the Else value's live range, and the allocation
// result depends on downstream coalescing. Shuffles are excluded because an
// observed zero-cost shuffle increased register allocation and copy costs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/HoistZeroCostConditionalValues.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "hoist-zero-cost-conditional-values"

STATISTIC(NumReplaced, "Number of Else merge inputs replaced with Flow values");

namespace {

// Reusing the Flow value on both Merge inputs can avoid copies and register
// interference introduced by structurization. To replace the Else input, the
// Flow value must equal ElseValue whenever Else executes. This class checks
// that requirement and collects poison operands that can be filled with the
// hoisted ElseValue to establish it, before the caller changes any IR.
//
// Follow corresponding value and condition PHIs. A constant condition that
// skips Else imposes no requirement on the value. Every remaining input must
// be ElseValue, poison, or another pair of PHIs in the same block. In
// particular, a real value on an Else path cannot be treated as an unused
// input.
//
// For example, with ElseValue = %e and a true condition selecting Else:
//
//   Flow: (%v, %cond)
//     |
//     +-- Entry: (poison, true)        -> collect %v's Entry operand
//     |
//     `-- Flow1: (%inner, %inner_cond) -> follow this value/condition PHI pair
//           |
//           +-- Check: (poison, true)  -> collect %inner's Check operand
//           `-- Then: (%modified, false) -> skip; this path does not take Else
//
//   Flow branches to Else or directly to Merge using %cond.
//   Merge: %result = phi [ %v, Flow ], [ %e, Else ]
//
// Collection records the two poison operand Uses without changing IR. After
// the match and hoisting checks succeed, the caller hoists %e and fills them:
//
//   Flow1: %inner = phi [ %modified, Then ], [ %e, Check ]
//   Flow:  %v = phi [ %inner, Flow1 ], [ %e, Entry ]
//
// Now %v equals %e whenever Else executes. The caller can replace %result with
// %v and erase the Merge PHI. A failed match leaves IR unchanged.
class ElseInputs {
  static constexpr unsigned MaxPhiDepth = 32;
  Instruction *ElseValue;
  bool ElseOnTrue;
  using PhiPair = std::pair<PHINode *, PHINode *>;
  // Active pairs detect cycles; checked pairs avoid revisiting shared inputs.
  DenseSet<PhiPair> ActivePhiPairs;
  DenseSet<PhiPair> CheckedPhiPairs;
  DenseMap<PHINode *, DenseMap<BasicBlock *, Value *>> ConditionInputs;

  // Build one predecessor lookup per condition PHI during this read-only walk.
  // Paired value PHIs may list their incoming blocks in a different order.
  Value *getIncomingCondition(PHINode *ConditionPhi, BasicBlock *Pred) {
    auto [It, Inserted] = ConditionInputs.try_emplace(ConditionPhi);
    if (Inserted) {
      auto &ByPred = It->second;
      ByPred.reserve(ConditionPhi->getNumIncomingValues());
      for (unsigned I = 0, E = ConditionPhi->getNumIncomingValues(); I != E;
           ++I)
        ByPred.try_emplace(ConditionPhi->getIncomingBlock(I),
                           ConditionPhi->getIncomingValue(I));
    }
    return It->second.lookup(Pred);
  }

public:
  SmallVector<Use *, 8> PoisonInputs;

  ElseInputs(Instruction *ElseValue, bool ElseOnTrue)
      : ElseValue(ElseValue), ElseOnTrue(ElseOnTrue) {}

  // Return true if the value/condition match succeeds: when Condition selects
  // Else, Input is ElseValue or can be made so by filling the collected poison
  // operands, subject to the caller's hoisting and dominance checks.
  // Return false if this cannot be established; the caller must reject the
  // candidate and discard any partially collected inputs. No IR is changed.
  bool collect(Use &Input, Value *Condition) {
    if (auto *C = dyn_cast<ConstantInt>(Condition))
      if (C->isOne() != ElseOnTrue)
        return true;
    Value *V = Input.get();
    if (V == ElseValue)
      return true;
    if (isa<PoisonValue>(V)) {
      PoisonInputs.push_back(&Input);
      return true;
    }

    auto *ValuePhi = dyn_cast<PHINode>(V);
    auto *ConditionPhi = dyn_cast<PHINode>(Condition);
    if (!ValuePhi || !ConditionPhi ||
        ValuePhi->getParent() != ConditionPhi->getParent())
      return false;

    PhiPair Pair(ValuePhi, ConditionPhi);
    if (CheckedPhiPairs.contains(Pair))
      return true;
    // ActivePhiPairs contains one pair per recursive PHI level. Stop before
    // walking a deeper chain; rejecting the candidate leaves its IR unchanged.
    if (ActivePhiPairs.size() >= MaxPhiDepth)
      return false;
    // Do not infer equivalence through a cyclic PHI dependency.
    if (!ActivePhiPairs.insert(Pair).second)
      return false;
    for (unsigned I = 0, E = ValuePhi->getNumIncomingValues(); I != E; ++I) {
      BasicBlock *Pred = ValuePhi->getIncomingBlock(I);
      if (!collect(ValuePhi->getOperandUse(I),
                   getIncomingCondition(ConditionPhi, Pred)))
        return false;
    }
    ActivePhiPairs.erase(Pair);
    CheckedPhiPairs.insert(Pair);
    return true;
  }

  bool hoistDominatesPhis(BasicBlock *HoistTo, const DominatorTree &DT) const {
    // All matched PHIs must see the same execution of the hoisted instruction.
    // Hoisting into a loop-header PHI's block can otherwise fill a nested PHI
    // with this iteration's value and read it through the header next
    // iteration.
    return all_of(CheckedPhiPairs, [&](const PhiPair &Pair) {
      PHINode *ValuePhi = Pair.first;
      return DT.properlyDominates(HoistTo, ValuePhi->getParent());
    });
  }
};

// Apply the instruction-kind and target-cost filters before walking PHIs.
static bool isZeroCostCandidate(Instruction *I,
                                const TargetTransformInfo &TTI) {
  if (isa<PHINode>(I) || I->mayReadOrWriteMemory())
    return false;
  // Zero-cost shuffles can still need copies for PHI coalescing.
  // In rename-independent-subregs-unused-lanes.ll, hoisting the shuffle
  // increased allocated VGPRs from 14 to 16 on gfx1100.
  // Conservatively skip shuffles.
  if (isa<ShuffleVectorInst>(I))
    return false;
  auto Cost = TTI.getInstructionCost(I, TargetTransformInfo::TCK_Latency);
  return Cost.isValid() && Cost.getValue() == 0;
}

// Find a destination that dominates Flow and every new PHI incoming-edge use.
// If the Else value was already hoisted, include its existing definition block
// so it cannot move downward and invalidate uses from an earlier replacement.
// Return nullptr if a predecessor lacks a dominator-tree node.
static BasicBlock *findHoistDestination(BasicBlock *Flow,
                                        ArrayRef<Use *> PoisonInputs,
                                        BasicBlock *ExistingDef,
                                        const DominatorTree &DT) {
  BasicBlock *HoistTo = Flow;
  for (Use *Input : PoisonInputs) {
    auto *P = cast<PHINode>(Input->getUser());
    BasicBlock *Pred = P->getIncomingBlock(Input->getOperandNo());
    // findNearestCommonDominator requires both blocks to be in the tree.
    if (!DT.getNode(Pred))
      return nullptr;
    HoistTo = DT.findNearestCommonDominator(HoistTo, Pred);
  }
  if (ExistingDef)
    HoistTo = DT.findNearestCommonDominator(HoistTo, ExistingDef);
  return HoistTo;
}

// Check whether I can execute safely at the intended insertion point and
// whether all operands are available there. The caller filters instruction
// kinds and target cost before matching the PHI chain.
static bool isHoistableInstruction(Instruction *I, BasicBlock *HoistTo,
                                   const DominatorTree &DT) {
  Instruction *InsertBefore = HoistTo->getTerminator();
  // Hoisting retains call attributes. Attributes such as noundef can make a
  // previously skipped call undefined when it executes on an additional path.
  if (!isSafeToSpeculativelyExecute(I, InsertBefore, nullptr, &DT,
                                    /*TLI=*/nullptr, /*UseVariableInfo=*/true,
                                    /*IgnoreUBImplyingAttrs=*/false))
    return false;
  return all_of(I->operands(), [&](const Use &Op) {
    return DT.dominates(Op.get(), InsertBefore);
  });
}

static bool optimizeTriangle(BasicBlock &Else, DominatorTree &DT,
                             const TargetTransformInfo &TTI) {
  BasicBlock *Flow = Else.getSinglePredecessor();
  BasicBlock *Merge = Else.getSingleSuccessor();
  if (!Flow || !Merge || Flow == Merge || !Merge->hasNPredecessors(2) ||
      !DT.isReachableFromEntry(&Else))
    return false;
  auto *FlowBranch = dyn_cast<CondBrInst>(Flow->getTerminator());
  if (!FlowBranch || !isa<PHINode>(FlowBranch->getCondition()))
    return false;
  bool ElseOnTrue = FlowBranch->getSuccessor(0) == &Else;
  if (FlowBranch->getSuccessor(ElseOnTrue ? 1 : 0) != Merge)
    return false;

  bool Changed = false;
  SmallPtrSet<Instruction *, 8> HoistedElseValues;
  for (PHINode &Phi : make_early_inc_range(Merge->phis())) {
    auto *ElseValue =
        dyn_cast<Instruction>(Phi.getIncomingValueForBlock(&Else));
    if (!ElseValue)
      continue;
    bool AlreadyHoisted = HoistedElseValues.contains(ElseValue);
    if ((ElseValue->getParent() != &Else && !AlreadyHoisted) ||
        !isZeroCostCandidate(ElseValue, TTI))
      continue;
    // The Flow input needs both its value and operand Use for collect().
    unsigned FlowIndex = Phi.getBasicBlockIndex(Flow);
    auto *FlowValue = dyn_cast<PHINode>(Phi.getIncomingValue(FlowIndex));
    if (!FlowValue || FlowValue->getParent() != Flow)
      continue;
    ElseInputs Inputs(ElseValue, ElseOnTrue);
    if (!Inputs.collect(Phi.getOperandUse(FlowIndex),
                        FlowBranch->getCondition()) ||
        (Inputs.PoisonInputs.empty() && !AlreadyHoisted))
      continue;

    BasicBlock *ExistingDef = AlreadyHoisted ? ElseValue->getParent() : nullptr;
    BasicBlock *HoistTo =
        findHoistDestination(Flow, Inputs.PoisonInputs, ExistingDef, DT);
    if (!HoistTo)
      continue;

    if (!isHoistableInstruction(ElseValue, HoistTo, DT))
      continue;
    if (!Inputs.hoistDominatesPhis(HoistTo, DT))
      continue;

    // All checks passed. Refining these poison operands preserves other uses
    // of shared PHIs; real inputs, including all Then values, stay unchanged.
    ElseValue->moveBefore(*HoistTo, HoistTo->getTerminator()->getIterator());
    HoistedElseValues.insert(ElseValue);
    for (Use *Input : Inputs.PoisonInputs)
      Input->set(ElseValue);
    Phi.replaceAllUsesWith(FlowValue);
    Phi.eraseFromParent();
    ++NumReplaced;
    Changed = true;
  }
  return Changed;
}

static bool runImpl(Function &F, DominatorTree &DT,
                    const TargetTransformInfo &TTI) {
  bool Changed = false;
  for (BasicBlock &BB : F)
    Changed |= optimizeTriangle(BB, DT, TTI);
  return Changed;
}

class HoistZeroCostConditionalValuesLegacyPass : public FunctionPass {
public:
  static char ID;
  HoistZeroCostConditionalValuesLegacyPass() : FunctionPass(ID) {
    initializeHoistZeroCostConditionalValuesLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    return runImpl(F, getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                   getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F));
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // namespace

char HoistZeroCostConditionalValuesLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(HoistZeroCostConditionalValuesLegacyPass,
                      "hoist-zero-cost-conditional-values",
                      "Hoist zero-cost conditional values", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(HoistZeroCostConditionalValuesLegacyPass,
                    "hoist-zero-cost-conditional-values",
                    "Hoist zero-cost conditional values", false, false)

Pass *llvm::createHoistZeroCostConditionalValuesPass() {
  return new HoistZeroCostConditionalValuesLegacyPass();
}

PreservedAnalyses
HoistZeroCostConditionalValuesPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  if (!runImpl(F, AM.getResult<DominatorTreeAnalysis>(F),
               AM.getResult<TargetIRAnalysis>(F)))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
