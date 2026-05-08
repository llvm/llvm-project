//===- LoadStoreVec.cpp - Vectorizer pass short load-store chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

namespace sandboxir {

std::optional<Type *> LoadStoreVec::canVectorize(ArrayRef<Instruction *> Bndl,
                                                 Scheduler &Sched) {
  // Check if in the same BB.
  if (LegalityAnalysis::differentBlock(Bndl))
    return std::nullopt;

  // Check if instructions repeat.
  if (!LegalityAnalysis::areUnique(Bndl))
    return std::nullopt;

  // TODO: This is target-dependent.
  // Don't mix integer with floating point.
  bool IsFloat = false;
  bool IsInteger = false;
  for ([[maybe_unused]] auto *I : Bndl) {
    if (Utils::getExpectedType(I)->getScalarType()->isFloatingPointTy())
      IsFloat = true;
    else
      IsInteger = true;
  }
  if (IsFloat && IsInteger)
    return std::nullopt;

  // Check scheduling.
  if (!Sched.trySchedule(Bndl))
    return std::nullopt;

  return VecUtils::getCombinedVectorTypeFor(Bndl, *DL);
}

void LoadStoreVec::tryEraseDeadInstrs(ArrayRef<Instruction *> Stores,
                                      ArrayRef<Value *> Operands) {
  SmallPtrSet<Instruction *, 8> DeadCandidates;
  for (auto *SI : Stores) {
    if (auto *PtrI =
            dyn_cast<Instruction>(cast<StoreInst>(SI)->getPointerOperand()))
      DeadCandidates.insert(PtrI);
    SI->eraseFromParent();
  }
  for (auto *Op : Operands) {
    if (auto *LI = dyn_cast<LoadInst>(Op)) {
      if (auto *PtrI =
              dyn_cast<Instruction>(cast<LoadInst>(LI)->getPointerOperand()))
        DeadCandidates.insert(PtrI);
      cast<LoadInst>(LI)->eraseFromParent();
    }
  }
  for (auto *PtrI : DeadCandidates)
    if (!PtrI->hasNUsesOrMore(1))
      PtrI->eraseFromParent();
}

bool LoadStoreVec::runOnRegion(Region &Rgn, const Analyses &A) {
  SmallVector<Instruction *, 8> Bndl(Rgn.getAux().begin(), Rgn.getAux().end());
  if (Bndl.size() < 2)
    return false;
  Function &F = *Bndl[0]->getParent()->getParent();
  DL = &F.getParent()->getDataLayout();
  auto &Ctx = F.getContext();
  Scheduler Sched(A.getAA(), Ctx);
  if (!VecUtils::areConsecutive<StoreInst, Instruction>(
          Bndl, A.getScalarEvolution(), *DL))
    return false;
  if (!canVectorize(Bndl, Sched))
    return false;

  SmallVector<Value *, 4> Operands;
  Operands.reserve(Bndl.size());
  for (auto *I : Bndl) {
    auto *Op = cast<StoreInst>(I)->getValueOperand();
    Operands.push_back(Op);
  }
  BasicBlock *BB = Bndl[0]->getParent();
  // TODO: For now we only support load operands.
  // TODO: For now we don't cross BBs.
  // TODO: For now don't vectorize if the loads have external uses.
  bool AllLoads = all_of(Operands, [BB](Value *V) {
    auto *LI = dyn_cast<LoadInst>(V);
    if (LI == nullptr)
      return false;
    // TODO: For now we don't cross BBs.
    if (LI->getParent() != BB)
      return false;
    if (LI->hasNUsesOrMore(2))
      return false;
    return true;
  });
  bool AllConstants =
      all_of(Operands, [](Value *V) { return isa<Constant>(V); });
  if (!AllLoads && !AllConstants)
    return false;

  Value *VecOp = nullptr;
  if (AllLoads) {
    // TODO: Try to avoid the extra copy to an instruction vector.
    SmallVector<Instruction *, 8> Loads;
    Loads.reserve(Operands.size());
    for (Value *Op : Operands)
      Loads.push_back(cast<Instruction>(Op));

    bool Consecutive = VecUtils::areConsecutive<LoadInst, Instruction>(
        Loads, A.getScalarEvolution(), *DL);
    if (!Consecutive)
      return false;
    if (!canVectorize(Loads, Sched))
      return false;

    // Generate vector load.
    Type *Ty = VecUtils::getCombinedVectorTypeFor(Bndl, *DL);
    Value *LdPtr = cast<LoadInst>(Loads[0])->getPointerOperand();
    // TODO: Compute alignment.
    Align LdAlign(1);
    auto LdWhereIt = std::next(VecUtils::getLowest(Loads)->getIterator());
    VecOp = LoadInst::create(Ty, LdPtr, LdAlign, LdWhereIt, Ctx, "VecIinitL");
  } else if (AllConstants) {
    SmallVector<Constant *, 8> Constants;
    Constants.reserve(Operands.size());
    for (Value *Op : Operands) {
      auto *COp = cast<Constant>(Op);
      if (auto *AggrCOp = dyn_cast<ConstantAggregate>(COp)) {
        // If the operand is a constant aggregate, then append all its elements.
        for (Value *Elm : AggrCOp->operands())
          Constants.push_back(cast<Constant>(Elm));
      } else if (auto *SeqCOp = dyn_cast<ConstantDataSequential>(COp)) {
        for (auto ElmIdx : seq<unsigned>(SeqCOp->getNumElements()))
          Constants.push_back(SeqCOp->getElementAsConstant(ElmIdx));
      } else if (auto *Zero = dyn_cast<ConstantAggregateZero>(COp)) {
        auto *ZeroElm = Zero->getSequentialElement();
        for ([[maybe_unused]] auto Cnt :
             seq<unsigned>(Zero->getElementCount().getFixedValue()))
          Constants.push_back(ZeroElm);
      } else {
        Constants.push_back(COp);
      }
    }
    VecOp = ConstantVector::get(Constants);
  }

  // Generate vector store.
  Value *StPtr = cast<StoreInst>(Bndl[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align StAlign(1);
  auto StWhereIt = std::next(VecUtils::getLowest(Bndl)->getIterator());
  StoreInst::create(VecOp, StPtr, StAlign, StWhereIt, Ctx);

  tryEraseDeadInstrs(Bndl, Operands);
  return true;
}

} // namespace sandboxir

} // namespace llvm
