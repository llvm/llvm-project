//===- LoadStoreVec.cpp - Vectorizer pass short load-store chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

extern cl::opt<int> CostThreshold; // Defined in TransactionAcceptOrRevert.cpp

namespace sandboxir {

#define DEBUG_PREFIX_LOCAL DEBUG_PREFIX "LoadStoreVec: "

std::optional<Type *> LoadStoreVec::canVectorize(ArrayRef<Instruction *> Bndl,
                                                 Scheduler &Sched) {
  // Check if in the same BB.
  if (LegalityAnalysis::differentBlock(Bndl))
    return std::nullopt;

  // Check if instructions repeat.
  if (!LegalityAnalysis::areUnique(Bndl))
    return std::nullopt;

  // Check scheduling.
  if (!Sched.trySchedule(Bndl))
    return std::nullopt;

  return VecUtils::getCombinedVectorTypeFor(Bndl, *DL);
}

void LoadStoreVec::saveIR(Region &Rgn) {
  SavedRgn = &Rgn;
  const auto &SB = cast<RegionWithScore>(Rgn).getScoreboard();
  CostBefore = SB.getAfterCost() - SB.getBeforeCost();
  Rgn.getContext().save();
}

bool LoadStoreVec::acceptOrRevert() {
  auto &Ctx = SavedRgn->getContext();
  const auto &SB = cast<RegionWithScore>(*SavedRgn).getScoreboard();
  InstructionCost CostAfter = SB.getAfterCost() - SB.getBeforeCost();
  InstructionCost CostGain = CostAfter - CostBefore;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "CostGain=" << CostGain
                    << " (After=" << CostAfter << " Before=" << CostBefore
                    << ")\n");
  if (CostGain > CostThreshold) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Not profitable, reverting.\n");
    Ctx.revert();
    return false;
  }
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Profitable accepting.\n");
  Ctx.accept();
  return true;
}

Value *LoadStoreVec::createVectorLoad(ArrayRef<Value *> Operands,
                                      Scheduler &Sched, const Analyses &A,
                                      Context &Ctx) {
  SmallVector<Instruction *, 8> Loads;
  Loads.reserve(Operands.size());
  for (Value *Op : Operands)
    Loads.push_back(cast<Instruction>(Op));

  if (!VecUtils::areConsecutive<LoadInst, Instruction>(
          Loads, A.getScalarEvolution(), *DL))
    return nullptr;
  if (!canVectorize(Loads, Sched))
    return nullptr;

  Type *Ty = VecUtils::getCombinedVectorTypeFor(Loads, *DL);
  Value *LdPtr = cast<LoadInst>(Loads[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align LdAlign(1);
  auto LdWhereIt = std::next(VecUtils::getLowest(Loads)->getIterator());
  return LoadInst::create(Ty, LdPtr, LdAlign, LdWhereIt, Ctx, "VecIinitL");
}

Value *LoadStoreVec::createConstantVector(ArrayRef<Value *> Operands,
                                          Context &Ctx) {
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
    } else if (isa<ConstantInt>(COp) && isa<VectorType>(COp->getType())) {
      auto *Elm = ConstantInt::get(Ctx, cast<ConstantInt>(COp)->getValue());
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(cast<VectorType>(COp->getType())
                             ->getElementCount()
                             .getFixedValue()))
        Constants.push_back(Elm);
    } else if (isa<ConstantFP>(COp) && isa<VectorType>(COp->getType())) {
      auto *Elm = ConstantFP::get(cast<ConstantFP>(COp)->getValue(), Ctx);
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(cast<VectorType>(COp->getType())
                             ->getElementCount()
                             .getFixedValue()))
        Constants.push_back(Elm);
    } else {
      Constants.push_back(COp);
    }
  }
  return ConstantVector::get(Constants);
}

bool LoadStoreVec::vectorizeStores(ArrayRef<Instruction *> Bndl, Region &Rgn,
                                   Scheduler &Sched, const Analyses &A) {
  assert(all_of(Bndl, [](Instruction *I) { return isa<StoreInst>(I); }) &&
         "vectorizeStores expects a store bundle!");
  Function &F = *Bndl[0]->getParent()->getParent();
  auto &Ctx = F.getContext();
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

  saveIR(Rgn);

  Value *VecOp = nullptr;
  if (AllLoads) {
    SmallVector<Instruction *, 8> Loads;
    Loads.reserve(Operands.size());
    for (Value *Op : Operands)
      Loads.push_back(cast<Instruction>(Op));
    VecOp = vectorizeLoads(Loads, Rgn, Sched, A, /*OwnTransaction=*/false);
    if (VecOp == nullptr) {
      Ctx.accept();
      return false;
    }
  } else if (AllConstants) {
    VecOp = createConstantVector(Operands, Ctx);
  }

  // Generate vector store.
  Value *StPtr = cast<StoreInst>(Bndl[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align StAlign(1);
  auto StWhereIt = std::next(VecUtils::getLowest(Bndl)->getIterator());
  StoreInst::create(VecOp, StPtr, StAlign, StWhereIt, Ctx);

  DenseSet<Instruction *> DeadInstrCandidates;
  VecUtils::collectPotentiallyDeadInstrs(Bndl, DeadInstrCandidates);
  if (AllLoads)
    VecUtils::collectPotentiallyDeadInstrs(ArrayRef<Value *>(Operands),
                                           DeadInstrCandidates);
  VecUtils::tryEraseDeadInstrs(DeadInstrCandidates);

  return acceptOrRevert();
}

Value *LoadStoreVec::vectorizeLoads(ArrayRef<Instruction *> Bndl, Region &Rgn,
                                    Scheduler &Sched, const Analyses &A,
                                    bool OwnTransaction) {
  Function &F = *Bndl[0]->getParent()->getParent();
  auto &Ctx = F.getContext();
  if (OwnTransaction)
    saveIR(Rgn);

  SmallVector<Value *, 8> Operands(Bndl.begin(), Bndl.end());
  Value *VecLoad = createVectorLoad(Operands, Sched, A, Ctx);
  if (VecLoad == nullptr) {
    if (OwnTransaction)
      Ctx.accept();
    return nullptr;
  }

  if (!OwnTransaction)
    return VecLoad;

  // TODO: Support mixed-type top-level load chains.
  Type *VecElemTy = cast<FixedVectorType>(VecLoad->getType())->getElementType();
  if (!all_of(Bndl, [VecElemTy](Instruction *I) {
        return VecUtils::getElementType(I->getType()) == VecElemTy;
      })) {
    Ctx.revert();
    return nullptr;
  }

  auto *VecLoadI = cast<Instruction>(VecLoad);
  BasicBlock::iterator WhereIt = std::next(VecLoadI->getIterator());
  for (auto [Lane, OrigV] : VecUtils::enumerateLanes(Bndl)) {
    auto *OrigLoad = cast<LoadInst>(OrigV);
    if (OrigLoad->hasNUses(0))
      continue;
    Value *Unpacked =
        VecUtils::unpack(VecLoad, OrigLoad->getType(), Lane, WhereIt);
    OrigLoad->replaceAllUsesWith(Unpacked);
  }

  DenseSet<Instruction *> DeadInstrCandidates;
  VecUtils::collectPotentiallyDeadInstrs(Bndl, DeadInstrCandidates);
  VecUtils::tryEraseDeadInstrs(DeadInstrCandidates);

  if (!acceptOrRevert())
    return nullptr;
  return VecLoad;
}

bool LoadStoreVec::runOnRegion(Region &Rgn, const Analyses &A) {
  SmallVector<Instruction *, 8> Bndl(Rgn.getAux().begin(), Rgn.getAux().end());
  if (Bndl.size() < 2)
    return false;
  Function &F = *Bndl[0]->getParent()->getParent();
  DL = &F.getParent()->getDataLayout();

  // SeedCollection only ever gives us a homogeneous seed slice: stores and
  // loads are collected in separate passes over the BB, never mixed into one
  // Aux (see SeedCollection::runOnFunction).
  bool IsStoreKind = isa<StoreInst>(Bndl[0]);
  assert(all_of(Bndl,
                [&](Instruction *I) {
                  return isa<StoreInst>(I) == IsStoreKind;
                }) &&
         "Expected a homogeneous seed slice!");

  Scheduler Sched(A.getAA(), F.getContext(), SchedDirection::BottomUp);
  return IsStoreKind ? vectorizeStores(Bndl, Rgn, Sched, A)
                     : vectorizeLoads(Bndl, Rgn, Sched, A) != nullptr;
}

} // namespace sandboxir

} // namespace llvm
