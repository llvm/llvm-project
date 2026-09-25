//===- LoadStoreVec.cpp - Vectorizer pass short load-store chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Instruction.h"
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

std::optional<Type *> LoadStoreVec::canVectorize(BndlRef<Instruction *> Bndl) {
  // Check if in the same BB.
  if (LegalityAnalysis::differentBlock(Bndl))
    return std::nullopt;

  // Check if instructions repeat.
  if (!LegalityAnalysis::areUnique(Bndl))
    return std::nullopt;

  // Check scheduling.
  if (!Sched->trySchedule(Bndl))
    return std::nullopt;

  return VecUtils::getCombinedVectorTypeFor(Bndl, *DL);
}

void LoadStoreVec::saveIR(Region &R) {
  Rgn = &R;
  const auto &SB = cast<RegionWithScore>(Rgn)->getScoreboard();
  CostBefore = SB.getAfterCost() - SB.getBeforeCost();
  Rgn->getContext().save();
}

bool LoadStoreVec::acceptOrRevert() {
  const auto &SB = cast<RegionWithScore>(*Rgn).getScoreboard();
  InstructionCost CostAfter = SB.getAfterCost() - SB.getBeforeCost();
  InstructionCost CostGain = CostAfter - CostBefore;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "CostGain=" << CostGain
                    << " (After=" << CostAfter << " Before=" << CostBefore
                    << ")\n");
  if (CostGain > CostThreshold) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Not profitable, reverting.\n");
    Ctx->revert();
    return false;
  }
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Profitable accepting.\n");
  Ctx->accept();
  return true;
}

LoadInst *LoadStoreVec::createVectorLoad(BndlRef<Instruction *> Loads) {
  if (!VecUtils::areConsecutive<LoadInst, Instruction>(
          Loads, A->getScalarEvolution(), *DL))
    return nullptr;
  if (!canVectorize(Loads))
    return nullptr;

  Type *Ty = VecUtils::getCombinedVectorTypeFor(Loads, *DL);
  Value *LdPtr = cast<LoadInst>(Loads[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align LdAlign(1);
  auto LdWhereIt = std::next(VecUtils::getLowest(Loads)->getIterator());
  return LoadInst::create(Ty, LdPtr, LdAlign, LdWhereIt, *Ctx, "VecIinitL");
}

Constant *LoadStoreVec::getEquivalentConstantWithType(Constant *C,
                                                      Type *DestTy) {
  Type *SrcTy = C->getType();
  if (SrcTy == DestTy)
    return C;
  auto IsNonIntegralPtr = [this](Type *Ty) {
    return Ty->isPointerTy() &&
           DL->isNonIntegralAddressSpace(Ty->getPointerAddressSpace());
  };
  if (IsNonIntegralPtr(SrcTy) || IsNonIntegralPtr(DestTy))
    return nullptr;

  Constant *AsInt = C;
  if (!SrcTy->isIntegerTy()) {
    Type *IntTy = IntegerType::get(*Ctx, Utils::getNumBits(SrcTy, *DL));
    AsInt = SrcTy->isPointerTy() ? ConstantExpr::getPtrToInt(C, IntTy)
                                 : ConstantExpr::getBitCast(C, IntTy);
  }
  if (DestTy->isIntegerTy())
    return AsInt;
  return DestTy->isPointerTy() ? ConstantExpr::getIntToPtr(AsInt, DestTy)
                               : ConstantExpr::getBitCast(AsInt, DestTy);
}

Value *LoadStoreVec::createConstantVector(ArrayRef<Value *> Constants,
                                          Type *LaneTy) {
  SmallVector<Constant *, 8> ConstantElements;
  ConstantElements.reserve(Constants.size());
  for (Value *Op : Constants) {
    auto *COp = cast<Constant>(Op);
    if (auto *AggrCOp = dyn_cast<ConstantAggregate>(COp)) {
      // If the operand is a constant aggregate, then append all its elements.
      for (Value *Elm : AggrCOp->operands())
        ConstantElements.push_back(cast<Constant>(Elm));
    } else if (auto *SeqCOp = dyn_cast<ConstantDataSequential>(COp)) {
      for (auto ElmIdx : seq<unsigned>(SeqCOp->getNumElements()))
        ConstantElements.push_back(SeqCOp->getElementAsConstant(ElmIdx));
    } else if (auto *Zero = dyn_cast<ConstantAggregateZero>(COp)) {
      auto *ZeroElm = Zero->getSequentialElement();
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(Zero->getElementCount().getFixedValue()))
        ConstantElements.push_back(ZeroElm);
    } else if (isa<ConstantInt>(COp) && isa<VectorType>(COp->getType())) {
      auto *Elm = ConstantInt::get(*Ctx, cast<ConstantInt>(COp)->getValue());
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(cast<VectorType>(COp->getType())
                             ->getElementCount()
                             .getFixedValue()))
        ConstantElements.push_back(Elm);
    } else if (isa<ConstantFP>(COp) && isa<VectorType>(COp->getType())) {
      auto *Elm = ConstantFP::get(cast<ConstantFP>(COp)->getValue(), *Ctx);
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(cast<VectorType>(COp->getType())
                             ->getElementCount()
                             .getFixedValue()))
        ConstantElements.push_back(Elm);
    } else if (isa<VectorType>(COp->getType())) {
      // TODO: Flatten the remaining vector constants, e.g. undef or poison.
      return nullptr;
    } else {
      ConstantElements.push_back(COp);
    }
  }

  // Convert each constant to LaneTy. A wider constant is split into multiple
  // LaneTy-sized pieces, in memory order. Relocatable values like the address
  // of a global do not fold to a ConstantInt, so they cannot be split.
  unsigned LaneBits = Utils::getNumBits(LaneTy, *DL);
  if (any_of(ConstantElements, [&](Constant *C) {
        return Utils::getNumBits(C->getType(), *DL) > LaneBits &&
               !isa<ConstantInt, ConstantFP, ConstantPointerNull>(C);
      }))
    return nullptr;

  SmallVector<Constant *, 8> ConstantLanes;
  ConstantLanes.reserve(ConstantElements.size());
  for (Constant *C : ConstantElements) {
    unsigned Bits = Utils::getNumBits(C->getType(), *DL);
    assert(Bits % LaneBits == 0);
    if (Bits == LaneBits) {
      Constant *Lane = getEquivalentConstantWithType(C, LaneTy);
      if (Lane == nullptr)
        return nullptr;
      ConstantLanes.push_back(Lane);
    } else if (Bits > LaneBits) {
      APInt Val;
      if (auto *CI = dyn_cast<ConstantInt>(C))
        Val = CI->getValue();
      else if (auto *CFP = dyn_cast<ConstantFP>(C))
        Val = CFP->getValue().bitcastToAPInt();
      else
        Val = APInt::getZero(Bits);
      unsigned NumSlices = Bits / LaneBits;
      for (unsigned SliceIdx : seq<unsigned>(NumSlices)) {
        unsigned EndianAwarePart =
            DL->isLittleEndian() ? SliceIdx : NumSlices - 1 - SliceIdx;
        Constant *SliceCInt = ConstantInt::get(
            *Ctx, Val.extractBits(LaneBits, EndianAwarePart * LaneBits));
        Constant *SliceC = getEquivalentConstantWithType(SliceCInt, LaneTy);
        if (SliceC == nullptr)
          return nullptr;
        ConstantLanes.push_back(SliceC);
      }
    } else {
      llvm_unreachable("Vector element type size was calculated incorrectly");
    }
  }
  return ConstantVector::get(ConstantLanes);
}

bool LoadStoreVec::vectorizeStores(BndlRef<Instruction *> Stores, Region &Rgn) {
  if (!VecUtils::areConsecutive<StoreInst, Instruction>(
          Stores, A->getScalarEvolution(), *DL))
    return false;
  if (!canVectorize(Stores))
    return false;
  SmallVector<Value *, 4> Operands;
  Operands.reserve(Stores.size());
  for (auto *I : Stores) {
    auto *Op = cast<StoreInst>(I)->getValueOperand();
    Operands.push_back(Op);
  }
  BasicBlock *BB = Stores[0]->getParent();
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

  // Vectorizing mixed floats and integers with external uses may not be
  // profitable on some targets, so save state here.
  saveIR(Rgn);
  Value *VecOp = nullptr;
  if (AllLoads) {
    // TODO: Try to avoid the extra copy to an instruction vector.
    SmallVector<Instruction *, 8> Loads;
    Loads.reserve(Operands.size());
    for (Value *Op : Operands)
      Loads.push_back(cast<Instruction>(Op));
    VecOp = createVectorLoad(Loads);
    if (VecOp == nullptr) {
      Ctx->accept();
      return false;
    }
  } else if (AllConstants) {
    auto *VecTy =
        cast<FixedVectorType>(VecUtils::getCombinedVectorTypeFor(Stores, *DL));
    VecOp = createConstantVector(Operands, VecTy->getElementType());
    if (VecOp == nullptr) {
      Ctx->accept();
      return false;
    }
  }

  // Generate vector store.
  Value *StPtr = cast<StoreInst>(Stores[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align StAlign(1);
  auto StWhereIt = std::next(VecUtils::getLowest(Stores)->getIterator());
  StoreInst::create(VecOp, StPtr, StAlign, StWhereIt, *Ctx);

  DeadInstrMorgue.collectPotentiallyDeadInstrs(Stores);
  if (AllLoads)
    DeadInstrMorgue.collectPotentiallyDeadInstrs<Value>(Operands);
  DeadInstrMorgue.tryEraseDeadInstrs();

  return acceptOrRevert();
}

LoadInst *LoadStoreVec::vectorizeLoads(BndlRef<Instruction *> Loads,
                                       Region &Rgn) {
  if (!VecUtils::areConsecutive<LoadInst, Instruction>(
          Loads, A->getScalarEvolution(), *DL))
    return nullptr;
  auto VecTy = canVectorize(Loads);
  if (!VecTy)
    return nullptr;

  // TODO: Support mixed-type top-level load chains.
  Type *VecElemTy = cast<FixedVectorType>(*VecTy)->getElementType();
  if (!all_of(Loads, [VecElemTy](Instruction *I) {
        return VecUtils::getElementType(I->getType()) == VecElemTy;
      }))
    return nullptr;

  saveIR(Rgn);

  auto *VecLoad = createVectorLoad(Loads);
  if (VecLoad == nullptr) {
    Ctx->accept();
    return nullptr;
  }

  BasicBlock::iterator WhereIt = std::next(VecLoad->getIterator());
  for (auto [Lane, OrigV] : VecUtils::enumerateLanes(Loads)) {
    auto *OrigLoad = cast<LoadInst>(OrigV);
    if (OrigLoad->hasNUses(0))
      continue;
    Value *Unpacked =
        VecUtils::unpack(VecLoad, OrigLoad->getType(), Lane, WhereIt);
    OrigLoad->replaceAllUsesWith(Unpacked);
  }

  DeadInstrMorgue.collectPotentiallyDeadInstrs(Loads);
  DeadInstrMorgue.tryEraseDeadInstrs();

  if (!acceptOrRevert())
    return nullptr;
  return VecLoad;
}

bool LoadStoreVec::runOnRegion(Region &Rgn, const Analyses &RegionAnalyses) {
  SmallVector<Instruction *, 8> Bndl(Rgn.getAux().begin(), Rgn.getAux().end());
  if (Bndl.size() < 2)
    return false;
  Function &F = *Bndl[0]->getParent()->getParent();
  DL = &F.getParent()->getDataLayout();
  Ctx = &F.getContext();
  A = &RegionAnalyses;
  Sched =
      std::make_unique<Scheduler>(A->getAA(), *Ctx, SchedDirection::BottomUp);

  auto Opc = Bndl[0]->getOpcode();
  assert(
      all_of(Bndl, [Opc](Instruction *I) { return I->getOpcode() == Opc; }) &&
      "Expected a homogeneous seed slice!");

  bool Changed = false;
  switch (Opc) {
  case Instruction::Opcode::Load:
    Changed = vectorizeLoads(Bndl, Rgn) != nullptr;
    break;
  case Instruction::Opcode::Store:
    Changed = vectorizeStores(Bndl, Rgn);
    break;
  default:
    llvm_unreachable("Expected Load or Store");
  }
  Sched.reset();
  return Changed;
}

} // namespace sandboxir

} // namespace llvm
