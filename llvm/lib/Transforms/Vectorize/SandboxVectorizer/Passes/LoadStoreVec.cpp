//===- LoadStoreVec.cpp - Vectorizer pass short load-store chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.h"
#include "../../VectorizeOptions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

namespace sandboxir {

#define DEBUG_PREFIX_LOCAL DEBUG_PREFIX "LoadStoreVec: "

std::optional<Type *> LoadStoreVec::canVectorize(BndlRef<Instruction *> Bndl) {
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "canVectorize() Bundle:\n";
             VecUtils::dump(Bndl));
  const auto &LegalityRes = Legality->canVectorizeLSV(Bndl);
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Legality: " << LegalityRes << "\n");

  if (LegalityRes.getSubclassID() != LegalityResultID::Widen)
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
  const VectorizeOptions &Opts = getVectorizeOptions(*Ctx);
  if (CostGain > Opts.sbvec_cost_threshold) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Not profitable, reverting.\n");
    Ctx->revert();
    return false;
  }
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Profitable accepting.\n");
  Ctx->accept();
  return true;
}

/// \returns the type of the lanes of \p LaneTy that hold a value of type \p Ty
/// in a packed vector: \p LaneTy itself, or <N x LaneTy> if \p Ty is wider.
static Type *getLanesTypeFor(Type *Ty, Type *LaneTy, const DataLayout &DL) {
  unsigned NumLanes = Utils::getNumBits(Ty, DL) / Utils::getNumBits(LaneTy, DL);
  return NumLanes > 1 ? VecUtils::getWideType(LaneTy, NumLanes) : LaneTy;
}

bool LoadStoreVec::canUnpackLoads(BndlRef<Instruction *> Loads,
                                  FixedVectorType *VecTy) {
  Type *VecElemTy = VecTy->getElementType();
  return all_of(Loads, [this, VecElemTy](Instruction *I) {
    return Reinterpret::canReinterpret(
        getLanesTypeFor(I->getType(), VecElemTy, *DL), I->getType(), *DL);
  });
}

void LoadStoreVec::unpackLoads(LoadInst *VecLoad,
                               BndlRef<Instruction *> Loads) {
  // Lanes are counted in units of the vector's element type, so a load wider
  // than that spans several lanes.
  Type *VecElemTy = cast<FixedVectorType>(VecLoad->getType())->getElementType();
  BasicBlock::iterator WhereIt = std::next(VecLoad->getIterator());
  unsigned Lane = 0;
  for (Instruction *I : Loads) {
    auto *OrigLoad = cast<LoadInst>(I);
    Type *VecTy = getLanesTypeFor(OrigLoad->getType(), VecElemTy, *DL);
    if (!OrigLoad->hasNUses(0)) {
      Value *Unpacked = VecUtils::unpack(VecLoad, VecTy, Lane, WhereIt);
      Unpacked = Reinterpret::reinterpretValue(Unpacked, OrigLoad->getType(),
                                               WhereIt, *Ctx, *DL);
      OrigLoad->replaceAllUsesWith(Unpacked);
    }
    Lane += VecUtils::getNumLanes(VecTy);
  }
}

LoadInst *LoadStoreVec::createVectorLoad(BndlRef<Instruction *> Loads) {
  if (!canVectorize(Loads))
    return nullptr;

  Type *Ty = VecUtils::getCombinedVectorTypeFor(Loads, *DL);
  Value *LdPtr = cast<LoadInst>(Loads[0])->getPointerOperand();
  Align LdAlign = VecUtils::getVecAlignment(Loads);
  auto LdWhereIt = std::next(VecUtils::getLowest(Loads)->getIterator());
  auto *VecL =
      LoadInst::create(Ty, LdPtr, LdAlign, LdWhereIt, *Ctx, "VecIinitL");
  Utils::propagateMetadata(VecL, Loads);
  return VecL;
}

Type *Reinterpret::getIntTypeFor(Type *Ty, const DataLayout &DL) {
  Context &Ctx = Ty->getContext();
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    auto *ElmIntTy =
        IntegerType::get(Ctx, Utils::getNumBits(VecTy->getElementType(), DL));
    return FixedVectorType::get(ElmIntTy, VecTy->getNumElements());
  }
  return IntegerType::get(Ctx, Utils::getNumBits(Ty, DL));
}

std::optional<SmallVector<Reinterpret::CastStep, 3>>
Reinterpret::getCastChain(Type *SrcTy, Type *DestTy, const DataLayout &DL) {
  assert(Utils::getNumBits(SrcTy, DL) == Utils::getNumBits(DestTy, DL) &&
         "Expected types of the same size!");
  // A non-integral pointer has no fixed ptrtoint/inttoptr representation.
  if (VecUtils::isNonIntegralPtr(SrcTy, DL) ||
      VecUtils::isNonIntegralPtr(DestTy, DL))
    return std::nullopt;
  SmallVector<CastStep, 3> Chain;
  if (SrcTy == DestTy)
    return Chain;
  Type *Ty = SrcTy;
  if (Ty->getScalarType()->isPointerTy()) {
    Ty = getIntTypeFor(Ty, DL);
    Chain.emplace_back(Instruction::Opcode::PtrToInt, Ty);
  }
  bool DestIsPtr = DestTy->getScalarType()->isPointerTy();
  Type *BitCastTy = DestIsPtr ? getIntTypeFor(DestTy, DL) : DestTy;
  if (Ty != BitCastTy)
    Chain.emplace_back(Instruction::Opcode::BitCast, BitCastTy);
  if (DestIsPtr)
    Chain.emplace_back(Instruction::Opcode::IntToPtr, DestTy);
  return Chain;
}

Value *Reinterpret::reinterpretValue(Value *V, Type *DestTy,
                                     BasicBlock::iterator WhereIt, Context &Ctx,
                                     const DataLayout &DL) {
  auto Chain = getCastChain(V->getType(), DestTy, DL);
  if (!Chain)
    return nullptr;
  for (auto [Opc, Ty] : *Chain)
    V = CastInst::create(Ty, Opc, V, WhereIt, Ctx, "VCast");
  return V;
}

Constant *Reinterpret::reinterpretConstant(Constant *C, Type *DestTy,
                                           const DataLayout &DL) {
  auto Chain = getCastChain(C->getType(), DestTy, DL);
  if (!Chain)
    return nullptr;
  for (auto [Opc, Ty] : *Chain) {
    switch (Opc) {
    case Instruction::Opcode::PtrToInt:
      C = ConstantExpr::getPtrToInt(C, Ty);
      break;
    case Instruction::Opcode::BitCast:
      C = ConstantExpr::getBitCast(C, Ty);
      break;
    case Instruction::Opcode::IntToPtr:
      C = ConstantExpr::getIntToPtr(C, Ty);
      break;
    default:
      llvm_unreachable("Unexpected cast in reinterpret chain!");
    }
  }
  return C;
}

Constant *LoadStoreVec::createConstantVector(BndlRef<Constant *> Constants) {
  auto *VecTy =
      cast<FixedVectorType>(VecUtils::getCombinedVectorTypeFor(Constants, *DL));
  auto *LaneTy = VecTy->getScalarType();
  unsigned LaneBits = Utils::getNumBits(LaneTy, *DL);
  assert(all_of(Constants,
                [&](Value *V) {
                  return Utils::getNumBits(V, *DL) % LaneBits == 0;
                }) &&
         "Chain contains a constant which cannot be converted or split");

  // Some of the contents of Constants may be vectors, like <i8 1, i8 2>.
  // Collect each scalar element in ConstantElements.
  SmallVector<Constant *, 4> ConstantElements;
  for (Constant *COp : Constants) {
    if (auto *AggrCOp = dyn_cast<ConstantAggregate>(COp)) {
      // If the operand is a constant aggregate, then append all its elements.
      // TODO: Flatten nested aggregates such as {i32, {i32, <2 x i32>}} in a
      // follow-up.
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
    } else if (isa<PoisonValue>(COp) && isa<VectorType>(COp->getType())) {
      auto *Elm =
          PoisonValue::get(cast<VectorType>(COp->getType())->getElementType());
      for ([[maybe_unused]] auto Cnt :
           seq<unsigned>(cast<VectorType>(COp->getType())
                             ->getElementCount()
                             .getFixedValue()))
        ConstantElements.push_back(Elm);
    } else if (isa<VectorType>(COp->getType())) {
      // TODO: Flatten the remaining vector constants.
      return nullptr;
    } else {
      ConstantElements.push_back(COp);
    }
  }

  // The elements in ConstantElements may not be of the same type, so we need to
  // convert them to LaneTy such that we end up getting a vector
  // <ConstantElements.size() x LaneTy>. This may involve converting integers to
  // floats (e.g., float 1.0 to i32 0x3f800000), widening or slicing values if
  // not at LaneTy bit-width (e.g., i16 0x0101 to i8 0x01, i8 0x01 on
  // little-endian), or a combination of the two.
  SmallVector<Constant *, 8> LaneTyConstants;
  LaneTyConstants.reserve(ConstantElements.size());
  for (Constant *C : ConstantElements) {
    unsigned Bits = Utils::getNumBits(C, *DL);
    if (Bits == LaneBits) {
      Constant *Lane = Reinterpret::reinterpretConstant(C, LaneTy, *DL);
      if (Lane == nullptr)
        return nullptr;
      LaneTyConstants.push_back(Lane);
    } else if (Bits > LaneBits) {
      // C is wider than LaneTy, so slice it into LaneTy-sized pieces. That
      // needs a known bit pattern. A ConstantExpr (for example, ptrtoint of a
      // global) or a global's address is only known at link time and cannot
      // be sliced.
      if (!isa<ConstantInt, ConstantFP, ConstantPointerNull>(C))
        return nullptr;
      APInt Val;
      if (auto *CI = dyn_cast<ConstantInt>(C))
        Val = CI->getValue();
      else if (auto *CFP = dyn_cast<ConstantFP>(C))
        Val = CFP->getValue().bitcastToAPInt();
      else
        Val = APInt::getZero(Bits);
      unsigned NumSlices = Bits / LaneBits;
      for (unsigned SliceIdx : seq<unsigned>(NumSlices)) {
        unsigned EndianAwareSlice =
            DL->isLittleEndian() ? SliceIdx : NumSlices - 1 - SliceIdx;
        // extractBits returns an integer. Convert that slice back to LaneTy
        // without changing its bits, so a float lane stays a float.
        Constant *SliceCInt = ConstantInt::get(
            *Ctx, Val.extractBits(LaneBits, EndianAwareSlice * LaneBits));
        Constant *SliceC =
            Reinterpret::reinterpretConstant(SliceCInt, LaneTy, *DL);
        if (SliceC == nullptr)
          return nullptr;
        LaneTyConstants.push_back(SliceC);
      }
    } else {
      llvm_unreachable("Vector element type size was calculated incorrectly");
    }
  }
  return ConstantVector::get(LaneTyConstants);
}

Value *LoadStoreVec::createPack(BndlRef<Value *> Operands,
                                FixedVectorType *VecTy,
                                BasicBlock::iterator WhereIt) {
  Type *LaneTy = VecTy->getElementType();
  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  Value *Vec = PoisonValue::get(VecTy);
  unsigned Lane = 0;
  for (Value *Op : Operands) {
    Type *LanesTy = getLanesTypeFor(Op->getType(), LaneTy, *DL);
    Value *Elms =
        Reinterpret::reinterpretValue(Op, LanesTy, WhereIt, *Ctx, *DL);
    assert(Elms != nullptr && "Expected the caller to check canReinterpret()");
    // An operand that spans several lanes is inserted lane by lane. Like
    // all the other instructions created here, these may fold to constants.
    auto *LanesVecTy = dyn_cast<FixedVectorType>(LanesTy);
    if (LanesVecTy == nullptr) {
      Vec = InsertElementInst::create(
          Vec, Elms, ConstantInt::get(Int32Ty, Lane++), WhereIt, *Ctx, "VPack");
      continue;
    }
    for (unsigned Idx : seq<unsigned>(LanesVecTy->getNumElements())) {
      Value *Elm = ExtractElementInst::create(
          Elms, ConstantInt::get(Int32Ty, Idx), WhereIt, *Ctx, "VPack");
      Vec = InsertElementInst::create(
          Vec, Elm, ConstantInt::get(Int32Ty, Lane++), WhereIt, *Ctx, "VPack");
    }
  }
  return Vec;
}

bool LoadStoreVec::vectorizeStores(BndlRef<Instruction *> Stores, Region &Rgn) {
  auto VecTy = canVectorize(Stores);
  if (!VecTy)
    return false;
  SmallVector<Value *, 4> Operands;
  Operands.reserve(Stores.size());
  for (auto *I : Stores) {
    auto *Op = cast<StoreInst>(I)->getValueOperand();
    Operands.push_back(Op);
  }
  BasicBlock *BB = Stores[0]->getParent();
  // TODO: For now we don't cross BBs.
  bool AllLoads = all_of(Operands, [BB](Value *V) {
    auto *LI = dyn_cast<LoadInst>(V);
    return LI != nullptr && LI->getParent() == BB;
  });
  bool AllConstants =
      all_of(Operands, [](Value *V) { return isa<Constant>(V); });
  // Any other operands are packed into a vector one by one, after
  // reinterpreting each of them as the lanes it occupies.
  auto *LaneTy = cast<FixedVectorType>(*VecTy)->getElementType();
  if (!AllLoads && !AllConstants && any_of(Operands, [this, LaneTy](Value *Op) {
        return !Reinterpret::canReinterpret(
            Op->getType(), getLanesTypeFor(Op->getType(), LaneTy, *DL), *DL);
      }))
    return false;

  // Packing operands, or vectorizing mixed floats and integers, may not be
  // profitable on some targets, so save state here.
  saveIR(Rgn);
  auto StWhereIt = std::next(VecUtils::getLowest(Stores)->getIterator());
  Value *VecOp = nullptr;
  SmallVector<Instruction *, 8> Loads;
  if (AllLoads) {
    Loads.reserve(Operands.size());
    for (Value *Op : Operands)
      Loads.push_back(cast<Instruction>(Op));
    // Loads with uses other than the stores are unpacked from the vector load.
    bool HasExternalUses =
        any_of(Loads, [](Instruction *I) { return I->hasNUsesOrMore(2); });
    if (HasExternalUses) {
      auto LdVecTy = canVectorize(Loads);
      if (!LdVecTy || !canUnpackLoads(Loads, cast<FixedVectorType>(*LdVecTy))) {
        Ctx->accept();
        return false;
      }
    }
    VecOp = createVectorLoad(Loads);
    if (VecOp == nullptr) {
      Ctx->accept();
      return false;
    }
  } else if (AllConstants) {
    SmallVector<Constant *, 8> Constants;
    Constants.reserve(Operands.size());
    for (Value *Op : Operands)
      Constants.push_back(cast<Constant>(Op));
    VecOp = createConstantVector(Constants);
    if (VecOp == nullptr) {
      Ctx->accept();
      return false;
    }
  } else {
    // All operands dominate their stores, so they are available right
    // before the vector store.
    VecOp = createPack(Operands, cast<FixedVectorType>(*VecTy), StWhereIt);
  }

  // Generate vector store.
  Value *StPtr = cast<StoreInst>(Stores[0])->getPointerOperand();
  Align StAlign = VecUtils::getVecAlignment(Stores);
  auto *VecS = StoreInst::create(VecOp, StPtr, StAlign, StWhereIt, *Ctx);
  Utils::propagateMetadata(VecS, Stores);

  DeadInstrMorgue.collectPotentiallyDeadInstrs(Stores);
  DeadInstrMorgue.tryEraseDeadInstrs();
  if (AllLoads) {
    // The scalar stores are gone, so any remaining uses of the loads are
    // external.
    unpackLoads(cast<LoadInst>(VecOp), Loads);
    DeadInstrMorgue.collectPotentiallyDeadInstrs<Instruction>(Loads);
    DeadInstrMorgue.tryEraseDeadInstrs();
  }

  return acceptOrRevert();
}

LoadInst *LoadStoreVec::vectorizeLoads(BndlRef<Instruction *> Loads,
                                       Region &Rgn) {
  auto VecTy = canVectorize(Loads);
  if (!VecTy)
    return nullptr;

  if (!canUnpackLoads(Loads, cast<FixedVectorType>(*VecTy)))
    return nullptr;

  saveIR(Rgn);

  auto *VecLoad = createVectorLoad(Loads);
  if (VecLoad == nullptr) {
    Ctx->accept();
    return nullptr;
  }

  unpackLoads(VecLoad, Loads);
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
  InstrMaps IMaps;
  Legality = std::make_unique<LegalityAnalysis>(
      A->getAA(), A->getScalarEvolution(), F.getParent()->getDataLayout(),
      A->getTTI(), F.getContext(), IMaps, SchedDirection::BottomUp);

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
  return Changed;
}

} // namespace sandboxir

} // namespace llvm
