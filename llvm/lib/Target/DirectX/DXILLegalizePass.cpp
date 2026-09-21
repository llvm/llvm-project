//===- DXILLegalizePass.cpp - Legalizes llvm IR for DXIL ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "DXILLegalizePass.h"
#include "DirectX.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <functional>

#define DEBUG_TYPE "dxil-legalize"

using namespace llvm;

// Map an unsupported integer type to the smallest legal DXIL carrier type.
static IntegerType *getLegalIntegerType(Type *Ty) {
  auto *IntTy = dyn_cast<IntegerType>(Ty);
  if (!IntTy)
    return nullptr;

  unsigned Width = IntTy->getBitWidth();
  // DXIL uses i1 for SSA predicates even though booleans occupy i32 in memory.
  if (Width == 1 || Width == 16 || Width == 32 || Width == 64)
    return nullptr;
  if (Width < 32)
    return Type::getInt32Ty(Ty->getContext());
  if (Width < 64)
    return Type::getInt64Ty(Ty->getContext());
  return nullptr;
}

enum class IntegerExtension { None, Zero, Sign };

// Zero-extend the low Width bits of a legal-width carrier.
static Value *maskToIntegerWidth(Value *V, unsigned Width,
                                 IRBuilder<> &Builder) {
  auto *LegalTy = cast<IntegerType>(V->getType());
  APInt Mask = APInt::getLowBitsSet(LegalTy->getBitWidth(), Width);
  return Builder.CreateAnd(V, ConstantInt::get(LegalTy, Mask));
}

// Return true when V is already a zero-extended Width-bit value.
static bool isKnownZeroExtendedFromWidth(Value *V, unsigned Width,
                                         const DataLayout &DL) {
  unsigned LegalWidth = V->getType()->getIntegerBitWidth();
  ConstantRange Range = computeConstantRangeIncludingKnownBits(
      V, /*ForSigned=*/false, SimplifyQuery(DL, dyn_cast<Instruction>(V)));
  return Range.getUnsignedMax().ult(APInt::getOneBitSet(LegalWidth, Width));
}

// Return true when V is already a sign-extended Width-bit value.
static bool isKnownSignExtendedFromWidth(Value *V, unsigned Width,
                                         const DataLayout &DL) {
  unsigned LegalWidth = V->getType()->getIntegerBitWidth();
  ConstantRange Range = computeConstantRangeIncludingKnownBits(
      V, /*ForSigned=*/true, SimplifyQuery(DL, dyn_cast<Instruction>(V)));
  APInt SignedMin = APInt::getSignedMinValue(Width).sext(LegalWidth);
  APInt SignedMax = APInt::getSignedMaxValue(Width).sext(LegalWidth);
  return Range.getSignedMin().sge(SignedMin) &&
         Range.getSignedMax().sle(SignedMax);
}

// Get an operand's legal carrier, normalizing it only when a consumer requires
// signed or unsigned narrow-integer semantics.
static Value *
getLegalizedIntegerOperand(Value *Operand, IntegerType *LegalTy,
                           IntegerExtension Extension, IRBuilder<> &Builder,
                           DenseMap<Value *, Value *> &ReplacedValues,
                           const DataLayout &DL) {
  if (Value *Replacement = ReplacedValues.lookup(Operand)) {
    Replacement = Builder.CreateZExtOrTrunc(Replacement, LegalTy);
    if (Extension == IntegerExtension::None)
      return Replacement;
    unsigned Width = cast<IntegerType>(Operand->getType())->getBitWidth();
    if (Extension == IntegerExtension::Zero) {
      if (auto *OverflowingOp = dyn_cast<OverflowingBinaryOperator>(Operand);
          OverflowingOp && OverflowingOp->hasNoUnsignedWrap())
        return Replacement;
      return isKnownZeroExtendedFromWidth(Replacement, Width, DL)
                 ? Replacement
                 : maskToIntegerWidth(Replacement, Width, Builder);
    }
    if (isKnownSignExtendedFromWidth(Replacement, Width, DL))
      return Replacement;
    if (auto *OverflowingOp = dyn_cast<OverflowingBinaryOperator>(Operand);
        OverflowingOp && OverflowingOp->hasNoSignedWrap())
      return Replacement;
    unsigned Shift = LegalTy->getBitWidth() - Width;
    return Builder.CreateAShr(Builder.CreateShl(Replacement, Shift), Shift);
  }
  if (auto *C = dyn_cast<ConstantInt>(Operand))
    return ConstantInt::get(
        LegalTy, Extension == IntegerExtension::Sign
                     ? C->getValue().sextOrTrunc(LegalTy->getBitWidth())
                     : C->getValue().zextOrTrunc(LegalTy->getBitWidth()));
  return Operand->getType() == LegalTy ? Operand : nullptr;
}

// bitcast <N x iM> to illegal iK -> extract and pack into an i32/i64 carrier.
static bool
legalizeNonStandardIntegerBitCast(BitCastInst &BitCast,
                                  SmallVectorImpl<Instruction *> &ToRemove,
                                  DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalTy = getLegalIntegerType(BitCast.getDestTy());
  auto *SourceTy = dyn_cast<FixedVectorType>(BitCast.getSrcTy());
  if (!LegalTy || !SourceTy || !SourceTy->getElementType()->isIntegerTy() ||
      SourceTy->getPrimitiveSizeInBits() !=
          BitCast.getDestTy()->getPrimitiveSizeInBits())
    return false;

  IRBuilder<> Builder(&BitCast);
  Value *Packed = ConstantInt::get(LegalTy, 0);
  unsigned ElementWidth = SourceTy->getScalarSizeInBits();
  for (unsigned Index = 0; Index < SourceTy->getNumElements(); ++Index) {
    Value *Element =
        Builder.CreateExtractElement(BitCast.getOperand(0), uint64_t(Index));
    Element = Builder.CreateZExt(Element, LegalTy);
    if (Index != 0)
      Element = Builder.CreateShl(Element, Index * ElementWidth);
    Packed = Builder.CreateOr(Packed, Element);
  }
  ReplacedValues[&BitCast] = Packed;
  ToRemove.push_back(&BitCast);
  return true;
}

// trunc illegal iM or iN -> retain its low bits in a legal result type.
static bool
legalizeNonStandardIntegerTrunc(TruncInst &Trunc,
                                SmallVectorImpl<Instruction *> &ToRemove,
                                DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalSrcTy = getLegalIntegerType(Trunc.getSrcTy());
  auto *LegalDstTy = getLegalIntegerType(Trunc.getDestTy());
  if (!LegalSrcTy && !LegalDstTy)
    return false;

  IRBuilder<> Builder(&Trunc);
  Value *Source = ReplacedValues.lookup(Trunc.getOperand(0));
  if (!Source)
    Source = Trunc.getOperand(0);
  Type *ResultTy = LegalDstTy ? LegalDstTy : Trunc.getDestTy();
  Value *Replacement = Builder.CreateZExtOrTrunc(Source, ResultTy);
  if (LegalDstTy)
    ReplacedValues[&Trunc] = Replacement;
  else
    Trunc.replaceAllUsesWith(Replacement);
  ToRemove.push_back(&Trunc);
  return true;
}

// binop illegal iN -> perform the operation in an i32/i64 carrier.
static bool
legalizeNonStandardIntegerBinOp(BinaryOperator &BO,
                                SmallVectorImpl<Instruction *> &ToRemove,
                                DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalTy = getLegalIntegerType(BO.getType());
  if (!LegalTy)
    return false;

  IRBuilder<> Builder(&BO);
  IntegerExtension LHSExtension = IntegerExtension::None;
  IntegerExtension RHSExtension = IntegerExtension::None;
  if (auto *OverflowingOp = dyn_cast<OverflowingBinaryOperator>(&BO)) {
    if (OverflowingOp->hasNoUnsignedWrap())
      LHSExtension = RHSExtension = IntegerExtension::Zero;
    else if (OverflowingOp->hasNoSignedWrap())
      LHSExtension = RHSExtension = IntegerExtension::Sign;
  }
  switch (BO.getOpcode()) {
  case Instruction::SDiv:
  case Instruction::SRem:
    LHSExtension = RHSExtension = IntegerExtension::Sign;
    break;
  case Instruction::AShr:
    LHSExtension = IntegerExtension::Sign;
    RHSExtension = IntegerExtension::Zero;
    break;
  case Instruction::UDiv:
  case Instruction::URem:
  case Instruction::LShr:
    LHSExtension = RHSExtension = IntegerExtension::Zero;
    break;
  case Instruction::Shl:
    RHSExtension = IntegerExtension::Zero;
    break;
  default:
    break;
  }
  Value *LHS =
      getLegalizedIntegerOperand(BO.getOperand(0), LegalTy, LHSExtension,
                                 Builder, ReplacedValues, BO.getDataLayout());
  Value *RHS =
      getLegalizedIntegerOperand(BO.getOperand(1), LegalTy, RHSExtension,
                                 Builder, ReplacedValues, BO.getDataLayout());
  if (!LHS || !RHS)
    return false;

  Value *NewBO = Builder.CreateBinOp(BO.getOpcode(), LHS, RHS);
  if (auto *NewBOInst = dyn_cast<BinaryOperator>(NewBO))
    NewBOInst->copyIRFlags(&BO);
  ReplacedValues[&BO] = NewBO;
  ToRemove.push_back(&BO);
  return true;
}

// select illegal iN -> select between i32/i64 carrier values.
static bool
legalizeNonStandardIntegerSelect(SelectInst &Select,
                                 SmallVectorImpl<Instruction *> &ToRemove,
                                 DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalTy = getLegalIntegerType(Select.getType());
  if (!LegalTy)
    return false;

  IRBuilder<> Builder(&Select);
  Value *True = getLegalizedIntegerOperand(
      Select.getTrueValue(), LegalTy, IntegerExtension::None, Builder,
      ReplacedValues, Select.getDataLayout());
  Value *False = getLegalizedIntegerOperand(
      Select.getFalseValue(), LegalTy, IntegerExtension::None, Builder,
      ReplacedValues, Select.getDataLayout());
  if (!True || !False)
    return false;

  ReplacedValues[&Select] = Builder.CreateSelect(
      Select.getCondition(), True, False, Select.getName(), &Select);
  ToRemove.push_back(&Select);
  return true;
}

// icmp illegal iN -> compare normalized i32/i64 carrier values.
static bool
legalizeNonStandardIntegerICmp(ICmpInst &Cmp,
                               SmallVectorImpl<Instruction *> &ToRemove,
                               DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalTy = getLegalIntegerType(Cmp.getOperand(0)->getType());
  if (!LegalTy)
    return false;

  IRBuilder<> Builder(&Cmp);
  IntegerExtension Extension =
      Cmp.isSigned() ? IntegerExtension::Sign : IntegerExtension::Zero;
  Value *LHS =
      getLegalizedIntegerOperand(Cmp.getOperand(0), LegalTy, Extension, Builder,
                                 ReplacedValues, Cmp.getDataLayout());
  Value *RHS =
      getLegalizedIntegerOperand(Cmp.getOperand(1), LegalTy, Extension, Builder,
                                 ReplacedValues, Cmp.getDataLayout());
  if (!LHS || !RHS)
    return false;

  Cmp.replaceAllUsesWith(Builder.CreateICmp(Cmp.getPredicate(), LHS, RHS));
  ToRemove.push_back(&Cmp);
  return true;
}

// cast with an illegal integer endpoint -> cast using legal carrier types.
static bool
legalizeNonStandardIntegerCast(CastInst &Cast,
                               SmallVectorImpl<Instruction *> &ToRemove,
                               DenseMap<Value *, Value *> &ReplacedValues) {
  auto *LegalSrcTy = getLegalIntegerType(Cast.getSrcTy());
  auto *LegalDstTy = getLegalIntegerType(Cast.getDestTy());
  if (!LegalSrcTy && !LegalDstTy)
    return false;

  IRBuilder<> Builder(&Cast);
  bool IsSigned = Cast.getOpcode() == Instruction::SExt ||
                  Cast.getOpcode() == Instruction::SIToFP;
  Value *Source = Cast.getOperand(0);
  if (LegalSrcTy)
    Source = getLegalizedIntegerOperand(
        Source, LegalSrcTy,
        IsSigned ? IntegerExtension::Sign : IntegerExtension::Zero, Builder,
        ReplacedValues, Cast.getDataLayout());
  if (!Source)
    return false;

  Type *ResultTy = LegalDstTy ? LegalDstTy : Cast.getDestTy();
  Value *Replacement = nullptr;
  switch (Cast.getOpcode()) {
  case Instruction::ZExt:
    Replacement = Builder.CreateZExtOrTrunc(Source, ResultTy);
    break;
  case Instruction::SExt:
    Replacement = Builder.CreateSExtOrTrunc(Source, ResultTy);
    break;
  case Instruction::FPToUI:
    Replacement = Builder.CreateFPToUI(Source, ResultTy);
    break;
  case Instruction::FPToSI:
    Replacement = Builder.CreateFPToSI(Source, ResultTy);
    break;
  case Instruction::UIToFP:
    Replacement = Builder.CreateUIToFP(Source, ResultTy);
    break;
  case Instruction::SIToFP:
    Replacement = Builder.CreateSIToFP(Source, ResultTy);
    break;
  case Instruction::PtrToInt:
    Replacement = Builder.CreatePtrToInt(Source, ResultTy);
    break;
  case Instruction::IntToPtr:
    Replacement = Builder.CreateIntToPtr(Source, ResultTy);
    break;
  default:
    return false;
  }

  if (LegalDstTy)
    ReplacedValues[&Cast] = Replacement;
  else
    Cast.replaceAllUsesWith(Replacement);
  ToRemove.push_back(&Cast);
  return true;
}

// Promote unsupported integer SSA operations to legal carriers while
// preserving the original width for normalization at semantic consumers.
static bool
legalizeNonStandardInteger(Instruction &I,
                           SmallVectorImpl<Instruction *> &ToRemove,
                           DenseMap<Value *, Value *> &ReplacedValues) {
  if (auto *BitCast = dyn_cast<BitCastInst>(&I))
    return legalizeNonStandardIntegerBitCast(*BitCast, ToRemove,
                                             ReplacedValues);
  if (auto *Trunc = dyn_cast<TruncInst>(&I))
    return legalizeNonStandardIntegerTrunc(*Trunc, ToRemove, ReplacedValues);
  if (auto *BO = dyn_cast<BinaryOperator>(&I))
    return legalizeNonStandardIntegerBinOp(*BO, ToRemove, ReplacedValues);
  if (auto *Select = dyn_cast<SelectInst>(&I))
    return legalizeNonStandardIntegerSelect(*Select, ToRemove, ReplacedValues);
  if (auto *Cmp = dyn_cast<ICmpInst>(&I))
    return legalizeNonStandardIntegerICmp(*Cmp, ToRemove, ReplacedValues);
  if (auto *Cast = dyn_cast<CastInst>(&I))
    return legalizeNonStandardIntegerCast(*Cast, ToRemove, ReplacedValues);
  return false;
}

static bool legalizeFreeze(Instruction &I,
                           SmallVectorImpl<Instruction *> &ToRemove,
                           DenseMap<Value *, Value *>) {
  auto *FI = dyn_cast<FreezeInst>(&I);
  if (!FI)
    return false;

  FI->replaceAllUsesWith(FI->getOperand(0));
  ToRemove.push_back(FI);
  return true;
}

static bool legalizeI8MemoryUses(Instruction &I,
                                 SmallVectorImpl<Instruction *> &ToRemove,
                                 DenseMap<Value *, Value *> &ReplacedValues) {

  auto ProcessOperands = [&](SmallVector<Value *> &NewOperands) {
    Type *InstrType = IntegerType::get(I.getContext(), 32);

    for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
      Value *Op = I.getOperand(OpIdx);
      if (ReplacedValues.count(Op) &&
          ReplacedValues[Op]->getType()->isIntegerTy())
        InstrType = ReplacedValues[Op]->getType();
    }

    for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
      Value *Op = I.getOperand(OpIdx);
      if (ReplacedValues.count(Op))
        NewOperands.push_back(ReplacedValues[Op]);
      else if (auto *Imm = dyn_cast<ConstantInt>(Op)) {
        APInt Value = Imm->getValue();
        unsigned NewBitWidth = InstrType->getIntegerBitWidth();
        // Note: options here are sext or sextOrTrunc.
        // Since i8 isn't supported, we assume new values
        // will always have a higher bitness.
        assert(NewBitWidth > Value.getBitWidth() &&
               "Replacement's BitWidth should be larger than Current.");
        APInt NewValue = Value.sext(NewBitWidth);
        NewOperands.push_back(ConstantInt::get(InstrType, NewValue));
      } else {
        assert(!Op->getType()->isIntegerTy(8));
        NewOperands.push_back(Op);
      }
    }
  };
  IRBuilder<> Builder(&I);
  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    if (!Store->getValueOperand()->getType()->isIntegerTy(8))
      return false;

    Value *StoredValue = Store->getValueOperand();
    if (Value *Replacement = ReplacedValues.lookup(StoredValue))
      StoredValue = Replacement;
    Value *Pointer = Store->getPointerOperand();
    if (Value *Replacement = ReplacedValues.lookup(Pointer))
      Pointer = Replacement;

    Type *StorageTy = nullptr;
    if (auto *AI = dyn_cast<AllocaInst>(Pointer))
      StorageTy = AI->getAllocatedType();
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer))
      StorageTy = GEP->getSourceElementType();
    else if (auto *GV = dyn_cast<GlobalVariable>(Pointer))
      StorageTy = GV->getValueType();
    if (auto *ArrayTy = dyn_cast_or_null<ArrayType>(StorageTy))
      StorageTy = ArrayTy->getArrayElementType();
    if (!StorageTy || !StorageTy->isIntegerTy())
      return false;

    StoredValue = Builder.CreateZExtOrTrunc(StoredValue, StorageTy);
    Value *NewStore = Builder.CreateStore(StoredValue, Pointer);
    ReplacedValues[Store] = NewStore;
    ToRemove.push_back(Store);
    return true;
  }

  if (auto *Load = dyn_cast<LoadInst>(&I);
      Load && I.getType()->isIntegerTy(8)) {
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Type *ElementType = NewOperands[0]->getType();
    if (auto *AI = dyn_cast<AllocaInst>(NewOperands[0]))
      ElementType = AI->getAllocatedType();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(NewOperands[0])) {
      ElementType = GEP->getSourceElementType();
    }
    if (ElementType->isArrayTy())
      ElementType = ElementType->getArrayElementType();
    LoadInst *NewLoad = Builder.CreateLoad(ElementType, NewOperands[0]);
    ReplacedValues[Load] = NewLoad;
    ToRemove.push_back(Load);
    return true;
  }

  if (auto *Load = dyn_cast<LoadInst>(&I);
      Load && isa<ConstantExpr>(Load->getPointerOperand())) {
    auto *CE = dyn_cast<ConstantExpr>(Load->getPointerOperand());
    if (!(CE->getOpcode() == Instruction::GetElementPtr))
      return false;
    auto *GEP = dyn_cast<GEPOperator>(CE);
    if (!GEP->getSourceElementType()->isIntegerTy(8))
      return false;

    Type *ElementType = Load->getType();
    ConstantInt *Offset = dyn_cast<ConstantInt>(GEP->getOperand(1));
    uint32_t ByteOffset = Offset->getZExtValue();
    uint32_t ElemSize = Load->getDataLayout().getTypeAllocSize(ElementType);
    uint32_t Index = ByteOffset / ElemSize;

    Value *PtrOperand = GEP->getPointerOperand();
    Type *GEPType = GEP->getPointerOperandType();

    if (auto *GV = dyn_cast<GlobalVariable>(PtrOperand))
      GEPType = GV->getValueType();
    if (auto *AI = dyn_cast<AllocaInst>(PtrOperand))
      GEPType = AI->getAllocatedType();

    if (auto *ArrTy = dyn_cast<ArrayType>(GEPType))
      GEPType = ArrTy;
    else
      GEPType = ArrayType::get(ElementType, 1); // its a scalar

    Value *NewGEP = Builder.CreateGEP(
        GEPType, PtrOperand, {Builder.getInt32(0), Builder.getInt32(Index)},
        GEP->getName(), GEP->getNoWrapFlags());

    LoadInst *NewLoad = Builder.CreateLoad(ElementType, NewGEP);
    ReplacedValues[Load] = NewLoad;
    Load->replaceAllUsesWith(NewLoad);
    ToRemove.push_back(Load);
    return true;
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    if (!GEP->getType()->isPointerTy() ||
        !GEP->getSourceElementType()->isIntegerTy(8))
      return false;

    Value *BasePtr = GEP->getPointerOperand();
    if (ReplacedValues.count(BasePtr))
      BasePtr = ReplacedValues[BasePtr];

    Type *ElementType = BasePtr->getType();

    if (auto *AI = dyn_cast<AllocaInst>(BasePtr))
      ElementType = AI->getAllocatedType();
    if (auto *GV = dyn_cast<GlobalVariable>(BasePtr))
      ElementType = GV->getValueType();

    Type *GEPType = ElementType;
    if (auto *ArrTy = dyn_cast<ArrayType>(ElementType))
      ElementType = ArrTy->getArrayElementType();
    else
      GEPType = ArrayType::get(ElementType, 1); // its a scalar

    ConstantInt *Offset = dyn_cast<ConstantInt>(GEP->getOperand(1));
    // Note: i8 to i32 offset conversion without emitting IR requires constant
    // ints. Since offset conversion is common, we can safely assume Offset is
    // always a ConstantInt, so no need to have a conditional bail out on
    // nullptr, instead assert this is the case.
    assert(Offset && "Offset is expected to be a ConstantInt");
    uint32_t ByteOffset = Offset->getZExtValue();
    uint32_t ElemSize = GEP->getDataLayout().getTypeAllocSize(ElementType);
    assert(ElemSize > 0 && "ElementSize must be set");
    uint32_t Index = ByteOffset / ElemSize;
    Value *NewGEP = Builder.CreateGEP(
        GEPType, BasePtr, {Builder.getInt32(0), Builder.getInt32(Index)},
        GEP->getName(), GEP->getNoWrapFlags());
    ReplacedValues[GEP] = NewGEP;
    GEP->replaceAllUsesWith(NewGEP);
    ToRemove.push_back(GEP);
    return true;
  }
  return false;
}

static bool upcastI8AllocasAndUses(Instruction &I,
                                   SmallVectorImpl<Instruction *> &ToRemove,
                                   DenseMap<Value *, Value *> &ReplacedValues) {
  auto *AI = dyn_cast<AllocaInst>(&I);
  if (!AI || !AI->getAllocatedType()->isIntegerTy(8))
    return false;

  Type *SmallestType = nullptr;

  auto ProcessLoad = [&](LoadInst *Load) {
    for (User *LU : Load->users()) {
      CastInst *Cast = dyn_cast<CastInst>(LU);
      if (!Cast)
        continue;
      Type *Ty = Cast->getType();

      if (!SmallestType ||
          Ty->getPrimitiveSizeInBits() < SmallestType->getPrimitiveSizeInBits())
        SmallestType = Ty;
    }
  };

  for (User *U : AI->users()) {
    if (auto *Load = dyn_cast<LoadInst>(U))
      ProcessLoad(Load);
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      for (User *GU : GEP->users()) {
        if (auto *Load = dyn_cast<LoadInst>(GU))
          ProcessLoad(Load);
      }
    }
  }

  if (!SmallestType)
    return false; // no valid casts found

  // Replace alloca
  IRBuilder<> Builder(AI);
  auto *NewAlloca = Builder.CreateAlloca(SmallestType);
  ReplacedValues[AI] = NewAlloca;
  ToRemove.push_back(AI);
  return true;
}

static bool
downcastI64toI32InsertExtractElements(Instruction &I,
                                      SmallVectorImpl<Instruction *> &ToRemove,
                                      DenseMap<Value *, Value *> &) {

  if (auto *Extract = dyn_cast<ExtractElementInst>(&I)) {
    Value *Idx = Extract->getIndexOperand();
    auto *CI = dyn_cast<ConstantInt>(Idx);
    if (CI && CI->getBitWidth() == 64) {
      IRBuilder<> Builder(Extract);
      int64_t IndexValue = CI->getSExtValue();
      auto *Idx32 =
          ConstantInt::get(Type::getInt32Ty(I.getContext()), IndexValue);
      Value *NewExtract = Builder.CreateExtractElement(
          Extract->getVectorOperand(), Idx32, Extract->getName());

      Extract->replaceAllUsesWith(NewExtract);
      ToRemove.push_back(Extract);
      return true;
    }
  }

  if (auto *Insert = dyn_cast<InsertElementInst>(&I)) {
    Value *Idx = Insert->getOperand(2);
    auto *CI = dyn_cast<ConstantInt>(Idx);
    if (CI && CI->getBitWidth() == 64) {
      int64_t IndexValue = CI->getSExtValue();
      auto *Idx32 =
          ConstantInt::get(Type::getInt32Ty(I.getContext()), IndexValue);
      IRBuilder<> Builder(Insert);
      Value *Insert32Index = Builder.CreateInsertElement(
          Insert->getOperand(0), Insert->getOperand(1), Idx32,
          Insert->getName());

      Insert->replaceAllUsesWith(Insert32Index);
      ToRemove.push_back(Insert);
      return true;
    }
  }
  return false;
}

static bool updateFnegToFsub(Instruction &I,
                             SmallVectorImpl<Instruction *> &ToRemove,
                             DenseMap<Value *, Value *> &) {
  const Intrinsic::ID ID = I.getOpcode();
  if (ID != Instruction::FNeg)
    return false;

  IRBuilder<> Builder(&I);
  Value *In = I.getOperand(0);
  Value *Zero = ConstantFP::get(In->getType(), -0.0);
  I.replaceAllUsesWith(Builder.CreateFSub(Zero, In));
  ToRemove.push_back(&I);
  return true;
}

static bool
resolveUnreachableSwitchDefault(Instruction &I,
                                SmallVectorImpl<Instruction *> &ToRemove,
                                DenseMap<Value *, Value *> &) {
  auto *SI = dyn_cast<SwitchInst>(&I);
  if (!SI || SI->getNumCases() == 0)
    return false;

  BasicBlock *DefaultBB = SI->getDefaultDest();

  // Check if the default destination ends with an unreachable instruction.
  if (DefaultBB->size() == 0 ||
      !isa<UnreachableInst>(DefaultBB->getTerminator()))
    return false;

  // Try to find a common successor of all case destinations. If all case
  // blocks unconditionally branch to the same block, that is the common
  // successor. This is just a best effort, and is done as the original form of
  // the switch statement was likely in this form before being transformed to
  // an unreachable branch.
  BasicBlock *CommonSuccessor = nullptr;
  for (auto &Case : SI->cases()) {
    BasicBlock *CaseBB = Case.getCaseSuccessor();
    auto *BI = dyn_cast<UncondBrInst>(CaseBB->getTerminator());
    if (!BI) {
      CommonSuccessor = nullptr;
      break;
    }
    BasicBlock *Succ = BI->getSuccessor(0);
    if (!CommonSuccessor)
      CommonSuccessor = Succ;
    else if (CommonSuccessor != Succ) {
      CommonSuccessor = nullptr;
      break;
    }
  }

  BasicBlock *NewDefault =
      CommonSuccessor ? CommonSuccessor : SI->case_begin()->getCaseSuccessor();

  BasicBlock *SwitchBB = SI->getParent();
  SI->setDefaultDest(NewDefault);

  // Ensure all phi nodes are legal by adding an incoming poison value from the
  // unreachable branch.
  for (PHINode &Phi : NewDefault->phis())
    Phi.addIncoming(PoisonValue::get(Phi.getType()), SwitchBB);

  return true;
}

static bool
legalizeScalarLoadStoreOnArrays(Instruction &I,
                                SmallVectorImpl<Instruction *> &ToRemove,
                                DenseMap<Value *, Value *> &) {

  Value *PtrOp;
  unsigned PtrOpIndex;
  [[maybe_unused]] Type *LoadStoreTy;
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    PtrOp = LI->getPointerOperand();
    PtrOpIndex = LI->getPointerOperandIndex();
    LoadStoreTy = LI->getType();
  } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
    PtrOp = SI->getPointerOperand();
    PtrOpIndex = SI->getPointerOperandIndex();
    LoadStoreTy = SI->getValueOperand()->getType();
  } else
    return false;

  // If the load/store is not of a single-value type (i.e., scalar or vector)
  // then we do not modify it. It shouldn't be a vector either because the
  // dxil-data-scalarization pass is expected to run before this, but it's not
  // incorrect to apply this transformation to vector load/stores.
  if (!LoadStoreTy->isSingleValueType())
    return false;

  Type *ArrayTy;
  if (auto *GlobalVarPtrOp = dyn_cast<GlobalVariable>(PtrOp))
    ArrayTy = GlobalVarPtrOp->getValueType();
  else if (auto *AllocaPtrOp = dyn_cast<AllocaInst>(PtrOp))
    ArrayTy = AllocaPtrOp->getAllocatedType();
  else
    return false;

  if (!isa<ArrayType>(ArrayTy))
    return false;

  assert(ArrayTy->getArrayElementType() == LoadStoreTy &&
         "Expected array element type to be the same as to the scalar load or "
         "store type");

  Value *Zero = ConstantInt::get(Type::getInt32Ty(I.getContext()), 0);
  Value *GEP = GetElementPtrInst::Create(
      ArrayTy, PtrOp, {Zero, Zero}, GEPNoWrapFlags::all(), "", I.getIterator());
  I.setOperand(PtrOpIndex, GEP);
  return true;
}

namespace {
class DXILLegalizationPipeline {

public:
  DXILLegalizationPipeline() { initializeLegalizationPipeline(); }

  bool runLegalizationPipeline(Function &F) {
    bool MadeChange = false;
    SmallVector<Instruction *> ToRemove;
    DenseMap<Value *, Value *> ReplacedValues;
    for (int Stage = 0; Stage < NumStages; ++Stage) {
      ToRemove.clear();
      ReplacedValues.clear();
      for (auto &I : instructions(F)) {
        for (auto &LegalizationFn : LegalizationPipeline[Stage])
          MadeChange |= LegalizationFn(I, ToRemove, ReplacedValues);
      }

      for (auto *Inst : reverse(ToRemove))
        Inst->eraseFromParent();
    }

    if (MadeChange)
      MadeChange |= removeUnreachableBlocks(F);
    return MadeChange;
  }

private:
  enum LegalizationStage { Stage1 = 0, Stage2, NumStages };

  using LegalizationFnTy =
      std::function<bool(Instruction &, SmallVectorImpl<Instruction *> &,
                         DenseMap<Value *, Value *> &)>;

  SmallVector<LegalizationFnTy> LegalizationPipeline[NumStages];

  void initializeLegalizationPipeline() {
    LegalizationPipeline[Stage1].push_back(upcastI8AllocasAndUses);
    LegalizationPipeline[Stage1].push_back(legalizeI8MemoryUses);
    LegalizationPipeline[Stage1].push_back(legalizeNonStandardInteger);
    LegalizationPipeline[Stage1].push_back(legalizeFreeze);
    LegalizationPipeline[Stage1].push_back(updateFnegToFsub);
    LegalizationPipeline[Stage1].push_back(
        downcastI64toI32InsertExtractElements);
    LegalizationPipeline[Stage2].push_back(legalizeScalarLoadStoreOnArrays);
    LegalizationPipeline[Stage2].push_back(resolveUnreachableSwitchDefault);
  }
};

class DXILLegalizeLegacy : public FunctionPass {

public:
  bool runOnFunction(Function &F) override;
  DXILLegalizeLegacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
};
} // namespace

PreservedAnalyses DXILLegalizePass::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  DXILLegalizationPipeline DXLegalize;
  bool MadeChanges = DXLegalize.runLegalizationPipeline(F);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  return PA;
}

bool DXILLegalizeLegacy::runOnFunction(Function &F) {
  DXILLegalizationPipeline DXLegalize;
  return DXLegalize.runLegalizationPipeline(F);
}

char DXILLegalizeLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILLegalizeLegacy, DEBUG_TYPE, "DXIL Legalizer", false,
                      false)
INITIALIZE_PASS_END(DXILLegalizeLegacy, DEBUG_TYPE, "DXIL Legalizer", false,
                    false)

FunctionPass *llvm::createDXILLegalizeLegacyPass() {
  return new DXILLegalizeLegacy();
}
