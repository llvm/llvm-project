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
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <functional>

#define DEBUG_TYPE "dxil-legalize"

using namespace llvm;

namespace {

// Map an unsupported integer type to the smallest legal DXIL carrier type.
// Return nullptr only for non-integer types or already legal integer types.
static IntegerType *getLegalIntegerType(Type *Ty) {
  IntegerType *IntTy = dyn_cast<IntegerType>(Ty);
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
  report_fatal_error("DXIL does not support integer types wider than 64 bits",
                     /*gen_crash_diag=*/false);
}

enum class IntegerExtension { None, Zero, Sign };

// Clear all but the low Width bits of a legal-width carrier.
static Value *maskToIntegerWidth(Value *V, unsigned Width,
                                 IRBuilder<> &Builder) {
  auto *LegalTy = cast<IntegerType>(V->getType());
  APInt Mask = APInt::getLowBitsSet(LegalTy->getBitWidth(), Width);
  return Builder.CreateAnd(V, ConstantInt::get(LegalTy, Mask));
}

// Promote unsupported integer SSA operations to legal carriers while
// preserving the original width for normalization at semantic consumers.
// Poison inputs may become defined; do not propagate narrow-width IR flags.
static bool
legalizeNonStandardInteger(Instruction &I,
                           SmallVectorImpl<Instruction *> &ToRemove,
                           DenseMap<Value *, Value *> &ReplacedValues) {
  IRBuilder<> Builder(&I);

  // Get an operand's legal carrier, normalizing it only when a consumer
  // requires signed or unsigned narrow-integer semantics.
  auto operand = [&](Value *Operand, IntegerExtension Extension =
                                         IntegerExtension::None) -> Value * {
    IntegerType *LegalTy = getLegalIntegerType(Operand->getType());
    assert(LegalTy && "Expected an unsupported-width integer operand");
    Value *Replacement =
        ReplacedValues.lookup_or(Operand, dyn_cast<ConstantInt>(Operand));
    if (!Replacement)
      report_fatal_error(
          "DXIL legalization is missing an integer operand replacement",
          /*gen_crash_diag=*/false);
    Replacement = Builder.CreateZExtOrTrunc(Replacement, LegalTy);
    if (Extension == IntegerExtension::None)
      return Replacement;
    unsigned Width = Operand->getType()->getIntegerBitWidth();
    unsigned LegalWidth = LegalTy->getBitWidth();
    auto *ContextI = dyn_cast<Instruction>(Replacement);
    if (Extension == IntegerExtension::Zero)
      return MaskedValueIsZero(Replacement,
                               APInt::getBitsSetFrom(LegalWidth, Width),
                               SimplifyQuery(I.getDataLayout(), ContextI))
                 ? Replacement
                 : maskToIntegerWidth(Replacement, Width, Builder);
    unsigned Shift = LegalWidth - Width;
    if (ComputeNumSignBits(Replacement, I.getDataLayout(), /*AC=*/nullptr,
                           ContextI) > Shift)
      return Replacement;
    return Builder.CreateAShr(Builder.CreateShl(Replacement, Shift), Shift);
  };

  auto replace = [&](Value *Replacement) {
    if (I.getType() == Replacement->getType())
      I.replaceAllUsesWith(Replacement);
    else
      ReplacedValues[&I] = Replacement;
    ToRemove.push_back(&I);
    return true;
  };

  // Legalize bitcasts between an illegal integer and a fixed integer vector.
  if (auto *BitCast = dyn_cast<BitCastInst>(&I)) {
    IntegerType *LegalDstTy = getLegalIntegerType(BitCast->getDestTy());
    IntegerType *LegalSrcTy = getLegalIntegerType(BitCast->getSrcTy());
    if (!LegalDstTy && !LegalSrcTy)
      return false;
    FixedVectorType *VectorTy = dyn_cast<FixedVectorType>(
        LegalDstTy ? BitCast->getSrcTy() : BitCast->getDestTy());
    if (!VectorTy || !VectorTy->getElementType()->isIntegerTy() ||
        getLegalIntegerType(VectorTy->getElementType()))
      report_fatal_error(
          "DXIL legalization does not support this integer bitcast",
          /*gen_crash_diag=*/false);
    assert(BitCast->getSrcTy()->getPrimitiveSizeInBits() ==
               BitCast->getDestTy()->getPrimitiveSizeInBits() &&
           "Bitcast source and destination must have equal sizes");

    unsigned ElementWidth = VectorTy->getScalarSizeInBits();
    unsigned VecSize = VectorTy->getNumElements();
    if (LegalDstTy) {
      Value *Packed = ConstantInt::get(LegalDstTy, 0);
      for (unsigned Index = 0; Index < VecSize; ++Index) {
        Value *Element = Builder.CreateExtractElement(BitCast->getOperand(0),
                                                      Builder.getInt32(Index));
        Element = Builder.CreateZExt(Element, LegalDstTy);
        if (Index != 0)
          Element = Builder.CreateShl(Element, Index * ElementWidth);
        Packed = Builder.CreateOr(Packed, Element);
      }
      return replace(Packed);
    }
    Value *Packed = operand(BitCast->getOperand(0));
    Value *Unpacked = PoisonValue::get(VectorTy);
    for (unsigned Index = 0; Index < VecSize; ++Index) {
      Value *Element = Packed;
      if (Index != 0)
        Element = Builder.CreateLShr(Element, Index * ElementWidth);
      Element = Builder.CreateTrunc(Element, VectorTy->getElementType());
      Unpacked = Builder.CreateInsertElement(Unpacked, Element,
                                             Builder.getInt32(Index));
    }
    return replace(Unpacked);
  }

  // binop illegal iN -> perform the operation in an i32/i64 carrier.
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    if (!getLegalIntegerType(BO->getType()))
      return false;

    IntegerExtension LHSExtension = IntegerExtension::None;
    switch (BO->getOpcode()) {
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::AShr:
      LHSExtension = IntegerExtension::Sign;
      break;
    case Instruction::UDiv:
    case Instruction::URem:
    case Instruction::LShr:
      LHSExtension = IntegerExtension::Zero;
      break;
    default:
      break;
    }
    IntegerExtension RHSExtension =
        BO->isShift() ? IntegerExtension::Zero : LHSExtension;
    Value *LHS = operand(BO->getOperand(0), LHSExtension);
    Value *RHS = operand(BO->getOperand(1), RHSExtension);
    return replace(Builder.CreateBinOp(BO->getOpcode(), LHS, RHS));
  }

  // select illegal iN -> select between i32/i64 carrier values.
  if (auto *Select = dyn_cast<SelectInst>(&I)) {
    if (!getLegalIntegerType(Select->getType()))
      return false;

    Value *True = operand(Select->getTrueValue());
    Value *False = operand(Select->getFalseValue());
    Value *Replacement = Builder.CreateSelect(Select->getCondition(), True,
                                              False, Select->getName(), Select);
    if (auto *NewSelect = dyn_cast<SelectInst>(Replacement);
        NewSelect && !NewSelect->getMetadata(LLVMContext::MD_prof))
      setExplicitlyUnknownBranchWeightsIfProfiled(*NewSelect, DEBUG_TYPE);
    return replace(Replacement);
  }

  // icmp illegal iN -> compare normalized i32/i64 carrier values.
  if (auto *Cmp = dyn_cast<ICmpInst>(&I)) {
    if (!getLegalIntegerType(Cmp->getOperand(0)->getType()))
      return false;

    IntegerExtension Extension =
        Cmp->isSigned() ? IntegerExtension::Sign : IntegerExtension::Zero;
    Value *LHS = operand(Cmp->getOperand(0), Extension);
    Value *RHS = operand(Cmp->getOperand(1), Extension);
    return replace(Builder.CreateICmp(Cmp->getPredicate(), LHS, RHS));
  }

  // cast with an illegal integer endpoint -> cast using legal carrier types.
  if (auto *Cast = dyn_cast<CastInst>(&I)) {
    auto *LegalSrcTy = getLegalIntegerType(Cast->getSrcTy());
    auto *LegalDstTy = getLegalIntegerType(Cast->getDestTy());
    if (!LegalSrcTy && !LegalDstTy)
      return false;

    IntegerExtension Extension = IntegerExtension::Zero;
    if (Cast->getOpcode() == Instruction::Trunc)
      Extension = IntegerExtension::None;
    else if (Cast->getOpcode() == Instruction::SExt ||
             Cast->getOpcode() == Instruction::SIToFP)
      Extension = IntegerExtension::Sign;
    Value *Source = Cast->getOperand(0);
    if (LegalSrcTy)
      Source = operand(Source, Extension);
    Type *ResultTy = LegalDstTy ? LegalDstTy : Cast->getDestTy();
    if (Cast->isIntegerCast())
      return replace(Builder.CreateIntCast(
          Source, ResultTy, Extension == IntegerExtension::Sign));
    return replace(Builder.CreateCast(Cast->getOpcode(), Source, ResultTy));
  }

  if (ReplacedValues.contains(&I) || isa<FreezeInst>(I))
    return false;
  if (getLegalIntegerType(I.getType()))
    report_fatal_error(
        Twine("DXIL legalization does not support non-standard integer result "
              "type for instruction '") +
            I.getOpcodeName() + "'",
        /*gen_crash_diag=*/false);
  for (Value *Operand : I.operands())
    if (getLegalIntegerType(Operand->getType()))
      report_fatal_error(Twine("DXIL legalization does not support "
                               "non-standard integer operand "
                               "type for instruction '") +
                             I.getOpcodeName() + "'",
                         /*gen_crash_diag=*/false);
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
  IRBuilder<> Builder(&I);
  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    if (!Store->getValueOperand()->getType()->isIntegerTy(8))
      return false;

    Value *StoredValue = ReplacedValues.lookup_or(Store->getValueOperand(),
                                                  Store->getValueOperand());
    Value *Pointer = ReplacedValues.lookup_or(Store->getPointerOperand(),
                                              Store->getPointerOperand());

    Type *StorageTy = nullptr;
    if (auto *AI = dyn_cast<AllocaInst>(Pointer))
      StorageTy = AI->getAllocatedType();
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer))
      StorageTy = GEP->getSourceElementType();
    else if (auto *GV = dyn_cast<GlobalVariable>(Pointer))
      StorageTy = GV->getValueType();
    if (auto *ArrayTy = dyn_cast_or_null<ArrayType>(StorageTy))
      StorageTy = ArrayTy->getArrayElementType();
    if (!StorageTy && isa<Argument>(Pointer))
      StorageTy = Builder.getInt32Ty();
    else if (!StorageTy)
      report_fatal_error(
          "DXIL legalization cannot determine the i8 store's storage type",
          /*gen_crash_diag=*/false);
    else if (!StorageTy->isIntegerTy())
      return false;

    StoredValue = maskToIntegerWidth(StoredValue, 8, Builder);
    StoredValue = Builder.CreateZExtOrTrunc(StoredValue, StorageTy);
    Value *NewStore = Builder.CreateStore(StoredValue, Pointer);
    ReplacedValues[Store] = NewStore;
    ToRemove.push_back(Store);
    return true;
  }

  if (auto *Load = dyn_cast<LoadInst>(&I);
      Load && I.getType()->isIntegerTy(8)) {
    Value *Pointer = ReplacedValues.lookup_or(Load->getPointerOperand(),
                                              Load->getPointerOperand());
    Type *ElementType = Pointer->getType();
    if (auto *AI = dyn_cast<AllocaInst>(Pointer))
      ElementType = AI->getAllocatedType();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
      ElementType = GEP->getSourceElementType();
    }
    if (ElementType->isArrayTy())
      ElementType = ElementType->getArrayElementType();
    LoadInst *NewLoad = Builder.CreateLoad(ElementType, Pointer);
    ReplacedValues[Load] = NewLoad;
    ToRemove.push_back(Load);
    return true;
  }

  // Loads supply their access type; standalone GEPs use the storage element.
  auto createLegalGEP = [&](GEPOperator *GEP, Value *BasePtr,
                            Type *ElementType = nullptr) {
    Type *GEPType = BasePtr->getType();
    if (auto *AI = dyn_cast<AllocaInst>(BasePtr))
      GEPType = AI->getAllocatedType();
    if (auto *GV = dyn_cast<GlobalVariable>(BasePtr))
      GEPType = GV->getValueType();
    if (!ElementType)
      ElementType =
          GEPType->isArrayTy() ? GEPType->getArrayElementType() : GEPType;
    if (!GEPType->isArrayTy())
      GEPType = ArrayType::get(ElementType, 1);
    auto *Offset = dyn_cast<ConstantInt>(GEP->getOperand(1));
    assert(Offset && "Offset is expected to be a ConstantInt");
    uint32_t ByteOffset = Offset->getZExtValue();
    uint32_t ElemSize = I.getDataLayout().getTypeAllocSize(ElementType);
    assert(ElemSize > 0 && "ElementSize must be set");
    uint32_t Index = ByteOffset / ElemSize;
    return GetElementPtrInst::Create(
        GEPType, BasePtr, {Builder.getInt32(0), Builder.getInt32(Index)},
        GEP->getNoWrapFlags(), GEP->getName(), I.getIterator());
  };

  if (auto *Load = dyn_cast<LoadInst>(&I);
      Load && isa<ConstantExpr>(Load->getPointerOperand())) {
    auto *GEP = dyn_cast<GEPOperator>(Load->getPointerOperand());
    if (!GEP || !GEP->getSourceElementType()->isIntegerTy(8))
      return false;

    Type *ElementType = Load->getType();
    Value *NewGEP = createLegalGEP(GEP, GEP->getPointerOperand(), ElementType);
    LoadInst *NewLoad = Builder.CreateLoad(ElementType, NewGEP);
    ReplacedValues[Load] = NewLoad;
    Load->replaceAllUsesWith(NewLoad);
    ToRemove.push_back(Load);
    return true;
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    if (!GEP->getSourceElementType()->isIntegerTy(8))
      return false;

    Value *BasePtr = ReplacedValues.lookup_or(GEP->getPointerOperand(),
                                              GEP->getPointerOperand());

    Value *NewGEP = createLegalGEP(cast<GEPOperator>(GEP), BasePtr);
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

// DXIL has no floating-point atomic operation. A float exchange only moves the
// bit pattern, so exchange an integer of the same width instead. Opaque
// pointers keep the pointer operand type-agnostic, so only the value and the
// result need a cast. This matches what DXC emits for groupshared memory.
static bool
legalizeFloatAtomicExchange(Instruction &I,
                            SmallVectorImpl<Instruction *> &ToRemove,
                            DenseMap<Value *, Value *> &) {
  auto *AI = dyn_cast<AtomicRMWInst>(&I);
  if (!AI || AI->getOperation() != AtomicRMWInst::Xchg)
    return false;

  Type *ValTy = AI->getValOperand()->getType();
  if (!ValTy->isFloatingPointTy())
    return false;

  // DXIL has 32-bit and 64-bit atomics only. A float of any other width has no
  // integer exchange to lower to.
  unsigned Width = ValTy->getPrimitiveSizeInBits();
  if (Width != 32 && Width != 64)
    reportFatalUsageError("DXIL atomic exchange requires a 32-bit or 64-bit "
                          "floating-point value");

  IRBuilder<> Builder(AI);
  Type *IntTy = Builder.getIntNTy(Width);
  Value *Val = Builder.CreateBitCast(AI->getValOperand(), IntTy);
  AtomicRMWInst *NewAI = Builder.CreateAtomicRMW(
      AtomicRMWInst::Xchg, AI->getPointerOperand(), Val, AI->getAlign(),
      AI->getOrdering(), AI->getSyncScopeID());
  NewAI->copyMetadata(*AI);
  AI->replaceAllUsesWith(Builder.CreateBitCast(NewAI, ValTy));
  ToRemove.push_back(AI);
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
    LegalizationPipeline[Stage1].push_back(legalizeFloatAtomicExchange);
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
