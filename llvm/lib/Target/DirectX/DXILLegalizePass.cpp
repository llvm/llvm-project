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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <functional>

#define DEBUG_TYPE "dxil-legalize"

using namespace llvm;

static IntegerType *getLegalIntegerType(Type *Ty) {
  auto *IT = dyn_cast<IntegerType>(Ty);
  if (!IT)
    return nullptr;
  unsigned Width = IT->getBitWidth();
  if (Width == 1 || Width == 16 || Width == 32 || Width == 64)
    return nullptr;
  if (Width > 64)
    reportFatalUsageError(
        "DXIL does not support integer types wider than 64 bits");
  return IntegerType::get(Ty->getContext(), Width <= 32 ? 32 : 64);
}

enum class IntegerExtension { None, Zero, Sign };

static bool legalizeInteger(Instruction &I,
                            SmallVectorImpl<Instruction *> &ToRemove,
                            DenseMap<Value *, Value *> &ReplacedValues) {
  IRBuilder<> B(&I);
  auto GetOperand = [&](Value *V,
                        IntegerExtension Ext = IntegerExtension::None) {
    auto *Ty = getLegalIntegerType(V->getType());
    if (!Ty)
      return V;
    Value *R = ReplacedValues.lookup_or(V, dyn_cast<Constant>(V));
    if (!R)
      reportFatalUsageError("DXIL legalization is missing an integer operand "
                            "replacement");
    R = B.CreateZExtOrTrunc(R, Ty);
    unsigned Width = V->getType()->getIntegerBitWidth();
    unsigned Shift = Ty->getBitWidth() - Width;
    auto *ContextI = dyn_cast<Instruction>(R);
    if (Ext == IntegerExtension::Sign &&
        ComputeNumSignBits(R, I.getDataLayout(), nullptr, ContextI) <= Shift)
      R = B.CreateAShr(B.CreateShl(R, Shift), Shift);
    if (Ext == IntegerExtension::Zero &&
        !MaskedValueIsZero(R, APInt::getBitsSetFrom(Ty->getBitWidth(), Width),
                           SimplifyQuery(I.getDataLayout(), ContextI)))
      R = B.CreateAnd(R, APInt::getLowBitsSet(Ty->getBitWidth(), Width));
    return R;
  };
  auto Replace = [&](Value *V) {
    if (I.getType() == V->getType())
      I.replaceAllUsesWith(V);
    else
      ReplacedValues[&I] = V;
    ToRemove.push_back(&I);
  };

  if (auto *BC = dyn_cast<BitCastInst>(&I)) {
    auto *Dst = getLegalIntegerType(BC->getDestTy());
    auto *Src = getLegalIntegerType(BC->getSrcTy());
    if (!Dst && !Src)
      return false;
    auto *VT =
        dyn_cast<FixedVectorType>(Dst ? BC->getSrcTy() : BC->getDestTy());
    if (!VT || !VT->getElementType()->isIntegerTy() ||
        getLegalIntegerType(VT->getElementType()))
      reportFatalUsageError("DXIL legalization does not support this integer "
                            "bitcast");
    unsigned Width = VT->getScalarSizeInBits();
    Value *Op = GetOperand(BC->getOperand(0));
    if (Dst) {
      Value *Result = B.getIntN(Dst->getBitWidth(), 0);
      for (unsigned Idx = 0; Idx < VT->getNumElements(); ++Idx) {
        Value *Elt =
            B.CreateZExt(B.CreateExtractElement(Op, B.getInt32(Idx)), Dst);
        Result = B.CreateOr(Result, Idx ? B.CreateShl(Elt, Idx * Width) : Elt);
      }
      Replace(Result);
    } else {
      Value *Result = PoisonValue::get(VT);
      for (unsigned Idx = 0; Idx < VT->getNumElements(); ++Idx) {
        Value *Elt = B.CreateTrunc(Idx ? B.CreateLShr(Op, Idx * Width) : Op,
                                   VT->getElementType());
        Result = B.CreateInsertElement(Result, Elt, B.getInt32(Idx));
      }
      Replace(Result);
    }
    return true;
  }
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    if (!getLegalIntegerType(I.getType()))
      return false;
    IntegerExtension Ext = IntegerExtension::None;
    switch (BO->getOpcode()) {
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::AShr:
      Ext = IntegerExtension::Sign;
      break;
    case Instruction::UDiv:
    case Instruction::URem:
    case Instruction::LShr:
      Ext = IntegerExtension::Zero;
      break;
    default:
      break;
    }
    Value *LHS = GetOperand(BO->getOperand(0), Ext);
    Value *RHS = GetOperand(BO->getOperand(1),
                            BO->isShift() ? IntegerExtension::Zero : Ext);
    // Narrow no-wrap/exact flags do not describe the carrier operation.
    Replace(B.CreateBinOp(BO->getOpcode(), LHS, RHS));
    return true;
  }
  if (auto *SI = dyn_cast<SelectInst>(&I)) {
    if (!getLegalIntegerType(I.getType()))
      return false;
    Value *True = GetOperand(SI->getTrueValue());
    Value *False = GetOperand(SI->getFalseValue());
    Value *V =
        B.CreateSelect(SI->getCondition(), True, False, SI->getName(), SI);
    if (auto *NewSI = dyn_cast<SelectInst>(V);
        NewSI && !NewSI->getMetadata(LLVMContext::MD_prof))
      setExplicitlyUnknownBranchWeightsIfProfiled(*NewSI, DEBUG_TYPE);
    Replace(V);
    return true;
  }
  if (auto *CI = dyn_cast<ICmpInst>(&I)) {
    if (!getLegalIntegerType(CI->getOperand(0)->getType()))
      return false;
    auto Ext = CI->isSigned() ? IntegerExtension::Sign : IntegerExtension::Zero;
    Value *LHS = GetOperand(CI->getOperand(0), Ext);
    Value *RHS = GetOperand(CI->getOperand(1), Ext);
    Replace(B.CreateICmp(CI->getPredicate(), LHS, RHS));
    return true;
  }
  if (auto *CI = dyn_cast<CastInst>(&I)) {
    auto *Dst = getLegalIntegerType(CI->getDestTy());
    if (!Dst && !getLegalIntegerType(CI->getSrcTy()))
      return false;
    auto Ext = IntegerExtension::Zero;
    if (CI->getOpcode() == Instruction::Trunc)
      Ext = IntegerExtension::None;
    else if (CI->getOpcode() == Instruction::SExt ||
             CI->getOpcode() == Instruction::SIToFP)
      Ext = IntegerExtension::Sign;
    Value *V = GetOperand(CI->getOperand(0), Ext);
    Type *Ty = Dst ? Dst : CI->getDestTy();
    Replace(CI->isIntegerCast()
                ? B.CreateIntCast(V, Ty, Ext == IntegerExtension::Sign)
                : B.CreateCast(CI->getOpcode(), V, Ty));
    return true;
  }
  if (ReplacedValues.contains(&I))
    return false;
  if (getLegalIntegerType(I.getType()))
    reportFatalUsageError(Twine("DXIL legalization does not support integer "
                                "result for instruction '") +
                          I.getOpcodeName() + "'");
  for (Use &Op : I.operands()) {
    // Fixed intrinsic immediates are ABI operands, not integer SSA. These
    // DirectX intrinsics do not currently mark their immediates with immarg.
    // TODO: Use immarg once https://github.com/llvm/llvm-project/issues/217423
    // is addressed.
    if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isArgOperand(&Op)) {
      unsigned Arg = CB->getArgOperandNo(&Op);
      Intrinsic::ID ID = CB->getIntrinsicID();
      if (CB->paramHasAttr(Arg, Attribute::ImmArg) ||
          (isa<ConstantInt>(Op) &&
           ((ID == Intrinsic::dx_resource_updatecounter && Arg == 1) ||
            ((ID == Intrinsic::dx_load_input ||
              ID == Intrinsic::dx_store_output) &&
             Arg == 2))))
        continue;
    }
    if (getLegalIntegerType(Op->getType()))
      reportFatalUsageError(Twine("DXIL legalization does not support integer "
                                  "operand for instruction '") +
                            I.getOpcodeName() + "'");
  }
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

static bool legalizeI8Memory(Instruction &I,
                             SmallVectorImpl<Instruction *> &ToRemove,
                             DenseMap<Value *, Value *> &ReplacedValues) {

  IRBuilder<> Builder(&I);
  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    if (!Store->getValueOperand()->getType()->isIntegerTy(8))
      return false;
    Value *Pointer = ReplacedValues.lookup_or(Store->getPointerOperand(),
                                              Store->getPointerOperand());
    Type *Ty = nullptr;
    if (auto *AI = dyn_cast<AllocaInst>(Pointer))
      Ty = AI->getAllocatedType();
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(Pointer))
      Ty = GEP->getResultElementType();
    else if (auto *GV = dyn_cast<GlobalVariable>(Pointer))
      Ty = GV->getValueType();
    else if (isa<Argument>(Pointer))
      Ty = Builder.getInt32Ty();
    if (auto *AT = dyn_cast_or_null<ArrayType>(Ty))
      Ty = AT->getElementType();
    if (!Ty || !Ty->isIntegerTy() || getLegalIntegerType(Ty))
      reportFatalUsageError("DXIL legalization cannot determine the i8 store's "
                            "storage type");
    Value *V = Builder.CreateZExtOrTrunc(
        ReplacedValues.lookup_or(Store->getValueOperand(),
                                 Store->getValueOperand()),
        Builder.getInt32Ty());
    V = Builder.CreateAnd(V, 255);
    Value *NewStore =
        Builder.CreateStore(Builder.CreateZExtOrTrunc(V, Ty), Pointer);
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
      ElementType = GEP->getResultElementType();
    }
    if (ElementType->isArrayTy())
      ElementType = ElementType->getArrayElementType();
    LoadInst *NewLoad = Builder.CreateLoad(ElementType, Pointer);
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

    Value *NewGEP = GetElementPtrInst::Create(
        GEPType, PtrOperand, {Builder.getInt32(0), Builder.getInt32(Index)},
        GEP->getNoWrapFlags(), GEP->getName(), I.getIterator());

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
    Value *NewGEP = GetElementPtrInst::Create(
        GEPType, BasePtr, {Builder.getInt32(0), Builder.getInt32(Index)},
        GEP->getNoWrapFlags(), GEP->getName(), GEP->getIterator());
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

namespace {
class DXILLegalizationPipeline {

public:
  DXILLegalizationPipeline() { initializeLegalizationPipeline(); }

  bool runLegalizationPipeline(Function &F) {
    bool MadeChange = removeUnreachableBlocks(F);
    SmallVector<Instruction *> ToRemove;
    DenseMap<Value *, Value *> ReplacedValues;
    for (int Stage = 0; Stage < NumStages; ++Stage) {
      ToRemove.clear();
      ReplacedValues.clear();
      for (BasicBlock *BB : ReversePostOrderTraversal<Function *>(&F))
        for (Instruction &I : *BB)
          for (auto &LegalizationFn : LegalizationPipeline[Stage])
            if (LegalizationFn(I, ToRemove, ReplacedValues)) {
              MadeChange = true;
              break;
            }

      for (auto *Inst : reverse(ToRemove))
        Inst->eraseFromParent();
    }

    if (MadeChange)
      MadeChange |= removeUnreachableBlocks(F);
    return MadeChange;
  }

private:
  enum LegalizationStage { Stage1 = 0, Stage2 = 1, NumStages };

  using LegalizationFnTy =
      std::function<bool(Instruction &, SmallVectorImpl<Instruction *> &,
                         DenseMap<Value *, Value *> &)>;

  SmallVector<LegalizationFnTy> LegalizationPipeline[NumStages];

  void initializeLegalizationPipeline() {
    LegalizationPipeline[Stage1].push_back(upcastI8AllocasAndUses);
    LegalizationPipeline[Stage1].push_back(legalizeFreeze);
    LegalizationPipeline[Stage1].push_back(legalizeI8Memory);
    LegalizationPipeline[Stage1].push_back(legalizeInteger);
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
