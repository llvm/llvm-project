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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <functional>

#define DEBUG_TYPE "dxil-legalize"

using namespace llvm;

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

static bool fixI8UseChain(Instruction &I,
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
  if (auto *Trunc = dyn_cast<TruncInst>(&I)) {
    if (Trunc->getDestTy()->isIntegerTy(8)) {
      ReplacedValues[Trunc] = Trunc->getOperand(0);
      ToRemove.push_back(Trunc);
      return true;
    }
  }

  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    if (!Store->getValueOperand()->getType()->isIntegerTy(8))
      return false;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewStore = Builder.CreateStore(NewOperands[0], NewOperands[1]);
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

  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    if (!I.getType()->isIntegerTy(8))
      return false;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewInst =
        Builder.CreateBinOp(BO->getOpcode(), NewOperands[0], NewOperands[1]);
    if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I)) {
      auto *NewBO = dyn_cast<BinaryOperator>(NewInst);
      if (NewBO && OBO->hasNoSignedWrap())
        NewBO->setHasNoSignedWrap();
      if (NewBO && OBO->hasNoUnsignedWrap())
        NewBO->setHasNoUnsignedWrap();
    }
    ReplacedValues[BO] = NewInst;
    ToRemove.push_back(BO);
    return true;
  }

  if (auto *Sel = dyn_cast<SelectInst>(&I)) {
    if (!I.getType()->isIntegerTy(8))
      return false;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewInst = Builder.CreateSelect(Sel->getCondition(), NewOperands[1],
                                          NewOperands[2]);
    ReplacedValues[Sel] = NewInst;
    ToRemove.push_back(Sel);
    return true;
  }

  if (auto *Cmp = dyn_cast<CmpInst>(&I)) {
    if (!Cmp->getOperand(0)->getType()->isIntegerTy(8))
      return false;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewInst =
        Builder.CreateCmp(Cmp->getPredicate(), NewOperands[0], NewOperands[1]);
    Cmp->replaceAllUsesWith(NewInst);
    ReplacedValues[Cmp] = NewInst;
    ToRemove.push_back(Cmp);
    return true;
  }

  if (auto *Cast = dyn_cast<CastInst>(&I)) {
    if (!Cast->getSrcTy()->isIntegerTy(8))
      return false;

    ToRemove.push_back(Cast);
    auto *Replacement = ReplacedValues[Cast->getOperand(0)];
    if (Cast->getType() == Replacement->getType()) {
      Cast->replaceAllUsesWith(Replacement);
      return true;
    }

    Value *AdjustedCast = nullptr;
    if (Cast->getOpcode() == Instruction::ZExt)
      AdjustedCast = Builder.CreateZExtOrTrunc(Replacement, Cast->getType());
    if (Cast->getOpcode() == Instruction::SExt)
      AdjustedCast = Builder.CreateSExtOrTrunc(Replacement, Cast->getType());

    if (AdjustedCast)
      Cast->replaceAllUsesWith(AdjustedCast);
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
      Type *Ty = nullptr;
      if (CastInst *Cast = dyn_cast<CastInst>(LU))
        Ty = Cast->getType();
      else if (CallInst *CI = dyn_cast<CallInst>(LU)) {
        assert(CI->getIntrinsicID() != Intrinsic::memset &&
               "memset should have been eliminated in an earlier pass");
      }

      if (!Ty)
        continue;

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
legalizeGetHighLowi64Bytes(Instruction &I,
                           SmallVectorImpl<Instruction *> &ToRemove,
                           DenseMap<Value *, Value *> &ReplacedValues) {
  if (auto *BitCast = dyn_cast<BitCastInst>(&I)) {
    if (BitCast->getDestTy() ==
            FixedVectorType::get(Type::getInt32Ty(I.getContext()), 2) &&
        BitCast->getSrcTy()->isIntegerTy(64)) {
      ToRemove.push_back(BitCast);
      ReplacedValues[BitCast] = BitCast->getOperand(0);
      return true;
    }
  }

  if (auto *Extract = dyn_cast<ExtractElementInst>(&I)) {
    if (!dyn_cast<BitCastInst>(Extract->getVectorOperand()))
      return false;
    auto *VecTy = dyn_cast<FixedVectorType>(Extract->getVectorOperandType());
    if (VecTy && VecTy->getElementType()->isIntegerTy(32) &&
        VecTy->getNumElements() == 2) {
      if (auto *Index = dyn_cast<ConstantInt>(Extract->getIndexOperand())) {
        unsigned Idx = Index->getZExtValue();
        IRBuilder<> Builder(&I);

        auto *Replacement = ReplacedValues[Extract->getVectorOperand()];
        assert(Replacement && "The BitCast replacement should have been set "
                              "before working on ExtractElementInst.");
        if (Idx == 0) {
          Value *LowBytes = Builder.CreateTrunc(
              Replacement, Type::getInt32Ty(I.getContext()));
          ReplacedValues[Extract] = LowBytes;
        } else {
          assert(Idx == 1);
          Value *LogicalShiftRight = Builder.CreateLShr(
              Replacement,
              ConstantInt::get(
                  Replacement->getType(),
                  APInt(Replacement->getType()->getIntegerBitWidth(), 32)));
          Value *HighBytes = Builder.CreateTrunc(
              LogicalShiftRight, Type::getInt32Ty(I.getContext()));
          ReplacedValues[Extract] = HighBytes;
        }
        ToRemove.push_back(Extract);
        Extract->replaceAllUsesWith(ReplacedValues[Extract]);
        return true;
      }
    }
  }
  return false;
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
    LegalizationPipeline[Stage1].push_back(fixI8UseChain);
    LegalizationPipeline[Stage1].push_back(legalizeGetHighLowi64Bytes);
    LegalizationPipeline[Stage1].push_back(legalizeFreeze);
    LegalizationPipeline[Stage1].push_back(updateFnegToFsub);
    // Note: legalizeGetHighLowi64Bytes and
    // downcastI64toI32InsertExtractElements both modify extractelement, so they
    // must run staggered stages. legalizeGetHighLowi64Bytes runs first b\c it
    // removes extractelements, reducing the number that
    // downcastI64toI32InsertExtractElements needs to handle.
    LegalizationPipeline[Stage2].push_back(
        downcastI64toI32InsertExtractElements);
    LegalizationPipeline[Stage2].push_back(legalizeScalarLoadStoreOnArrays);
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
