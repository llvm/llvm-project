//===- DXILLegalizePass.cpp - Legalizes llvm IR for DXIL ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "DXILLegalizePass.h"
#include "DirectX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <functional>

#define DEBUG_TYPE "dxil-legalize"

using namespace llvm;

static void legalizeFreeze(Instruction &I,
                           SmallVectorImpl<Instruction *> &ToRemove,
                           DenseMap<Value *, Value *>) {
  auto *FI = dyn_cast<FreezeInst>(&I);
  if (!FI)
    return;

  FI->replaceAllUsesWith(FI->getOperand(0));
  ToRemove.push_back(FI);
}

static void fixI8UseChain(Instruction &I,
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
      return;
    }
  }

  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    if (!Store->getValueOperand()->getType()->isIntegerTy(8))
      return;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewStore = Builder.CreateStore(NewOperands[0], NewOperands[1]);
    ReplacedValues[Store] = NewStore;
    ToRemove.push_back(Store);
    return;
  }

  if (auto *Load = dyn_cast<LoadInst>(&I)) {
    if (!I.getType()->isIntegerTy(8))
      return;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Type *ElementType = NewOperands[0]->getType();
    if (auto *AI = dyn_cast<AllocaInst>(NewOperands[0]))
      ElementType = AI->getAllocatedType();
    LoadInst *NewLoad = Builder.CreateLoad(ElementType, NewOperands[0]);
    ReplacedValues[Load] = NewLoad;
    ToRemove.push_back(Load);
    return;
  }

  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    if (!I.getType()->isIntegerTy(8))
      return;
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
    return;
  }

  if (auto *Sel = dyn_cast<SelectInst>(&I)) {
    if (!I.getType()->isIntegerTy(8))
      return;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewInst = Builder.CreateSelect(Sel->getCondition(), NewOperands[1],
                                          NewOperands[2]);
    ReplacedValues[Sel] = NewInst;
    ToRemove.push_back(Sel);
    return;
  }

  if (auto *Cmp = dyn_cast<CmpInst>(&I)) {
    if (!Cmp->getOperand(0)->getType()->isIntegerTy(8))
      return;
    SmallVector<Value *> NewOperands;
    ProcessOperands(NewOperands);
    Value *NewInst =
        Builder.CreateCmp(Cmp->getPredicate(), NewOperands[0], NewOperands[1]);
    Cmp->replaceAllUsesWith(NewInst);
    ReplacedValues[Cmp] = NewInst;
    ToRemove.push_back(Cmp);
    return;
  }

  if (auto *Cast = dyn_cast<CastInst>(&I)) {
    if (!Cast->getSrcTy()->isIntegerTy(8))
      return;

    ToRemove.push_back(Cast);
    auto *Replacement = ReplacedValues[Cast->getOperand(0)];
    if (Cast->getType() == Replacement->getType()) {
      Cast->replaceAllUsesWith(Replacement);
      return;
    }
    Value *AdjustedCast = nullptr;
    if (Cast->getOpcode() == Instruction::ZExt)
      AdjustedCast = Builder.CreateZExtOrTrunc(Replacement, Cast->getType());
    if (Cast->getOpcode() == Instruction::SExt)
      AdjustedCast = Builder.CreateSExtOrTrunc(Replacement, Cast->getType());

    if (AdjustedCast)
      Cast->replaceAllUsesWith(AdjustedCast);
  }
}

static void upcastI8AllocasAndUses(Instruction &I,
                                   SmallVectorImpl<Instruction *> &ToRemove,
                                   DenseMap<Value *, Value *> &ReplacedValues) {
  auto *AI = dyn_cast<AllocaInst>(&I);
  if (!AI || !AI->getAllocatedType()->isIntegerTy(8))
    return;

  Type *SmallestType = nullptr;

  // Gather all cast targets
  for (User *U : AI->users()) {
    auto *Load = dyn_cast<LoadInst>(U);
    if (!Load)
      continue;
    for (User *LU : Load->users()) {
      auto *Cast = dyn_cast<CastInst>(LU);
      if (!Cast)
        continue;
      Type *Ty = Cast->getType();
      if (!SmallestType ||
          Ty->getPrimitiveSizeInBits() < SmallestType->getPrimitiveSizeInBits())
        SmallestType = Ty;
    }
  }

  if (!SmallestType)
    return; // no valid casts found

  // Replace alloca
  IRBuilder<> Builder(AI);
  auto *NewAlloca = Builder.CreateAlloca(SmallestType);
  ReplacedValues[AI] = NewAlloca;
  ToRemove.push_back(AI);
}

static void
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
    }
  }
}

namespace {
class DXILLegalizationPipeline {

public:
  DXILLegalizationPipeline() { initializeLegalizationPipeline(); }

  bool runLegalizationPipeline(Function &F) {
    SmallVector<Instruction *> ToRemove;
    DenseMap<Value *, Value *> ReplacedValues;
    for (auto &I : instructions(F)) {
      for (auto &LegalizationFn : LegalizationPipeline)
        LegalizationFn(I, ToRemove, ReplacedValues);
    }

    for (auto *Inst : reverse(ToRemove))
      Inst->eraseFromParent();

    return !ToRemove.empty();
  }

private:
  SmallVector<
      std::function<void(Instruction &, SmallVectorImpl<Instruction *> &,
                         DenseMap<Value *, Value *> &)>>
      LegalizationPipeline;

  void initializeLegalizationPipeline() {
    LegalizationPipeline.push_back(upcastI8AllocasAndUses);
    LegalizationPipeline.push_back(fixI8UseChain);
    LegalizationPipeline.push_back(downcastI64toI32InsertExtractElements);
    LegalizationPipeline.push_back(legalizeFreeze);
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
