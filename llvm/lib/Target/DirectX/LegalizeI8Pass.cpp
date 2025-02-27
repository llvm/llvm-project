//===- LegalizeI8Pass.cpp - A pass that reverts i8 conversions-*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//===---------------------------------------------------------------------===//
///
/// \file This file contains a pass to remove i8 truncations.
///
//===----------------------------------------------------------------------===//
#include "DirectX.h"
#include "LegalizeI8Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <map>
#include <stack>

#define DEBUG_TYPE "dxil-legalize-i8"

using namespace llvm;
namespace {

class LegalizeI8Legacy : public FunctionPass {

public:
  bool runOnFunction(Function &F) override;
  LegalizeI8Legacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
};
} // namespace

static bool fixI8TruncUseChain(Function &F) {
    std::stack<Instruction*> ToRemove;
    std::map<Value*, Value*> ReplacedValues;
    
    for (auto &I : instructions(F)) {
        if (auto *Trunc = dyn_cast<TruncInst>(&I)) {
            if (Trunc->getDestTy()->isIntegerTy(8)) {
                ReplacedValues[Trunc] = Trunc->getOperand(0);
                ToRemove.push(Trunc);
            }
        } else if (I.getType()->isIntegerTy(8)) {
            IRBuilder<> Builder(&I);
            
            std::vector<Value*> NewOperands;
            Type* InstrType = nullptr;
            for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
                Value *Op = I.getOperand(OpIdx);
                if (ReplacedValues.count(Op)) {
                    InstrType = ReplacedValues[Op]->getType();
                    NewOperands.push_back(ReplacedValues[Op]);
                }
                else if (auto *Imm = dyn_cast<ConstantInt>(Op)) {
                    APInt Value = Imm->getValue();
                    unsigned NewBitWidth = InstrType->getIntegerBitWidth();
                    // Note: options here are sext or sextOrTrunc. 
                    // Since i8 isn't suppport we assume new values
                    // will always have a higher bitness.
                    APInt NewValue = Value.sext(NewBitWidth);
                    NewOperands.push_back(ConstantInt::get(InstrType, NewValue));
                } else {
                    assert(!Op->getType()->isIntegerTy(8));
                    NewOperands.push_back(Op);
                }
                
            }
            
            Value *NewInst = nullptr;
            if (auto *BO = dyn_cast<BinaryOperator>(&I))
                NewInst = Builder.CreateBinOp(BO->getOpcode(), NewOperands[0], NewOperands[1]);
            else if (auto *Cmp = dyn_cast<CmpInst>(&I))
                NewInst = Builder.CreateCmp(Cmp->getPredicate(), NewOperands[0], NewOperands[1]);
            else if (auto *Cast = dyn_cast<CastInst>(&I))
                NewInst = Builder.CreateCast(Cast->getOpcode(), NewOperands[0], Cast->getDestTy());
            else if (auto *UnaryOp = dyn_cast<UnaryOperator>(&I))
                NewInst = Builder.CreateUnOp(UnaryOp->getOpcode(), NewOperands[0]);
                
            if (NewInst) {
                ReplacedValues[&I] = NewInst;
                ToRemove.push(&I);
            }
        } else if (auto *Sext = dyn_cast<SExtInst>(&I)) {
            if (Sext->getSrcTy()->isIntegerTy(8)) {
                ToRemove.push(Sext);
                Sext->replaceAllUsesWith(ReplacedValues[Sext->getOperand(0)]);
            }
        }
    }
    
    while (!ToRemove.empty()) {
        Instruction *I = ToRemove.top();
        I->eraseFromParent();
        ToRemove.pop();
    }
    
    return true;
}

PreservedAnalyses LegalizeI8Pass::run(Function &F, FunctionAnalysisManager &FAM) {
    bool MadeChanges = fixI8TruncUseChain(F);
    if (!MadeChanges)
      return PreservedAnalyses::all();
    PreservedAnalyses PA;
    return PA;
  }
  
  bool LegalizeI8Legacy::runOnFunction(Function &F) {
    return fixI8TruncUseChain(F);
  }
  
  char LegalizeI8Legacy::ID = 0;
  
  INITIALIZE_PASS_BEGIN(LegalizeI8Legacy, DEBUG_TYPE,
                        "DXIL I8 Legalizer", false, false)
  INITIALIZE_PASS_END(LegalizeI8Legacy, DEBUG_TYPE, "DXIL I8 Legalizer",
                      false, false)
  
FunctionPass *llvm::createLegalizeI8LegacyPass() {
    return new LegalizeI8Legacy();
  }