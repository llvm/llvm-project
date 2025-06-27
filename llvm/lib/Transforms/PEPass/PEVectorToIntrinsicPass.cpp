/* --- PEVectorToIntrinsicPass.cpp --- */

/* ------------------------------------------
author: 
date: 6/23/2025
------------------------------------------ */
#include "llvm/Transforms/PEPass/PEVectorToIntrinsicPass.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
PreservedAnalyses PEVectorToIntrinsicPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  // 添加日志输出以确认 pass 被调用
  errs() << "PEVectorToIntrinsicPass is running on function: " << F.getName() << "\n";

  bool Changed = false;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();

  // 声明自定义 intrinsic
  Type *VecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 8);
  FunctionType *FTy = FunctionType::get(VecTy, {VecTy, VecTy}, false);
  FunctionCallee MyAdd = M->getOrInsertFunction("llvm.pe.v8i32.add", FTy);

  // 收集要替换的指令，避免遍历时修改IR导致迭代器失效
  SmallVector<Instruction *, 8> ToReplace;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
        if (BinOp->getOpcode() == Instruction::Add &&
            BinOp->getType()->isVectorTy() &&
            cast<VectorType>(BinOp->getType())
                    ->getElementCount()
                    .getFixedValue() == 8 &&
            BinOp->getType()->getScalarSizeInBits() == 32) {
          ToReplace.push_back(BinOp);
        }
      }
    }
  }

  // 替换为 intrinsic 调用
  for (Instruction *I : ToReplace) {
    IRBuilder<> Builder(I);
    Value *LHS = I->getOperand(0);
    Value *RHS = I->getOperand(1);
    CallInst *Call = Builder.CreateCall(MyAdd, {LHS, RHS}); // 创建调用指令
    I->replaceAllUsesWith(Call); // 替换所有使用该指令的地方
    I->eraseFromParent();        // 删除原指令
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
