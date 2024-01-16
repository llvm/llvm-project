//===- DXILStrengthReduce.cpp - Prepare LLVM Module for DXIL encoding------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains strength reduction pass to convert a modern LLVM
/// module into an LLVM module with LLVM intrinsics amenable for lowering to
/// LLVM 3.7-based DirectX Intermediate Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DirectX.h"
#include "DirectXIRPasses/PointerTypeAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

#define DEBUG_TYPE "dxil-strength-reduce"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class DXILStrengthReduce : public ModulePass {

public:
  bool runOnModule(Module &M) override {
    for (auto &F : make_early_inc_range(M.functions())) {
      IRBuilder<> IRB(M.getContext());
      // Reduce strength of LLVM intrinsics
      // Flag to indicate if the intrinsic has been replaced. This ensures any
      // other functions with no uses are not deleted in this pass.
      bool IntrinsicReplaced = false;
      if (F.isDeclaration()) {
        Intrinsic::ID IntrinsicId = F.getIntrinsicID();
        // Convert
        //    %ret = call i32 @llvm.abs.i32(i32 %arg, i1 false)
        // to
        //    %NegArg = sub 0, %arg
        //    %ret = call i32 @llvm.imax(NegArg, %arg)
        if (IntrinsicId == Intrinsic::abs) {
          // Get to uses of the intrinsic
          for (User *U : make_early_inc_range(F.users())) {
            auto *IntrinsicCall = dyn_cast<CallInst>(U);
            if (!IntrinsicCall)
              continue;
            Value *Input = IntrinsicCall->getOperand(0);
            Value *Poison = IntrinsicCall->getOperand(1);

            // Get Poison argument value
            const ConstantInt *CI = dyn_cast<ConstantInt>(Poison);
            assert(
                CI != nullptr &&
                "Expect second argument of abs intrinsic to be constant type.");
            assert(CI->getType()->isIntegerTy(1) &&
                   "Expect second argument of abs intrinsic to be constant int "
                   "type.");
            bool isPoison = CI->getZExtValue();

            // Construct the Instruction sub(0, Input)
            Value *ZeroValue = ConstantInt::get(Input->getType(), 0);
            IRB.SetInsertPoint(IntrinsicCall);
            auto *SubInst =
                IRB.CreateSub(ZeroValue, Input, "NegArg", isPoison, isPoison);

            // Replace
            //   call i32 @llvm.abs.i32(i32 %arg, i1 false)
            // with
            //   call i32 @llvm.max.i32(i32 %NegArg, %arg)
            // Generate Intrinsic function call
            Value *IntrinsicCallArgs[] = {Input, SubInst};
            auto *IMaxCall = IRB.CreateIntrinsic(
                Input->getType(), Intrinsic::smax,
                ArrayRef<Value *>(IntrinsicCallArgs), nullptr, "IMax");
            // Retain the tail call and attributes of the intrinsic being
            // replaced.
            IMaxCall->setTailCall(IntrinsicCall->isTailCall());
            IMaxCall->setAttributes(IntrinsicCall->getAttributes());
            IntrinsicCall->replaceAllUsesWith(IMaxCall);
            IntrinsicCall->eraseFromParent();
            IntrinsicReplaced = true;
          }
        }
        if (F.user_empty() && IntrinsicReplaced)
          F.eraseFromParent();

      } else {
        // Reduce strength of instructions
        for (auto &BB : F) {
          IRBuilder<> Builder(&BB);
          for (auto &I : make_early_inc_range(BB)) {
            // Rewrite
            //    %nval = fneg double %val
            // to
            //    %nval = fsub double -0.000000e+00, %val

            if (I.getOpcode() == Instruction::FNeg) {
              Builder.SetInsertPoint(&I);
              Value *In = I.getOperand(0);
              Value *Zero = ConstantFP::get(In->getType(), -0.0);
              I.replaceAllUsesWith(Builder.CreateFSub(Zero, In));
              I.eraseFromParent();
            }
          }
        }
      }
    }
    return true;
  }

  DXILStrengthReduce() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILStrengthReduce::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILStrengthReduce, DEBUG_TYPE, "DXIL Strength Reduce",
                      false, false)
INITIALIZE_PASS_END(DXILStrengthReduce, DEBUG_TYPE, "DXIL Strength Reduce",
                    false, false)

ModulePass *llvm::createDXILStrengthReducePass() {
  return new DXILStrengthReduce();
}
