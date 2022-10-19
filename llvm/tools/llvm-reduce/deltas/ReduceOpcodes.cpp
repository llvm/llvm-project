//===- ReduceOpcodes.cpp - Specialized Delta Pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to replace instructions that are likely to codegen to simpler or smaller
// sequences. This is a fuzzy and target specific concept.
//
//===----------------------------------------------------------------------===//

#include "ReduceOpcodes.h"
#include "Delta.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

static Value *replaceIntrinsic(Module &M, IntrinsicInst *II,
                               Intrinsic::ID NewIID,
                               ArrayRef<Type *> Tys = None) {
  Function *NewFunc = Intrinsic::getDeclaration(&M, NewIID, Tys);
  II->setCalledFunction(NewFunc);
  return II;
}

static Value *reduceInstruction(Module &M, Instruction &I) {
  IRBuilder<> B(&I);
  switch (I.getOpcode()) {
  case Instruction::FDiv:
  case Instruction::FRem:
    // Divisions tends to codegen into a long sequence or a library call.
    return B.CreateFMul(I.getOperand(0), I.getOperand(1));
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    // Divisions tends to codegen into a long sequence or a library call.
    return B.CreateMul(I.getOperand(0), I.getOperand(1));
  case Instruction::Add:
  case Instruction::Sub: {
    // Add/sub are more likely codegen to instructions with carry out side
    // effects.
    return B.CreateOr(I.getOperand(0), I.getOperand(1));
  }
  case Instruction::Call: {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      return nullptr;

    switch (II->getIntrinsicID()) {
    case Intrinsic::sqrt:
      return B.CreateFMul(II->getArgOperand(0),
                          ConstantFP::get(I.getType(), 2.0));
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::amdgcn_fmul_legacy:
      return B.CreateFMul(II->getArgOperand(0), II->getArgOperand(1));
    case Intrinsic::amdgcn_workitem_id_y:
    case Intrinsic::amdgcn_workitem_id_z:
      return replaceIntrinsic(M, II, Intrinsic::amdgcn_workitem_id_x);
    case Intrinsic::amdgcn_workgroup_id_y:
    case Intrinsic::amdgcn_workgroup_id_z:
      return replaceIntrinsic(M, II, Intrinsic::amdgcn_workgroup_id_x);
    case Intrinsic::amdgcn_div_fixup:
    case Intrinsic::amdgcn_fma_legacy:
      return replaceIntrinsic(M, II, Intrinsic::fma, {II->getType()});
    default:
      return nullptr;
    }

    return nullptr;
  }
  default:
    return nullptr;
  }

  return nullptr;
}

static void replaceOpcodesInModule(Oracle &O, Module &Mod) {
  for (Function &F : Mod) {
    for (BasicBlock &BB : F)
      for (Instruction &I : make_early_inc_range(BB)) {

        Instruction *Replacement =
            dyn_cast_or_null<Instruction>(reduceInstruction(Mod, I));
        if (Replacement && Replacement != &I) {
          if (O.shouldKeep())
            continue;

          if (isa<FPMathOperator>(Replacement))
            Replacement->copyFastMathFlags(&I);

          Replacement->copyIRFlags(&I);
          Replacement->copyMetadata(I);
          Replacement->takeName(&I);
          I.replaceAllUsesWith(Replacement);
          I.eraseFromParent();
        }
      }
  }
}

void llvm::reduceOpcodesDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, replaceOpcodesInModule, "Reducing Opcodes");
}
