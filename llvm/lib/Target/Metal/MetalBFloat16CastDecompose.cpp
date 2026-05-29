//===- MetalBFloat16CastDecompose.cpp - Decompose bf16 casts -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalBFloat16CastDecompose.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-bfloat16-cast-decompose"

static constexpr StringLiteral kAIRConvertPrefix("air.convert");

static bool bfloat16CastDecompose(Module &M) {
  bool Changed = false;
  Type *BF16 = Type::getBFloatTy(M.getContext());
  Type *F32 = Type::getFloatTy(M.getContext());
  Type *I32 = Type::getInt32Ty(M.getContext());
  Type *I16 = Type::getInt16Ty(M.getContext());

  // Phase 1: decompose sitofp/uitofp iN -> bfloat via f32 + bit manipulation.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if ((isa<SIToFPInst>(I) || isa<UIToFPInst>(I)) &&
            I->getType() == BF16) {
          IRBuilder<> B(I);
          Value *ToFloat =
              isa<SIToFPInst>(I)
                  ? B.CreateSIToFP(I->getOperand(0), F32, "to_f32")
                  : B.CreateUIToFP(I->getOperand(0), F32, "to_f32");
          // bf16 = upper 16 bits of f32.
          Value *AsInt = B.CreateBitCast(ToFloat, I32, "f32_bits");
          Value *Shifted = B.CreateLShr(AsInt, 16, "bf16_bits");
          Value *Narrow = B.CreateTrunc(Shifted, I16, "bf16_i16");
          Value *Trunc = B.CreateBitCast(Narrow, BF16, I->getName());
          I->replaceAllUsesWith(Trunc);
          I->eraseFromParent();
          Changed = true;
        }
      }
    }
  }

  // Phase 2: widen sitofp/uitofp i8/i16 -> float via i32.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        bool IsSigned = isa<SIToFPInst>(I);
        if (!IsSigned && !isa<UIToFPInst>(I))
          continue;
        if (I->getType() != F32)
          continue;
        Type *SrcTy = I->getOperand(0)->getType();
        unsigned Bits = SrcTy->getIntegerBitWidth();
        if (Bits >= 32)
          continue;

        IRBuilder<> B(I);
        Value *Wide = IsSigned
                          ? B.CreateSExt(I->getOperand(0), I32, "sext_i32")
                          : B.CreateZExt(I->getOperand(0), I32, "zext_i32");
        Value *FP = IsSigned ? B.CreateSIToFP(Wide, F32, I->getName())
                             : B.CreateUIToFP(Wide, F32, I->getName());
        I->replaceAllUsesWith(FP);
        I->eraseFromParent();
        Changed = true;
      }
    }
  }

  // Phase 3: fold sext/zext(trunc i32 to iN) to i32 into bit ops.
  SmallVector<Instruction *, 16> DeadInsts;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *Ext = dyn_cast<CastInst>(&I);
        if (!Ext)
          continue;
        bool IsSExt = isa<SExtInst>(Ext);
        if (!IsSExt && !isa<ZExtInst>(Ext))
          continue;
        if (Ext->getType() != I32)
          continue;
        auto *Trn = dyn_cast<TruncInst>(Ext->getOperand(0));
        if (!Trn || Trn->getOperand(0)->getType() != I32)
          continue;

        unsigned NarrowBits = Trn->getType()->getIntegerBitWidth();
        unsigned ShiftAmt = 32 - NarrowBits;
        IRBuilder<> B(Ext);
        Value *Src = Trn->getOperand(0);
        Value *Result;
        if (IsSExt) {
          Value *Shl = B.CreateShl(Src, ShiftAmt, "sext_shl");
          Result = B.CreateAShr(Shl, ShiftAmt, Ext->getName());
        } else {
          uint32_t Mask = (1u << NarrowBits) - 1;
          Result = B.CreateAnd(Src, Mask, Ext->getName());
        }
        Ext->replaceAllUsesWith(Result);
        DeadInsts.push_back(Ext);
        Changed = true;
      }
    }
  }
  for (Instruction *I : DeadInsts)
    I->eraseFromParent();

  // Phase 4: erase dead sub-32-bit trunc instructions (iteratively).
  bool Progress = true;
  while (Progress) {
    Progress = false;
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (auto It = BB.begin(); It != BB.end();) {
          auto *Trn = dyn_cast<TruncInst>(&*It++);
          if (!Trn)
            continue;
          unsigned Bits = Trn->getType()->getIntegerBitWidth();
          if (Bits >= 32)
            continue;
          if (Trn->use_empty()) {
            Trn->eraseFromParent();
            Progress = true;
          }
        }
      }
    }
  }

  // Phase 5: remove dead air.convert declarations.
  for (auto It = M.begin(); It != M.end();) {
    Function &Fn = *It++;
    if (Fn.isDeclaration() && Fn.use_empty() &&
        Fn.getName().starts_with(kAIRConvertPrefix))
      Fn.eraseFromParent();
  }

  return Changed;
}

PreservedAnalyses
MetalBFloat16CastDecomposePass::run(Module &M, ModuleAnalysisManager &AM) {
  return bfloat16CastDecompose(M) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

bool MetalBFloat16CastDecomposeLegacy::runOnModule(Module &M) {
  return bfloat16CastDecompose(M);
}

char MetalBFloat16CastDecomposeLegacy::ID = 0;

INITIALIZE_PASS(MetalBFloat16CastDecomposeLegacy, DEBUG_TYPE,
                "Metal BFloat16 Cast Decompose", false, false)

ModulePass *llvm::createMetalBFloat16CastDecomposeLegacyPass() {
  return new MetalBFloat16CastDecomposeLegacy();
}
