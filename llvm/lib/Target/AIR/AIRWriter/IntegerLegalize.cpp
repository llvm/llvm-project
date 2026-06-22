//===- IntegerLegalize.cpp - AGX-JIT integer/intrinsic legalization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-serialization legalizations that bring integer arithmetic, freeze, and a
// handful of intrinsics into the subset the AGX JIT and AIR v1 bitcode accept.
// Each enforces a single AIR/AGX limitation; see the per-function notes.
//
//===----------------------------------------------------------------------===//

#include "IntegerLegalize.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace llvm {
namespace metal {

void lowerCmpIntrinsics(Module &M) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Intrinsic::ID ID = CI->getIntrinsicID();
          if (ID == Intrinsic::scmp || ID == Intrinsic::ucmp)
            Calls.push_back(CI);
        }
    for (CallInst *CI : Calls) {
      bool Signed = CI->getIntrinsicID() == Intrinsic::scmp;
      IRBuilder<> B(CI);
      Value *A = CI->getArgOperand(0);
      Value *Bv = CI->getArgOperand(1);
      Value *Gt = Signed ? B.CreateICmpSGT(A, Bv) : B.CreateICmpUGT(A, Bv);
      Value *Lt = Signed ? B.CreateICmpSLT(A, Bv) : B.CreateICmpULT(A, Bv);
      Type *RetTy = CI->getType();
      Value *Res =
          B.CreateSub(B.CreateZExt(Gt, RetTy), B.CreateZExt(Lt, RetTy));
      CI->replaceAllUsesWith(Res);
      CI->eraseFromParent();
    }
  }

  for (auto It = M.begin(); It != M.end();) {
    Function &F = *It++;
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::scmp || ID == Intrinsic::ucmp) && F.use_empty())
      F.eraseFromParent();
  }
}

void expandWideIntegers(Module &M) {
  auto isWide = [](Type *T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() > 64 &&
           T->getIntegerBitWidth() <= 128;
  };
  Type *I64 = Type::getInt64Ty(M.getContext());
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    DenseMap<Value *, std::pair<Value *, Value *>> Limbs;
    SmallVector<Instruction *, 8> Wide;
    for (auto &BB : F)
      for (auto &I : BB)
        if (isWide(I.getType()) || llvm::any_of(I.operands(), [&](Value *Op) {
              return isWide(Op->getType());
            }))
          Wide.push_back(&I);
    if (Wide.empty())
      continue;
    auto umulh = [&](IRBuilder<> &B, Value *A, Value *Bv) -> Value * {
      Value *Mask = ConstantInt::get(I64, 0xffffffffull);
      Value *AL = B.CreateAnd(A, Mask), *AH = B.CreateLShr(A, 32);
      Value *BL = B.CreateAnd(Bv, Mask), *BH = B.CreateLShr(Bv, 32);
      Value *LL = B.CreateMul(AL, BL);
      Value *LH = B.CreateMul(AL, BH);
      Value *HL = B.CreateMul(AH, BL);
      Value *HH = B.CreateMul(AH, BH);
      Value *Mid =
          B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                      B.CreateAnd(HL, Mask));
      return B.CreateAdd(B.CreateAdd(HH, B.CreateAdd(B.CreateLShr(LH, 32),
                                                     B.CreateLShr(HL, 32))),
                         B.CreateLShr(Mid, 32));
    };
    for (Instruction *I : Wide) {
      IRBuilder<> B(I);
      auto limbsOf = [&](Value *V) -> std::pair<Value *, Value *> {
        auto It = Limbs.find(V);
        if (It != Limbs.end())
          return It->second;
        if (auto *C = dyn_cast<ConstantInt>(V)) {
          APInt A = C->getValue();
          return {ConstantInt::get(I64, A.trunc(64)),
                  ConstantInt::get(I64, A.lshr(64).trunc(64))};
        }
        report_fatal_error("AIRWriter: unmapped wide integer operand");
      };
      if (auto *ZE = dyn_cast<ZExtInst>(I)) {
        Limbs[I] = {B.CreateZExtOrTrunc(ZE->getOperand(0), I64),
                    ConstantInt::get(I64, 0)};
      } else if (auto *SE2 = dyn_cast<SExtInst>(I)) {
        Value *Lo = B.CreateSExtOrTrunc(SE2->getOperand(0), I64);
        Limbs[I] = {Lo, B.CreateAShr(Lo, 63)};
      } else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        auto [L1, H1] = limbsOf(BO->getOperand(0));
        switch (BO->getOpcode()) {
        case Instruction::Mul: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateMul(L1, L2);
          Value *Hi =
              B.CreateAdd(umulh(B, L1, L2), B.CreateAdd(B.CreateMul(L1, H2),
                                                        B.CreateMul(H1, L2)));
          Limbs[I] = {Lo, Hi};
          break;
        }
        case Instruction::Add: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateAdd(L1, L2);
          Value *Carry = B.CreateZExt(B.CreateICmpULT(Lo, L1), I64);
          Limbs[I] = {Lo, B.CreateAdd(B.CreateAdd(H1, H2), Carry)};
          break;
        }
        case Instruction::Sub: {
          auto [L2, H2] = limbsOf(BO->getOperand(1));
          Value *Lo = B.CreateSub(L1, L2);
          Value *Borrow = B.CreateZExt(B.CreateICmpULT(L1, L2), I64);
          Limbs[I] = {Lo, B.CreateSub(B.CreateSub(H1, H2), Borrow)};
          break;
        }
        case Instruction::LShr: {
          auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
          if (!CI)
            report_fatal_error("AIRWriter: wide lshr by non-constant");
          uint64_t Sh = CI->getZExtValue();
          if (Sh == 0) {
            Limbs[I] = {L1, H1};
          } else if (Sh < 64) {
            Limbs[I] = {
                B.CreateOr(B.CreateLShr(L1, Sh), B.CreateShl(H1, 64 - Sh)),
                B.CreateLShr(H1, Sh)};
          } else {
            Limbs[I] = {B.CreateLShr(H1, Sh - 64), ConstantInt::get(I64, 0)};
          }
          break;
        }
        default:
          report_fatal_error(Twine("AIRWriter: unhandled wide integer op '") +
                             BO->getOpcodeName() + "'");
        }
      } else if (auto *TR = dyn_cast<TruncInst>(I)) {
        auto [Lo, Hi] = limbsOf(TR->getOperand(0));
        (void)Hi;
        Value *R = B.CreateZExtOrTrunc(Lo, TR->getType());
        TR->replaceAllUsesWith(R);
      } else {
        report_fatal_error(Twine("AIRWriter: unhandled wide integer user '") +
                           I->getOpcodeName() + "'");
      }
    }
    for (auto It = Wide.rbegin(); It != Wide.rend(); ++It)
      (*It)->eraseFromParent();
  }
}

void stripLifetimeIntrinsics(Module &M) {
  SmallVector<Instruction *, 16> Dead;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (auto *Callee = CI->getCalledFunction()) {
            Intrinsic::ID ID = Callee->getIntrinsicID();
            if (ID == Intrinsic::lifetime_start ||
                ID == Intrinsic::lifetime_end)
              Dead.push_back(CI);
          }
  for (auto *I : Dead)
    I->eraseFromParent();

  SmallVector<Function *, 4> DeadDecls;
  for (auto &F : M) {
    Intrinsic::ID ID = F.getIntrinsicID();
    if ((ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end) &&
        F.use_empty())
      DeadDecls.push_back(&F);
  }
  for (auto *F : DeadDecls)
    F->eraseFromParent();
}

void canonicalizeNNegZExt(Module &M) {
  SmallVector<ZExtInst *, 16> Zexts;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *ZE = dyn_cast<ZExtInst>(&I))
          if (ZE->hasNonNeg())
            Zexts.push_back(ZE);
  for (auto *ZE : Zexts) {
    auto *SE = CastInst::Create(Instruction::SExt, ZE->getOperand(0),
                                ZE->getType(), "", ZE->getIterator());
    SE->takeName(ZE);
    ZE->replaceAllUsesWith(SE);
    ZE->eraseFromParent();
  }
}

void lowerFreezeInsts(Module &M) {
  SmallVector<FreezeInst *, 8> Frozen;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *FI = dyn_cast<FreezeInst>(&I))
          Frozen.push_back(FI);
  for (auto *FI : Frozen) {
    FI->replaceAllUsesWith(FI->getOperand(0));
    FI->eraseFromParent();
  }
}

void stripDisjointFlags(Module &M) {
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *BO = dyn_cast<PossiblyDisjointInst>(&I))
          if (BO->isDisjoint())
            BO->setIsDisjoint(false);
}

}
}
