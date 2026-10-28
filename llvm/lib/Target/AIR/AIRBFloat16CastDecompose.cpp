//===- AIRBFloat16CastDecompose.cpp - Decompose bf16 casts -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRBFloat16CastDecompose.h"
#include "AIR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-bfloat16-cast-decompose"

static constexpr StringLiteral kAIRConvertPrefix("air.convert");

static bool bfloat16CastDecompose(Module &M) {
  bool Changed = false;
  Type *BF16 = Type::getBFloatTy(M.getContext());
  Type *F32 = Type::getFloatTy(M.getContext());
  Type *I32 = Type::getInt32Ty(M.getContext());
  Type *I16 = Type::getInt16Ty(M.getContext());

  // Widen a scalar type to match the vector shape of Ref (no-op for scalars).
  auto matchShape = [](Type *Elem, Type *Ref) -> Type * {
    if (auto *VT = dyn_cast<FixedVectorType>(Ref))
      return FixedVectorType::get(Elem, VT->getNumElements());
    return Elem;
  };

  // Phase -1: fptosi/fptoui from half or bfloat go through f32, which holds
  // every half and bfloat value exactly. Metal reads a bfloat operand of
  // fptosi as half, and its compiler can hang on half -> i64 (a
  // PTX-formalization cvt.rmi.s64.f16 request never finished compiling). The
  // fpext bfloat this creates is decomposed by phase 0.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if (!isa<FPToSIInst>(I) && !isa<FPToUIInst>(I))
          continue;
        Value *Src = I->getOperand(0);
        Type *SrcElem = Src->getType()->getScalarType();
        if (!SrcElem->isHalfTy() && !SrcElem->isBFloatTy())
          continue;
        IRBuilder<> B(I);
        Value *Wide = B.CreateFPExt(Src, matchShape(F32, Src->getType()), "to_f32");
        Value *R = isa<FPToSIInst>(I) ? B.CreateFPToSI(Wide, I->getType())
                                      : B.CreateFPToUI(Wide, I->getType());
        R->takeName(I);
        I->replaceAllUsesWith(R);
        I->eraseFromParent();
        Changed = true;
      }
    }
  }

  // Phase 0: decompose fptrunc f32 -> bfloat (round-to-nearest-even) and
  // fpext bfloat -> f32, both as integer bit manipulation — the native casts
  // are miscompiled by the GPU JIT. Scalar and vector.
  //
  // NB: no NaN-quieting branch. A `fcmp uno` + `select` to force the QNaN bit
  // is undefined under the module's `air.compile.fast_math_enable` (which
  // implies `nnan`); the AGX JIT miscompiles the resulting select and its
  // surrounding bitcast chain — most visibly inside the bf16 atomic-add CAS
  // loop, which corrupts adjacent lanes into NaN. Plain RTNE truncation is
  // the correct lowering for a no-NaN target.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if (isa<FPTruncInst>(I) && I->getType()->getScalarType() == BF16 &&
            I->getOperand(0)->getType()->getScalarType() == F32) {
          IRBuilder<> B(I);
          Value *Src = I->getOperand(0);
          Type *I32T = matchShape(I32, I->getType());
          Type *I16T = matchShape(I16, I->getType());
          Value *Bits = B.CreateBitCast(Src, I32T, "f32_bits");
          Value *Lsb = B.CreateAnd(B.CreateLShr(Bits, 16), 1, "bf16_lsb");
          Value *Rounded = B.CreateAdd(
              Bits, B.CreateAdd(Lsb, ConstantInt::get(I32T, 0x7fff)),
              "bf16_rnd");
          Value *Res = B.CreateTrunc(B.CreateLShr(Rounded, 16), I16T);
          Value *BF = B.CreateBitCast(Res, I->getType(), I->getName());
          I->replaceAllUsesWith(BF);
          I->eraseFromParent();
          Changed = true;
        } else if (isa<FPExtInst>(I) &&
                   I->getOperand(0)->getType()->getScalarType() == BF16 &&
                   I->getType()->getScalarType() == F32) {
          IRBuilder<> B(I);
          Value *Src = I->getOperand(0);
          Type *I32T = matchShape(I32, I->getType());
          Type *I16T = matchShape(I16, I->getType());
          Value *Bits = B.CreateBitCast(Src, I16T, "bf16_bits");
          Value *Wide = B.CreateZExt(Bits, I32T, "bf16_zext");
          Value *Shifted = B.CreateShl(Wide, 16, "f32_bits");
          Value *FP = B.CreateBitCast(Shifted, I->getType(), I->getName());
          I->replaceAllUsesWith(FP);
          I->eraseFromParent();
          Changed = true;
        }
      }
    }
  }

  // Phase 0.5: sink a vector `bitcast <N x i16> -> <N x bfloat>` (the RNE
  // decompose tail) through its extractelement users to scalars. When the i16
  // vector traces back to a simdgroup_matrix accumulator, the AGX JIT
  // miscompiles MULTI-LANE extraction of the narrow <N x i16> register to zero
  // (a single-lane extract, and the integer RNE math itself, are fine — the
  // bug is narrowing the wide simdgroup f32 register to i16 *as a vector* and
  // then pulling several lanes out of it). If the i16 vector is a
  // `trunc <N x i32> -> <N x i16>`, sink the trunc too so the extract reads the
  // <N x i32> (full-width) register and the trunc+reinterpret run on scalars.
  // Every vector op is preserved; only the free trunc/reinterpret moves to
  // scalars — no extra work, no scalarized arithmetic. Same shape for half.
  for (Function &F : M) {
    SmallVector<Instruction *, 8> DeadBC;
    for (BasicBlock &BB : F) {
      for (Instruction &Inst : BB) {
        auto *BC = dyn_cast<BitCastInst>(&Inst);
        if (!BC)
          continue;
        auto *DstVT = dyn_cast<FixedVectorType>(BC->getType());
        auto *SrcVT = dyn_cast<FixedVectorType>(BC->getSrcTy());
        if (!DstVT || !SrcVT)
          continue;
        Type *DstElem = DstVT->getElementType();
        if (!DstElem->isBFloatTy() && !DstElem->isHalfTy())
          continue;
        if (!SrcVT->getElementType()->isIntegerTy(16))
          continue;
        bool AllExtract = !BC->use_empty();
        for (User *U : BC->users())
          if (!isa<ExtractElementInst>(U))
            AllExtract = false;
        if (!AllExtract)
          continue;
        Value *I16Vec = BC->getOperand(0);
        auto *Trn = dyn_cast<TruncInst>(I16Vec);
        Value *WideVec =
            (Trn && isa<FixedVectorType>(Trn->getSrcTy())) ? Trn->getOperand(0)
                                                           : nullptr;
        for (User *U : llvm::make_early_inc_range(BC->users())) {
          auto *EE = cast<ExtractElementInst>(U);
          IRBuilder<> B(EE);
          Value *Si;
          if (WideVec) {
            Value *Wi = B.CreateExtractElement(WideVec, EE->getIndexOperand());
            Si = B.CreateTrunc(Wi, Type::getInt16Ty(M.getContext()));
          } else {
            Si = B.CreateExtractElement(I16Vec, EE->getIndexOperand());
          }
          Value *Sf = B.CreateBitCast(Si, DstElem);
          EE->replaceAllUsesWith(Sf);
          DeadBC.push_back(EE);
        }
        DeadBC.push_back(BC);
      }
    }
    for (Instruction *I : DeadBC)
      if (I->use_empty())
        I->eraseFromParent();
  }

  // Phase 1: decompose sitofp/uitofp iN -> bfloat via f32 + bit manipulation.
  // The magnitude is first rounded to bfloat's 8 significant bits, to nearest
  // even, in the integer domain: converting to f32 first and truncating to
  // bfloat would round twice (and the truncation alone rounds toward zero).
  // The rounded magnitude q * 2^sh (q < 2^9) then converts to f32 exactly.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction *I = &*It++;
        if ((isa<SIToFPInst>(I) || isa<UIToFPInst>(I)) &&
            I->getType()->getScalarType() == BF16) {
          IRBuilder<> B(I);
          Type *F32T = matchShape(F32, I->getType());
          Type *I32T = matchShape(I32, I->getType());
          Type *I16T = matchShape(I16, I->getType());
          Value *X = I->getOperand(0);
          Type *XT = X->getType();
          unsigned W = XT->getScalarSizeInBits();
          if (W < 32) {
            X = isa<SIToFPInst>(I) ? B.CreateSExt(X, I32T) : B.CreateZExt(X, I32T);
            XT = I32T;
            W = 32;
          }
          bool Signed = isa<SIToFPInst>(I);
          Value *Neg = Signed ? B.CreateICmpSLT(X, Constant::getNullValue(XT))
                              : nullptr;
          // |x| as an unsigned value (INT_MIN's magnitude is its bit pattern).
          Value *Mag = Signed ? B.CreateSelect(Neg, B.CreateNeg(X), X) : X;
          auto C = [&](uint64_t V) { return ConstantInt::get(XT, V); };
          // The bit length of |x| from the exponent of its f32 conversion
          // (no llvm.ctlz: LLVMToAIRIntrinsics has run), one less where the
          // conversion rounded up to the next power of two; 0 for 0.
          Value *MagF = B.CreateBitCast(B.CreateUIToFP(Mag, F32T), I32T);
          Value *E = B.CreateSub(B.CreateLShr(MagF, 23), ConstantInt::get(I32T, 126));
          E = B.CreateSelect(B.CreateICmpSGT(E, ConstantInt::get(I32T, W)),
                             ConstantInt::get(I32T, W), E);
          Value *Bits = B.CreateZExtOrTrunc(E, XT);
          Value *Top = B.CreateLShr(Mag, B.CreateSub(Bits, C(1)));
          Bits = B.CreateSelect(B.CreateICmpEQ(Top, C(0)), B.CreateSub(Bits, C(1)),
                                Bits);
          Bits = B.CreateSelect(B.CreateICmpEQ(Mag, C(0)), C(0), Bits);
          // sh = max(bits - 8, 0): the bits below bfloat's precision.
          Value *Sh = B.CreateSelect(B.CreateICmpUGT(Bits, C(8)),
                                     B.CreateSub(Bits, C(8)), C(0));
          Value *Q = B.CreateLShr(Mag, Sh);
          Value *Rem = B.CreateSub(Mag, B.CreateShl(Q, Sh));
          // Half of the dropped unit; 0 when nothing is dropped, so that
          // rem > half never holds then.
          Value *Half = B.CreateSelect(B.CreateICmpEQ(Sh, C(0)), C(0),
                                       B.CreateShl(C(1), B.CreateSub(Sh, C(1))));
          Value *Up = B.CreateOr(
              B.CreateICmpUGT(Rem, Half),
              B.CreateAnd(B.CreateICmpEQ(Rem, Half),
                          B.CreateAnd(B.CreateICmpNE(Sh, C(0)),
                                      B.CreateICmpNE(B.CreateAnd(Q, C(1)), C(0)))));
          Q = B.CreateAdd(Q, B.CreateZExt(Up, XT));
          // q <= 2^8 converts exactly; scale by 2^sh through the exponent.
          Value *QF = B.CreateUIToFP(B.CreateTrunc(Q, I32T), F32T);
          Value *Scale = B.CreateBitCast(
              B.CreateShl(B.CreateAdd(B.CreateTrunc(Sh, I32T),
                                      ConstantInt::get(I32T, 127)),
                          23),
              F32T);
          Value *ToFloat = B.CreateFMul(QF, Scale, "to_f32");
          // bf16 = upper 16 bits of f32 (exact: 8 significant bits), with
          // the sign set in the bits (an fneg here would reach Metal as is:
          // AIRLowerFNeg has run).
          Value *AsInt = B.CreateBitCast(ToFloat, I32T, "f32_bits");
          if (Signed)
            AsInt = B.CreateOr(
                AsInt, B.CreateSelect(Neg, ConstantInt::get(I32T, 0x80000000u),
                                      Constant::getNullValue(I32T)));
          Value *Shifted = B.CreateLShr(AsInt, 16, "bf16_bits");
          Value *Narrow = B.CreateTrunc(Shifted, I16T, "bf16_i16");
          Value *Trunc = B.CreateBitCast(Narrow, I->getType(), I->getName());
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
          if (!Trn || !Trn->getType()->isIntegerTy())
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
AIRBFloat16CastDecomposePass::run(Module &M, ModuleAnalysisManager &AM) {
  return bfloat16CastDecompose(M) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

bool AIRBFloat16CastDecomposeLegacy::runOnModule(Module &M) {
  return bfloat16CastDecompose(M);
}

char AIRBFloat16CastDecomposeLegacy::ID = 0;

INITIALIZE_PASS(AIRBFloat16CastDecomposeLegacy, DEBUG_TYPE,
                "AIR BFloat16 Cast Decompose", false, false)

ModulePass *llvm::createAIRBFloat16CastDecomposeLegacyPass() {
  return new AIRBFloat16CastDecomposeLegacy();
}
