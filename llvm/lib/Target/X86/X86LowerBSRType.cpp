//===- Target/X86/X86LowerBSRType.cpp - -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to lower x86_bsr type operations.
///
/// The x86_bsr type represents a 1024-bit BSR (Block Scale Register) value.
/// This pass lowers cast_vector_to_bsr / cast_bsr_to_vector bridging
/// intrinsics and converts bsr_create / bsr_get_hi / bsr_get_lo / bsr_set_hi
/// / bsr_set_lo intrinsics into the legacy implicit-BSR intrinsics
/// (bsrmovf, bsrmovh_set, bsrmovh_get, bsrmovl_set, bsrmovl_get).
///
/// This pass runs before instruction selection, parallel to X86LowerAMXType
/// for AMX tiles.
///
//===----------------------------------------------------------------------===//
//
#include "X86.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "x86-lower-bsr-type"

namespace {

static bool isBSRIntrinsic(IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::x86_bsr_create:
  case Intrinsic::x86_bsr_get_hi:
  case Intrinsic::x86_bsr_get_lo:
  case Intrinsic::x86_bsr_set_hi:
  case Intrinsic::x86_bsr_set_lo:
  case Intrinsic::x86_cast_vector_to_bsr:
  case Intrinsic::x86_cast_bsr_to_vector:
  case Intrinsic::x86_top4mxhf8ps_bsr_internal:
  case Intrinsic::x86_top4mxbhf8ps_bsr_internal:
  case Intrinsic::x86_top4mxhbf8ps_bsr_internal:
  case Intrinsic::x86_top4mxbf8ps_bsr_internal:
    return true;
  }
}

/// Map a BSR-carrying TOP4MX intrinsic to its non-BSR legacy equivalent.
static Intrinsic::ID getBSRToLegacyID(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::x86_top4mxhf8ps_bsr_internal:
    return Intrinsic::x86_top4mxhf8ps_internal;
  case Intrinsic::x86_top4mxbhf8ps_bsr_internal:
    return Intrinsic::x86_top4mxbhf8ps_internal;
  case Intrinsic::x86_top4mxhbf8ps_bsr_internal:
    return Intrinsic::x86_top4mxhbf8ps_internal;
  case Intrinsic::x86_top4mxbf8ps_bsr_internal:
    return Intrinsic::x86_top4mxbf8ps_internal;
  default:
    return Intrinsic::not_intrinsic;
  }
}

/// Lower a single BSR intrinsic call. Returns true if lowered.
static bool lowerBSRIntrinsic(IntrinsicInst *II, IRBuilder<> &Builder) {
  Builder.SetInsertPoint(II);

  switch (II->getIntrinsicID()) {
  default:
    return false;

  case Intrinsic::x86_bsr_create: {
    // bsr_create(lo, hi) -> bsrmovf(hi, lo) + implicit BSR
    // Per ACE spec: BSRMOVF(src1, src2) where src1→A-scales (upper=hi),
    // src2→B-scales (lower=lo). The legacy intrinsic writes to BSR implicitly.
    Value *Lo = II->getArgOperand(0);
    Value *Hi = II->getArgOperand(1);
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           II->getModule(), Intrinsic::x86_bsrmovf),
                       {Hi, Lo});
    // The result (x86_bsr SSA value) is only used by get_hi/get_lo/set_hi/
    // set_lo which will also be lowered. For now, replace uses with poison.
    II->replaceAllUsesWith(PoisonValue::get(II->getType()));
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_bsr_get_hi: {
    // bsr_get_hi(bsr) -> bsrmovh_get()
    Value *Result = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::x86_bsrmovh_get));
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_bsr_get_lo: {
    // bsr_get_lo(bsr) -> bsrmovl_get()
    Value *Result = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::x86_bsrmovl_get));
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_bsr_set_hi: {
    // bsr_set_hi(bsr, val) -> bsrmovh_set(val)
    Value *Val = II->getArgOperand(1);
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           II->getModule(), Intrinsic::x86_bsrmovh_set),
                       {Val});
    II->replaceAllUsesWith(PoisonValue::get(II->getType()));
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_bsr_set_lo: {
    // bsr_set_lo(bsr, val) -> bsrmovl_set(val)
    Value *Val = II->getArgOperand(1);
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           II->getModule(), Intrinsic::x86_bsrmovl_set),
                       {Val});
    II->replaceAllUsesWith(PoisonValue::get(II->getType()));
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_cast_vector_to_bsr: {
    // cast_vector_to_bsr(vec) -> split vector and call bsrmovf
    // For v32i32 (1024 bits), split into two v16i32 halves.
    // Input vector layout: [lo=B-scales, hi=A-scales] from __bsr_combine_v32.
    //
    // If this cast's result has no uses (e.g., because all consumers already
    // extracted the operands directly during their lowering), skip emitting
    // bsrmovf to avoid duplicate instructions.
    if (II->use_empty()) {
      II->eraseFromParent();
      return true;
    }

    Value *Vec = II->getArgOperand(0);
    auto *VecTy = cast<FixedVectorType>(Vec->getType());
    unsigned NumElts = VecTy->getNumElements();
    Type *EltTy = VecTy->getElementType();
    unsigned HalfElts = NumElts / 2;
    auto *HalfVecTy = FixedVectorType::get(EltTy, HalfElts);

    // Extract low and high halves using shufflevector.
    // Lo = Vec[0:15] = B-scales (lower), Hi = Vec[16:31] = A-scales (upper).
    SmallVector<int, 32> LoMask(HalfElts), HiMask(HalfElts);
    for (unsigned i = 0; i < HalfElts; ++i) {
      LoMask[i] = i;
      HiMask[i] = i + HalfElts;
    }
    Value *Lo = Builder.CreateShuffleVector(Vec, LoMask);
    Value *Hi = Builder.CreateShuffleVector(Vec, HiMask);

    // Bitcast to v16i32 if needed.
    auto *V16I32Ty = FixedVectorType::get(Builder.getInt32Ty(), 16);
    if (HalfVecTy != V16I32Ty) {
      Lo = Builder.CreateBitCast(Lo, V16I32Ty);
      Hi = Builder.CreateBitCast(Hi, V16I32Ty);
    }

    // Per ACE spec: BSRMOVF(src1, src2) where src1→A-scales (upper=Hi),
    // src2→B-scales (lower=Lo).
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           II->getModule(), Intrinsic::x86_bsrmovf),
                       {Hi, Lo});
    II->replaceAllUsesWith(PoisonValue::get(II->getType()));
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_cast_bsr_to_vector: {
    // cast_bsr_to_vector(bsr) -> get low + get high, combine
    auto *V16I32Ty = FixedVectorType::get(Builder.getInt32Ty(), 16);
    Value *Lo = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::x86_bsrmovl_get));
    Value *Hi = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
        II->getModule(), Intrinsic::x86_bsrmovh_get));

    auto *RetTy = cast<FixedVectorType>(II->getType());
    unsigned NumElts = RetTy->getNumElements();
    unsigned HalfElts = NumElts / 2;

    // Bitcast from v16i32 to the half-type if needed.
    auto *HalfTy = FixedVectorType::get(RetTy->getElementType(), HalfElts);
    if (V16I32Ty != HalfTy) {
      Lo = Builder.CreateBitCast(Lo, HalfTy);
      Hi = Builder.CreateBitCast(Hi, HalfTy);
    }

    // Combine via shufflevector.
    SmallVector<int, 32> CombineMask(NumElts);
    for (unsigned i = 0; i < HalfElts; ++i) {
      CombineMask[i] = i;
      CombineMask[i + HalfElts] = i + HalfElts;
    }
    Value *Result = Builder.CreateShuffleVector(Lo, Hi, CombineMask);
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
    return true;
  }

  case Intrinsic::x86_top4mxhf8ps_bsr_internal:
  case Intrinsic::x86_top4mxbhf8ps_bsr_internal:
  case Intrinsic::x86_top4mxhbf8ps_bsr_internal:
  case Intrinsic::x86_top4mxbf8ps_bsr_internal: {
    // Lower BSR-carrying TOP4MX: emit bsrmovf before the call, replace with
    // legacy (non-BSR) intrinsic call.
    //
    // Input:  %result = call x86_amx @llvm.x86.top4mxhf8ps.bsr.internal(
    //             i16 %m, i16 %n, i16 %k, i8 %imm, x86_amx %acc,
    //             <16 x i32> %src1, <16 x i32> %src2, x86_bsr %bsr)
    //
    // Output: ; extract lo/hi from BSR source
    //         call void @llvm.x86.bsrmovf(<16 x i32> %lo, <16 x i32> %hi)
    //         %result = call x86_amx @llvm.x86.top4mxhf8ps.internal(
    //             i16 %m, i16 %n, i16 %k, i8 %imm, x86_amx %acc,
    //             <16 x i32> %src1, <16 x i32> %src2)

    Value *BSRVal = II->getArgOperand(7); // x86_bsr argument

    // Extract lo and hi vectors from the x86_bsr value.
    Value *Lo = nullptr;
    Value *Hi = nullptr;
    auto *V16I32Ty = FixedVectorType::get(Builder.getInt32Ty(), 16);

    if (auto *SrcII = dyn_cast<IntrinsicInst>(BSRVal)) {
      if (SrcII->getIntrinsicID() == Intrinsic::x86_bsr_create) {
        Lo = SrcII->getArgOperand(0);
        Hi = SrcII->getArgOperand(1);
      } else if (SrcII->getIntrinsicID() == Intrinsic::x86_cast_vector_to_bsr) {
        // The cast_vector_to_bsr takes a single large vector and splits it.
        // We need to do the same split here.
        Value *Vec = SrcII->getArgOperand(0);
        auto *VecTy = cast<FixedVectorType>(Vec->getType());
        unsigned NumElts = VecTy->getNumElements();
        unsigned HalfElts = NumElts / 2;

        SmallVector<int, 32> LoMask(HalfElts), HiMask(HalfElts);
        for (unsigned i = 0; i < HalfElts; ++i) {
          LoMask[i] = i;
          HiMask[i] = i + HalfElts;
        }
        Lo = Builder.CreateShuffleVector(Vec, LoMask);
        Hi = Builder.CreateShuffleVector(Vec, HiMask);

        auto *HalfVecTy =
            FixedVectorType::get(VecTy->getElementType(), HalfElts);
        if (HalfVecTy != V16I32Ty) {
          Lo = Builder.CreateBitCast(Lo, V16I32Ty);
          Hi = Builder.CreateBitCast(Hi, V16I32Ty);
        }
      }
    }

    if (!Lo || !Hi) {
      // Fallback: emit bsr_get_lo/bsr_get_hi calls.
      // These will be lowered by this same pass (they're also in
      // BSRIntrinsics).
      Lo = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                                  II->getModule(), Intrinsic::x86_bsr_get_lo),
                              {BSRVal});
      Hi = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                                  II->getModule(), Intrinsic::x86_bsr_get_hi),
                              {BSRVal});
    }

    // Emit bsrmovf(hi, lo) immediately before the TOP4MX call.
    // Per ACE spec: BSRMOVF(src1, src2) where src1→A-scales (upper=Hi),
    // src2→B-scales (lower=Lo).
    Builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           II->getModule(), Intrinsic::x86_bsrmovf),
                       {Hi, Lo});

    // Create the legacy (non-BSR) TOP4MX intrinsic call with args 0-6.
    Intrinsic::ID LegacyID = getBSRToLegacyID(II->getIntrinsicID());
    SmallVector<Value *, 7> LegacyArgs;
    for (unsigned i = 0; i < 7; ++i)
      LegacyArgs.push_back(II->getArgOperand(i));

    // The legacy intrinsic has an ImmArg at index 3; we need to use the
    // overloaded declaration getter with no type overloads (non-overloaded).
    Value *NewCall = Builder.CreateCall(
        Intrinsic::getOrInsertDeclaration(II->getModule(), LegacyID),
        LegacyArgs);

    II->replaceAllUsesWith(NewCall);
    II->eraseFromParent();
    return true;
  }
  }
}

static bool runOnFunction(Function &F) {
  bool Changed = false;
  SmallVector<IntrinsicInst *, 8> BSRConsumers; // TOP4MX *_bsr_internal
  SmallVector<IntrinsicInst *, 8> BSRProducers; // bsr_create, bsr_set_*, casts

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (!isBSRIntrinsic(II))
          continue;
        // Separate consumers (TOP4MX BSR) from producers (bsr_create etc.)
        // so we lower consumers first (they need to read producer operands).
        if (getBSRToLegacyID(II->getIntrinsicID()) != Intrinsic::not_intrinsic)
          BSRConsumers.push_back(II);
        else
          BSRProducers.push_back(II);
      }
    }
  }

  if (BSRConsumers.empty() && BSRProducers.empty())
    return false;

  IRBuilder<> Builder(F.getContext());

  // Lower consumers first: they may reference bsr_create operands that would
  // be replaced with poison if we lowered producers first.
  for (auto *II : BSRConsumers)
    Changed |= lowerBSRIntrinsic(II, Builder);
  for (auto *II : BSRProducers)
    Changed |= lowerBSRIntrinsic(II, Builder);

  return Changed;
}

class X86LowerBSRTypeLegacyPass : public FunctionPass {
public:
  static char ID;

  X86LowerBSRTypeLegacyPass() : FunctionPass(ID) {
    initializeX86LowerBSRTypeLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override { return ::runOnFunction(F); }

  StringRef getPassName() const override { return "X86 Lower BSR Type"; }
};

} // anonymous namespace

char X86LowerBSRTypeLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(X86LowerBSRTypeLegacyPass, DEBUG_TYPE,
                      "X86 Lower BSR Type", false, false)
INITIALIZE_PASS_END(X86LowerBSRTypeLegacyPass, DEBUG_TYPE, "X86 Lower BSR Type",
                    false, false)

FunctionPass *llvm::createX86LowerBSRTypeLegacyPass() {
  return new X86LowerBSRTypeLegacyPass();
}

PreservedAnalyses X86LowerBSRTypePass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  if (!runOnFunction(F))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
