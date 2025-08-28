//===-- AMDGPUShuffleOptimizer.cpp - Optimize shuffle patterns -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes generic shuffle intrinsics by detecting constant
// patterns and replacing them with efficient hardware-specific instructions
// (DPP, PERMLANE*, etc.) when beneficial, falling back to DS_BPERMUTE_B32
// for unmatched patterns.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-shuffle-optimizer"

static cl::opt<bool>
    EnableShuffleOptimization("amdgpu-enable-shuffle-optimization",
                              cl::desc("Enable AMDGPU shuffle optimization"),
                              cl::init(true), cl::Hidden);

namespace {

// Represents a detected shuffle pattern that can be optimized
struct ShufflePattern {
  enum PatternKind {
    DPP_QUAD_PERM, // DPP quad permutation
    DPP_ROW_SHL,   // DPP row shift left
    DPP_ROW_SHR,   // DPP row shift right
    DPP_WAVE_SHL,  // DPP wave shift left
    DPP_WAVE_SHR,  // DPP wave shift right
    PERMLANE16,    // V_PERMLANE16_B32
    PERMLANEX16,   // V_PERMLANEX16_B32
    PERMLANE64,    // V_PERMLANE64_B32
    DS_BPERMUTE,   // Fallback to DS_BPERMUTE_B32
    UNSUPPORTED    // Cannot be optimized
  };

  PatternKind Kind;
  uint32_t DPPCtrl = 0;    // DPP control value
  uint32_t RowMask = 0xf;  // DPP row mask
  uint32_t BankMask = 0xf; // DPP bank mask
  bool BoundCtrl = false;  // DPP bound control

  ShufflePattern() : Kind(UNSUPPORTED) {}
};

class AMDGPUShuffleOptimizer : public FunctionPass {
public:
  static char ID;

  AMDGPUShuffleOptimizer() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  static bool runShuffleOptimizer(Function &F, const GCNSubtarget &ST);

private:
  const GCNSubtarget *ST = nullptr;

  bool optimizeShuffleIntrinsic(CallInst *CI);
  ShufflePattern analyzeShufflePattern(CallInst *CI);
  ShufflePattern analyzeShuffleIdx(int Width, int Offset);
  ShufflePattern analyzeShuffleUp(int Width, int Delta);
  ShufflePattern analyzeShuffleDown(int Width, int Delta);
  ShufflePattern analyzeShuffleXor(int Width, int Mask);

  bool tryOptimizeToDPP(CallInst *CI, const ShufflePattern &Pattern);
  bool tryOptimizeToPermlane(CallInst *CI, const ShufflePattern &Pattern);
  bool fallbackToBpermute(CallInst *CI);

  Value *createDPPIntrinsic(IRBuilder<> &Builder, Value *OldVal, Value *SrcVal,
                            const ShufflePattern &Pattern);
  Value *createPermlaneIntrinsic(IRBuilder<> &Builder, Value *Val,
                                 const ShufflePattern &Pattern);
  Value *createBpermuteIntrinsic(IRBuilder<> &Builder, Value *Val,
                                 Value *Index);

  bool processShuffleIntrinsics(Function &F);
};

char AMDGPUShuffleOptimizer::ID = 0;

bool AMDGPUShuffleOptimizer::runOnFunction(Function &F) {
  if (!EnableShuffleOptimization)
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  ST = &TM.getSubtarget<GCNSubtarget>(F);

  return processShuffleIntrinsics(F);
}

bool AMDGPUShuffleOptimizer::optimizeShuffleIntrinsic(CallInst *CI) {
  ShufflePattern Pattern = analyzeShufflePattern(CI);

  if (Pattern.Kind == ShufflePattern::UNSUPPORTED)
    return fallbackToBpermute(CI);

  switch (Pattern.Kind) {
  case ShufflePattern::DPP_QUAD_PERM:
  case ShufflePattern::DPP_ROW_SHL:
  case ShufflePattern::DPP_ROW_SHR:
  case ShufflePattern::DPP_WAVE_SHL:
  case ShufflePattern::DPP_WAVE_SHR:
    return tryOptimizeToDPP(CI, Pattern);

  case ShufflePattern::PERMLANE16:
  case ShufflePattern::PERMLANEX16:
  case ShufflePattern::PERMLANE64:
    return tryOptimizeToPermlane(CI, Pattern);

  case ShufflePattern::DS_BPERMUTE:
    return fallbackToBpermute(CI);

  default:
    return false;
  }
}

ShufflePattern AMDGPUShuffleOptimizer::analyzeShufflePattern(CallInst *CI) {
  auto *II = cast<IntrinsicInst>(CI);

  // Get width parameter (must be constant)
  auto *WidthConst = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!WidthConst)
    return ShufflePattern();

  int Width = WidthConst->getSExtValue();

  // Get offset/delta/mask parameter (must be constant for pattern optimization)
  auto *ParamConst = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!ParamConst)
    return ShufflePattern();

  int Param = ParamConst->getSExtValue();

  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_generic_shuffle:
    return analyzeShuffleIdx(Width, Param);
  case Intrinsic::amdgcn_generic_shuffle_up:
    return analyzeShuffleUp(Width, Param);
  case Intrinsic::amdgcn_generic_shuffle_down:
    return analyzeShuffleDown(Width, Param);
  case Intrinsic::amdgcn_generic_shuffle_xor:
    return analyzeShuffleXor(Width, Param);
  default:
    return ShufflePattern();
  }
}

ShufflePattern AMDGPUShuffleOptimizer::analyzeShuffleIdx(int Width,
                                                         int Offset) {
  // For idx shuffle, all lanes read from the same offset
  if (Offset == 0) {
    // Broadcast from lane 0 - use DPP if supported, otherwise bpermute
    if (ST->hasDPP() && Width == 16) {
      // Use DPP quad permutation to broadcast lane 0 within each group of 4
      ShufflePattern Pattern;
      Pattern.Kind = ShufflePattern::DPP_QUAD_PERM;
      Pattern.DPPCtrl = 0x00; // [0,0,0,0] quad perm - broadcast lane 0
      return Pattern;
    } else {
      // Fall back to bpermute for other cases
      ShufflePattern Pattern;
      Pattern.Kind = ShufflePattern::DS_BPERMUTE;
      return Pattern;
    }
  }

  // For other constant broadcasts, fall back to bpermute
  ShufflePattern Pattern;
  Pattern.Kind = ShufflePattern::DS_BPERMUTE;
  return Pattern;
}

ShufflePattern AMDGPUShuffleOptimizer::analyzeShuffleUp(int Width, int Delta) {
  if (Width == 32 && Delta == 1) {
    // Simple wave shift up by 1 - can use DPP
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::DPP_WAVE_SHL;
    Pattern.DPPCtrl = 0x101; // WAVE_SHL1
    return Pattern;
  }

  if (Width == 16 && Delta <= 15) {
    // Row shift within 16 lanes - can use DPP row shift
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::DPP_ROW_SHL;
    Pattern.DPPCtrl = 0x100 + Delta; // ROW_SHL0 + delta
    return Pattern;
  }

  // Check for permlane patterns
  if (Width == 16) {
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::PERMLANE16;
    return Pattern;
  }

  // Fall back to bpermute
  ShufflePattern Pattern;
  Pattern.Kind = ShufflePattern::DS_BPERMUTE;
  return Pattern;
}

ShufflePattern AMDGPUShuffleOptimizer::analyzeShuffleDown(int Width,
                                                          int Delta) {
  if (Width == 32 && Delta == 1) {
    // Simple wave shift down by 1 - can use DPP
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::DPP_WAVE_SHR;
    Pattern.DPPCtrl = 0x111; // WAVE_SHR1
    return Pattern;
  }

  if (Width == 16 && Delta <= 15) {
    // Row shift within 16 lanes - can use DPP row shift
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::DPP_ROW_SHR;
    Pattern.DPPCtrl = 0x110 + Delta; // ROW_SHR0 + delta
    return Pattern; 
  }

  // Check for permlane patterns
  if (Width == 16) {
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::PERMLANE16;
    return Pattern;
  }

  // Fall back to bpermute
  ShufflePattern Pattern;
  Pattern.Kind = ShufflePattern::DS_BPERMUTE;
  return Pattern;
}

ShufflePattern AMDGPUShuffleOptimizer::analyzeShuffleXor(int Width, int Mask) {
  // XOR with mask 1 within quads - can use DPP quad permutation
  if (Width == 32 && Mask == 1) {
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::DPP_QUAD_PERM;
    Pattern.DPPCtrl = 0xB1; // [2,3,0,1] - swap pairs within quads
    return Pattern;
  }

  // XOR with mask 16 - cross-row exchange within 32 lanes
  if (Width == 32 && Mask == 16) {
    ShufflePattern Pattern;
    Pattern.Kind = ShufflePattern::PERMLANE16;
    return Pattern;
  }

  // XOR with mask 32 - cross-half exchange for Wave64 on GFX11+
  if (Width == 64 && Mask == 32) {
    if (ST->getGeneration() >= AMDGPUSubtarget::GFX11) {
      ShufflePattern Pattern;
      Pattern.Kind = ShufflePattern::PERMLANE64;
      return Pattern;
    } else {
      // GFX10 doesn't have PERMLANE64, fall back to bpermute
      ShufflePattern Pattern;
      Pattern.Kind = ShufflePattern::DS_BPERMUTE;
      return Pattern;
    }
  }

  // Fall back to bpermute for other patterns
  ShufflePattern Pattern;
  Pattern.Kind = ShufflePattern::DS_BPERMUTE;
  return Pattern;
}

bool AMDGPUShuffleOptimizer::tryOptimizeToDPP(CallInst *CI,
                                              const ShufflePattern &Pattern) {
  IRBuilder<> Builder(CI);
  Value *SrcVal = CI->getArgOperand(0);
  Value *OldVal = PoisonValue::get(SrcVal->getType());

  Value *DPPResult = createDPPIntrinsic(Builder, OldVal, SrcVal, Pattern);

  CI->replaceAllUsesWith(DPPResult);
  CI->eraseFromParent();

  return true;
}

bool AMDGPUShuffleOptimizer::tryOptimizeToPermlane(
    CallInst *CI, const ShufflePattern &Pattern) {
  IRBuilder<> Builder(CI);
  Value *Val = CI->getArgOperand(0);

  Value *PermlaneResult = createPermlaneIntrinsic(Builder, Val, Pattern);

  CI->replaceAllUsesWith(PermlaneResult);
  CI->eraseFromParent();

  return true;
}

bool AMDGPUShuffleOptimizer::fallbackToBpermute(CallInst *CI) {
  IRBuilder<> Builder(CI);
  Value *Val = CI->getArgOperand(0);

  // Create lane ID
  Value *LaneId =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                              {Builder.getInt32(-1), Builder.getInt32(0)});
  if (ST->isWave64()) {
    Value *LaneIdHi = Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_hi, {},
                                              {Builder.getInt32(-1), LaneId});
    LaneId = LaneIdHi;
  }

  // Calculate target lane based on shuffle type
  Value *TargetLane = nullptr;
  auto *II = cast<IntrinsicInst>(CI);
  Value *Param = CI->getArgOperand(1);

  switch (II->getIntrinsicID()) {
  case Intrinsic::amdgcn_generic_shuffle:
    TargetLane = Param;
    break;
  case Intrinsic::amdgcn_generic_shuffle_up:
    TargetLane = Builder.CreateSub(LaneId, Param);
    break;
  case Intrinsic::amdgcn_generic_shuffle_down:
    TargetLane = Builder.CreateAdd(LaneId, Param);
    break;
  case Intrinsic::amdgcn_generic_shuffle_xor:
    TargetLane = Builder.CreateXor(LaneId, Param);
    break;
  default:
    return false;
  }

  // Create byte-aligned index for bpermute
  Value *ByteIndex = Builder.CreateShl(TargetLane, 2);

  Value *BpermuteResult = createBpermuteIntrinsic(Builder, Val, ByteIndex);

  CI->replaceAllUsesWith(BpermuteResult);
  CI->eraseFromParent();

  return true;
}

Value *
AMDGPUShuffleOptimizer::createDPPIntrinsic(IRBuilder<> &Builder, Value *OldVal,
                                           Value *SrcVal,
                                           const ShufflePattern &Pattern) {
  return Builder.CreateIntrinsic(
      Intrinsic::amdgcn_update_dpp, {SrcVal->getType()},
      {OldVal, SrcVal, Builder.getInt32(Pattern.DPPCtrl),
       Builder.getInt32(Pattern.RowMask), Builder.getInt32(Pattern.BankMask),
       Builder.getInt1(Pattern.BoundCtrl)});
}

Value *AMDGPUShuffleOptimizer::createPermlaneIntrinsic(
    IRBuilder<> &Builder, Value *Val, const ShufflePattern &Pattern) {
  switch (Pattern.Kind) {
  case ShufflePattern::PERMLANE16:
    return Builder.CreateIntrinsic(
        Intrinsic::amdgcn_permlane16, {Val->getType()},
        {PoisonValue::get(Val->getType()), Val, Builder.getInt32(0),
         Builder.getInt32(0), Builder.getInt1(false), Builder.getInt1(false)});

  case ShufflePattern::PERMLANEX16:
    return Builder.CreateIntrinsic(
        Intrinsic::amdgcn_permlanex16, {Val->getType()},
        {PoisonValue::get(Val->getType()), Val, Builder.getInt32(0),
         Builder.getInt32(0), Builder.getInt1(false), Builder.getInt1(false)});

  case ShufflePattern::PERMLANE64:
    return Builder.CreateIntrinsic(Intrinsic::amdgcn_permlane64,
                                   {Val->getType()}, {Val});
  default:
    llvm_unreachable("Invalid permlane pattern");
  }
}

Value *AMDGPUShuffleOptimizer::createBpermuteIntrinsic(IRBuilder<> &Builder,
                                                       Value *Val,
                                                       Value *Index) {
  // Convert value to i32 for bpermute
  Type *OrigType = Val->getType();
  Value *I32Val = Val;

  if (OrigType != Builder.getInt32Ty())
    I32Val = Builder.CreateBitCast(Val, Builder.getInt32Ty());

  Value *Result = Builder.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {},
                                          {Index, I32Val});

  if (OrigType != Builder.getInt32Ty())
    Result = Builder.CreateBitCast(Result, OrigType);

  return Result;
}

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUShuffleOptimizer, DEBUG_TYPE,
                      "AMDGPU Shuffle Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUShuffleOptimizer, DEBUG_TYPE,
                    "AMDGPU Shuffle Optimizer", false, false)

FunctionPass *llvm::createAMDGPUShuffleOptimizerPass() {
  return new AMDGPUShuffleOptimizer();
}

bool AMDGPUShuffleOptimizer::processShuffleIntrinsics(Function &F) {
  bool Changed = false;

  for (auto &BB : F)
    for (auto &I : llvm::make_early_inc_range(BB))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
          switch (II->getIntrinsicID()) {
          case Intrinsic::amdgcn_generic_shuffle:
          case Intrinsic::amdgcn_generic_shuffle_up:
          case Intrinsic::amdgcn_generic_shuffle_down:
          case Intrinsic::amdgcn_generic_shuffle_xor:
            Changed |= optimizeShuffleIntrinsic(CI);
            break;
          default:
            break;
          }
        }

  return Changed;
}

bool AMDGPUShuffleOptimizer::runShuffleOptimizer(Function &F,
                                                 const GCNSubtarget &ST) {
  if (!EnableShuffleOptimization)
    return false;

  AMDGPUShuffleOptimizer TempOptimizer;
  TempOptimizer.ST = &ST;
  return TempOptimizer.processShuffleIntrinsics(F);
}

PreservedAnalyses AMDGPUShuffleOptimizerPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  bool Changed = AMDGPUShuffleOptimizer::runShuffleOptimizer(F, ST);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
