//===-- AMDGPULowerIntrinsics.cpp -------------------------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower intrinsics that would otherwise require separate handling in both
// SelectionDAG and GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "amdgpu-lower-intrinsics"

using namespace llvm;

namespace {

struct MatrixIntrinsicNarrowingResult {
  IntrinsicInst *Current;
  bool Changed;
};

class AMDGPULowerIntrinsicsImpl {
public:
  Module &M;
  const AMDGPUTargetMachine &TM;

  AMDGPULowerIntrinsicsImpl(Module &M, const AMDGPUTargetMachine &TM)
      : M(M), TM(TM) {}

  bool run();

private:
  bool visitBarrier(IntrinsicInst &I);
  bool visitPtrSBufferLoad(IntrinsicInst &I);
  bool visitMonitorSleep(IntrinsicInst &I);
  bool visitCvtScale(IntrinsicInst &I);
  MatrixIntrinsicNarrowingResult narrowF8F6F4MatrixOperands(IntrinsicInst &I);
  bool visitUnscaledWMMA(IntrinsicInst &I);
};

class AMDGPULowerIntrinsicsLegacy : public ModulePass {
public:
  static char ID;

  AMDGPULowerIntrinsicsLegacy() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

template <class T> static void forEachCall(Function &Intrin, T Callback) {
  for (User *U : make_early_inc_range(Intrin.users())) {
    if (auto *CI = dyn_cast<IntrinsicInst>(U))
      Callback(CI);
  }
}

} // anonymous namespace

bool AMDGPULowerIntrinsicsImpl::run() {
  bool Changed = false;

  for (Function &F : M) {
    switch (F.getIntrinsicID()) {
    default:
      continue;
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::amdgcn_s_barrier_signal:
    case Intrinsic::amdgcn_s_barrier_signal_isfirst:
    case Intrinsic::amdgcn_s_barrier_wait:
    case Intrinsic::amdgcn_s_cluster_barrier:
      forEachCall(F, [&](IntrinsicInst *II) { Changed |= visitBarrier(*II); });
      break;
    case Intrinsic::amdgcn_ptr_s_buffer_load:
      forEachCall(
          F, [&](IntrinsicInst *II) { Changed |= visitPtrSBufferLoad(*II); });
      break;
    case Intrinsic::amdgcn_s_monitor_sleep:
      forEachCall(
          F, [&](IntrinsicInst *II) { Changed |= visitMonitorSleep(*II); });
      break;
    case Intrinsic::amdgcn_cvt_scale_pk8_f16_fp8:
    case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp8:
    case Intrinsic::amdgcn_cvt_scale_pk8_f16_bf8:
    case Intrinsic::amdgcn_cvt_scale_pk8_bf16_bf8:
    case Intrinsic::amdgcn_cvt_scale_pk8_f16_fp4:
    case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4:
    case Intrinsic::amdgcn_cvt_scale_pk8_f32_fp8:
    case Intrinsic::amdgcn_cvt_scale_pk8_f32_bf8:
    case Intrinsic::amdgcn_cvt_scale_pk8_f32_fp4:
    case Intrinsic::amdgcn_cvt_scale_pk16_f16_bf6:
    case Intrinsic::amdgcn_cvt_scale_pk16_bf16_bf6:
    case Intrinsic::amdgcn_cvt_scale_pk16_f16_fp6:
    case Intrinsic::amdgcn_cvt_scale_pk16_bf16_fp6:
    case Intrinsic::amdgcn_cvt_scale_pk16_f32_fp6:
    case Intrinsic::amdgcn_cvt_scale_pk16_f32_bf6:
      forEachCall(F, [&](IntrinsicInst *II) { Changed |= visitCvtScale(*II); });
      break;
    case Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4:
      forEachCall(F, [&](IntrinsicInst *II) {
        MatrixIntrinsicNarrowingResult Result = narrowF8F6F4MatrixOperands(*II);
        Changed |= Result.Changed;
        Changed |= visitUnscaledWMMA(*Result.Current);
      });
      break;
    case Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
    case Intrinsic::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
    case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
    case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
      forEachCall(F, [&](IntrinsicInst *II) {
        Changed |= narrowF8F6F4MatrixOperands(*II).Changed;
      });
      break;
    }
  }

  return Changed;
}

// Optimize barriers and lower s_(cluster_)barrier to a sequence of split
// barrier intrinsics.
bool AMDGPULowerIntrinsicsImpl::visitBarrier(IntrinsicInst &I) {
  assert(I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal_isfirst ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_wait ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_cluster_barrier);

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*I.getFunction());
  bool IsSingleWaveWG = false;

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    IsSingleWaveWG = ST.isSingleWavefrontWorkgroup(*I.getFunction());

  IRBuilder<> B(&I);

  // Lower the s_cluster_barrier intrinsic first. There is no corresponding
  // hardware instruction in any subtarget.
  if (I.getIntrinsicID() == Intrinsic::amdgcn_s_cluster_barrier) {
    // The default cluster barrier expects one signal per workgroup. So we need
    // a workgroup barrier first.
    if (IsSingleWaveWG) {
      B.CreateIntrinsicWithoutFolding(B.getVoidTy(),
                                      Intrinsic::amdgcn_wave_barrier, {})
          ->copyMetadata(I);
    } else {
      Value *BarrierID_32 = B.getInt32(AMDGPU::Barrier::WORKGROUP);
      Value *BarrierID_16 = B.getInt16(AMDGPU::Barrier::WORKGROUP);
      CallInst *IsFirst = B.CreateIntrinsicWithoutFolding(
          B.getInt1Ty(), Intrinsic::amdgcn_s_barrier_signal_isfirst,
          {BarrierID_32});
      IsFirst->copyMetadata(I);
      B.CreateIntrinsicWithoutFolding(
           B.getVoidTy(), Intrinsic::amdgcn_s_barrier_wait, {BarrierID_16})
          ->copyMetadata(I);

      Instruction *ThenTerm =
          SplitBlockAndInsertIfThen(IsFirst, I.getIterator(), false);
      B.SetInsertPoint(ThenTerm);
    }

    // Now we can signal the cluster barrier from a single wave and wait for the
    // barrier in all waves.
    Value *BarrierID_32 = B.getInt32(AMDGPU::Barrier::CLUSTER);
    Value *BarrierID_16 = B.getInt16(AMDGPU::Barrier::CLUSTER);
    B.CreateIntrinsicWithoutFolding(
         B.getVoidTy(), Intrinsic::amdgcn_s_barrier_signal, {BarrierID_32})
        ->copyMetadata(I);

    B.SetInsertPoint(&I);
    B.CreateIntrinsicWithoutFolding(
         B.getVoidTy(), Intrinsic::amdgcn_s_barrier_wait, {BarrierID_16})
        ->copyMetadata(I);

    I.eraseFromParent();
    return true;
  }

  bool IsWorkgroupScope = false;

  if (I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_wait ||
      I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal ||
      I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal_isfirst) {
    int BarrierID = cast<ConstantInt>(I.getArgOperand(0))->getSExtValue();
    if (BarrierID == AMDGPU::Barrier::TRAP ||
        BarrierID == AMDGPU::Barrier::WORKGROUP ||
        (BarrierID >= AMDGPU::Barrier::NAMED_BARRIER_FIRST &&
         BarrierID <= AMDGPU::Barrier::NAMED_BARRIER_LAST))
      IsWorkgroupScope = true;
    else if (I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal_isfirst &&
             BarrierID == AMDGPU::Barrier::CLUSTER) {
      I.getContext().diagnose(
          DiagnosticInfoUnsupported(*I.getFunction(),
                                    "s_barrier_signal_isfirst does not support "
                                    "user_cluster_barrier_id (-3)",
                                    I.getDebugLoc()));
    }
  } else {
    assert(I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier);
    IsWorkgroupScope = true;
  }

  if (IsWorkgroupScope && IsSingleWaveWG) {
    // Down-grade waits, remove split signals.
    if (I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier ||
        I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_wait) {
      B.CreateIntrinsicWithoutFolding(B.getVoidTy(),
                                      Intrinsic::amdgcn_wave_barrier, {})
          ->copyMetadata(I);
    } else if (I.getIntrinsicID() ==
               Intrinsic::amdgcn_s_barrier_signal_isfirst) {
      // If we're the only wave of the workgroup, we're always first.
      I.replaceAllUsesWith(B.getInt1(true));
    }
    I.eraseFromParent();
    return true;
  }

  if (I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier &&
      ST.hasSplitBarriers()) {
    // Lower to split barriers.
    Value *BarrierID_32 = B.getInt32(AMDGPU::Barrier::WORKGROUP);
    Value *BarrierID_16 = B.getInt16(AMDGPU::Barrier::WORKGROUP);
    B.CreateIntrinsicWithoutFolding(
         B.getVoidTy(), Intrinsic::amdgcn_s_barrier_signal, {BarrierID_32})
        ->copyMetadata(I);
    B.CreateIntrinsicWithoutFolding(
         B.getVoidTy(), Intrinsic::amdgcn_s_barrier_wait, {BarrierID_16})
        ->copyMetadata(I);
    I.eraseFromParent();
    return true;
  }

  return false;
}

bool AMDGPULowerIntrinsicsImpl::visitPtrSBufferLoad(IntrinsicInst &I) {
  assert(I.getIntrinsicID() == Intrinsic::amdgcn_ptr_s_buffer_load);

  if (I.hasMetadata(LLVMContext::MD_invariant_load))
    return false;

  I.setMetadata(LLVMContext::MD_invariant_load,
                MDNode::get(I.getContext(), {}));
  return true;
}

bool AMDGPULowerIntrinsicsImpl::visitMonitorSleep(IntrinsicInst &I) {
  assert(I.getIntrinsicID() == Intrinsic::amdgcn_s_monitor_sleep);

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*I.getFunction());
  if (!ST.hasNoSleepForever())
    return false;

  int Sleep = cast<ConstantInt>(I.getArgOperand(0))->getSExtValue();
  if (!(Sleep & 0x8000))
    return false;

  IRBuilder<> B(&I);
  Value *NewSleep = B.getInt16(127); // Maximum
  I.setArgOperand(0, NewSleep);

  return true;
}

bool AMDGPULowerIntrinsicsImpl::visitCvtScale(IntrinsicInst &I) {
  int MaxSel = 0;
  switch (I.getIntrinsicID()) {
  default:
    llvm_unreachable("expected cvt_scale_* intrinsic");
  case Intrinsic::amdgcn_cvt_scale_pk8_f16_fp8:
  case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp8:
  case Intrinsic::amdgcn_cvt_scale_pk8_f16_bf8:
  case Intrinsic::amdgcn_cvt_scale_pk8_bf16_bf8:
  case Intrinsic::amdgcn_cvt_scale_pk8_f32_fp8:
  case Intrinsic::amdgcn_cvt_scale_pk8_f32_bf8:
    MaxSel = 8;
    break;
  case Intrinsic::amdgcn_cvt_scale_pk8_f16_fp4:
  case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4:
  case Intrinsic::amdgcn_cvt_scale_pk8_f32_fp4:
  case Intrinsic::amdgcn_cvt_scale_pk16_f16_bf6:
  case Intrinsic::amdgcn_cvt_scale_pk16_bf16_bf6:
  case Intrinsic::amdgcn_cvt_scale_pk16_f16_fp6:
  case Intrinsic::amdgcn_cvt_scale_pk16_bf16_fp6:
  case Intrinsic::amdgcn_cvt_scale_pk16_f32_fp6:
  case Intrinsic::amdgcn_cvt_scale_pk16_f32_bf6:
    MaxSel = 4;
    break;
  }

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*I.getFunction());
  if (ST.hasBlock16ConversionScaleInsts())
    MaxSel *= 2;

  int ScaleSel = cast<ConstantInt>(I.getArgOperand(2))->getSExtValue();
  if (ScaleSel < MaxSel)
    return false;

  I.getContext().diagnose(DiagnosticInfoUnsupported(
      *I.getFunction(),
      I.getCalledFunction()->getName() +
          Twine(" scale_sel maximum supported value is ") + Twine(MaxSel - 1),
      I.getDebugLoc()));

  return false;
}

MatrixIntrinsicNarrowingResult
AMDGPULowerIntrinsicsImpl::narrowF8F6F4MatrixOperands(IntrinsicInst &I) {
  unsigned Src0Idx;
  unsigned Src1Idx;
  unsigned Fmt0Idx;
  unsigned Fmt1Idx;
  bool IsWMMA;

  switch (I.getIntrinsicID()) {
  default:
    llvm_unreachable("expected an f8f6f4 matrix intrinsic");
  case Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
    Src0Idx = 1;
    Src1Idx = 3;
    Fmt0Idx = 0;
    Fmt1Idx = 2;
    IsWMMA = true;
    break;
  case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
    Src0Idx = 0;
    Src1Idx = 1;
    Fmt0Idx = 3;
    Fmt1Idx = 4;
    IsWMMA = false;
    break;
  }

  auto *Fmt0 = dyn_cast<ConstantInt>(I.getArgOperand(Fmt0Idx));
  auto *Fmt1 = dyn_cast<ConstantInt>(I.getArgOperand(Fmt1Idx));
  if (!Fmt0 || !Fmt1)
    return {&I, false};

  uint64_t Fmt0Val = Fmt0->getZExtValue();
  uint64_t Fmt1Val = Fmt1->getZExtValue();
  unsigned MaxFmt =
      IsWMMA ? static_cast<unsigned>(AMDGPU::WMMA::MATRIX_FMT_FP4)
             : static_cast<unsigned>(AMDGPU::MFMAScaleFormats::FP4_E2M1);
  if (Fmt0Val > MaxFmt || Fmt1Val > MaxFmt)
    return {&I, false};

  Value *Src0 = I.getArgOperand(Src0Idx);
  Value *Src1 = I.getArgOperand(Src1Idx);
  auto *Src0Ty = dyn_cast<FixedVectorType>(Src0->getType());
  auto *Src1Ty = dyn_cast<FixedVectorType>(Src1->getType());
  auto IsLegalSourceType = [IsWMMA](FixedVectorType *Ty) {
    if (!Ty || !Ty->getElementType()->isIntegerTy(32))
      return false;

    unsigned NumElts = Ty->getNumElements();
    if (IsWMMA)
      return NumElts == 8 || NumElts == 12 || NumElts == 16;
    return NumElts == 4 || NumElts == 6 || NumElts == 8;
  };
  if (!IsLegalSourceType(Src0Ty) || !IsLegalSourceType(Src1Ty))
    return {&I, false};

  unsigned Src0NumElts = IsWMMA
                             ? AMDGPU::wmmaScaleF8F6F4FormatToNumRegs(Fmt0Val)
                             : AMDGPU::mfmaScaleF8F6F4FormatToNumRegs(Fmt0Val);
  unsigned Src1NumElts = IsWMMA
                             ? AMDGPU::wmmaScaleF8F6F4FormatToNumRegs(Fmt1Val)
                             : AMDGPU::mfmaScaleF8F6F4FormatToNumRegs(Fmt1Val);
  if (Src0Ty->getNumElements() < Src0NumElts ||
      Src1Ty->getNumElements() < Src1NumElts)
    return {&I, false};

  IRBuilder<> B(&I);
  B.SetCurrentDebugLocation(I.getDebugLoc());
  auto Narrow = [&B](Value *Src, FixedVectorType *SrcTy,
                     unsigned NumElts) -> Value * {
    if (SrcTy->getNumElements() == NumElts)
      return Src;

    SmallVector<int, 16> Mask;
    for (unsigned Idx = 0; Idx != NumElts; ++Idx)
      Mask.push_back(Idx);
    return B.CreateShuffleVector(Src, Mask);
  };

  Value *NarrowedSrc0 = Narrow(Src0, Src0Ty, Src0NumElts);
  Value *NarrowedSrc1 = Narrow(Src1, Src1Ty, Src1NumElts);
  if (NarrowedSrc0 == Src0 && NarrowedSrc1 == Src1)
    return {&I, false};

  SmallVector<Value *, 14> Args(I.args());
  Args[Src0Idx] = NarrowedSrc0;
  Args[Src1Idx] = NarrowedSrc1;

  SmallVector<Type *, 3> OverloadTypes;
  if (IsWMMA)
    OverloadTypes.push_back(I.getType());
  OverloadTypes.push_back(NarrowedSrc0->getType());
  OverloadTypes.push_back(NarrowedSrc1->getType());

  SmallVector<OperandBundleDef, 2> OpBundles;
  I.getOperandBundlesAsDefs(OpBundles);
  CallInst *NewCall = B.CreateIntrinsicWithoutFolding(
      I.getIntrinsicID(), OverloadTypes, Args, &I, "", OpBundles);
  auto *NewI = cast<IntrinsicInst>(NewCall);
  NewI->copyMetadata(I);
  NewI->setAttributes(I.getAttributes());
  NewI->setTailCallKind(I.getTailCallKind());
  NewI->setCallingConv(I.getCallingConv());
  NewI->takeName(&I);
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  return {NewI, true};
}

bool AMDGPULowerIntrinsicsImpl::visitUnscaledWMMA(IntrinsicInst &I) {
  assert(I.getIntrinsicID() == Intrinsic::amdgcn_wmma_f32_16x16x128_f8f6f4);

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*I.getFunction());
  if (!ST.hasGFX1250_STRICT())
    return false;

  IRBuilder<> B(&I);

  // Convert unscaled WMMA into a scaled version with scale factors 0.
  SmallVector<Value *, 14> Args(I.args());
  Args.push_back(B.getInt32(0)); // matrix_a_scale
  Args.push_back(
      B.getInt32(AMDGPU::WMMA::MATRIX_SCALE_FMT_E8)); // matrix_a_scale_fmt
  Args.push_back(B.getInt32(0)); // matrix a scale exponential
  Args.push_back(B.getInt32(0)); // matrix_b_scale
  Args.push_back(
      B.getInt32(AMDGPU::WMMA::MATRIX_SCALE_FMT_E8)); // matrix_b_scale_fmt
  Args.push_back(B.getInt32(0)); // matrix b scale exponential
  Args.push_back(B.getInt1(0));  // matrix_a_reuse
  Args.push_back(B.getInt1(0));  // matrix_b_reuse

  CallInst *NewI = B.CreateIntrinsicWithoutFolding(
      I.getType(), Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4, Args);
  I.replaceAllUsesWith(NewI);
  NewI->copyMetadata(I);
  NewI->takeName(&I);
  I.eraseFromParent();

  return true;
}

PreservedAnalyses AMDGPULowerIntrinsicsPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  AMDGPULowerIntrinsicsImpl Impl(M, TM);
  if (!Impl.run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

bool AMDGPULowerIntrinsicsLegacy::runOnModule(Module &M) {
  auto &TPC = getAnalysis<TargetPassConfig>();
  const AMDGPUTargetMachine &TM = TPC.getTM<AMDGPUTargetMachine>();

  AMDGPULowerIntrinsicsImpl Impl(M, TM);
  return Impl.run();
}

#define PASS_DESC "AMDGPU lower intrinsics"
INITIALIZE_PASS_BEGIN(AMDGPULowerIntrinsicsLegacy, DEBUG_TYPE, PASS_DESC, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPULowerIntrinsicsLegacy, DEBUG_TYPE, PASS_DESC, false,
                    false)

char AMDGPULowerIntrinsicsLegacy::ID = 0;

ModulePass *llvm::createAMDGPULowerIntrinsicsLegacyPass() {
  return new AMDGPULowerIntrinsicsLegacy;
}
