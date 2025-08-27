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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-lower-intrinsics"

using namespace llvm;

namespace {

class AMDGPULowerIntrinsicsImpl {
public:
  Module &M;
  const AMDGPUTargetMachine &TM;

  AMDGPULowerIntrinsicsImpl(Module &M, const AMDGPUTargetMachine &TM)
      : M(M), TM(TM) {}

  bool run();

private:
  bool visitBarrier(IntrinsicInst &I);
};

class AMDGPULowerIntrinsicsLegacy : public ModulePass {
public:
  static char ID;

  AMDGPULowerIntrinsicsLegacy() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
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
      forEachCall(F, [&](IntrinsicInst *II) { Changed |= visitBarrier(*II); });
      break;
    }
  }

  return Changed;
}

// Optimize barriers and lower s_barrier to a sequence of split barrier
// intrinsics.
bool AMDGPULowerIntrinsicsImpl::visitBarrier(IntrinsicInst &I) {
  assert(I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_signal_isfirst ||
         I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_wait);

  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(*I.getFunction());
  bool IsSingleWaveWG = false;

  if (TM.getOptLevel() > CodeGenOptLevel::None) {
    unsigned WGMaxSize = ST.getFlatWorkGroupSizes(*I.getFunction()).second;
    IsSingleWaveWG = WGMaxSize <= ST.getWavefrontSize();
  }

  IRBuilder<> B(&I);

  if (IsSingleWaveWG) {
    // Down-grade waits, remove split signals.
    if (I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier ||
        I.getIntrinsicID() == Intrinsic::amdgcn_s_barrier_wait) {
      B.CreateIntrinsic(B.getVoidTy(), Intrinsic::amdgcn_wave_barrier, {});
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
    B.CreateIntrinsic(B.getVoidTy(), Intrinsic::amdgcn_s_barrier_signal,
                      {BarrierID_32});
    B.CreateIntrinsic(B.getVoidTy(), Intrinsic::amdgcn_s_barrier_wait,
                      {BarrierID_16});
    I.eraseFromParent();
    return true;
  }

  return false;
}

PreservedAnalyses AMDGPULowerIntrinsicsPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  AMDGPULowerIntrinsicsImpl Impl(M, TM);
  if (!Impl.run())
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
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
