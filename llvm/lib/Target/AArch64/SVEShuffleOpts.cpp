//===------- SVEShuffleOpts - SVE Shuffle Optimization --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tries to pattern match and combine scalable vector shuffles that could
// be more efficiently performed by tbl instructions.
//
// An example would be a loop with 4 multiply-accumulate reductions, where the
// new data in each vector iterations comes from a 4-way deinterleaving of
// smaller datatypes loaded from memory,  which are then extended and multiplied
// by a common term loaded in reverse order from memory before being added to
// the accumulator.
//
// If the initial load is a legal vector rather than 4x the size (generating a
// structured ld4 instead), we would see multiple uunpkhi/lo instructions for
// the extensions, followed by uzp1/2 instructions for the deinterleave, and rev
// instructions for the common terms. Instead, we can replace all of those with
// 4 tbl instructions. The tradeoff, of course, is that we now have 4 mask
// values to maintain which increases register pressure.
//
// We should also be able to introduce new shuffles in order to balance out
// SVE's bottom/top instruction pairs, which act on even/odd lanes instead of
// the high or low half of a register.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "aarch64-sve-shuffle-opts"

namespace {

class SVEShuffleImpl {
  const AArch64TargetMachine *TM = nullptr;
  const LoopInfo *LI = nullptr;

public:
  SVEShuffleImpl() {};
  SVEShuffleImpl(const AArch64TargetMachine *TM) : TM(TM) {};

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
  bool runOnFunction(Function &F, Pass &P);

private:
  bool processLoop(Loop &L);
};

struct SVEShuffleOpts : public FunctionPass {
  SVEShuffleImpl Impl;
  static char ID; // Pass identification, replacement for typeid
  SVEShuffleOpts() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    return Impl.runOnFunction(F, *this);
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  StringRef getPassName() const override { return "SVE Tbl Folding Opts"; }

private:
};
} // end anonymous namespace

/// A mapping between a vector_deinterleaveN intrinsic and extending cast
/// instructions used on the resulting subvectors.
using DeinterleaveMap =
    SmallDenseMap<CallInst *, SmallVector<std::pair<CastInst *, unsigned>, 4>>;

static void evaluateDeinterleave(IntrinsicInst *I, DeinterleaveMap &Candidates,
                                 Loop &L) {
  // TODO: 'Legalize' if the input is wider than nx128b but not wide enough
  //       to match a structured load?
  if (I->getOperand(0)
          ->getType()
          ->getPrimitiveSizeInBits()
          .getKnownMinValue() != AArch64::SVEBitsPerBlock)
    return;

  unsigned IntId = I->getIntrinsicID();
  assert(IntId == Intrinsic::vector_deinterleave4 &&
         "Only deinterleave4 supported currently");
  SmallVector<std::pair<CastInst *, unsigned>, 4> Extends;
  unsigned Opcode = 0;
  Type *DestTy = nullptr;
  for (User *U : I->users()) {
    auto *Extract = dyn_cast<ExtractValueInst>(U);
    // We expect only a single cast instruction as a user.
    if (!Extract || Extract->getNumIndices() != 1)
      return;

    auto *Extend = dyn_cast<CastInst>(Extract->getUniqueUndroppableUser());
    if (!Extend || (!isa<ZExtInst>(Extend) && !isa<UIToFPInst>(Extend)))
      return;

    // We're only interested if the uses are in the loop. This is almost
    // certainly the case.
    if (!L.contains(Extract) || !L.contains(Extend))
      return;

    Opcode = Extend->getOpcode();
    DestTy = Extend->getDestTy();
    Type *SrcTy = Extend->getSrcTy();

    // For now, we only want to handle scalable vectors here.
    if (!DestTy->isScalableTy())
      return;

    unsigned SrcBits = SrcTy->getScalarSizeInBits();
    unsigned DestBits = DestTy->getScalarSizeInBits();

    // Looking to match the deinterleave factor.
    if (DestBits / SrcBits != 4)
      return;

    // We can't abuse the invalid index trick for tbls of bytes, since the
    // largest possible SVE vector (2048b) would have 256 bytes, leaving no
    // way of zeroing.
    // TODO: If we know vscale is 8 or less, then we could use tbls for bytes.
    if (SrcBits <= 8)
      return;

    Extends.push_back({Extend, Extract->getIndices()[0]});
  }

  // Check that all extracted values are being extended the same way, and that
  // we have the expected number of extensions.
  if (Extends.size() != 4 ||
      !all_of(Extends, [&](std::pair<CastInst *, unsigned> Ext) {
        CastInst *CI = Ext.first;
        return CI->getDestTy() == DestTy && CI->getOpcode() == Opcode;
      }))
    return;

  Candidates.insert({I, Extends});
}

// Optimize zext and uitofp from a 4-way deinterleaved load.
static void optimizeSVEDeinterleavedExtends(DeinterleaveMap Deinterleaves) {
  // TODO: Cache tbl patterns and reuse, and abandon transforms for a particular
  //       deinterleave if it would introduce too many. We probably want a
  //       hardcoded number of tbls to start with, but if we can estimate
  //       register pressure then we could make better decisions.
  for (auto &[Deinterleave, Extends] : Deinterleaves) {
    VectorType *DestTy = cast<VectorType>(Extends[0].first->getDestTy());
    VectorType *SrcTy = cast<VectorType>(Extends[0].first->getSrcTy());
    unsigned DstBits = DestTy->getScalarSizeInBits();
    unsigned SrcBits = SrcTy->getScalarSizeInBits();
    bool IsUIToFP = isa<UIToFPInst>(Extends[0].first);
    VectorType *StepVecTy = VectorType::getInteger(DestTy);
    Type *StepTy = StepVecTy->getScalarType();
    Value *Input = Deinterleave->getOperand(0);
    Type *InputTy = Input->getType();

    APInt Invalid = APInt::getAllOnes(DstBits);
    for (auto &[Extend, Idx] : Extends) {
      // Build mask
      APInt StartIdx = Invalid << SrcBits;
      StartIdx += Idx;
      IRBuilder<> Builder(Extend);
      Value *StepVector = Builder.CreateStepVector(StepVecTy);
      Value *ScaledSteps = Builder.CreateMul(
          StepVector, Builder.CreateVectorSplat(StepVecTy->getElementCount(),
                                                ConstantInt::get(StepTy, 4)));
      Value *Start = ConstantInt::get(StepTy, StartIdx);
      Value *ZextTbl = Builder.CreateAdd(
          ScaledSteps,
          Builder.CreateVectorSplat(StepVecTy->getElementCount(), Start));
      Value *FinalMask = Builder.CreateBitCast(ZextTbl, InputTy);

      // Replace the deinterleave, extractvalue, and extension chain with
      // a tbl directly on the input value.
      Value *Tbl = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_tbl,
                                           {InputTy}, {Input, FinalMask});
      Value *Widen = Builder.CreateBitCast(Tbl, StepVecTy);
      if (IsUIToFP)
        Widen = Builder.CreateUIToFP(Widen, DestTy);
      LLVM_DEBUG(dbgs() << "SVETBLOPT: Replaced " << *Extend << " with "
                        << *Widen << "\n");
      Extend->replaceAllUsesWith(Widen);
      Extend->eraseFromParent();
    }
  }
}

bool SVEShuffleImpl::processLoop(Loop &L) {
  // TODO: Pull other shuffles into the tbl where possible.
  // TODO: Add more advanced cases, such as introducing shuffles so that
  //       the SVE odd/even BT narrowing instructions can be used.
  // TODO: Support other deinterleaves.
  DeinterleaveMap Candidates;
  for (auto *BB : L.blocks())
    for (auto &I : *BB)
      if (match(&I, m_Intrinsic<Intrinsic::vector_deinterleave4>(m_Value())))
        evaluateDeinterleave(cast<IntrinsicInst>(&I), Candidates, L);

  if (Candidates.empty())
    return false;

  optimizeSVEDeinterleavedExtends(Candidates);
  return true;
}

void SVEShuffleOpts::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
}

char SVEShuffleOpts::ID = 0;
static const char *name = "SVE VLA shuffle optimizations";
INITIALIZE_PASS_BEGIN(SVEShuffleOpts, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(SVEShuffleOpts, DEBUG_TYPE, name, false, false)

FunctionPass *llvm::createSVEShuffleOptsPass() { return new SVEShuffleOpts(); }

namespace llvm {
class SVEShuffleOptsPass : public PassInfoMixin<SVEShuffleOptsPass> {
  const AArch64TargetMachine *TM;

public:
  explicit SVEShuffleOptsPass(const AArch64TargetMachine &TM) : TM(&TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    SVEShuffleImpl Impl(TM);
    return Impl.run(F, FAM);
  }
};
} // end namespace llvm

bool SVEShuffleImpl::runOnFunction(Function &F, Pass &P) {
  // Make sure we can use SVE
  TargetPassConfig &TPC = P.getAnalysis<TargetPassConfig>();
  TM = &TPC.getTM<AArch64TargetMachine>();
  const AArch64Subtarget *ST = TM->getSubtargetImpl(F);
  if (!ST->isSVEorStreamingSVEAvailable())
    return false;

  LI = &P.getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  bool Changed = false;
  // Only looking to tranform innermost loops, given the increase in
  // register usage.
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->isInnermost())
      Changed |= processLoop(*L);
  }

  return Changed;
}

PreservedAnalyses SVEShuffleImpl::run(Function &F,
                                      FunctionAnalysisManager &FAM) {
  const AArch64Subtarget *ST = TM->getSubtargetImpl(F);
  if (!ST->isSVEorStreamingSVEAvailable())
    return PreservedAnalyses::all();

  LI = &FAM.getResult<LoopAnalysis>(F);

  bool Changed = false;
  // Only looking to tranform innermost loops, given the increase in
  // register usage.
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->isInnermost())
      Changed |= processLoop(*L);
  }

  // Can we do better than 'none'?
  // We're not actually using the new pass manager though.
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
