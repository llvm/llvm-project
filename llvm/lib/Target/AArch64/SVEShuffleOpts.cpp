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
// smaller datatypes loaded from memory which are then zero extended.
//
// Something like the following:
//   %bgra = call ... @llvm.masked.load
//   %deinterleave = call ... @llvm.vector.deinterleave4(%bgra)
// If the load was of a <vscale x 8 x i16>, we now have 4 deinterleaved
// <vscale x 2 x i16> values.
//   %b.i16 = extractvalue %deinterleave, 0
//   %b.i64 = zext <vscale x 2 x i16> %b.i16 to <vscale x 2 x i64>
//   %acc.b.next = add <vscale x 2 x i64> %acc.b, %b.i64
//   <repeat for the other 3 subvectors>
//
// If the initial load is a legal vector rather than 4x the size (generating a
// structured ld4 instead), we would see multiple uunpkhi/lo instructions for
// the extensions, followed by uzp1/2 instructions for the deinterleave.
// Instead, we can replace all of those with 4 tbl instructions. The tradeoff,
// of course, is that we now have 4 mask values to maintain which may increase
// register pressure.
//
// This basic transform could be performed in CodeGenPrepare (as the equivalent
// for NEON is), or in a DAG Combine. However, we hope to extend it to detect
// other shuffles that we can fold into the tbl. Extending the above example,
// if instead of directly adding to the accumulator we multiplied it by a
// common term for all 4 components that had been reversed:
//   %common.load = call @llvm.masked.load
//   %common.reverse = call @llvm.vector.reverse
// These would be loaded at the extended size, <vscale x 2 x i64> in our
// example.
//   %b.mul = mul <vscale x 2 x i64> %b.i64, %common.reverse
//   %acc.b.next = add <vscale x 2 x i64> %acc.b, %b.mul
//   <repeat for the other 3 subvectors, using %common.reverse for each)
//
// In this case, the reverse isn't applied to the deinterleaved data in the
// original IR, but to the common term multiplied by the individual bgra
// elements. If the order of the elements in the accumulator is important, we
// cannot change that. If, however, we know that the accumulator is reduced to
// a single scalar after the loop and the data is either integers or floating
// point with reassociation allowed, we could instead choose a different mask
// for the tbls to reverse the individual bgra elements instead, removing an
// additional instruction from the loop. This does require looking beyond the
// blocks in the loop, so DAGCombine won't help.
//
// We should also be able to introduce new shuffles in order to balance out
// SVE's bottom/top instruction pairs, which act on even/odd lanes instead of
// the high or low half of a register.
//
// This pass may end up being a temporary solution that is removed if we can
// create a generic vector shuffle intrinsic and move this feature to
// LoopVectorize itself, as that would allow for better cost modelling.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include <array>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "aarch64-sve-shuffle-opts"

/// A mapping between a vector_deinterleaveN intrinsic and extending cast
/// instructions used on the resulting subvectors.
using DeinterleaveMap = SmallDenseMap<CallInst *, std::array<CastInst *, 4>>;

/// Evaluate a deinterleave and see what the uses are. If we find other
/// operations that we can combine into a tbl shuffle, add the deinterleave and
/// the operations (currently only zext or uitofp) to the candidates map.
static void evaluateDeinterleave(IntrinsicInst *I, DeinterleaveMap &Candidates,
                                 Loop &L, const AArch64TargetLowering &TL,
                                 const DataLayout DL) {
  assert(I->getIntrinsicID() == Intrinsic::vector_deinterleave4 &&
         "Only deinterleave4 supported currently");

  ConstantRange VScaleRange = getVScaleRange(I->getFunction(), 64);
  // TBL zeroes elements with an out-of-bounds index, but for the largest
  // possible SVE vector (2048b) the maximum value for i8 elements (255) is not
  // large enough to encode an 'out of bounds' value. So we can only perform
  // this optimization for i8 elements if we know vscale is < 16.
  EVT InputVT = TL.getValueType(DL, I->getOperand(0)->getType());
  if (!InputVT.isScalableVector() ||
      (InputVT.getScalarSizeInBits() < 16 &&
       (!VScaleRange.getUpper().ult(16) || VScaleRange.isUpperWrapped())) ||
      TL.getTypeConversion(I->getContext(), InputVT).first !=
          TargetLoweringBase::TypeLegal)
    return;

  std::array<CastInst *, 4> Extends = {};
  unsigned Opcode = 0;
  Type *DestTy = nullptr;
  for (User *U : I->users()) {
    auto *Extract = dyn_cast<ExtractValueInst>(U);
    if (!Extract || !Extract->hasOneUse())
      return;

    // We expect only a single cast instruction as a user for the extract.
    auto *Extend = dyn_cast_if_present<CastInst>(*Extract->users().begin());
    if (!Extend || (!isa<ZExtInst>(Extend) && !isa<UIToFPInst>(Extend)))
      return;

    // We're only interested if the uses are in the loop. This is almost
    // certainly the case.
    if (!L.contains(Extend))
      return;

    Opcode = Extend->getOpcode();
    DestTy = Extend->getDestTy();

    // Make sure DestTy matches the input size.
    if (DestTy->getPrimitiveSizeInBits() != InputVT.getSizeInBits())
      return;

    Extends[Extract->getIndices().front()] = Extend;
  }

  // Check that all extracted values are being extended the same way, and that
  // we have the expected number of extensions.
  if (!all_of(Extends, [DestTy, Opcode](CastInst *CI) {
        return !CI || (CI->getDestTy() == DestTy && CI->getOpcode() == Opcode);
      }))
    return;

  Candidates.try_emplace(I, Extends);
}

/// Given a map of deinterleaves to zext or uitofp casts, remove the operations
/// and replace them with tbl shuffles.
static void optimizeSVEDeinterleavedExtends(DeinterleaveMap Deinterleaves) {
  for (auto &[Deinterleave, Extends] : Deinterleaves) {
    VectorType *DestTy = cast<VectorType>(Extends[0]->getDestTy());
    VectorType *SrcTy = cast<VectorType>(Extends[0]->getSrcTy());
    unsigned DstBits = DestTy->getScalarSizeInBits();
    unsigned SrcBits = SrcTy->getScalarSizeInBits();
    bool IsUIToFP = isa<UIToFPInst>(Extends[0]);
    VectorType *StepVecTy = VectorType::getInteger(DestTy);
    Value *Input = Deinterleave->getOperand(0);
    Type *InputTy = Input->getType();

    APInt Invalid = APInt::getAllOnes(DstBits);
    for (auto [Idx, Extend] : enumerate(Extends)) {
      // If not all lanes were extracted, we can have gaps. Skip over them.
      if (!Extend)
        continue;
      // Build the mask using stepvectors and casting.
      // We want to select the Idx'th element, and every 4 elements after that.
      // Each element needs to be zero extended; we can do that by providing
      // tbl index values that are out of range. We can't do that nicely with
      // a stepvector of the same element type as the input type, but we can
      // do it with elements the size of the output type.
      // E.g. for element 0 of a 16b -> 64b zext, we would start with a mask of
      // 0xFFFF_FFFF_FFFF_0000 + Idx for the start of the stepvector, and use a
      // step of 4. We then cast that back to an element size of 16b, yielding
      // <0x0000 + Idx, 0xFFFF, 0xFFFF, 0xFFFF, 0x0004 + Idx, 0xFFFF...>.
      APInt StartIdx = Invalid << SrcBits;
      StartIdx += Idx;
      IRBuilder<> Builder(Extend);
      Value *StepVector = Builder.CreateStepVector(StepVecTy);
      Value *ScaledSteps =
          Builder.CreateNUWMul(StepVector, ConstantInt::get(StepVecTy, 4));
      Value *ZextTbl = Builder.CreateNUWAdd(
          ScaledSteps, ConstantInt::get(StepVecTy, StartIdx));
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

    // Delete the unused extracts and deinterleave.
    for (User *U : make_early_inc_range(Deinterleave->users()))
      cast<Instruction>(U)->eraseFromParent();
    Deinterleave->eraseFromParent();
  }
}

static bool processLoop(Loop &L, const AArch64Subtarget &ST, DataLayout DL) {
  // At present, we only want to do this for innermost loops when SVE
  // is available.
  if (!L.isInnermost() || !ST.isSVEorStreamingSVEAvailable())
    return false;

  // TODO: Pull other shuffles into the tbl where possible.
  // TODO: Add more advanced cases, such as introducing shuffles so that
  //       the SVE odd/even BT narrowing instructions can be used.
  // TODO: Support other deinterleaves.
  const AArch64TargetLowering &TL = *ST.getTargetLowering();
  assert(DL.isLittleEndian() &&
         "Shuffle optimizations unsupported for big endian targets.");
  DeinterleaveMap Candidates;
  for (auto *BB : L.blocks())
    for (auto &I : *BB)
      if (match(&I, m_Intrinsic<Intrinsic::vector_deinterleave4>(m_Value())))
        evaluateDeinterleave(cast<IntrinsicInst>(&I), Candidates, L, TL, DL);

  if (Candidates.empty())
    return false;

  optimizeSVEDeinterleavedExtends(Candidates);
  return true;
}

namespace {
struct SVEShuffleOpts : public LoopPass {
  static char ID; // Pass identification, replacement for typeid
  SVEShuffleOpts() : LoopPass(ID) {}

  bool runOnLoop(Loop *L, LPPassManager &PM) override {
    if (skipLoop(L))
      return false;

    TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    const AArch64TargetMachine &TM = TPC.getTM<AArch64TargetMachine>();
    const AArch64Subtarget &ST =
        *TM.getSubtargetImpl(*L->getHeader()->getParent());

    return processLoop(*L, ST, TM.createDataLayout());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override { return "SVE Shuffle Optimizations"; }
};
} // end anonymous namespace

char SVEShuffleOpts::ID = 0;
static const char *name = "SVE Shuffle Optimizations";
INITIALIZE_PASS_BEGIN(SVEShuffleOpts, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(SVEShuffleOpts, DEBUG_TYPE, name, false, false)

Pass *llvm::createSVEShuffleOptsPass() { return new SVEShuffleOpts(); }

PreservedAnalyses SVEShuffleOptsPass::run(Loop &L, LoopAnalysisManager &AM,
                                          LoopStandardAnalysisResults &AR,
                                          LPMUpdater &U) {
  const AArch64Subtarget &ST =
      *TM.getSubtargetImpl(*L.getHeader()->getParent());

  if (processLoop(L, ST, TM.createDataLayout())) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<TargetIRAnalysis>();
    PA.preserve<AssumptionAnalysis>();
    PA.preserve<MemorySSAAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}
