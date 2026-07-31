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
#include "Utils/AArch64BaseInfo.h"
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
static bool optimizeSVEDeinterleavedExtends(Loop &L,
                                            const AArch64TargetLowering &TL,
                                            const DataLayout DL) {
  DeinterleaveMap Deinterleaves;
  for (auto *BB : L.blocks())
    for (auto &I : *BB)
      if (match(&I, m_Intrinsic<Intrinsic::vector_deinterleave4>(m_Value())))
        evaluateDeinterleave(cast<IntrinsicInst>(&I), Deinterleaves, L, TL, DL);

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

  return !Deinterleaves.empty();
}

// Match a bitcasted tbl intrinsic, and bind the tbl along with the mask.
template <typename TblT, typename MaskT>
static auto m_Tbl(TblT &&Tbl, MaskT &&Mask) {
  return m_OneUse(
      m_BitCast(m_OneUse(m_Value(Tbl, m_Intrinsic<Intrinsic::aarch64_sve_tbl>(
                                          m_Value(), m_Value(Mask))))));
}

// Match a tbl intrinsic whose result is converted to a floating point value.
template <typename TblT, typename MaskT>
static auto m_UIToFPTbl(TblT &&Tbl, MaskT &&Mask) {
  return m_OneUse(m_UIToFP(m_Tbl(Tbl, Mask)));
}

// Match either of the above tbls, and recalculate the index of the
// deinterleaved subvector. Bind the tbl and the index.
template <typename T> struct deinterleaving_tbl_match {
  T *&Tbl;
  unsigned &Idx;

  deinterleaving_tbl_match(T *&Tbl, unsigned &Idx) : Tbl(Tbl), Idx(Idx) {}

  template <typename ITy> bool match(ITy *V) const {
    // Match the tbl.
    Value *Mask;
    if (!PatternMatch::match(
            V, m_CombineOr(m_Tbl(Tbl, Mask), m_UIToFPTbl(Tbl, Mask))))
      return false;

    // For a deinterleaving+extending tbl, we will have a known constant values
    // for the starting index and the step.
    const APInt *Start;
    const APInt *Step;
    if (!PatternMatch::match(
            Mask, m_BitCast(m_Add(m_Mul(m_Intrinsic<Intrinsic::stepvector>(),
                                        m_APInt(Step)),
                                  m_APInt(Start)))))
      return false;

    unsigned SrcSize = Tbl->getType()->getScalarType()->getScalarSizeInBits();
    unsigned ResSize = V->getType()->getScalarType()->getScalarSizeInBits();
    // If the top bits are all ones, we know we're forcing an out-of-range
    // index. With a deinterleave of 4, we should have 3 invalid indices for
    // every valid one.
    if (Start->countLeadingOnes() != ResSize - SrcSize)
      return false;

    // The start of the valid indices must be between 0 and 3, for the 4
    // subvectors we're extracting.
    Idx = Start->getZExtValue() & SrcSize - 1;
    return Idx >= 0 && Idx < 4;
  }
};

template <typename T> static auto m_DeinterleavingTbl(T *&Tbl, unsigned &Idx) {
  return deinterleaving_tbl_match<T>(Tbl, Idx);
}

/// We want to find a reverse used only by BinOps where the other term comes
/// from one of the deinterleave-and-extend tbls we created before. If this
/// BinOp is only used in a reduction operation in the loop (so the order of
/// elements within it do not matter), then we can potentially fold the reverse
/// into the tbl and remove the separate reverse operation.
/// Something like the following (possibly repeated multiple times):
///
/// %acc.b.f64 = phi <vscale x 2 x double> [ splat(double 0.000000e+00),
///                                          %entry ], [ %fadd.b.f64, %loop ]
/// ...
/// %rev.load = load <vscale x 2 x double>, ptr %rev.ptr
/// %reversed = call <vscale x 2 x double> @llvm.vector.reverse.nxv2f64(
///                                            <vscale x 2 x double> %rev.load)
/// %bgra = load <vscale x 8 x i16>, ptr %src.gep
/// %stepvec = call <vscale x 2 x i64> @llvm.stepvector.nxv2i64()
/// %stride = mul nuw <vscale x 2 x i64> %stepvec, splat (i64 4)
/// %start = add nuw <vscale x 2 x i64> %stride, splat (i64 -65536)
/// %bc.to = bitcast <vscale x 2 x i64> %start to <vscale x 8 x i16>
/// %tbl = call <vscale x 8 x i16> @llvm.aarch64.sve.tbl.nxv8i16(
//                         <vscale x 8 x i16> %bgra, <vscale x 8 x i16> %bc.to)
/// %bc.from = bitcast <vscale x 8 x i16> %tbl to <vscale x 2 x i64>
/// %b.f64 = uitofp <vscale x 2 x i64> %bc.from to <vscale x 2 x double>
/// %b.mul.f64 = fmul <vscale x 2 x double> %b.f64, %reversed
/// %fadd.b.f64 = fadd <vscale x 2 x double> %acc.b.f64, %b.mul.f64
/// ...
/// %b.acc = call fast double @llvm.vector.reduce.fadd.nxv2f64(double
///                            0.000000e+00, <vscale x 2 x double> %fadd.b.f64)
static bool foldAdjacentReversesIntoTbls(Loop &L,
                                         const AArch64TargetLowering &TL,
                                         const DataLayout DL) {
  struct RevTblData {
    Instruction *Tbl;
    Use *RevUse;
    unsigned Idx;
  };

  // Look for reverse intrinsics used with the results of tbl instructions.
  SmallVector<RevTblData, 4> Tbls;
  for (auto *BB : L.blocks())
    for (auto &I : *BB) {
      if (!match(&I, m_Intrinsic<Intrinsic::vector_reverse>(m_Value())))
        continue;

      // Check all uses of the reverse; if there's anything which doesn't
      // match our expected patterns, then give up on it. The intention is
      // to remove the reverse, so any remaining users would prevent that.
      SmallVector<RevTblData, 4> RevUses;
      for (Use &U : I.uses()) {
        Instruction *UI = cast<Instruction>(U.getUser());

        // Look for a tbl used to deinterleave and zero extend (and optionally
        // convert to FP), the result of which is then used in a BinOp with
        // the reverse.
        Value *Tbl;
        unsigned Idx;
        if (!match(UI, m_OneUse(m_c_BinOp(m_DeinterleavingTbl(Tbl, Idx),
                                          m_Specific(&I))))) {
          RevUses.clear();
          break;
        }

        // Check that the only use of the BinOp is part of a reduction such that
        // the order of elements in the vector doesn't matter.
        // TODO: Allow more operations in the chain.
        // Look for 2 users -- a phi in the loop, and an outside
        // reduction intrinsic.
        Instruction *RdxUpdate = cast<Instruction>(UI->user_back());
        SmallVector<User *, 2> Users(RdxUpdate->users());
        if (!match(RdxUpdate, m_BinOp()) || Users.size() != 2) {
          RevUses.clear();
          break;
        }

        Instruction *Phi = cast<Instruction>(Users[0]);
        Instruction *Reduce = cast<Instruction>(Users[1]);

        // We're looking for an in-loop phi to confirm a reduction.
        if (!L.contains(Phi))
          std::swap(Phi, Reduce);

        // If the in-loop user is not a header Phi, or isn't one of the
        // operands for the RdxUpdate operation, or the reduction op isn't
        // outside of the loop, abandon this reverse.
        if (!isa<PHINode>(Phi) || Phi->getParent() != L.getHeader() ||
            !is_contained(RdxUpdate->operands(), Phi) || L.contains(Reduce)) {
          RevUses.clear();
          break;
        }

        // The out-of-loop user may be an LCSSA phi; look through that.
        if (isa<PHINode>(Reduce) && Reduce->getNumOperands() == 1 &&
            Reduce->hasOneUser())
          Reduce = Reduce->user_back();

        // Make sure the outside user is a supported reduction operation, and if
        // FP that reassociation is allowed.
        if (!match(Reduce,
                   m_CombineOr(
                       m_Intrinsic<Intrinsic::vector_reduce_add>(),
                       m_AllowReassoc(
                           m_Intrinsic<Intrinsic::vector_reduce_fadd>())))) {
          RevUses.clear();
          break;
        }
        RevUses.push_back({cast<Instruction>(Tbl), &U, Idx});
      }
      append_range(Tbls, RevUses);
    }

  // Convert each candidate we found to perform the reverse in the tbl instead,
  // then remove the reverse.
  for (auto [Tbl, RevUse, Idx] : Tbls) {
    Instruction *Rev = cast<Instruction>(RevUse->get());
    VectorType *SrcTy = cast<VectorType>(Tbl->getType());
    VectorType *DstTy = cast<VectorType>(Rev->getType());
    unsigned SrcBits = SrcTy->getScalarSizeInBits();
    unsigned DstBits = DstTy->getScalarSizeInBits();
    VectorType *StepVecTy = VectorType::getInteger(DstTy);

    // We need to create a new mask for the tbl, so that we effectively reverse
    // the elements with the tbl. This means the resulting vector will be
    // backwards compared to the original, which is why we check for reductions
    // where we don't care about the order.
    //
    // Similar to the normal deinterleaving+extending mask, we will use out-of
    // range indices to perform the zero extension. However, we need a negative
    // stride starting from the highest group of elements. Since we're grouping
    // by 4 (for now), we need to subtract 4 from the total source element count
    // for the vector to get the start, then add the index of the extraction.
    //
    // The result should be the following:
    // <EltCnt - 4 + Idx, 0xFFFF, 0xFFFF, 0xFFFF, EltCnt - 8 + Idx, 0xFFFF...>.
    IRBuilder<> Builder(Tbl);

    // Create and splat the starting value.
    APInt Invalid = APInt::getAllOnes(DstBits);
    APInt StartIdx = Invalid << SrcBits;
    StartIdx -= (4 - Idx);
    Value *EltCnt = Builder.CreateVScale(StepVecTy->getScalarType());
    EltCnt = Builder.CreateNUWMul(
        EltCnt, ConstantInt::get(EltCnt->getType(),
                                 AArch64::SVEBitsPerBlock / SrcBits));
    Value *StartVal = Builder.CreateNUWAdd(
        EltCnt, ConstantInt::get(EltCnt->getType(), StartIdx));
    StartVal =
        Builder.CreateVectorSplat(StepVecTy->getElementCount(), StartVal);

    // Create the negative stride.
    Value *StepVector = Builder.CreateStepVector(StepVecTy);
    Value *ScaledSteps =
        Builder.CreateMul(StepVector, ConstantInt::get(StepVecTy, -4));

    // Add the start to the stride, replace the old mask.
    ScaledSteps = Builder.CreateNUWAdd(ScaledSteps, StartVal);
    Value *RevExtMask = Builder.CreateBitCast(ScaledSteps, SrcTy);
    Tbl->setOperand(1, RevExtMask);

    // Skip the original reverse now that we've migrated it to the tbl.
    RevUse->set(Rev->getOperand(0));

    // And erase the reverse if that was the last use.
    if (Rev->uses().empty()) {
      LLVM_DEBUG(dbgs() << "SVETBLOPT: Erasing " << *Rev << "\n");
      Rev->eraseFromParent();
    }
  }

  return !Tbls.empty();
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

  bool Changed = false;
  Changed |= optimizeSVEDeinterleavedExtends(L, TL, DL);
  Changed |= foldAdjacentReversesIntoTbls(L, TL, DL);
  return Changed;
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

PreservedAnalyses
AArch64SVEShuffleOptsPass::run(Loop &L, LoopAnalysisManager &AM,
                               LoopStandardAnalysisResults &AR, LPMUpdater &U) {
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
