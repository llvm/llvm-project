//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <cmath>
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

static cl::opt<unsigned> RVVRegisterWidthLMUL(
    "riscv-v-register-bit-width-lmul",
    cl::desc(
        "The LMUL to use for getRegisterBitWidth queries. Affects LMUL used "
        "by autovectorized code. Fractional LMULs are not supported."),
    cl::init(1), cl::Hidden);

InstructionCost RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                            TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    getST()->getFeatureBits());
}

InstructionCost RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                const APInt &Imm, Type *Ty,
                                                TTI::TargetCostKind CostKind,
                                                Instruction *Inst) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in RISC-V can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::And:
    // zext.h
    if (Imm == UINT64_C(0xffff) && ST->hasStdExtZbb())
      return TTI::TCC_Free;
    // zext.w
    if (Imm == UINT64_C(0xffffffff) && ST->hasStdExtZbb())
      return TTI::TCC_Free;
    LLVM_FALLTHROUGH;
  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getMinSignedBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty, CostKind);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

InstructionCost
RISCVTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                  const APInt &Imm, Type *Ty,
                                  TTI::TargetCostKind CostKind) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

TargetTransformInfo::PopcntSupportKind
RISCVTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasStdExtZbb() ? TTI::PSK_FastHardware : TTI::PSK_Software;
}

bool RISCVTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  // Currently, the ExpandReductions pass can't expand scalable-vector
  // reductions, but we still request expansion as RVV doesn't support certain
  // reductions and the SelectionDAG can't legalize them either.
  switch (II->getIntrinsicID()) {
  default:
    return false;
  // These reductions have no equivalent in RVV
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_fmul:
    return true;
  }
}

Optional<unsigned> RISCVTTIImpl::getMaxVScale() const {
  // There is no assumption of the maximum vector length in V specification.
  // We use the value specified by users as the maximum vector length.
  // This function will use the assumed maximum vector length to get the
  // maximum vscale for LoopVectorizer.
  // If users do not specify the maximum vector length, we have no way to
  // know whether the LoopVectorizer is safe to do or not.
  // We only consider to use single vector register (LMUL = 1) to vectorize.
  unsigned MaxVectorSizeInBits = ST->getMaxRVVVectorSizeInBits();
  if (ST->hasVInstructions() && MaxVectorSizeInBits != 0)
    return MaxVectorSizeInBits / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}

TypeSize
RISCVTTIImpl::getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
  unsigned LMUL = PowerOf2Floor(
      std::max<unsigned>(std::min<unsigned>(RVVRegisterWidthLMUL, 8), 1));
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(ST->getXLen());
  case TargetTransformInfo::RGK_FixedWidthVector:
    return TypeSize::getFixed(
        ST->hasVInstructions() ? LMUL * ST->getMinRVVVectorSizeInBits() : 0);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(
        ST->hasVInstructions() ? LMUL * RISCV::RVVBitsPerBlock : 0);
  }

  llvm_unreachable("Unsupported register kind");
}

InstructionCost RISCVTTIImpl::getSpliceCost(VectorType *Tp, int Index) {
  std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);

  unsigned Cost = 2; // vslidedown+vslideup.
  // TODO: LMUL should increase cost.
  // TODO: Multiplying by LT.first implies this legalizes into multiple copies
  // of similar code, but I think we expand through memory.
  return Cost * LT.first;
}

InstructionCost RISCVTTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                             VectorType *Tp, ArrayRef<int> Mask,
                                             int Index, VectorType *SubTp,
                                             ArrayRef<const Value *> Args) {
  if (isa<ScalableVectorType>(Tp)) {
    std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);
    switch (Kind) {
    default:
      // Fallthrough to generic handling.
      // TODO: Most of these cases will return getInvalid in generic code, and
      // must be implemented here.
      break;
    case TTI::SK_Broadcast: {
      return LT.first * 1;
    }
    case TTI::SK_Splice:
      return getSpliceCost(Tp, Index);
    case TTI::SK_Reverse:
      // Most of the cost here is producing the vrgather index register
      // Example sequence:
      //   csrr a0, vlenb
      //   srli a0, a0, 3
      //   addi a0, a0, -1
      //   vsetvli a1, zero, e8, mf8, ta, mu (ignored)
      //   vid.v v9
      //   vrsub.vx v10, v9, a0
      //   vrgather.vv v9, v8, v10
      return LT.first * 6;
    }
  }

  return BaseT::getShuffleCost(Kind, Tp, Mask, Index, SubTp);
}

InstructionCost
RISCVTTIImpl::getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                    unsigned AddressSpace,
                                    TTI::TargetCostKind CostKind) {
  if (!isa<ScalableVectorType>(Src))
    return BaseT::getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                        CostKind);

  return getMemoryOpCost(Opcode, Src, Alignment, AddressSpace, CostKind);
}

InstructionCost RISCVTTIImpl::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) {
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  if ((Opcode == Instruction::Load &&
       !isLegalMaskedGather(DataTy, Align(Alignment))) ||
      (Opcode == Instruction::Store &&
       !isLegalMaskedScatter(DataTy, Align(Alignment))))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  // FIXME: Only supporting fixed vectors for now.
  if (!isa<FixedVectorType>(DataTy))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  auto *VTy = cast<FixedVectorType>(DataTy);
  unsigned NumLoads = VTy->getNumElements();
  InstructionCost MemOpCost =
      getMemoryOpCost(Opcode, VTy->getElementType(), Alignment, 0, CostKind, I);
  return NumLoads * MemOpCost;
}

InstructionCost
RISCVTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                    TTI::TargetCostKind CostKind) {
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
  // TODO: add more intrinsic
  case Intrinsic::experimental_stepvector: {
    unsigned Cost = 1; // vid
    auto LT = TLI->getTypeLegalizationCost(DL, RetTy);
    return Cost + (LT.first - 1);
  }
  default:
    break;
  }
  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

InstructionCost RISCVTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                               Type *Src,
                                               TTI::CastContextHint CCH,
                                               TTI::TargetCostKind CostKind,
                                               const Instruction *I) {
  if (isa<VectorType>(Dst) && isa<VectorType>(Src)) {
    // FIXME: Need to compute legalizing cost for illegal types.
    if (!isTypeLegal(Src) || !isTypeLegal(Dst))
      return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

    // Skip if element size of Dst or Src is bigger than ELEN.
    if (Src->getScalarSizeInBits() > ST->getELEN() ||
        Dst->getScalarSizeInBits() > ST->getELEN())
      return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");

    // FIXME: Need to consider vsetvli and lmul.
    int PowDiff = (int)Log2_32(Dst->getScalarSizeInBits()) -
                  (int)Log2_32(Src->getScalarSizeInBits());
    switch (ISD) {
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
      return 1;
    case ISD::TRUNCATE:
    case ISD::FP_EXTEND:
    case ISD::FP_ROUND:
      // Counts of narrow/widen instructions.
      return std::abs(PowDiff);
    case ISD::FP_TO_SINT:
    case ISD::FP_TO_UINT:
    case ISD::SINT_TO_FP:
    case ISD::UINT_TO_FP:
      if (std::abs(PowDiff) <= 1)
        return 1;
      // Backend could lower (v[sz]ext i8 to double) to vfcvt(v[sz]ext.f8 i8),
      // so it only need two conversion.
      if (Src->isIntOrIntVectorTy())
        return 2;
      // Counts of narrow/widen instructions.
      return std::abs(PowDiff);
    }
  }
  return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
}

InstructionCost
RISCVTTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                     bool IsUnsigned,
                                     TTI::TargetCostKind CostKind) {
  // FIXME: Only supporting fixed vectors for now.
  if (!isa<FixedVectorType>(Ty))
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  if (!ST->useRVVForFixedLengthVectors())
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (Ty->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  if (Ty->getElementType()->isIntegerTy(1))
    // vcpop sequences, see vreduction-mask.ll.  umax, smin actually only
    // cost 2, but we don't have enough info here so we slightly over cost.
    return (LT.first - 1) + 3;

  // IR Reduction is composed by two vmv and one rvv reduction instruction.
  InstructionCost BaseCost = 2;
  unsigned VL = cast<FixedVectorType>(Ty)->getNumElements();
  return (LT.first - 1) + BaseCost + Log2_32_Ceil(VL);
}

InstructionCost
RISCVTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *VTy,
                                         Optional<FastMathFlags> FMF,
                                         TTI::TargetCostKind CostKind) {
  // FIXME: Only supporting fixed vectors for now.
  if (!isa<FixedVectorType>(VTy))
    return BaseT::getArithmeticReductionCost(Opcode, VTy, FMF, CostKind);

  if (!ST->useRVVForFixedLengthVectors())
    return BaseT::getArithmeticReductionCost(Opcode, VTy, FMF, CostKind);

  // Skip if scalar size of VTy is bigger than ELEN.
  if (VTy->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getArithmeticReductionCost(Opcode, VTy, FMF, CostKind);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  if (ISD != ISD::ADD && ISD != ISD::OR && ISD != ISD::XOR && ISD != ISD::AND &&
      ISD != ISD::FADD)
    return BaseT::getArithmeticReductionCost(Opcode, VTy, FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = TLI->getTypeLegalizationCost(DL, VTy);
  if (VTy->getElementType()->isIntegerTy(1))
    // vcpop sequences, see vreduction-mask.ll
    return (LT.first - 1) + (ISD == ISD::AND ? 3 : 2);

  // IR Reduction is composed by two vmv and one rvv reduction instruction.
  InstructionCost BaseCost = 2;
  unsigned VL = cast<FixedVectorType>(VTy)->getNumElements();
  if (TTI::requiresOrderedReduction(FMF))
    return (LT.first - 1) + BaseCost + VL;
  return (LT.first - 1) + BaseCost + Log2_32_Ceil(VL);
}

void RISCVTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                           TTI::UnrollingPreferences &UP,
                                           OptimizationRemarkEmitter *ORE) {
  // TODO: More tuning on benchmarks and metrics with changes as needed
  //       would apply to all settings below to enable performance.


  if (ST->enableDefaultUnroll())
    return BasicTTIImplBase::getUnrollingPreferences(L, SE, UP, ORE);

  // Enable Upper bound unrolling universally, not dependant upon the conditions
  // below.
  UP.UpperBound = true;

  // Disable loop unrolling for Oz and Os.
  UP.OptSizeThreshold = 0;
  UP.PartialOptSizeThreshold = 0;
  if (L->getHeader()->getParent()->hasOptSize())
    return;

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  LLVM_DEBUG(dbgs() << "Loop has:\n"
                    << "Blocks: " << L->getNumBlocks() << "\n"
                    << "Exit blocks: " << ExitingBlocks.size() << "\n");

  // Only allow another exit other than the latch. This acts as an early exit
  // as it mirrors the profitability calculation of the runtime unroller.
  if (ExitingBlocks.size() > 2)
    return;

  // Limit the CFG of the loop body for targets with a branch predictor.
  // Allowing 4 blocks permits if-then-else diamonds in the body.
  if (L->getNumBlocks() > 4)
    return;

  // Don't unroll vectorized loops, including the remainder loop
  if (getBooleanLoopAttribute(L, "llvm.loop.isvectorized"))
    return;

  // Scan the loop: don't unroll loops with calls as this could prevent
  // inlining.
  InstructionCost Cost = 0;
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      // Initial setting - Don't unroll loops containing vectorized
      // instructions.
      if (I.getType()->isVectorTy())
        return;

      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (const Function *F = cast<CallBase>(I).getCalledFunction()) {
          if (!isLoweredToCall(F))
            continue;
        }
        return;
      }

      SmallVector<const Value *> Operands(I.operand_values());
      Cost +=
          getUserCost(&I, Operands, TargetTransformInfo::TCK_SizeAndLatency);
    }
  }

  LLVM_DEBUG(dbgs() << "Cost of loop: " << Cost << "\n");

  UP.Partial = true;
  UP.Runtime = true;
  UP.UnrollRemainder = true;
  UP.UnrollAndJam = true;
  UP.UnrollAndJamInnerLoopThreshold = 60;

  // Force unrolling small loops can be very useful because of the branch
  // taken cost of the backedge.
  if (Cost < 12)
    UP.Force = true;
}

void RISCVTTIImpl::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::PeelingPreferences &PP) {
  BaseT::getPeelingPreferences(L, SE, PP);
}

unsigned RISCVTTIImpl::getRegUsageForType(Type *Ty) {
  TypeSize Size = Ty->getPrimitiveSizeInBits();
  if (Ty->isVectorTy()) {
    if (Size.isScalable() && ST->hasVInstructions())
      return divideCeil(Size.getKnownMinValue(), RISCV::RVVBitsPerBlock);

    if (ST->useRVVForFixedLengthVectors())
      return divideCeil(Size, ST->getMinRVVVectorSizeInBits());
  }

  return BaseT::getRegUsageForType(Ty);
}
