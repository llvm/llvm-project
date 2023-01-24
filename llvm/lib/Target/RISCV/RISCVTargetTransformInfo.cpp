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
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <cmath>
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

static cl::opt<unsigned> RVVRegisterWidthLMUL(
    "riscv-v-register-bit-width-lmul",
    cl::desc(
        "The LMUL to use for getRegisterBitWidth queries. Affects LMUL used "
        "by autovectorized code. Fractional LMULs are not supported."),
    cl::init(1), cl::Hidden);

static cl::opt<unsigned> SLPMaxVF(
    "riscv-v-slp-max-vf",
    cl::desc(
        "Result used for getMaximumVF query which is used exclusively by "
        "SLP vectorizer.  Defaults to 1 which disables SLP."),
    cl::init(1), cl::Hidden);

InstructionCost RISCVTTIImpl::getLMULCost(MVT VT) {
  // TODO: Here assume reciprocal throughput is 1 for LMUL_1, it is
  // implementation-defined.
  if (!VT.isVector())
    return InstructionCost::getInvalid();
  unsigned Cost;
  if (VT.isScalableVector()) {
    unsigned LMul;
    bool Fractional;
    std::tie(LMul, Fractional) =
        RISCVVType::decodeVLMUL(RISCVTargetLowering::getLMUL(VT));
    if (Fractional)
      Cost = 1;
    else
      Cost = LMul;
  } else {
    Cost = VT.getSizeInBits() / ST->getRealMinVLen();
  }
  return std::max<unsigned>(Cost, 1);
}

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

// Look for patterns of shift followed by AND that can be turned into a pair of
// shifts. We won't need to materialize an immediate for the AND so these can
// be considered free.
static bool canUseShiftPair(Instruction *Inst, const APInt &Imm) {
  uint64_t Mask = Imm.getZExtValue();
  auto *BO = dyn_cast<BinaryOperator>(Inst->getOperand(0));
  if (!BO || !BO->hasOneUse())
    return false;

  if (BO->getOpcode() != Instruction::Shl)
    return false;

  if (!isa<ConstantInt>(BO->getOperand(1)))
    return false;

  unsigned ShAmt = cast<ConstantInt>(BO->getOperand(1))->getZExtValue();
  // (and (shl x, c2), c1) will be matched to (srli (slli x, c2+c3), c3) if c1
  // is a mask shifted by c2 bits with c3 leading zeros.
  if (isShiftedMask_64(Mask)) {
    unsigned Trailing = countTrailingZeros(Mask);
    if (ShAmt == Trailing)
      return true;
  }

  return false;
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
    if (Imm == UINT64_C(0xffffffff) && ST->hasStdExtZba())
      return TTI::TCC_Free;
    // bclri
    if (ST->hasStdExtZbs() && (~Imm).isPowerOf2())
      return TTI::TCC_Free;
    if (Inst && Idx == 1 && Imm.getBitWidth() <= ST->getXLen() &&
        canUseShiftPair(Inst, Imm))
      return TTI::TCC_Free;
    Takes12BitImm = true;
    break;
  case Instruction::Add:
    Takes12BitImm = true;
    break;
  case Instruction::Or:
  case Instruction::Xor:
    // bseti/binvi
    if (ST->hasStdExtZbs() && Imm.isPowerOf2())
      return TTI::TCC_Free;
    Takes12BitImm = true;
    break;
  case Instruction::Mul:
    // Negated power of 2 is a shift and a negate.
    if (Imm.isNegatedPowerOf2())
      return TTI::TCC_Free;
    // FIXME: There is no MULI instruction.
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

std::optional<unsigned> RISCVTTIImpl::getMaxVScale() const {
  if (ST->hasVInstructions())
    return ST->getRealMaxVLen() / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}

std::optional<unsigned> RISCVTTIImpl::getVScaleForTuning() const {
  if (ST->hasVInstructions())
    if (unsigned MinVLen = ST->getRealMinVLen();
        MinVLen >= RISCV::RVVBitsPerBlock)
      return MinVLen / RISCV::RVVBitsPerBlock;
  return BaseT::getVScaleForTuning();
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
        ST->useRVVForFixedLengthVectors() ? LMUL * ST->getRealMinVLen() : 0);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(
        (ST->hasVInstructions() &&
         ST->getRealMinVLen() >= RISCV::RVVBitsPerBlock)
            ? LMUL * RISCV::RVVBitsPerBlock
            : 0);
  }

  llvm_unreachable("Unsupported register kind");
}

InstructionCost RISCVTTIImpl::getSpliceCost(VectorType *Tp, int Index) {
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);

  unsigned Cost = 2; // vslidedown+vslideup.
  // TODO: Multiplying by LT.first implies this legalizes into multiple copies
  // of similar code, but I think we expand through memory.
  return Cost * LT.first * getLMULCost(LT.second);
}

InstructionCost RISCVTTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                             VectorType *Tp, ArrayRef<int> Mask,
                                             TTI::TargetCostKind CostKind,
                                             int Index, VectorType *SubTp,
                                             ArrayRef<const Value *> Args) {
  if (isa<ScalableVectorType>(Tp)) {
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
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
      if (Tp->getElementType()->isIntegerTy(1))
        // Mask operation additionally required extend and truncate
        return LT.first * 9;
      return LT.first * 6;
    }
  }

  if (isa<FixedVectorType>(Tp) && Kind == TargetTransformInfo::SK_Broadcast) {
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
    bool HasScalar = (Args.size() > 0) && (Operator::getOpcode(Args[0]) ==
                                           Instruction::InsertElement);
    if (LT.second.getScalarSizeInBits() == 1) {
      if (HasScalar) {
        // Example sequence:
        //   andi a0, a0, 1
        //   vsetivli zero, 2, e8, mf8, ta, ma (ignored)
        //   vmv.v.x v8, a0
        //   vmsne.vi v0, v8, 0
        return LT.first * getLMULCost(LT.second) * 3;
      }
      // Example sequence:
      //   vsetivli  zero, 2, e8, mf8, ta, mu (ignored)
      //   vmv.v.i v8, 0
      //   vmerge.vim      v8, v8, 1, v0
      //   vmv.x.s a0, v8
      //   andi    a0, a0, 1
      //   vmv.v.x v8, a0
      //   vmsne.vi  v0, v8, 0

      return LT.first * getLMULCost(LT.second) * 6;
    }

    if (HasScalar) {
      // Example sequence:
      //   vmv.v.x v8, a0
      return LT.first * getLMULCost(LT.second);
    }

    // Example sequence:
    //   vrgather.vi     v9, v8, 0
    // TODO: vrgather could be slower than vmv.v.x. It is
    // implementation-dependent.
    return LT.first * getLMULCost(LT.second);
  }

  return BaseT::getShuffleCost(Kind, Tp, Mask, CostKind, Index, SubTp);
}

InstructionCost
RISCVTTIImpl::getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                    unsigned AddressSpace,
                                    TTI::TargetCostKind CostKind) {
  if (!isLegalMaskedLoadStore(Src, Alignment) ||
      CostKind != TTI::TCK_RecipThroughput)
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

  // Cost is proportional to the number of memory operations implied.  For
  // scalable vectors, we use an estimate on that number since we don't
  // know exactly what VL will be.
  auto &VTy = *cast<VectorType>(DataTy);
  InstructionCost MemOpCost =
      getMemoryOpCost(Opcode, VTy.getElementType(), Alignment, 0, CostKind,
                      {TTI::OK_AnyValue, TTI::OP_None}, I);
  unsigned NumLoads = getEstimatedVLFor(&VTy);
  return NumLoads * MemOpCost;
}

// Currently, these represent both throughput and codesize costs
// for the respective intrinsics.  The costs in this table are simply
// instruction counts with the following adjustments made:
// * One vsetvli is considered free.
static const CostTblEntry VectorIntrinsicCostTable[]{
    {Intrinsic::floor, MVT::v2f32, 9},
    {Intrinsic::floor, MVT::v4f32, 9},
    {Intrinsic::floor, MVT::v8f32, 9},
    {Intrinsic::floor, MVT::v16f32, 9},
    {Intrinsic::floor, MVT::nxv1f32, 9},
    {Intrinsic::floor, MVT::nxv2f32, 9},
    {Intrinsic::floor, MVT::nxv4f32, 9},
    {Intrinsic::floor, MVT::nxv8f32, 9},
    {Intrinsic::floor, MVT::nxv16f32, 9},
    {Intrinsic::floor, MVT::v2f64, 9},
    {Intrinsic::floor, MVT::v4f64, 9},
    {Intrinsic::floor, MVT::v8f64, 9},
    {Intrinsic::floor, MVT::v16f64, 9},
    {Intrinsic::floor, MVT::nxv1f64, 9},
    {Intrinsic::floor, MVT::nxv2f64, 9},
    {Intrinsic::floor, MVT::nxv4f64, 9},
    {Intrinsic::floor, MVT::nxv8f64, 9},
    {Intrinsic::ceil, MVT::v2f32, 9},
    {Intrinsic::ceil, MVT::v4f32, 9},
    {Intrinsic::ceil, MVT::v8f32, 9},
    {Intrinsic::ceil, MVT::v16f32, 9},
    {Intrinsic::ceil, MVT::nxv1f32, 9},
    {Intrinsic::ceil, MVT::nxv2f32, 9},
    {Intrinsic::ceil, MVT::nxv4f32, 9},
    {Intrinsic::ceil, MVT::nxv8f32, 9},
    {Intrinsic::ceil, MVT::nxv16f32, 9},
    {Intrinsic::ceil, MVT::v2f64, 9},
    {Intrinsic::ceil, MVT::v4f64, 9},
    {Intrinsic::ceil, MVT::v8f64, 9},
    {Intrinsic::ceil, MVT::v16f64, 9},
    {Intrinsic::ceil, MVT::nxv1f64, 9},
    {Intrinsic::ceil, MVT::nxv2f64, 9},
    {Intrinsic::ceil, MVT::nxv4f64, 9},
    {Intrinsic::ceil, MVT::nxv8f64, 9},
    {Intrinsic::trunc, MVT::v2f32, 7},
    {Intrinsic::trunc, MVT::v4f32, 7},
    {Intrinsic::trunc, MVT::v8f32, 7},
    {Intrinsic::trunc, MVT::v16f32, 7},
    {Intrinsic::trunc, MVT::nxv1f32, 7},
    {Intrinsic::trunc, MVT::nxv2f32, 7},
    {Intrinsic::trunc, MVT::nxv4f32, 7},
    {Intrinsic::trunc, MVT::nxv8f32, 7},
    {Intrinsic::trunc, MVT::nxv16f32, 7},
    {Intrinsic::trunc, MVT::v2f64, 7},
    {Intrinsic::trunc, MVT::v4f64, 7},
    {Intrinsic::trunc, MVT::v8f64, 7},
    {Intrinsic::trunc, MVT::v16f64, 7},
    {Intrinsic::trunc, MVT::nxv1f64, 7},
    {Intrinsic::trunc, MVT::nxv2f64, 7},
    {Intrinsic::trunc, MVT::nxv4f64, 7},
    {Intrinsic::trunc, MVT::nxv8f64, 7},
    {Intrinsic::round, MVT::v2f32, 9},
    {Intrinsic::round, MVT::v4f32, 9},
    {Intrinsic::round, MVT::v8f32, 9},
    {Intrinsic::round, MVT::v16f32, 9},
    {Intrinsic::round, MVT::nxv1f32, 9},
    {Intrinsic::round, MVT::nxv2f32, 9},
    {Intrinsic::round, MVT::nxv4f32, 9},
    {Intrinsic::round, MVT::nxv8f32, 9},
    {Intrinsic::round, MVT::nxv16f32, 9},
    {Intrinsic::round, MVT::v2f64, 9},
    {Intrinsic::round, MVT::v4f64, 9},
    {Intrinsic::round, MVT::v8f64, 9},
    {Intrinsic::round, MVT::v16f64, 9},
    {Intrinsic::round, MVT::nxv1f64, 9},
    {Intrinsic::round, MVT::nxv2f64, 9},
    {Intrinsic::round, MVT::nxv4f64, 9},
    {Intrinsic::round, MVT::nxv8f64, 9},
    {Intrinsic::roundeven, MVT::v2f32, 9},
    {Intrinsic::roundeven, MVT::v4f32, 9},
    {Intrinsic::roundeven, MVT::v8f32, 9},
    {Intrinsic::roundeven, MVT::v16f32, 9},
    {Intrinsic::roundeven, MVT::nxv1f32, 9},
    {Intrinsic::roundeven, MVT::nxv2f32, 9},
    {Intrinsic::roundeven, MVT::nxv4f32, 9},
    {Intrinsic::roundeven, MVT::nxv8f32, 9},
    {Intrinsic::roundeven, MVT::nxv16f32, 9},
    {Intrinsic::roundeven, MVT::v2f64, 9},
    {Intrinsic::roundeven, MVT::v4f64, 9},
    {Intrinsic::roundeven, MVT::v8f64, 9},
    {Intrinsic::roundeven, MVT::v16f64, 9},
    {Intrinsic::roundeven, MVT::nxv1f64, 9},
    {Intrinsic::roundeven, MVT::nxv2f64, 9},
    {Intrinsic::roundeven, MVT::nxv4f64, 9},
    {Intrinsic::roundeven, MVT::nxv8f64, 9},
    {Intrinsic::bswap, MVT::v2i16, 3},
    {Intrinsic::bswap, MVT::v4i16, 3},
    {Intrinsic::bswap, MVT::v8i16, 3},
    {Intrinsic::bswap, MVT::v16i16, 3},
    {Intrinsic::bswap, MVT::nxv1i16, 3},
    {Intrinsic::bswap, MVT::nxv2i16, 3},
    {Intrinsic::bswap, MVT::nxv4i16, 3},
    {Intrinsic::bswap, MVT::nxv8i16, 3},
    {Intrinsic::bswap, MVT::nxv16i16, 3},
    {Intrinsic::bswap, MVT::v2i32, 12},
    {Intrinsic::bswap, MVT::v4i32, 12},
    {Intrinsic::bswap, MVT::v8i32, 12},
    {Intrinsic::bswap, MVT::v16i32, 12},
    {Intrinsic::bswap, MVT::nxv1i32, 12},
    {Intrinsic::bswap, MVT::nxv2i32, 12},
    {Intrinsic::bswap, MVT::nxv4i32, 12},
    {Intrinsic::bswap, MVT::nxv8i32, 12},
    {Intrinsic::bswap, MVT::nxv16i32, 12},
    {Intrinsic::bswap, MVT::v2i64, 31},
    {Intrinsic::bswap, MVT::v4i64, 31},
    {Intrinsic::bswap, MVT::v8i64, 31},
    {Intrinsic::bswap, MVT::v16i64, 31},
    {Intrinsic::bswap, MVT::nxv1i64, 31},
    {Intrinsic::bswap, MVT::nxv2i64, 31},
    {Intrinsic::bswap, MVT::nxv4i64, 31},
    {Intrinsic::bswap, MVT::nxv8i64, 31},
    {Intrinsic::vp_bswap, MVT::v2i16, 3},
    {Intrinsic::vp_bswap, MVT::v4i16, 3},
    {Intrinsic::vp_bswap, MVT::v8i16, 3},
    {Intrinsic::vp_bswap, MVT::v16i16, 3},
    {Intrinsic::vp_bswap, MVT::nxv1i16, 3},
    {Intrinsic::vp_bswap, MVT::nxv2i16, 3},
    {Intrinsic::vp_bswap, MVT::nxv4i16, 3},
    {Intrinsic::vp_bswap, MVT::nxv8i16, 3},
    {Intrinsic::vp_bswap, MVT::nxv16i16, 3},
    {Intrinsic::vp_bswap, MVT::v2i32, 12},
    {Intrinsic::vp_bswap, MVT::v4i32, 12},
    {Intrinsic::vp_bswap, MVT::v8i32, 12},
    {Intrinsic::vp_bswap, MVT::v16i32, 12},
    {Intrinsic::vp_bswap, MVT::nxv1i32, 12},
    {Intrinsic::vp_bswap, MVT::nxv2i32, 12},
    {Intrinsic::vp_bswap, MVT::nxv4i32, 12},
    {Intrinsic::vp_bswap, MVT::nxv8i32, 12},
    {Intrinsic::vp_bswap, MVT::nxv16i32, 12},
    {Intrinsic::vp_bswap, MVT::v2i64, 31},
    {Intrinsic::vp_bswap, MVT::v4i64, 31},
    {Intrinsic::vp_bswap, MVT::v8i64, 31},
    {Intrinsic::vp_bswap, MVT::v16i64, 31},
    {Intrinsic::vp_bswap, MVT::nxv1i64, 31},
    {Intrinsic::vp_bswap, MVT::nxv2i64, 31},
    {Intrinsic::vp_bswap, MVT::nxv4i64, 31},
    {Intrinsic::vp_bswap, MVT::nxv8i64, 31},
    {Intrinsic::vp_fshl, MVT::v2i8, 7},
    {Intrinsic::vp_fshl, MVT::v4i8, 7},
    {Intrinsic::vp_fshl, MVT::v8i8, 7},
    {Intrinsic::vp_fshl, MVT::v16i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv1i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv2i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv4i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv8i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv16i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv32i8, 7},
    {Intrinsic::vp_fshl, MVT::nxv64i8, 7},
    {Intrinsic::vp_fshl, MVT::v2i16, 7},
    {Intrinsic::vp_fshl, MVT::v4i16, 7},
    {Intrinsic::vp_fshl, MVT::v8i16, 7},
    {Intrinsic::vp_fshl, MVT::v16i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv1i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv2i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv4i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv8i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv16i16, 7},
    {Intrinsic::vp_fshl, MVT::nxv32i16, 7},
    {Intrinsic::vp_fshl, MVT::v2i32, 7},
    {Intrinsic::vp_fshl, MVT::v4i32, 7},
    {Intrinsic::vp_fshl, MVT::v8i32, 7},
    {Intrinsic::vp_fshl, MVT::v16i32, 7},
    {Intrinsic::vp_fshl, MVT::nxv1i32, 7},
    {Intrinsic::vp_fshl, MVT::nxv2i32, 7},
    {Intrinsic::vp_fshl, MVT::nxv4i32, 7},
    {Intrinsic::vp_fshl, MVT::nxv8i32, 7},
    {Intrinsic::vp_fshl, MVT::nxv16i32, 7},
    {Intrinsic::vp_fshl, MVT::v2i64, 7},
    {Intrinsic::vp_fshl, MVT::v4i64, 7},
    {Intrinsic::vp_fshl, MVT::v8i64, 7},
    {Intrinsic::vp_fshl, MVT::v16i64, 7},
    {Intrinsic::vp_fshl, MVT::nxv1i64, 7},
    {Intrinsic::vp_fshl, MVT::nxv2i64, 7},
    {Intrinsic::vp_fshl, MVT::nxv4i64, 7},
    {Intrinsic::vp_fshl, MVT::nxv8i64, 7},
    {Intrinsic::vp_fshr, MVT::v2i8, 7},
    {Intrinsic::vp_fshr, MVT::v4i8, 7},
    {Intrinsic::vp_fshr, MVT::v8i8, 7},
    {Intrinsic::vp_fshr, MVT::v16i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv1i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv2i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv4i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv8i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv16i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv32i8, 7},
    {Intrinsic::vp_fshr, MVT::nxv64i8, 7},
    {Intrinsic::vp_fshr, MVT::v2i16, 7},
    {Intrinsic::vp_fshr, MVT::v4i16, 7},
    {Intrinsic::vp_fshr, MVT::v8i16, 7},
    {Intrinsic::vp_fshr, MVT::v16i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv1i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv2i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv4i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv8i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv16i16, 7},
    {Intrinsic::vp_fshr, MVT::nxv32i16, 7},
    {Intrinsic::vp_fshr, MVT::v2i32, 7},
    {Intrinsic::vp_fshr, MVT::v4i32, 7},
    {Intrinsic::vp_fshr, MVT::v8i32, 7},
    {Intrinsic::vp_fshr, MVT::v16i32, 7},
    {Intrinsic::vp_fshr, MVT::nxv1i32, 7},
    {Intrinsic::vp_fshr, MVT::nxv2i32, 7},
    {Intrinsic::vp_fshr, MVT::nxv4i32, 7},
    {Intrinsic::vp_fshr, MVT::nxv8i32, 7},
    {Intrinsic::vp_fshr, MVT::nxv16i32, 7},
    {Intrinsic::vp_fshr, MVT::v2i64, 7},
    {Intrinsic::vp_fshr, MVT::v4i64, 7},
    {Intrinsic::vp_fshr, MVT::v8i64, 7},
    {Intrinsic::vp_fshr, MVT::v16i64, 7},
    {Intrinsic::vp_fshr, MVT::nxv1i64, 7},
    {Intrinsic::vp_fshr, MVT::nxv2i64, 7},
    {Intrinsic::vp_fshr, MVT::nxv4i64, 7},
    {Intrinsic::vp_fshr, MVT::nxv8i64, 7},
    {Intrinsic::bitreverse, MVT::v2i8, 17},
    {Intrinsic::bitreverse, MVT::v4i8, 17},
    {Intrinsic::bitreverse, MVT::v8i8, 17},
    {Intrinsic::bitreverse, MVT::v16i8, 17},
    {Intrinsic::bitreverse, MVT::nxv1i8, 17},
    {Intrinsic::bitreverse, MVT::nxv2i8, 17},
    {Intrinsic::bitreverse, MVT::nxv4i8, 17},
    {Intrinsic::bitreverse, MVT::nxv8i8, 17},
    {Intrinsic::bitreverse, MVT::nxv16i8, 17},
    {Intrinsic::bitreverse, MVT::v2i16, 24},
    {Intrinsic::bitreverse, MVT::v4i16, 24},
    {Intrinsic::bitreverse, MVT::v8i16, 24},
    {Intrinsic::bitreverse, MVT::v16i16, 24},
    {Intrinsic::bitreverse, MVT::nxv1i16, 24},
    {Intrinsic::bitreverse, MVT::nxv2i16, 24},
    {Intrinsic::bitreverse, MVT::nxv4i16, 24},
    {Intrinsic::bitreverse, MVT::nxv8i16, 24},
    {Intrinsic::bitreverse, MVT::nxv16i16, 24},
    {Intrinsic::bitreverse, MVT::v2i32, 33},
    {Intrinsic::bitreverse, MVT::v4i32, 33},
    {Intrinsic::bitreverse, MVT::v8i32, 33},
    {Intrinsic::bitreverse, MVT::v16i32, 33},
    {Intrinsic::bitreverse, MVT::nxv1i32, 33},
    {Intrinsic::bitreverse, MVT::nxv2i32, 33},
    {Intrinsic::bitreverse, MVT::nxv4i32, 33},
    {Intrinsic::bitreverse, MVT::nxv8i32, 33},
    {Intrinsic::bitreverse, MVT::nxv16i32, 33},
    {Intrinsic::bitreverse, MVT::v2i64, 52},
    {Intrinsic::bitreverse, MVT::v4i64, 52},
    {Intrinsic::bitreverse, MVT::v8i64, 52},
    {Intrinsic::bitreverse, MVT::v16i64, 52},
    {Intrinsic::bitreverse, MVT::nxv1i64, 52},
    {Intrinsic::bitreverse, MVT::nxv2i64, 52},
    {Intrinsic::bitreverse, MVT::nxv4i64, 52},
    {Intrinsic::bitreverse, MVT::nxv8i64, 52},
    {Intrinsic::vp_bitreverse, MVT::v2i8, 17},
    {Intrinsic::vp_bitreverse, MVT::v4i8, 17},
    {Intrinsic::vp_bitreverse, MVT::v8i8, 17},
    {Intrinsic::vp_bitreverse, MVT::v16i8, 17},
    {Intrinsic::vp_bitreverse, MVT::nxv1i8, 17},
    {Intrinsic::vp_bitreverse, MVT::nxv2i8, 17},
    {Intrinsic::vp_bitreverse, MVT::nxv4i8, 17},
    {Intrinsic::vp_bitreverse, MVT::nxv8i8, 17},
    {Intrinsic::vp_bitreverse, MVT::nxv16i8, 17},
    {Intrinsic::vp_bitreverse, MVT::v2i16, 24},
    {Intrinsic::vp_bitreverse, MVT::v4i16, 24},
    {Intrinsic::vp_bitreverse, MVT::v8i16, 24},
    {Intrinsic::vp_bitreverse, MVT::v16i16, 24},
    {Intrinsic::vp_bitreverse, MVT::nxv1i16, 24},
    {Intrinsic::vp_bitreverse, MVT::nxv2i16, 24},
    {Intrinsic::vp_bitreverse, MVT::nxv4i16, 24},
    {Intrinsic::vp_bitreverse, MVT::nxv8i16, 24},
    {Intrinsic::vp_bitreverse, MVT::nxv16i16, 24},
    {Intrinsic::vp_bitreverse, MVT::v2i32, 33},
    {Intrinsic::vp_bitreverse, MVT::v4i32, 33},
    {Intrinsic::vp_bitreverse, MVT::v8i32, 33},
    {Intrinsic::vp_bitreverse, MVT::v16i32, 33},
    {Intrinsic::vp_bitreverse, MVT::nxv1i32, 33},
    {Intrinsic::vp_bitreverse, MVT::nxv2i32, 33},
    {Intrinsic::vp_bitreverse, MVT::nxv4i32, 33},
    {Intrinsic::vp_bitreverse, MVT::nxv8i32, 33},
    {Intrinsic::vp_bitreverse, MVT::nxv16i32, 33},
    {Intrinsic::vp_bitreverse, MVT::v2i64, 52},
    {Intrinsic::vp_bitreverse, MVT::v4i64, 52},
    {Intrinsic::vp_bitreverse, MVT::v8i64, 52},
    {Intrinsic::vp_bitreverse, MVT::v16i64, 52},
    {Intrinsic::vp_bitreverse, MVT::nxv1i64, 52},
    {Intrinsic::vp_bitreverse, MVT::nxv2i64, 52},
    {Intrinsic::vp_bitreverse, MVT::nxv4i64, 52},
    {Intrinsic::vp_bitreverse, MVT::nxv8i64, 52},
    {Intrinsic::ctpop, MVT::v2i8, 12},
    {Intrinsic::ctpop, MVT::v4i8, 12},
    {Intrinsic::ctpop, MVT::v8i8, 12},
    {Intrinsic::ctpop, MVT::v16i8, 12},
    {Intrinsic::ctpop, MVT::nxv1i8, 12},
    {Intrinsic::ctpop, MVT::nxv2i8, 12},
    {Intrinsic::ctpop, MVT::nxv4i8, 12},
    {Intrinsic::ctpop, MVT::nxv8i8, 12},
    {Intrinsic::ctpop, MVT::nxv16i8, 12},
    {Intrinsic::ctpop, MVT::v2i16, 19},
    {Intrinsic::ctpop, MVT::v4i16, 19},
    {Intrinsic::ctpop, MVT::v8i16, 19},
    {Intrinsic::ctpop, MVT::v16i16, 19},
    {Intrinsic::ctpop, MVT::nxv1i16, 19},
    {Intrinsic::ctpop, MVT::nxv2i16, 19},
    {Intrinsic::ctpop, MVT::nxv4i16, 19},
    {Intrinsic::ctpop, MVT::nxv8i16, 19},
    {Intrinsic::ctpop, MVT::nxv16i16, 19},
    {Intrinsic::ctpop, MVT::v2i32, 20},
    {Intrinsic::ctpop, MVT::v4i32, 20},
    {Intrinsic::ctpop, MVT::v8i32, 20},
    {Intrinsic::ctpop, MVT::v16i32, 20},
    {Intrinsic::ctpop, MVT::nxv1i32, 20},
    {Intrinsic::ctpop, MVT::nxv2i32, 20},
    {Intrinsic::ctpop, MVT::nxv4i32, 20},
    {Intrinsic::ctpop, MVT::nxv8i32, 20},
    {Intrinsic::ctpop, MVT::nxv16i32, 20},
    {Intrinsic::ctpop, MVT::v2i64, 21},
    {Intrinsic::ctpop, MVT::v4i64, 21},
    {Intrinsic::ctpop, MVT::v8i64, 21},
    {Intrinsic::ctpop, MVT::v16i64, 21},
    {Intrinsic::ctpop, MVT::nxv1i64, 21},
    {Intrinsic::ctpop, MVT::nxv2i64, 21},
    {Intrinsic::ctpop, MVT::nxv4i64, 21},
    {Intrinsic::ctpop, MVT::nxv8i64, 21},
    {Intrinsic::vp_ctpop, MVT::v2i8, 12},
    {Intrinsic::vp_ctpop, MVT::v4i8, 12},
    {Intrinsic::vp_ctpop, MVT::v8i8, 12},
    {Intrinsic::vp_ctpop, MVT::v16i8, 12},
    {Intrinsic::vp_ctpop, MVT::nxv1i8, 12},
    {Intrinsic::vp_ctpop, MVT::nxv2i8, 12},
    {Intrinsic::vp_ctpop, MVT::nxv4i8, 12},
    {Intrinsic::vp_ctpop, MVT::nxv8i8, 12},
    {Intrinsic::vp_ctpop, MVT::nxv16i8, 12},
    {Intrinsic::vp_ctpop, MVT::v2i16, 19},
    {Intrinsic::vp_ctpop, MVT::v4i16, 19},
    {Intrinsic::vp_ctpop, MVT::v8i16, 19},
    {Intrinsic::vp_ctpop, MVT::v16i16, 19},
    {Intrinsic::vp_ctpop, MVT::nxv1i16, 19},
    {Intrinsic::vp_ctpop, MVT::nxv2i16, 19},
    {Intrinsic::vp_ctpop, MVT::nxv4i16, 19},
    {Intrinsic::vp_ctpop, MVT::nxv8i16, 19},
    {Intrinsic::vp_ctpop, MVT::nxv16i16, 19},
    {Intrinsic::vp_ctpop, MVT::v2i32, 20},
    {Intrinsic::vp_ctpop, MVT::v4i32, 20},
    {Intrinsic::vp_ctpop, MVT::v8i32, 20},
    {Intrinsic::vp_ctpop, MVT::v16i32, 20},
    {Intrinsic::vp_ctpop, MVT::nxv1i32, 20},
    {Intrinsic::vp_ctpop, MVT::nxv2i32, 20},
    {Intrinsic::vp_ctpop, MVT::nxv4i32, 20},
    {Intrinsic::vp_ctpop, MVT::nxv8i32, 20},
    {Intrinsic::vp_ctpop, MVT::nxv16i32, 20},
    {Intrinsic::vp_ctpop, MVT::v2i64, 21},
    {Intrinsic::vp_ctpop, MVT::v4i64, 21},
    {Intrinsic::vp_ctpop, MVT::v8i64, 21},
    {Intrinsic::vp_ctpop, MVT::v16i64, 21},
    {Intrinsic::vp_ctpop, MVT::nxv1i64, 21},
    {Intrinsic::vp_ctpop, MVT::nxv2i64, 21},
    {Intrinsic::vp_ctpop, MVT::nxv4i64, 21},
    {Intrinsic::vp_ctpop, MVT::nxv8i64, 21},
    {Intrinsic::vp_ctlz, MVT::v2i8, 19},
    {Intrinsic::vp_ctlz, MVT::v4i8, 19},
    {Intrinsic::vp_ctlz, MVT::v8i8, 19},
    {Intrinsic::vp_ctlz, MVT::v16i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv1i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv2i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv4i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv8i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv16i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv32i8, 19},
    {Intrinsic::vp_ctlz, MVT::nxv64i8, 19},
    {Intrinsic::vp_ctlz, MVT::v2i16, 28},
    {Intrinsic::vp_ctlz, MVT::v4i16, 28},
    {Intrinsic::vp_ctlz, MVT::v8i16, 28},
    {Intrinsic::vp_ctlz, MVT::v16i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv1i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv2i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv4i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv8i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv16i16, 28},
    {Intrinsic::vp_ctlz, MVT::nxv32i16, 28},
    {Intrinsic::vp_ctlz, MVT::v2i32, 31},
    {Intrinsic::vp_ctlz, MVT::v4i32, 31},
    {Intrinsic::vp_ctlz, MVT::v8i32, 31},
    {Intrinsic::vp_ctlz, MVT::v16i32, 31},
    {Intrinsic::vp_ctlz, MVT::nxv1i32, 31},
    {Intrinsic::vp_ctlz, MVT::nxv2i32, 31},
    {Intrinsic::vp_ctlz, MVT::nxv4i32, 31},
    {Intrinsic::vp_ctlz, MVT::nxv8i32, 31},
    {Intrinsic::vp_ctlz, MVT::nxv16i32, 31},
    {Intrinsic::vp_ctlz, MVT::v2i64, 35},
    {Intrinsic::vp_ctlz, MVT::v4i64, 35},
    {Intrinsic::vp_ctlz, MVT::v8i64, 35},
    {Intrinsic::vp_ctlz, MVT::v16i64, 35},
    {Intrinsic::vp_ctlz, MVT::nxv1i64, 35},
    {Intrinsic::vp_ctlz, MVT::nxv2i64, 35},
    {Intrinsic::vp_ctlz, MVT::nxv4i64, 35},
    {Intrinsic::vp_ctlz, MVT::nxv8i64, 35},
    {Intrinsic::vp_cttz, MVT::v2i8, 16},
    {Intrinsic::vp_cttz, MVT::v4i8, 16},
    {Intrinsic::vp_cttz, MVT::v8i8, 16},
    {Intrinsic::vp_cttz, MVT::v16i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv1i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv2i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv4i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv8i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv16i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv32i8, 16},
    {Intrinsic::vp_cttz, MVT::nxv64i8, 16},
    {Intrinsic::vp_cttz, MVT::v2i16, 23},
    {Intrinsic::vp_cttz, MVT::v4i16, 23},
    {Intrinsic::vp_cttz, MVT::v8i16, 23},
    {Intrinsic::vp_cttz, MVT::v16i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv1i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv2i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv4i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv8i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv16i16, 23},
    {Intrinsic::vp_cttz, MVT::nxv32i16, 23},
    {Intrinsic::vp_cttz, MVT::v2i32, 24},
    {Intrinsic::vp_cttz, MVT::v4i32, 24},
    {Intrinsic::vp_cttz, MVT::v8i32, 24},
    {Intrinsic::vp_cttz, MVT::v16i32, 24},
    {Intrinsic::vp_cttz, MVT::nxv1i32, 24},
    {Intrinsic::vp_cttz, MVT::nxv2i32, 24},
    {Intrinsic::vp_cttz, MVT::nxv4i32, 24},
    {Intrinsic::vp_cttz, MVT::nxv8i32, 24},
    {Intrinsic::vp_cttz, MVT::nxv16i32, 24},
    {Intrinsic::vp_cttz, MVT::v2i64, 25},
    {Intrinsic::vp_cttz, MVT::v4i64, 25},
    {Intrinsic::vp_cttz, MVT::v8i64, 25},
    {Intrinsic::vp_cttz, MVT::v16i64, 25},
    {Intrinsic::vp_cttz, MVT::nxv1i64, 25},
    {Intrinsic::vp_cttz, MVT::nxv2i64, 25},
    {Intrinsic::vp_cttz, MVT::nxv4i64, 25},
    {Intrinsic::vp_cttz, MVT::nxv8i64, 25},
};

static unsigned getISDForVPIntrinsicID(Intrinsic::ID ID) {
  switch (ID) {
#define HELPER_MAP_VPID_TO_VPSD(VPID, VPSD)                                    \
  case Intrinsic::VPID:                                                        \
    return ISD::VPSD;
#include "llvm/IR/VPIntrinsics.def"
#undef HELPER_MAP_VPID_TO_VPSD
  }
  return ISD::DELETED_NODE;
}

InstructionCost
RISCVTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                    TTI::TargetCostKind CostKind) {
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
  case Intrinsic::ceil:
  case Intrinsic::floor:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::roundeven: {
    // These all use the same code.
    auto LT = getTypeLegalizationCost(RetTy);
    if (!LT.second.isVector() && TLI->isOperationCustom(ISD::FCEIL, LT.second))
      return LT.first * 8;
    break;
  }
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax: {
    auto LT = getTypeLegalizationCost(RetTy);
    if ((ST->hasVInstructions() && LT.second.isVector()) ||
        (LT.second.isScalarInteger() && ST->hasStdExtZbb()))
      return LT.first;
    break;
  }
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::usub_sat: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector())
      return LT.first;
    break;
  }
  case Intrinsic::abs: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector()) {
      // vrsub.vi v10, v8, 0
      // vmax.vv v8, v8, v10
      return LT.first * 2;
    }
    break;
  }
  case Intrinsic::fabs:
  case Intrinsic::sqrt: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector())
      return LT.first;
    break;
  }
  // TODO: add more intrinsic
  case Intrinsic::experimental_stepvector: {
    unsigned Cost = 1; // vid
    auto LT = getTypeLegalizationCost(RetTy);
    return Cost + (LT.first - 1);
  }
  case Intrinsic::vp_rint: {
    // RISC-V target uses at least 5 instructions to lower rounding intrinsics.
    unsigned Cost = 5;
    auto LT = getTypeLegalizationCost(RetTy);
    if (TLI->isOperationCustom(ISD::VP_FRINT, LT.second))
      return Cost * LT.first;
    break;
  }
  case Intrinsic::vp_nearbyint: {
    // More one read and one write for fflags than vp_rint.
    unsigned Cost = 7;
    auto LT = getTypeLegalizationCost(RetTy);
    if (TLI->isOperationCustom(ISD::VP_FRINT, LT.second))
      return Cost * LT.first;
    break;
  }
  case Intrinsic::vp_ceil:
  case Intrinsic::vp_floor:
  case Intrinsic::vp_round:
  case Intrinsic::vp_roundeven:
  case Intrinsic::vp_roundtozero: {
    // Rounding with static rounding mode needs two more instructions to
    // swap/write FRM than vp_rint.
    unsigned Cost = 7;
    auto LT = getTypeLegalizationCost(RetTy);
    unsigned VPISD = getISDForVPIntrinsicID(ICA.getID());
    if (TLI->isOperationCustom(VPISD, LT.second))
      return Cost * LT.first;
    break;
  }
  }

  if (ST->hasVInstructions() && RetTy->isVectorTy()) {
    auto LT = getTypeLegalizationCost(RetTy);
    if (const auto *Entry = CostTableLookup(VectorIntrinsicCostTable,
                                            ICA.getID(), LT.second))
      return LT.first * Entry->Cost;
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
      if (Src->getScalarSizeInBits() == 1) {
        // We do not use vsext/vzext to extend from mask vector.
        // Instead we use the following instructions to extend from mask vector:
        // vmv.v.i v8, 0
        // vmerge.vim v8, v8, -1, v0
        return 2;
      }
      return 1;
    case ISD::TRUNCATE:
      if (Dst->getScalarSizeInBits() == 1) {
        // We do not use several vncvt to truncate to mask vector. So we could
        // not use PowDiff to calculate it.
        // Instead we use the following instructions to truncate to mask vector:
        // vand.vi v8, v8, 1
        // vmsne.vi v0, v8, 0
        return 2;
      }
      [[fallthrough]];
    case ISD::FP_EXTEND:
    case ISD::FP_ROUND:
      // Counts of narrow/widen instructions.
      return std::abs(PowDiff);
    case ISD::FP_TO_SINT:
    case ISD::FP_TO_UINT:
    case ISD::SINT_TO_FP:
    case ISD::UINT_TO_FP:
      if (Src->getScalarSizeInBits() == 1 || Dst->getScalarSizeInBits() == 1) {
        // The cost of convert from or to mask vector is different from other
        // cases. We could not use PowDiff to calculate it.
        // For mask vector to fp, we should use the following instructions:
        // vmv.v.i v8, 0
        // vmerge.vim v8, v8, -1, v0
        // vfcvt.f.x.v v8, v8

        // And for fp vector to mask, we use:
        // vfncvt.rtz.x.f.w v9, v8
        // vand.vi v8, v9, 1
        // vmsne.vi v0, v8, 0
        return 3;
      }
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

unsigned RISCVTTIImpl::getEstimatedVLFor(VectorType *Ty) {
  if (isa<ScalableVectorType>(Ty)) {
    const unsigned EltSize = DL.getTypeSizeInBits(Ty->getElementType());
    const unsigned MinSize = DL.getTypeSizeInBits(Ty).getKnownMinValue();
    const unsigned VectorBits = *getVScaleForTuning() * RISCV::RVVBitsPerBlock;
    return RISCVTargetLowering::computeVLMAX(VectorBits, EltSize, MinSize);
  }
  return cast<FixedVectorType>(Ty)->getNumElements();
}

InstructionCost
RISCVTTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                     bool IsUnsigned,
                                     TTI::TargetCostKind CostKind) {
  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (Ty->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  if (Ty->getElementType()->isIntegerTy(1))
    // vcpop sequences, see vreduction-mask.ll.  umax, smin actually only
    // cost 2, but we don't have enough info here so we slightly over cost.
    return (LT.first - 1) + 3;

  // IR Reduction is composed by two vmv and one rvv reduction instruction.
  InstructionCost BaseCost = 2;
  unsigned VL = getEstimatedVLFor(Ty);
  return (LT.first - 1) + BaseCost + Log2_32_Ceil(VL);
}

InstructionCost
RISCVTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                         std::optional<FastMathFlags> FMF,
                                         TTI::TargetCostKind CostKind) {
  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (Ty->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  if (ISD != ISD::ADD && ISD != ISD::OR && ISD != ISD::XOR && ISD != ISD::AND &&
      ISD != ISD::FADD)
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  if (Ty->getElementType()->isIntegerTy(1))
    // vcpop sequences, see vreduction-mask.ll
    return (LT.first - 1) + (ISD == ISD::AND ? 3 : 2);

  // IR Reduction is composed by two vmv and one rvv reduction instruction.
  InstructionCost BaseCost = 2;
  unsigned VL = getEstimatedVLFor(Ty);
  if (TTI::requiresOrderedReduction(FMF))
    return (LT.first - 1) + BaseCost + VL;
  return (LT.first - 1) + BaseCost + Log2_32_Ceil(VL);
}

InstructionCost RISCVTTIImpl::getExtendedReductionCost(
    unsigned Opcode, bool IsUnsigned, Type *ResTy, VectorType *ValTy,
    std::optional<FastMathFlags> FMF, TTI::TargetCostKind CostKind) {
  if (isa<FixedVectorType>(ValTy) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  // Skip if scalar size of ResTy is bigger than ELEN.
  if (ResTy->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  if (Opcode != Instruction::Add && Opcode != Instruction::FAdd)
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);

  if (ResTy->getScalarSizeInBits() != 2 * LT.second.getScalarSizeInBits())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  return (LT.first - 1) +
         getArithmeticReductionCost(Opcode, ValTy, FMF, CostKind);
}

InstructionCost RISCVTTIImpl::getStoreImmCost(Type *Ty,
                                              TTI::OperandValueInfo OpInfo,
                                              TTI::TargetCostKind CostKind) {
  assert(OpInfo.isConstant() && "non constant operand?");
  if (!isa<VectorType>(Ty))
    // FIXME: We need to account for immediate materialization here, but doing
    // a decent job requires more knowledge about the immediate than we
    // currently have here.
    return 0;

  if (OpInfo.isUniform())
    // vmv.x.i, vmv.v.x, or vfmv.v.f
    // We ignore the cost of the scalar constant materialization to be consistent
    // with how we treat scalar constants themselves just above.
    return 1;

  // Add a cost of address generation + the cost of the vector load. The
  // address is expected to be a PC relative offset to a constant pool entry
  // using auipc/addi.
  return 2 + getMemoryOpCost(Instruction::Load, Ty, DL.getABITypeAlign(Ty),
                             /*AddressSpace=*/0, CostKind);
}


InstructionCost RISCVTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                              MaybeAlign Alignment,
                                              unsigned AddressSpace,
                                              TTI::TargetCostKind CostKind,
                                              TTI::OperandValueInfo OpInfo,
                                              const Instruction *I) {
  InstructionCost Cost = 0;
  if (Opcode == Instruction::Store && OpInfo.isConstant())
    Cost += getStoreImmCost(Src, OpInfo, CostKind);
  return Cost + BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                       CostKind, OpInfo, I);
}

InstructionCost RISCVTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                                 Type *CondTy,
                                                 CmpInst::Predicate VecPred,
                                                 TTI::TargetCostKind CostKind,
                                                 const Instruction *I) {
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     I);

  if (isa<FixedVectorType>(ValTy) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     I);

  // Skip if scalar size of ValTy is bigger than ELEN.
  if (ValTy->isVectorTy() && ValTy->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     I);

  if (Opcode == Instruction::Select && ValTy->isVectorTy()) {
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);
    if (CondTy->isVectorTy()) {
      if (ValTy->getScalarSizeInBits() == 1) {
        // vmandn.mm v8, v8, v9
        // vmand.mm v9, v0, v9
        // vmor.mm v0, v9, v8
        return LT.first * 3;
      }
      // vselect and max/min are supported natively.
      return LT.first * 1;
    }

    if (ValTy->getScalarSizeInBits() == 1) {
      //  vmv.v.x v9, a0
      //  vmsne.vi v9, v9, 0
      //  vmandn.mm v8, v8, v9
      //  vmand.mm v9, v0, v9
      //  vmor.mm v0, v9, v8
      return LT.first * 5;
    }

    // vmv.v.x v10, a0
    // vmsne.vi v0, v10, 0
    // vmerge.vvm v8, v9, v8, v0
    return LT.first * 3;
  }

  if ((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
      ValTy->isVectorTy()) {
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);

    // Support natively.
    if (CmpInst::isIntPredicate(VecPred))
      return LT.first * 1;

    // If we do not support the input floating point vector type, use the base
    // one which will calculate as:
    // ScalarizeCost + Num * Cost for fixed vector,
    // InvalidCost for scalable vector.
    if ((ValTy->getScalarSizeInBits() == 16 && !ST->hasVInstructionsF16()) ||
        (ValTy->getScalarSizeInBits() == 32 && !ST->hasVInstructionsF32()) ||
        (ValTy->getScalarSizeInBits() == 64 && !ST->hasVInstructionsF64()))
      return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                       I);
    switch (VecPred) {
      // Support natively.
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_UNE:
      return LT.first * 1;
    // TODO: Other comparisons?
    default:
      break;
    }
  }

  // TODO: Add cost for scalar type.

  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
}

InstructionCost RISCVTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                 TTI::TargetCostKind CostKind,
                                                 unsigned Index, Value *Op0,
                                                 Value *Op1) {
  assert(Val->isVectorTy() && "This must be a vector type");

  if (Opcode != Instruction::ExtractElement &&
      Opcode != Instruction::InsertElement)
    return BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1);

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Val);

  // This type is legalized to a scalar type.
  if (!LT.second.isVector())
    return 0;

  // For unsupported scalable vector.
  if (LT.second.isScalableVector() && !LT.first.isValid())
    return LT.first;

  if (!isTypeLegal(Val))
    return BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1);

  // In RVV, we could use vslidedown + vmv.x.s to extract element from vector
  // and vslideup + vmv.s.x to insert element to vector.
  unsigned BaseCost = 1;
  // When insertelement we should add the index with 1 as the input of vslideup.
  unsigned SlideCost = Opcode == Instruction::InsertElement ? 2 : 1;

  if (Index != -1U) {
    // The type may be split. For fixed-width vectors we can normalize the
    // index to the new type.
    if (LT.second.isFixedLengthVector()) {
      unsigned Width = LT.second.getVectorNumElements();
      Index = Index % Width;
    }

    // We could extract/insert the first element without vslidedown/vslideup.
    if (Index == 0)
      SlideCost = 0;
    else if (Opcode == Instruction::InsertElement)
      SlideCost = 1; // With a constant index, we do not need to use addi.
  }

  // Mask vector extract/insert element is different from normal case.
  if (Val->getScalarSizeInBits() == 1) {
    // For extractelement, we need the following instructions:
    // vmv.v.i v8, 0
    // vmerge.vim v8, v8, 1, v0
    // vsetivli zero, 1, e8, m2, ta, mu (not count)
    // vslidedown.vx v8, v8, a0
    // vmv.x.s a0, v8

    // For insertelement, we need the following instructions:
    // vsetvli a2, zero, e8, m1, ta, mu (not count)
    // vmv.s.x v8, a0
    // vmv.v.i v9, 0
    // vmerge.vim v9, v9, 1, v0
    // addi a0, a1, 1
    // vsetvli zero, a0, e8, m1, tu, mu (not count)
    // vslideup.vx v9, v8, a1
    // vsetvli a0, zero, e8, m1, ta, mu (not count)
    // vand.vi v8, v9, 1
    // vmsne.vi v0, v8, 0

    // TODO: should we count these special vsetvlis?
    BaseCost = Opcode == Instruction::InsertElement ? 5 : 3;
  }
  // Extract i64 in the target that has XLEN=32 need more instruction.
  if (Val->getScalarType()->isIntegerTy() &&
      ST->getXLen() < Val->getScalarSizeInBits()) {
    // For extractelement, we need the following instructions:
    // vsetivli zero, 1, e64, m1, ta, mu (not count)
    // vslidedown.vx v8, v8, a0
    // vmv.x.s a0, v8
    // li a1, 32
    // vsrl.vx v8, v8, a1
    // vmv.x.s a1, v8

    // For insertelement, we need the following instructions:
    // vsetivli zero, 2, e32, m4, ta, mu (not count)
    // vmv.v.i v12, 0
    // vslide1up.vx v16, v12, a1
    // vslide1up.vx v12, v16, a0
    // addi a0, a2, 1
    // vsetvli zero, a0, e64, m4, tu, mu (not count)
    // vslideup.vx v8, v12, a2

    // TODO: should we count these special vsetvlis?
    BaseCost = Opcode == Instruction::InsertElement ? 3 : 4;
  }
  return BaseCost + SlideCost;
}

InstructionCost RISCVTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args, const Instruction *CxtI) {

  // TODO: Handle more cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (isa<VectorType>(Ty) && Ty->getScalarSizeInBits() > ST->getELEN())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);

  // TODO: Handle scalar type.
  if (!LT.second.isVector())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);


  auto getConstantMatCost =
    [&](unsigned Operand, TTI::OperandValueInfo OpInfo) -> InstructionCost {
    if (OpInfo.isUniform() && TLI->canSplatOperand(Opcode, Operand))
      // Two sub-cases:
      // * Has a 5 bit immediate operand which can be splatted.
      // * Has a larger immediate which must be materialized in scalar register
      // We return 0 for both as we currently ignore the cost of materializing
      // scalar constants in GPRs.
      return 0;

    // Add a cost of address generation + the cost of the vector load. The
    // address is expected to be a PC relative offset to a constant pool entry
    // using auipc/addi.
    return 2 + getMemoryOpCost(Instruction::Load, Ty, DL.getABITypeAlign(Ty),
                               /*AddressSpace=*/0, CostKind);
  };

  // Add the cost of materializing any constant vectors required.
  InstructionCost ConstantMatCost = 0;
  if (Op1Info.isConstant())
    ConstantMatCost += getConstantMatCost(0, Op1Info);
  if (Op2Info.isConstant())
    ConstantMatCost += getConstantMatCost(1, Op2Info);

  switch (TLI->InstructionOpcodeToISD(Opcode)) {
  case ISD::ADD:
  case ISD::SUB:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FNEG: {
    return ConstantMatCost + getLMULCost(LT.second) * LT.first * 1;
  }
  default:
    return ConstantMatCost +
           BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);
  }
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
      Cost += getInstructionCost(&I, Operands,
                                 TargetTransformInfo::TCK_SizeAndLatency);
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
  TypeSize Size = DL.getTypeSizeInBits(Ty);
  if (Ty->isVectorTy()) {
    if (Size.isScalable() && ST->hasVInstructions())
      return divideCeil(Size.getKnownMinValue(), RISCV::RVVBitsPerBlock);

    if (ST->useRVVForFixedLengthVectors())
      return divideCeil(Size, ST->getRealMinVLen());
  }

  return BaseT::getRegUsageForType(Ty);
}

unsigned RISCVTTIImpl::getMaximumVF(unsigned ElemWidth, unsigned Opcode) const {
  // This interface is currently only used by SLP.  Returning 1 (which is the
  // default value for SLPMaxVF) disables SLP. We currently have a cost modeling
  // problem w/ constant materialization which causes SLP to perform majorly
  // unprofitable transformations.
  // TODO: Figure out constant materialization cost modeling and remove.
  return SLPMaxVF;
}

bool RISCVTTIImpl::isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                                 const TargetTransformInfo::LSRCost &C2) {
  // RISCV specific here are "instruction number 1st priority".
  return std::tie(C1.Insns, C1.NumRegs, C1.AddRecCost,
                  C1.NumIVMuls, C1.NumBaseAdds,
                  C1.ScaleCost, C1.ImmCost, C1.SetupCost) <
         std::tie(C2.Insns, C2.NumRegs, C2.AddRecCost,
                  C2.NumIVMuls, C2.NumBaseAdds,
                  C2.ScaleCost, C2.ImmCost, C2.SetupCost);
}
