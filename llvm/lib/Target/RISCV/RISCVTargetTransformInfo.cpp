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
    if (Imm == UINT64_C(0xffffffff) && ST->hasStdExtZba())
      return TTI::TCC_Free;
    [[fallthrough]];
  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
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

Optional<unsigned> RISCVTTIImpl::getMaxVScale() const {
  if (ST->hasVInstructions())
    return ST->getRealMaxVLen() / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}

Optional<unsigned> RISCVTTIImpl::getVScaleForTuning() const {
  if (ST->hasVInstructions())
    return ST->getRealMinVLen() / RISCV::RVVBitsPerBlock;
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
        ST->hasVInstructions() ? LMUL * RISCV::RVVBitsPerBlock : 0);
  }

  llvm_unreachable("Unsupported register kind");
}

InstructionCost RISCVTTIImpl::getSpliceCost(VectorType *Tp, int Index) {
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);

  unsigned Cost = 2; // vslidedown+vslideup.
  // TODO: LMUL should increase cost.
  // TODO: Multiplying by LT.first implies this legalizes into multiple copies
  // of similar code, but I think we expand through memory.
  return Cost * LT.first;
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

  return BaseT::getShuffleCost(Kind, Tp, Mask, CostKind, Index, SubTp);
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
    {Intrinsic::fabs, MVT::v2f32, 1},
    {Intrinsic::fabs, MVT::v4f32, 1},
    {Intrinsic::fabs, MVT::v8f32, 1},
    {Intrinsic::fabs, MVT::v16f32, 1},
    {Intrinsic::fabs, MVT::nxv2f32, 1},
    {Intrinsic::fabs, MVT::nxv4f32, 1},
    {Intrinsic::fabs, MVT::nxv8f32, 1},
    {Intrinsic::fabs, MVT::nxv16f32, 1},
    {Intrinsic::fabs, MVT::v2f64, 1},
    {Intrinsic::fabs, MVT::v4f64, 1},
    {Intrinsic::fabs, MVT::v8f64, 1},
    {Intrinsic::fabs, MVT::v16f64, 1},
    {Intrinsic::fabs, MVT::nxv1f64, 1},
    {Intrinsic::fabs, MVT::nxv2f64, 1},
    {Intrinsic::fabs, MVT::nxv4f64, 1},
    {Intrinsic::fabs, MVT::nxv8f64, 1},
    {Intrinsic::sqrt, MVT::v2f32, 1},
    {Intrinsic::sqrt, MVT::v4f32, 1},
    {Intrinsic::sqrt, MVT::v8f32, 1},
    {Intrinsic::sqrt, MVT::v16f32, 1},
    {Intrinsic::sqrt, MVT::nxv2f32, 1},
    {Intrinsic::sqrt, MVT::nxv4f32, 1},
    {Intrinsic::sqrt, MVT::nxv8f32, 1},
    {Intrinsic::sqrt, MVT::nxv16f32, 1},
    {Intrinsic::sqrt, MVT::v2f64, 1},
    {Intrinsic::sqrt, MVT::v4f64, 1},
    {Intrinsic::sqrt, MVT::v8f64, 1},
    {Intrinsic::sqrt, MVT::v16f64, 1},
    {Intrinsic::sqrt, MVT::nxv1f64, 1},
    {Intrinsic::sqrt, MVT::nxv2f64, 1},
    {Intrinsic::sqrt, MVT::nxv4f64, 1},
    {Intrinsic::sqrt, MVT::nxv8f64, 1},
    {Intrinsic::bswap, MVT::v2i16, 3},
    {Intrinsic::bswap, MVT::v4i16, 3},
    {Intrinsic::bswap, MVT::v8i16, 3},
    {Intrinsic::bswap, MVT::v16i16, 3},
    {Intrinsic::bswap, MVT::nxv2i16, 3},
    {Intrinsic::bswap, MVT::nxv4i16, 3},
    {Intrinsic::bswap, MVT::nxv8i16, 3},
    {Intrinsic::bswap, MVT::nxv16i16, 3},
    {Intrinsic::bswap, MVT::v2i32, 12},
    {Intrinsic::bswap, MVT::v4i32, 12},
    {Intrinsic::bswap, MVT::v8i32, 12},
    {Intrinsic::bswap, MVT::v16i32, 12},
    {Intrinsic::bswap, MVT::nxv2i32, 12},
    {Intrinsic::bswap, MVT::nxv4i32, 12},
    {Intrinsic::bswap, MVT::nxv8i32, 12},
    {Intrinsic::bswap, MVT::nxv16i32, 12},
    {Intrinsic::bswap, MVT::v2i64, 31},
    {Intrinsic::bswap, MVT::v4i64, 31},
    {Intrinsic::bswap, MVT::v8i64, 31},
    {Intrinsic::bswap, MVT::v16i64, 31},
    {Intrinsic::bswap, MVT::nxv2i64, 31},
    {Intrinsic::bswap, MVT::nxv4i64, 31},
    {Intrinsic::bswap, MVT::nxv8i64, 31},
    {Intrinsic::bitreverse, MVT::v2i8, 17},
    {Intrinsic::bitreverse, MVT::v4i8, 17},
    {Intrinsic::bitreverse, MVT::v8i8, 17},
    {Intrinsic::bitreverse, MVT::v16i8, 17},
    {Intrinsic::bitreverse, MVT::nxv2i8, 17},
    {Intrinsic::bitreverse, MVT::nxv4i8, 17},
    {Intrinsic::bitreverse, MVT::nxv8i8, 17},
    {Intrinsic::bitreverse, MVT::nxv16i8, 17},
    {Intrinsic::bitreverse, MVT::v2i16, 24},
    {Intrinsic::bitreverse, MVT::v4i16, 24},
    {Intrinsic::bitreverse, MVT::v8i16, 24},
    {Intrinsic::bitreverse, MVT::v16i16, 24},
    {Intrinsic::bitreverse, MVT::nxv2i16, 24},
    {Intrinsic::bitreverse, MVT::nxv4i16, 24},
    {Intrinsic::bitreverse, MVT::nxv8i16, 24},
    {Intrinsic::bitreverse, MVT::nxv16i16, 24},
    {Intrinsic::bitreverse, MVT::v2i32, 33},
    {Intrinsic::bitreverse, MVT::v4i32, 33},
    {Intrinsic::bitreverse, MVT::v8i32, 33},
    {Intrinsic::bitreverse, MVT::v16i32, 33},
    {Intrinsic::bitreverse, MVT::nxv2i32, 33},
    {Intrinsic::bitreverse, MVT::nxv4i32, 33},
    {Intrinsic::bitreverse, MVT::nxv8i32, 33},
    {Intrinsic::bitreverse, MVT::nxv16i32, 33},
    {Intrinsic::bitreverse, MVT::v2i64, 52},
    {Intrinsic::bitreverse, MVT::v4i64, 52},
    {Intrinsic::bitreverse, MVT::v8i64, 52},
    {Intrinsic::bitreverse, MVT::v16i64, 52},
    {Intrinsic::bitreverse, MVT::nxv2i64, 52},
    {Intrinsic::bitreverse, MVT::nxv4i64, 52},
    {Intrinsic::bitreverse, MVT::nxv8i64, 52},
    {Intrinsic::ctpop, MVT::v2i8, 12},
    {Intrinsic::ctpop, MVT::v4i8, 12},
    {Intrinsic::ctpop, MVT::v8i8, 12},
    {Intrinsic::ctpop, MVT::v16i8, 12},
    {Intrinsic::ctpop, MVT::nxv2i8, 12},
    {Intrinsic::ctpop, MVT::nxv4i8, 12},
    {Intrinsic::ctpop, MVT::nxv8i8, 12},
    {Intrinsic::ctpop, MVT::nxv16i8, 12},
    {Intrinsic::ctpop, MVT::v2i16, 19},
    {Intrinsic::ctpop, MVT::v4i16, 19},
    {Intrinsic::ctpop, MVT::v8i16, 19},
    {Intrinsic::ctpop, MVT::v16i16, 19},
    {Intrinsic::ctpop, MVT::nxv2i16, 19},
    {Intrinsic::ctpop, MVT::nxv4i16, 19},
    {Intrinsic::ctpop, MVT::nxv8i16, 19},
    {Intrinsic::ctpop, MVT::nxv16i16, 19},
    {Intrinsic::ctpop, MVT::v2i32, 20},
    {Intrinsic::ctpop, MVT::v4i32, 20},
    {Intrinsic::ctpop, MVT::v8i32, 20},
    {Intrinsic::ctpop, MVT::v16i32, 20},
    {Intrinsic::ctpop, MVT::nxv2i32, 20},
    {Intrinsic::ctpop, MVT::nxv4i32, 20},
    {Intrinsic::ctpop, MVT::nxv8i32, 20},
    {Intrinsic::ctpop, MVT::nxv16i32, 20},
    {Intrinsic::ctpop, MVT::v2i64, 21},
    {Intrinsic::ctpop, MVT::v4i64, 21},
    {Intrinsic::ctpop, MVT::v8i64, 21},
    {Intrinsic::ctpop, MVT::v16i64, 21},
    {Intrinsic::ctpop, MVT::nxv2i64, 21},
    {Intrinsic::ctpop, MVT::nxv4i64, 21},
    {Intrinsic::ctpop, MVT::nxv8i64, 21},
};

InstructionCost
RISCVTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                    TTI::TargetCostKind CostKind) {
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
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
  // TODO: add more intrinsic
  case Intrinsic::experimental_stepvector: {
    unsigned Cost = 1; // vid
    auto LT = getTypeLegalizationCost(RetTy);
    return Cost + (LT.first - 1);
  }
  default:
    if (ST->hasVInstructions() && RetTy->isVectorTy()) {
      auto LT = getTypeLegalizationCost(RetTy);
      if (const auto *Entry = CostTableLookup(VectorIntrinsicCostTable,
                                              ICA.getID(), LT.second))
        return LT.first * Entry->Cost;
    }
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
                                         Optional<FastMathFlags> FMF,
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
    Optional<FastMathFlags> FMF, TTI::TargetCostKind CostKind) {
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

InstructionCost RISCVTTIImpl::getVectorImmCost(VectorType *VecTy,
                                               TTI::OperandValueInfo OpInfo,
                                               TTI::TargetCostKind CostKind) {
  assert(OpInfo.isConstant() && "non constant operand?");
  APInt PseudoAddr = APInt::getAllOnes(DL.getPointerSizeInBits());
  // Add a cost of address load + the cost of the vector load.
  return RISCVMatInt::getIntMatCost(PseudoAddr, DL.getPointerSizeInBits(),
                                    getST()->getFeatureBits()) +
    getMemoryOpCost(Instruction::Load, VecTy, DL.getABITypeAlign(VecTy),
                    /*AddressSpace=*/0, CostKind);
}


InstructionCost RISCVTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                              MaybeAlign Alignment,
                                              unsigned AddressSpace,
                                              TTI::TargetCostKind CostKind,
                                              TTI::OperandValueInfo OpInfo,
                                              const Instruction *I) {
  InstructionCost Cost = 0;
  if (Opcode == Instruction::Store && isa<VectorType>(Src) && OpInfo.isConstant())
    Cost += getVectorImmCost(cast<VectorType>(Src), OpInfo, CostKind);
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

    // TODO: Add cost for fp vector compare instruction.
  }

  // TODO: Add cost for scalar type.

  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
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
  TypeSize Size = Ty->getPrimitiveSizeInBits();
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
