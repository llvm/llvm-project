//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include <cmath>
#include <optional>
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "riscvtti"

static cl::opt<unsigned> RVVRegisterWidthLMUL(
    "riscv-v-register-bit-width-lmul",
    cl::desc(
        "The LMUL to use for getRegisterBitWidth queries. Affects LMUL used "
        "by autovectorized code. Fractional LMULs are not supported."),
    cl::init(2), cl::Hidden);

static cl::opt<unsigned> SLPMaxVF(
    "riscv-v-slp-max-vf",
    cl::desc(
        "Overrides result used for getMaximumVF query which is used "
        "exclusively by SLP vectorizer."),
    cl::Hidden);

static cl::opt<unsigned>
    RVVMinTripCount("riscv-v-min-trip-count",
                    cl::desc("Set the lower bound of a trip count to decide on "
                             "vectorization while tail-folding."),
                    cl::init(5), cl::Hidden);

InstructionCost
RISCVTTIImpl::getRISCVInstructionCost(ArrayRef<unsigned> OpCodes, MVT VT,
                                      TTI::TargetCostKind CostKind) const {
  // Check if the type is valid for all CostKind
  if (!VT.isVector())
    return InstructionCost::getInvalid();
  size_t NumInstr = OpCodes.size();
  if (CostKind == TTI::TCK_CodeSize)
    return NumInstr;
  InstructionCost LMULCost = TLI->getLMULCost(VT);
  if ((CostKind != TTI::TCK_RecipThroughput) && (CostKind != TTI::TCK_Latency))
    return LMULCost * NumInstr;
  InstructionCost Cost = 0;
  for (auto Op : OpCodes) {
    switch (Op) {
    case RISCV::VRGATHER_VI:
      Cost += TLI->getVRGatherVICost(VT);
      break;
    case RISCV::VRGATHER_VV:
      Cost += TLI->getVRGatherVVCost(VT);
      break;
    case RISCV::VSLIDEUP_VI:
    case RISCV::VSLIDEDOWN_VI:
      Cost += TLI->getVSlideVICost(VT);
      break;
    case RISCV::VSLIDEUP_VX:
    case RISCV::VSLIDEDOWN_VX:
      Cost += TLI->getVSlideVXCost(VT);
      break;
    case RISCV::VREDMAX_VS:
    case RISCV::VREDMIN_VS:
    case RISCV::VREDMAXU_VS:
    case RISCV::VREDMINU_VS:
    case RISCV::VREDSUM_VS:
    case RISCV::VREDAND_VS:
    case RISCV::VREDOR_VS:
    case RISCV::VREDXOR_VS:
    case RISCV::VFREDMAX_VS:
    case RISCV::VFREDMIN_VS:
    case RISCV::VFREDUSUM_VS: {
      unsigned VL = VT.getVectorMinNumElements();
      if (!VT.isFixedLengthVector())
        VL *= *getVScaleForTuning();
      Cost += Log2_32_Ceil(VL);
      break;
    }
    case RISCV::VFREDOSUM_VS: {
      unsigned VL = VT.getVectorMinNumElements();
      if (!VT.isFixedLengthVector())
        VL *= *getVScaleForTuning();
      Cost += VL;
      break;
    }
    case RISCV::VMV_X_S:
    case RISCV::VMV_S_X:
    case RISCV::VFMV_F_S:
    case RISCV::VFMV_S_F:
    case RISCV::VMOR_MM:
    case RISCV::VMXOR_MM:
    case RISCV::VMAND_MM:
    case RISCV::VMANDN_MM:
    case RISCV::VMNAND_MM:
    case RISCV::VCPOP_M:
    case RISCV::VFIRST_M:
      Cost += 1;
      break;
    default:
      Cost += LMULCost;
    }
  }
  return Cost;
}

static InstructionCost getIntImmCostImpl(const DataLayout &DL,
                                         const RISCVSubtarget *ST,
                                         const APInt &Imm, Type *Ty,
                                         TTI::TargetCostKind CostKind,
                                         bool FreeZeroes) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty), *ST,
                                    /*CompressionCost=*/false, FreeZeroes);
}

InstructionCost
RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                            TTI::TargetCostKind CostKind) const {
  return getIntImmCostImpl(getDataLayout(), getST(), Imm, Ty, CostKind, false);
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
    unsigned Trailing = llvm::countr_zero(Mask);
    if (ShAmt == Trailing)
      return true;
  }

  return false;
}

InstructionCost RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                const APInt &Imm, Type *Ty,
                                                TTI::TargetCostKind CostKind,
                                                Instruction *Inst) const {
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
  case Instruction::Store: {
    // Use the materialization cost regardless of if it's the address or the
    // value that is constant, except for if the store is misaligned and
    // misaligned accesses are not legal (experience shows constant hoisting
    // can sometimes be harmful in such cases).
    if (Idx == 1 || !Inst)
      return getIntImmCostImpl(getDataLayout(), getST(), Imm, Ty, CostKind,
                               /*FreeZeroes=*/true);

    StoreInst *ST = cast<StoreInst>(Inst);
    if (!getTLI()->allowsMemoryAccessForAlignment(
            Ty->getContext(), DL, getTLI()->getValueType(DL, Ty),
            ST->getPointerAddressSpace(), ST->getAlign()))
      return TTI::TCC_Free;

    return getIntImmCostImpl(getDataLayout(), getST(), Imm, Ty, CostKind,
                             /*FreeZeroes=*/true);
  }
  case Instruction::Load:
    // If the address is a constant, use the materialization cost.
    return getIntImmCost(Imm, Ty, CostKind);
  case Instruction::And:
    // zext.h
    if (Imm == UINT64_C(0xffff) && ST->hasStdExtZbb())
      return TTI::TCC_Free;
    // zext.w
    if (Imm == UINT64_C(0xffffffff) &&
        ((ST->hasStdExtZba() && ST->isRV64()) || ST->isRV32()))
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
    // Power of 2 is a shift. Negated power of 2 is a shift and a negate.
    if (Imm.isPowerOf2() || Imm.isNegatedPowerOf2())
      return TTI::TCC_Free;
    // One more or less than a power of 2 can use SLLI+ADD/SUB.
    if ((Imm + 1).isPowerOf2() || (Imm - 1).isPowerOf2())
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
      if (Imm.getSignificantBits() <= 64 &&
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
                                  TTI::TargetCostKind CostKind) const {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

bool RISCVTTIImpl::hasActiveVectorLength(unsigned, Type *DataTy, Align) const {
  return ST->hasVInstructions();
}

TargetTransformInfo::PopcntSupportKind
RISCVTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return ST->hasStdExtZbb() || (ST->hasVendorXCVbitmanip() && !ST->is64Bit())
             ? TTI::PSK_FastHardware
             : TTI::PSK_Software;
}

InstructionCost RISCVTTIImpl::getPartialReductionCost(
    unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
    ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
    TTI::PartialReductionExtendKind OpBExtend,
    std::optional<unsigned> BinOp) const {

  // zve32x is broken for partial_reduce_umla, but let's make sure we
  // don't generate them.
  if (!ST->hasStdExtZvqdotq() || ST->getELen() < 64 ||
      Opcode != Instruction::Add || !BinOp || *BinOp != Instruction::Mul ||
      InputTypeA != InputTypeB || !InputTypeA->isIntegerTy(8) ||
      OpAExtend != OpBExtend || !AccumType->isIntegerTy(32) ||
      !VF.isKnownMultipleOf(4) || !VF.isScalable())
    return InstructionCost::getInvalid();

  Type *Tp = VectorType::get(AccumType, VF);
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
  // Note: Asuming all vqdot* variants are equal cost
  // TODO: Thread CostKind through this API
  return LT.first * getRISCVInstructionCost(RISCV::VQDOT_VV, LT.second,
                                            TTI::TCK_RecipThroughput);
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
  unsigned LMUL =
      llvm::bit_floor(std::clamp<unsigned>(RVVRegisterWidthLMUL, 1, 8));
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

InstructionCost
RISCVTTIImpl::getConstantPoolLoadCost(Type *Ty,
                                      TTI::TargetCostKind CostKind) const {
  // Add a cost of address generation + the cost of the load. The address
  // is expected to be a PC relative offset to a constant pool entry
  // using auipc/addi.
  return 2 + getMemoryOpCost(Instruction::Load, Ty, DL.getABITypeAlign(Ty),
                             /*AddressSpace=*/0, CostKind);
}

static bool isRepeatedConcatMask(ArrayRef<int> Mask, int &SubVectorSize) {
  unsigned Size = Mask.size();
  if (!isPowerOf2_32(Size))
    return false;
  for (unsigned I = 0; I != Size; ++I) {
    if (static_cast<unsigned>(Mask[I]) == I)
      continue;
    if (Mask[I] != 0)
      return false;
    if (Size % I != 0)
      return false;
    for (unsigned J = I + 1; J != Size; ++J)
      // Check the pattern is repeated.
      if (static_cast<unsigned>(Mask[J]) != J % I)
        return false;
    SubVectorSize = I;
    return true;
  }
  // That means Mask is <0, 1, 2, 3>. This is not a concatenation.
  return false;
}

static VectorType *getVRGatherIndexType(MVT DataVT, const RISCVSubtarget &ST,
                                        LLVMContext &C) {
  assert((DataVT.getScalarSizeInBits() != 8 ||
          DataVT.getVectorNumElements() <= 256) && "unhandled case in lowering");
  MVT IndexVT = DataVT.changeTypeToInteger();
  if (IndexVT.getScalarType().bitsGT(ST.getXLenVT()))
    IndexVT = IndexVT.changeVectorElementType(MVT::i16);
  return cast<VectorType>(EVT(IndexVT).getTypeForEVT(C));
}

/// Attempt to approximate the cost of a shuffle which will require splitting
/// during legalization.  Note that processShuffleMasks is not an exact proxy
/// for the algorithm used in LegalizeVectorTypes, but hopefully it's a
/// reasonably close upperbound.
static InstructionCost costShuffleViaSplitting(const RISCVTTIImpl &TTI,
                                               MVT LegalVT, VectorType *Tp,
                                               ArrayRef<int> Mask,
                                               TTI::TargetCostKind CostKind) {
  assert(LegalVT.isFixedLengthVector() && !Mask.empty() &&
         "Expected fixed vector type and non-empty mask");
  unsigned LegalNumElts = LegalVT.getVectorNumElements();
  // Number of destination vectors after legalization:
  unsigned NumOfDests = divideCeil(Mask.size(), LegalNumElts);
  // We are going to permute multiple sources and the result will be in
  // multiple destinations. Providing an accurate cost only for splits where
  // the element type remains the same.
  if (NumOfDests <= 1 ||
      LegalVT.getVectorElementType().getSizeInBits() !=
          Tp->getElementType()->getPrimitiveSizeInBits() ||
      LegalNumElts >= Tp->getElementCount().getFixedValue())
    return InstructionCost::getInvalid();

  unsigned VecTySize = TTI.getDataLayout().getTypeStoreSize(Tp);
  unsigned LegalVTSize = LegalVT.getStoreSize();
  // Number of source vectors after legalization:
  unsigned NumOfSrcs = divideCeil(VecTySize, LegalVTSize);

  auto *SingleOpTy = FixedVectorType::get(Tp->getElementType(), LegalNumElts);

  unsigned NormalizedVF = LegalNumElts * std::max(NumOfSrcs, NumOfDests);
  unsigned NumOfSrcRegs = NormalizedVF / LegalNumElts;
  unsigned NumOfDestRegs = NormalizedVF / LegalNumElts;
  SmallVector<int> NormalizedMask(NormalizedVF, PoisonMaskElem);
  assert(NormalizedVF >= Mask.size() &&
         "Normalized mask expected to be not shorter than original mask.");
  copy(Mask, NormalizedMask.begin());
  InstructionCost Cost = 0;
  SmallDenseSet<std::pair<ArrayRef<int>, unsigned>> ReusedSingleSrcShuffles;
  processShuffleMasks(
      NormalizedMask, NumOfSrcRegs, NumOfDestRegs, NumOfDestRegs, []() {},
      [&](ArrayRef<int> RegMask, unsigned SrcReg, unsigned DestReg) {
        if (ShuffleVectorInst::isIdentityMask(RegMask, RegMask.size()))
          return;
        if (!ReusedSingleSrcShuffles.insert(std::make_pair(RegMask, SrcReg))
                 .second)
          return;
        Cost += TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, SingleOpTy,
                                   RegMask, CostKind, 0, nullptr);
      },
      [&](ArrayRef<int> RegMask, unsigned Idx1, unsigned Idx2, bool NewReg) {
        Cost += TTI.getShuffleCost(TTI::SK_PermuteTwoSrc, SingleOpTy, RegMask,
                                   CostKind, 0, nullptr);
      });
  return Cost;
}

/// Try to perform better estimation of the permutation.
/// 1. Split the source/destination vectors into real registers.
/// 2. Do the mask analysis to identify which real registers are
/// permuted. If more than 1 source registers are used for the
/// destination register building, the cost for this destination register
/// is (Number_of_source_register - 1) * Cost_PermuteTwoSrc. If only one
/// source register is used, build mask and calculate the cost as a cost
/// of PermuteSingleSrc.
/// Also, for the single register permute we try to identify if the
/// destination register is just a copy of the source register or the
/// copy of the previous destination register (the cost is
/// TTI::TCC_Basic). If the source register is just reused, the cost for
/// this operation is 0.
static InstructionCost
costShuffleViaVRegSplitting(const RISCVTTIImpl &TTI, MVT LegalVT,
                            std::optional<unsigned> VLen, VectorType *Tp,
                            ArrayRef<int> Mask, TTI::TargetCostKind CostKind) {
  assert(LegalVT.isFixedLengthVector());
  if (!VLen || Mask.empty())
    return InstructionCost::getInvalid();
  MVT ElemVT = LegalVT.getVectorElementType();
  unsigned ElemsPerVReg = *VLen / ElemVT.getFixedSizeInBits();
  LegalVT = TTI.getTypeLegalizationCost(
                   FixedVectorType::get(Tp->getElementType(), ElemsPerVReg))
                .second;
  // Number of destination vectors after legalization:
  InstructionCost NumOfDests =
      divideCeil(Mask.size(), LegalVT.getVectorNumElements());
  if (NumOfDests <= 1 ||
      LegalVT.getVectorElementType().getSizeInBits() !=
          Tp->getElementType()->getPrimitiveSizeInBits() ||
      LegalVT.getVectorNumElements() >= Tp->getElementCount().getFixedValue())
    return InstructionCost::getInvalid();

  unsigned VecTySize = TTI.getDataLayout().getTypeStoreSize(Tp);
  unsigned LegalVTSize = LegalVT.getStoreSize();
  // Number of source vectors after legalization:
  unsigned NumOfSrcs = divideCeil(VecTySize, LegalVTSize);

  auto *SingleOpTy = FixedVectorType::get(Tp->getElementType(),
                                          LegalVT.getVectorNumElements());

  unsigned E = NumOfDests.getValue();
  unsigned NormalizedVF =
      LegalVT.getVectorNumElements() * std::max(NumOfSrcs, E);
  unsigned NumOfSrcRegs = NormalizedVF / LegalVT.getVectorNumElements();
  unsigned NumOfDestRegs = NormalizedVF / LegalVT.getVectorNumElements();
  SmallVector<int> NormalizedMask(NormalizedVF, PoisonMaskElem);
  assert(NormalizedVF >= Mask.size() &&
         "Normalized mask expected to be not shorter than original mask.");
  copy(Mask, NormalizedMask.begin());
  InstructionCost Cost = 0;
  int NumShuffles = 0;
  SmallDenseSet<std::pair<ArrayRef<int>, unsigned>> ReusedSingleSrcShuffles;
  processShuffleMasks(
      NormalizedMask, NumOfSrcRegs, NumOfDestRegs, NumOfDestRegs, []() {},
      [&](ArrayRef<int> RegMask, unsigned SrcReg, unsigned DestReg) {
        if (ShuffleVectorInst::isIdentityMask(RegMask, RegMask.size()))
          return;
        if (!ReusedSingleSrcShuffles.insert(std::make_pair(RegMask, SrcReg))
                 .second)
          return;
        ++NumShuffles;
        Cost += TTI.getShuffleCost(TTI::SK_PermuteSingleSrc, SingleOpTy,
                                   RegMask, CostKind, 0, nullptr);
      },
      [&](ArrayRef<int> RegMask, unsigned Idx1, unsigned Idx2, bool NewReg) {
        Cost += TTI.getShuffleCost(TTI::SK_PermuteTwoSrc, SingleOpTy, RegMask,
                                   CostKind, 0, nullptr);
        NumShuffles += 2;
      });
  // Note: check that we do not emit too many shuffles here to prevent code
  // size explosion.
  // TODO: investigate, if it can be improved by extra analysis of the masks
  // to check if the code is more profitable.
  if ((NumOfDestRegs > 2 && NumShuffles <= static_cast<int>(NumOfDestRegs)) ||
      (NumOfDestRegs <= 2 && NumShuffles < 4))
    return Cost;
  return InstructionCost::getInvalid();
}

InstructionCost RISCVTTIImpl::getSlideCost(FixedVectorType *Tp,
                                           ArrayRef<int> Mask,
                                           TTI::TargetCostKind CostKind) const {
  // Avoid missing masks and length changing shuffles
  if (Mask.size() <= 2 || Mask.size() != Tp->getNumElements())
    return InstructionCost::getInvalid();

  int NumElts = Tp->getNumElements();
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
  // Avoid scalarization cases
  if (!LT.second.isFixedLengthVector())
    return InstructionCost::getInvalid();

  // Requires moving elements between parts, which requires additional
  // unmodeled instructions.
  if (LT.first != 1)
    return InstructionCost::getInvalid();

  auto GetSlideOpcode = [&](int SlideAmt) {
    assert(SlideAmt != 0);
    bool IsVI = isUInt<5>(std::abs(SlideAmt));
    if (SlideAmt < 0)
      return IsVI ? RISCV::VSLIDEDOWN_VI : RISCV::VSLIDEDOWN_VX;
    return IsVI ? RISCV::VSLIDEUP_VI : RISCV::VSLIDEUP_VX;
  };

  std::array<std::pair<int, int>, 2> SrcInfo;
  if (!isMaskedSlidePair(Mask, NumElts, SrcInfo))
    return InstructionCost::getInvalid();

  if (SrcInfo[1].second == 0)
    std::swap(SrcInfo[0], SrcInfo[1]);

  InstructionCost FirstSlideCost = 0;
  if (SrcInfo[0].second != 0) {
    unsigned Opcode = GetSlideOpcode(SrcInfo[0].second);
    FirstSlideCost = getRISCVInstructionCost(Opcode, LT.second, CostKind);
  }

  if (SrcInfo[1].first == -1)
    return FirstSlideCost;

  InstructionCost SecondSlideCost = 0;
  if (SrcInfo[1].second != 0) {
    unsigned Opcode = GetSlideOpcode(SrcInfo[1].second);
    SecondSlideCost = getRISCVInstructionCost(Opcode, LT.second, CostKind);
  } else {
    SecondSlideCost =
        getRISCVInstructionCost(RISCV::VMERGE_VVM, LT.second, CostKind);
  }

  auto EC = Tp->getElementCount();
  VectorType *MaskTy =
      VectorType::get(IntegerType::getInt1Ty(Tp->getContext()), EC);
  InstructionCost MaskCost = getConstantPoolLoadCost(MaskTy, CostKind);
  return FirstSlideCost + SecondSlideCost + MaskCost;
}

InstructionCost RISCVTTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                             VectorType *Tp, ArrayRef<int> Mask,
                                             TTI::TargetCostKind CostKind,
                                             int Index, VectorType *SubTp,
                                             ArrayRef<const Value *> Args,
                                             const Instruction *CxtI) const {
  Kind = improveShuffleKindFromMask(Kind, Mask, Tp, Index, SubTp);
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);

  // First, handle cases where having a fixed length vector enables us to
  // give a more accurate cost than falling back to generic scalable codegen.
  // TODO: Each of these cases hints at a modeling gap around scalable vectors.
  if (auto *FVTp = dyn_cast<FixedVectorType>(Tp);
      FVTp && ST->hasVInstructions() && LT.second.isFixedLengthVector()) {
    InstructionCost VRegSplittingCost = costShuffleViaVRegSplitting(
        *this, LT.second, ST->getRealVLen(), Tp, Mask, CostKind);
    if (VRegSplittingCost.isValid())
      return VRegSplittingCost;
    switch (Kind) {
    default:
      break;
    case TTI::SK_PermuteSingleSrc: {
      if (Mask.size() >= 2) {
        MVT EltTp = LT.second.getVectorElementType();
        // If the size of the element is < ELEN then shuffles of interleaves and
        // deinterleaves of 2 vectors can be lowered into the following
        // sequences
        if (EltTp.getScalarSizeInBits() < ST->getELen()) {
          // Example sequence:
          //   vsetivli     zero, 4, e8, mf4, ta, ma (ignored)
          //   vwaddu.vv    v10, v8, v9
          //   li       a0, -1                   (ignored)
          //   vwmaccu.vx   v10, a0, v9
          if (ShuffleVectorInst::isInterleaveMask(Mask, 2, Mask.size()))
            return 2 * LT.first * TLI->getLMULCost(LT.second);

          if (Mask[0] == 0 || Mask[0] == 1) {
            auto DeinterleaveMask = createStrideMask(Mask[0], 2, Mask.size());
            // Example sequence:
            //   vnsrl.wi   v10, v8, 0
            if (equal(DeinterleaveMask, Mask))
              return LT.first * getRISCVInstructionCost(RISCV::VNSRL_WI,
                                                        LT.second, CostKind);
          }
        }
        int SubVectorSize;
        if (LT.second.getScalarSizeInBits() != 1 &&
            isRepeatedConcatMask(Mask, SubVectorSize)) {
          InstructionCost Cost = 0;
          unsigned NumSlides = Log2_32(Mask.size() / SubVectorSize);
          // The cost of extraction from a subvector is 0 if the index is 0.
          for (unsigned I = 0; I != NumSlides; ++I) {
            unsigned InsertIndex = SubVectorSize * (1 << I);
            FixedVectorType *SubTp =
                FixedVectorType::get(Tp->getElementType(), InsertIndex);
            FixedVectorType *DestTp =
                FixedVectorType::getDoubleElementsVectorType(SubTp);
            std::pair<InstructionCost, MVT> DestLT =
                getTypeLegalizationCost(DestTp);
            // Add the cost of whole vector register move because the
            // destination vector register group for vslideup cannot overlap the
            // source.
            Cost += DestLT.first * TLI->getLMULCost(DestLT.second);
            Cost += getShuffleCost(TTI::SK_InsertSubvector, DestTp, {},
                                   CostKind, InsertIndex, SubTp);
          }
          return Cost;
        }
      }

      if (InstructionCost SlideCost = getSlideCost(FVTp, Mask, CostKind);
          SlideCost.isValid())
        return SlideCost;

      // vrgather + cost of generating the mask constant.
      // We model this for an unknown mask with a single vrgather.
      if (LT.first == 1 && (LT.second.getScalarSizeInBits() != 8 ||
                            LT.second.getVectorNumElements() <= 256)) {
        VectorType *IdxTy =
            getVRGatherIndexType(LT.second, *ST, Tp->getContext());
        InstructionCost IndexCost = getConstantPoolLoadCost(IdxTy, CostKind);
        return IndexCost +
               getRISCVInstructionCost(RISCV::VRGATHER_VV, LT.second, CostKind);
      }
      break;
    }
    case TTI::SK_Transpose:
    case TTI::SK_PermuteTwoSrc: {

      if (InstructionCost SlideCost = getSlideCost(FVTp, Mask, CostKind);
          SlideCost.isValid())
        return SlideCost;

      // 2 x (vrgather + cost of generating the mask constant) + cost of mask
      // register for the second vrgather. We model this for an unknown
      // (shuffle) mask.
      if (LT.first == 1 && (LT.second.getScalarSizeInBits() != 8 ||
                            LT.second.getVectorNumElements() <= 256)) {
        auto &C = Tp->getContext();
        auto EC = Tp->getElementCount();
        VectorType *IdxTy = getVRGatherIndexType(LT.second, *ST, C);
        VectorType *MaskTy = VectorType::get(IntegerType::getInt1Ty(C), EC);
        InstructionCost IndexCost = getConstantPoolLoadCost(IdxTy, CostKind);
        InstructionCost MaskCost = getConstantPoolLoadCost(MaskTy, CostKind);
        return 2 * IndexCost +
               getRISCVInstructionCost({RISCV::VRGATHER_VV, RISCV::VRGATHER_VV},
                                       LT.second, CostKind) +
               MaskCost;
      }
      break;
    }
    }

    auto shouldSplit = [](TTI::ShuffleKind Kind) {
      switch (Kind) {
      default:
        return false;
      case TTI::SK_PermuteSingleSrc:
      case TTI::SK_Transpose:
      case TTI::SK_PermuteTwoSrc:
        return true;
      }
    };

    if (!Mask.empty() && LT.first.isValid() && LT.first != 1 &&
        shouldSplit(Kind)) {
      InstructionCost SplitCost =
          costShuffleViaSplitting(*this, LT.second, FVTp, Mask, CostKind);
      if (SplitCost.isValid())
        return SplitCost;
    }
  }

  // Handle scalable vectors (and fixed vectors legalized to scalable vectors).
  switch (Kind) {
  default:
    // Fallthrough to generic handling.
    // TODO: Most of these cases will return getInvalid in generic code, and
    // must be implemented here.
    break;
  case TTI::SK_ExtractSubvector:
    // Extract at zero is always a subregister extract
    if (Index == 0)
      return TTI::TCC_Free;

    // If we're extracting a subvector of at most m1 size at a sub-register
    // boundary - which unfortunately we need exact vlen to identify - this is
    // a subregister extract at worst and thus won't require a vslidedown.
    // TODO: Extend for aligned m2, m4 subvector extracts
    // TODO: Extend for misalgined (but contained) extracts
    // TODO: Extend for scalable subvector types
    if (std::pair<InstructionCost, MVT> SubLT = getTypeLegalizationCost(SubTp);
        SubLT.second.isValid() && SubLT.second.isFixedLengthVector()) {
      if (std::optional<unsigned> VLen = ST->getRealVLen();
          VLen && SubLT.second.getScalarSizeInBits() * Index % *VLen == 0 &&
          SubLT.second.getSizeInBits() <= *VLen)
        return TTI::TCC_Free;
    }

    // Example sequence:
    // vsetivli     zero, 4, e8, mf2, tu, ma (ignored)
    // vslidedown.vi  v8, v9, 2
    return LT.first *
           getRISCVInstructionCost(RISCV::VSLIDEDOWN_VI, LT.second, CostKind);
  case TTI::SK_InsertSubvector:
    // Example sequence:
    // vsetivli     zero, 4, e8, mf2, tu, ma (ignored)
    // vslideup.vi  v8, v9, 2
    return LT.first *
           getRISCVInstructionCost(RISCV::VSLIDEUP_VI, LT.second, CostKind);
  case TTI::SK_Select: {
    // Example sequence:
    // li           a0, 90
    // vsetivli     zero, 8, e8, mf2, ta, ma (ignored)
    // vmv.s.x      v0, a0
    // vmerge.vvm   v8, v9, v8, v0
    // We use 2 for the cost of the mask materialization as this is the true
    // cost for small masks and most shuffles are small.  At worst, this cost
    // should be a very small constant for the constant pool load.  As such,
    // we may bias towards large selects slightly more than truly warranted.
    return LT.first *
           (1 + getRISCVInstructionCost({RISCV::VMV_S_X, RISCV::VMERGE_VVM},
                                        LT.second, CostKind));
  }
  case TTI::SK_Broadcast: {
    bool HasScalar = (Args.size() > 0) && (Operator::getOpcode(Args[0]) ==
                                           Instruction::InsertElement);
    if (LT.second.getScalarSizeInBits() == 1) {
      if (HasScalar) {
        // Example sequence:
        //   andi a0, a0, 1
        //   vsetivli zero, 2, e8, mf8, ta, ma (ignored)
        //   vmv.v.x v8, a0
        //   vmsne.vi v0, v8, 0
        return LT.first *
               (1 + getRISCVInstructionCost({RISCV::VMV_V_X, RISCV::VMSNE_VI},
                                            LT.second, CostKind));
      }
      // Example sequence:
      //   vsetivli  zero, 2, e8, mf8, ta, mu (ignored)
      //   vmv.v.i v8, 0
      //   vmerge.vim      v8, v8, 1, v0
      //   vmv.x.s a0, v8
      //   andi    a0, a0, 1
      //   vmv.v.x v8, a0
      //   vmsne.vi  v0, v8, 0

      return LT.first *
             (1 + getRISCVInstructionCost({RISCV::VMV_V_I, RISCV::VMERGE_VIM,
                                           RISCV::VMV_X_S, RISCV::VMV_V_X,
                                           RISCV::VMSNE_VI},
                                          LT.second, CostKind));
    }

    if (HasScalar) {
      // Example sequence:
      //   vmv.v.x v8, a0
      return LT.first *
             getRISCVInstructionCost(RISCV::VMV_V_X, LT.second, CostKind);
    }

    // Example sequence:
    //   vrgather.vi     v9, v8, 0
    return LT.first *
           getRISCVInstructionCost(RISCV::VRGATHER_VI, LT.second, CostKind);
  }
  case TTI::SK_Splice: {
    // vslidedown+vslideup.
    // TODO: Multiplying by LT.first implies this legalizes into multiple copies
    // of similar code, but I think we expand through memory.
    unsigned Opcodes[2] = {RISCV::VSLIDEDOWN_VX, RISCV::VSLIDEUP_VX};
    if (Index >= 0 && Index < 32)
      Opcodes[0] = RISCV::VSLIDEDOWN_VI;
    else if (Index < 0 && Index > -32)
      Opcodes[1] = RISCV::VSLIDEUP_VI;
    return LT.first * getRISCVInstructionCost(Opcodes, LT.second, CostKind);
  }
  case TTI::SK_Reverse: {
    // TODO: Cases to improve here:
    // * Illegal vector types
    // * i64 on RV32
    // * i1 vector
    // At low LMUL, most of the cost is producing the vrgather index register.
    // At high LMUL, the cost of the vrgather itself will dominate.
    // Example sequence:
    //   csrr a0, vlenb
    //   srli a0, a0, 3
    //   addi a0, a0, -1
    //   vsetvli a1, zero, e8, mf8, ta, mu (ignored)
    //   vid.v v9
    //   vrsub.vx v10, v9, a0
    //   vrgather.vv v9, v8, v10
    InstructionCost LenCost = 3;
    if (LT.second.isFixedLengthVector())
      // vrsub.vi has a 5 bit immediate field, otherwise an li suffices
      LenCost = isInt<5>(LT.second.getVectorNumElements() - 1) ? 0 : 1;
    unsigned Opcodes[] = {RISCV::VID_V, RISCV::VRSUB_VX, RISCV::VRGATHER_VV};
    if (LT.second.isFixedLengthVector() &&
        isInt<5>(LT.second.getVectorNumElements() - 1))
      Opcodes[1] = RISCV::VRSUB_VI;
    InstructionCost GatherCost =
        getRISCVInstructionCost(Opcodes, LT.second, CostKind);
    // Mask operation additionally required extend and truncate
    InstructionCost ExtendCost = Tp->getElementType()->isIntegerTy(1) ? 3 : 0;
    return LT.first * (LenCost + GatherCost + ExtendCost);
  }
  }
  return BaseT::getShuffleCost(Kind, Tp, Mask, CostKind, Index, SubTp);
}

static unsigned isM1OrSmaller(MVT VT) {
  RISCVVType::VLMUL LMUL = RISCVTargetLowering::getLMUL(VT);
  return (LMUL == RISCVVType::VLMUL::LMUL_F8 ||
          LMUL == RISCVVType::VLMUL::LMUL_F4 ||
          LMUL == RISCVVType::VLMUL::LMUL_F2 ||
          LMUL == RISCVVType::VLMUL::LMUL_1);
}

InstructionCost RISCVTTIImpl::getScalarizationOverhead(
    VectorType *Ty, const APInt &DemandedElts, bool Insert, bool Extract,
    TTI::TargetCostKind CostKind, bool ForPoisonSrc,
    ArrayRef<Value *> VL) const {
  if (isa<ScalableVectorType>(Ty))
    return InstructionCost::getInvalid();

  // A build_vector (which is m1 sized or smaller) can be done in no
  // worse than one vslide1down.vx per element in the type.  We could
  // in theory do an explode_vector in the inverse manner, but our
  // lowering today does not have a first class node for this pattern.
  InstructionCost Cost = BaseT::getScalarizationOverhead(
      Ty, DemandedElts, Insert, Extract, CostKind);
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  if (Insert && !Extract && LT.first.isValid() && LT.second.isVector()) {
    if (Ty->getScalarSizeInBits() == 1) {
      auto *WideVecTy = cast<VectorType>(Ty->getWithNewBitWidth(8));
      // Note: Implicit scalar anyextend is assumed to be free since the i1
      // must be stored in a GPR.
      return getScalarizationOverhead(WideVecTy, DemandedElts, Insert, Extract,
                                      CostKind) +
             getCastInstrCost(Instruction::Trunc, Ty, WideVecTy,
                              TTI::CastContextHint::None, CostKind, nullptr);
    }

    assert(LT.second.isFixedLengthVector());
    MVT ContainerVT = TLI->getContainerForFixedLengthVector(LT.second);
    if (isM1OrSmaller(ContainerVT)) {
      InstructionCost BV =
          cast<FixedVectorType>(Ty)->getNumElements() *
          getRISCVInstructionCost(RISCV::VSLIDE1DOWN_VX, LT.second, CostKind);
      if (BV < Cost)
        Cost = BV;
    }
  }
  return Cost;
}

InstructionCost
RISCVTTIImpl::getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                    unsigned AddressSpace,
                                    TTI::TargetCostKind CostKind) const {
  if (!isLegalMaskedLoadStore(Src, Alignment) ||
      CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                        CostKind);

  return getMemoryOpCost(Opcode, Src, Alignment, AddressSpace, CostKind);
}

InstructionCost RISCVTTIImpl::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) const {

  // The interleaved memory access pass will lower interleaved memory ops (i.e
  // a load and store followed by a specific shuffle) to vlseg/vsseg
  // intrinsics.
  if (!UseMaskForCond && !UseMaskForGaps &&
      Factor <= TLI->getMaxSupportedInterleaveFactor()) {
    auto *VTy = cast<VectorType>(VecTy);
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(VTy);
    // Need to make sure type has't been scalarized
    if (LT.second.isVector()) {
      auto *SubVecTy =
          VectorType::get(VTy->getElementType(),
                          VTy->getElementCount().divideCoefficientBy(Factor));
      if (VTy->getElementCount().isKnownMultipleOf(Factor) &&
          TLI->isLegalInterleavedAccessType(SubVecTy, Factor, Alignment,
                                            AddressSpace, DL)) {

        // Some processors optimize segment loads/stores as one wide memory op +
        // Factor * LMUL shuffle ops.
        if (ST->hasOptimizedSegmentLoadStore(Factor)) {
          InstructionCost Cost =
              getMemoryOpCost(Opcode, VTy, Alignment, AddressSpace, CostKind);
          MVT SubVecVT = getTLI()->getValueType(DL, SubVecTy).getSimpleVT();
          Cost += Factor * TLI->getLMULCost(SubVecVT);
          return LT.first * Cost;
        }

        // Otherwise, the cost is proportional to the number of elements (VL *
        // Factor ops).
        InstructionCost MemOpCost =
            getMemoryOpCost(Opcode, VTy->getElementType(), Alignment, 0,
                            CostKind, {TTI::OK_AnyValue, TTI::OP_None});
        unsigned NumLoads = getEstimatedVLFor(VTy);
        return NumLoads * MemOpCost;
      }
    }
  }

  // TODO: Return the cost of interleaved accesses for scalable vector when
  // unable to convert to segment accesses instructions.
  if (isa<ScalableVectorType>(VecTy))
    return InstructionCost::getInvalid();

  auto *FVTy = cast<FixedVectorType>(VecTy);
  InstructionCost MemCost =
      getMemoryOpCost(Opcode, VecTy, Alignment, AddressSpace, CostKind);
  unsigned VF = FVTy->getNumElements() / Factor;

  // An interleaved load will look like this for Factor=3:
  // %wide.vec = load <12 x i32>, ptr %3, align 4
  // %strided.vec = shufflevector %wide.vec, poison, <4 x i32> <stride mask>
  // %strided.vec1 = shufflevector %wide.vec, poison, <4 x i32> <stride mask>
  // %strided.vec2 = shufflevector %wide.vec, poison, <4 x i32> <stride mask>
  if (Opcode == Instruction::Load) {
    InstructionCost Cost = MemCost;
    for (unsigned Index : Indices) {
      FixedVectorType *VecTy =
          FixedVectorType::get(FVTy->getElementType(), VF * Factor);
      auto Mask = createStrideMask(Index, Factor, VF);
      Mask.resize(VF * Factor, -1);
      InstructionCost ShuffleCost =
          getShuffleCost(TTI::ShuffleKind::SK_PermuteSingleSrc, VecTy, Mask,
                         CostKind, 0, nullptr, {});
      Cost += ShuffleCost;
    }
    return Cost;
  }

  // TODO: Model for NF > 2
  // We'll need to enhance getShuffleCost to model shuffles that are just
  // inserts and extracts into subvectors, since they won't have the full cost
  // of a vrgather.
  // An interleaved store for 3 vectors of 4 lanes will look like
  // %11 = shufflevector <4 x i32> %4, <4 x i32> %6, <8 x i32> <0...7>
  // %12 = shufflevector <4 x i32> %9, <4 x i32> poison, <8 x i32> <0...3>
  // %13 = shufflevector <8 x i32> %11, <8 x i32> %12, <12 x i32> <0...11>
  // %interleaved.vec = shufflevector %13, poison, <12 x i32> <interleave mask>
  // store <12 x i32> %interleaved.vec, ptr %10, align 4
  if (Factor != 2)
    return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                             Alignment, AddressSpace, CostKind,
                                             UseMaskForCond, UseMaskForGaps);

  assert(Opcode == Instruction::Store && "Opcode must be a store");
  // For an interleaving store of 2 vectors, we perform one large interleaving
  // shuffle that goes into the wide store
  auto Mask = createInterleaveMask(VF, Factor);
  InstructionCost ShuffleCost =
      getShuffleCost(TTI::ShuffleKind::SK_PermuteSingleSrc, FVTy, Mask,
                     CostKind, 0, nullptr, {});
  return MemCost + ShuffleCost;
}

InstructionCost RISCVTTIImpl::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) const {
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

InstructionCost RISCVTTIImpl::getExpandCompressMemoryOpCost(
    unsigned Opcode, Type *DataTy, bool VariableMask, Align Alignment,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  bool IsLegal = (Opcode == Instruction::Store &&
                  isLegalMaskedCompressStore(DataTy, Alignment)) ||
                 (Opcode == Instruction::Load &&
                  isLegalMaskedExpandLoad(DataTy, Alignment));
  if (!IsLegal || CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getExpandCompressMemoryOpCost(Opcode, DataTy, VariableMask,
                                                Alignment, CostKind, I);
  // Example compressstore sequence:
  // vsetivli        zero, 8, e32, m2, ta, ma (ignored)
  // vcompress.vm    v10, v8, v0
  // vcpop.m a1, v0
  // vsetvli zero, a1, e32, m2, ta, ma
  // vse32.v v10, (a0)
  // Example expandload sequence:
  // vsetivli        zero, 8, e8, mf2, ta, ma (ignored)
  // vcpop.m a1, v0
  // vsetvli zero, a1, e32, m2, ta, ma
  // vle32.v v10, (a0)
  // vsetivli        zero, 8, e32, m2, ta, ma
  // viota.m v12, v0
  // vrgather.vv     v8, v10, v12, v0.t
  auto MemOpCost =
      getMemoryOpCost(Opcode, DataTy, Alignment, /*AddressSpace*/ 0, CostKind);
  auto LT = getTypeLegalizationCost(DataTy);
  SmallVector<unsigned, 4> Opcodes{RISCV::VSETVLI};
  if (VariableMask)
    Opcodes.push_back(RISCV::VCPOP_M);
  if (Opcode == Instruction::Store)
    Opcodes.append({RISCV::VCOMPRESS_VM});
  else
    Opcodes.append({RISCV::VSETIVLI, RISCV::VIOTA_M, RISCV::VRGATHER_VV});
  return MemOpCost +
         LT.first * getRISCVInstructionCost(Opcodes, LT.second, CostKind);
}

InstructionCost RISCVTTIImpl::getStridedMemoryOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) const {
  if (((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
       !isLegalStridedLoadStore(DataTy, Alignment)) ||
      (Opcode != Instruction::Load && Opcode != Instruction::Store))
    return BaseT::getStridedMemoryOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);

  if (CostKind == TTI::TCK_CodeSize)
    return TTI::TCC_Basic;

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

InstructionCost
RISCVTTIImpl::getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const {
  // FIXME: This is a property of the default vector convention, not
  // all possible calling conventions.  Fixing that will require
  // some TTI API and SLP rework.
  InstructionCost Cost = 0;
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  for (auto *Ty : Tys) {
    if (!Ty->isVectorTy())
      continue;
    Align A = DL.getPrefTypeAlign(Ty);
    Cost += getMemoryOpCost(Instruction::Store, Ty, A, 0, CostKind) +
            getMemoryOpCost(Instruction::Load, Ty, A, 0, CostKind);
  }
  return Cost;
}

// Currently, these represent both throughput and codesize costs
// for the respective intrinsics.  The costs in this table are simply
// instruction counts with the following adjustments made:
// * One vsetvli is considered free.
static const CostTblEntry VectorIntrinsicCostTable[]{
    {Intrinsic::floor, MVT::f32, 9},
    {Intrinsic::floor, MVT::f64, 9},
    {Intrinsic::ceil, MVT::f32, 9},
    {Intrinsic::ceil, MVT::f64, 9},
    {Intrinsic::trunc, MVT::f32, 7},
    {Intrinsic::trunc, MVT::f64, 7},
    {Intrinsic::round, MVT::f32, 9},
    {Intrinsic::round, MVT::f64, 9},
    {Intrinsic::roundeven, MVT::f32, 9},
    {Intrinsic::roundeven, MVT::f64, 9},
    {Intrinsic::rint, MVT::f32, 7},
    {Intrinsic::rint, MVT::f64, 7},
    {Intrinsic::lrint, MVT::i32, 1},
    {Intrinsic::lrint, MVT::i64, 1},
    {Intrinsic::llrint, MVT::i64, 1},
    {Intrinsic::nearbyint, MVT::f32, 9},
    {Intrinsic::nearbyint, MVT::f64, 9},
    {Intrinsic::bswap, MVT::i16, 3},
    {Intrinsic::bswap, MVT::i32, 12},
    {Intrinsic::bswap, MVT::i64, 31},
    {Intrinsic::vp_bswap, MVT::i16, 3},
    {Intrinsic::vp_bswap, MVT::i32, 12},
    {Intrinsic::vp_bswap, MVT::i64, 31},
    {Intrinsic::vp_fshl, MVT::i8, 7},
    {Intrinsic::vp_fshl, MVT::i16, 7},
    {Intrinsic::vp_fshl, MVT::i32, 7},
    {Intrinsic::vp_fshl, MVT::i64, 7},
    {Intrinsic::vp_fshr, MVT::i8, 7},
    {Intrinsic::vp_fshr, MVT::i16, 7},
    {Intrinsic::vp_fshr, MVT::i32, 7},
    {Intrinsic::vp_fshr, MVT::i64, 7},
    {Intrinsic::bitreverse, MVT::i8, 17},
    {Intrinsic::bitreverse, MVT::i16, 24},
    {Intrinsic::bitreverse, MVT::i32, 33},
    {Intrinsic::bitreverse, MVT::i64, 52},
    {Intrinsic::vp_bitreverse, MVT::i8, 17},
    {Intrinsic::vp_bitreverse, MVT::i16, 24},
    {Intrinsic::vp_bitreverse, MVT::i32, 33},
    {Intrinsic::vp_bitreverse, MVT::i64, 52},
    {Intrinsic::ctpop, MVT::i8, 12},
    {Intrinsic::ctpop, MVT::i16, 19},
    {Intrinsic::ctpop, MVT::i32, 20},
    {Intrinsic::ctpop, MVT::i64, 21},
    {Intrinsic::ctlz, MVT::i8, 19},
    {Intrinsic::ctlz, MVT::i16, 28},
    {Intrinsic::ctlz, MVT::i32, 31},
    {Intrinsic::ctlz, MVT::i64, 35},
    {Intrinsic::cttz, MVT::i8, 16},
    {Intrinsic::cttz, MVT::i16, 23},
    {Intrinsic::cttz, MVT::i32, 24},
    {Intrinsic::cttz, MVT::i64, 25},
    {Intrinsic::vp_ctpop, MVT::i8, 12},
    {Intrinsic::vp_ctpop, MVT::i16, 19},
    {Intrinsic::vp_ctpop, MVT::i32, 20},
    {Intrinsic::vp_ctpop, MVT::i64, 21},
    {Intrinsic::vp_ctlz, MVT::i8, 19},
    {Intrinsic::vp_ctlz, MVT::i16, 28},
    {Intrinsic::vp_ctlz, MVT::i32, 31},
    {Intrinsic::vp_ctlz, MVT::i64, 35},
    {Intrinsic::vp_cttz, MVT::i8, 16},
    {Intrinsic::vp_cttz, MVT::i16, 23},
    {Intrinsic::vp_cttz, MVT::i32, 24},
    {Intrinsic::vp_cttz, MVT::i64, 25},
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
                                    TTI::TargetCostKind CostKind) const {
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
  case Intrinsic::lrint:
  case Intrinsic::llrint:
    // We can't currently lower half or bfloat vector lrint/llrint.
    if (auto *VecTy = dyn_cast<VectorType>(ICA.getArgTypes()[0]);
        VecTy && VecTy->getElementType()->is16bitFPTy())
      return InstructionCost::getInvalid();
    [[fallthrough]];
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
    if (LT.second.isScalarInteger() && ST->hasStdExtZbb())
      return LT.first;

    if (ST->hasVInstructions() && LT.second.isVector()) {
      unsigned Op;
      switch (ICA.getID()) {
      case Intrinsic::umin:
        Op = RISCV::VMINU_VV;
        break;
      case Intrinsic::umax:
        Op = RISCV::VMAXU_VV;
        break;
      case Intrinsic::smin:
        Op = RISCV::VMIN_VV;
        break;
      case Intrinsic::smax:
        Op = RISCV::VMAX_VV;
        break;
      }
      return LT.first * getRISCVInstructionCost(Op, LT.second, CostKind);
    }
    break;
  }
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::usub_sat: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector()) {
      unsigned Op;
      switch (ICA.getID()) {
      case Intrinsic::sadd_sat:
        Op = RISCV::VSADD_VV;
        break;
      case Intrinsic::ssub_sat:
        Op = RISCV::VSSUBU_VV;
        break;
      case Intrinsic::uadd_sat:
        Op = RISCV::VSADDU_VV;
        break;
      case Intrinsic::usub_sat:
        Op = RISCV::VSSUBU_VV;
        break;
      }
      return LT.first * getRISCVInstructionCost(Op, LT.second, CostKind);
    }
    break;
  }
  case Intrinsic::fma:
  case Intrinsic::fmuladd: {
    // TODO: handle promotion with f16/bf16 with zvfhmin/zvfbfmin
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector())
      return LT.first *
             getRISCVInstructionCost(RISCV::VFMADD_VV, LT.second, CostKind);
    break;
  }
  case Intrinsic::fabs: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector()) {
      // lui a0, 8
      // addi a0, a0, -1
      // vsetvli a1, zero, e16, m1, ta, ma
      // vand.vx v8, v8, a0
      // f16 with zvfhmin and bf16 with zvfhbmin
      if (LT.second.getVectorElementType() == MVT::bf16 ||
          (LT.second.getVectorElementType() == MVT::f16 &&
           !ST->hasVInstructionsF16()))
        return LT.first * getRISCVInstructionCost(RISCV::VAND_VX, LT.second,
                                                  CostKind) +
               2;
      else
        return LT.first *
               getRISCVInstructionCost(RISCV::VFSGNJX_VV, LT.second, CostKind);
    }
    break;
  }
  case Intrinsic::sqrt: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector()) {
      SmallVector<unsigned, 4> ConvOp;
      SmallVector<unsigned, 2> FsqrtOp;
      MVT ConvType = LT.second;
      MVT FsqrtType = LT.second;
      // f16 with zvfhmin and bf16 with zvfbfmin and the type of nxv32[b]f16
      // will be spilt.
      if (LT.second.getVectorElementType() == MVT::bf16) {
        if (LT.second == MVT::nxv32bf16) {
          ConvOp = {RISCV::VFWCVTBF16_F_F_V, RISCV::VFWCVTBF16_F_F_V,
                    RISCV::VFNCVTBF16_F_F_W, RISCV::VFNCVTBF16_F_F_W};
          FsqrtOp = {RISCV::VFSQRT_V, RISCV::VFSQRT_V};
          ConvType = MVT::nxv16f16;
          FsqrtType = MVT::nxv16f32;
        } else {
          ConvOp = {RISCV::VFWCVTBF16_F_F_V, RISCV::VFNCVTBF16_F_F_W};
          FsqrtOp = {RISCV::VFSQRT_V};
          FsqrtType = TLI->getTypeToPromoteTo(ISD::FSQRT, FsqrtType);
        }
      } else if (LT.second.getVectorElementType() == MVT::f16 &&
                 !ST->hasVInstructionsF16()) {
        if (LT.second == MVT::nxv32f16) {
          ConvOp = {RISCV::VFWCVT_F_F_V, RISCV::VFWCVT_F_F_V,
                    RISCV::VFNCVT_F_F_W, RISCV::VFNCVT_F_F_W};
          FsqrtOp = {RISCV::VFSQRT_V, RISCV::VFSQRT_V};
          ConvType = MVT::nxv16f16;
          FsqrtType = MVT::nxv16f32;
        } else {
          ConvOp = {RISCV::VFWCVT_F_F_V, RISCV::VFNCVT_F_F_W};
          FsqrtOp = {RISCV::VFSQRT_V};
          FsqrtType = TLI->getTypeToPromoteTo(ISD::FSQRT, FsqrtType);
        }
      } else {
        FsqrtOp = {RISCV::VFSQRT_V};
      }

      return LT.first * (getRISCVInstructionCost(FsqrtOp, FsqrtType, CostKind) +
                         getRISCVInstructionCost(ConvOp, ConvType, CostKind));
    }
    break;
  }
  case Intrinsic::cttz:
  case Intrinsic::ctlz:
  case Intrinsic::ctpop: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && ST->hasStdExtZvbb() && LT.second.isVector()) {
      unsigned Op;
      switch (ICA.getID()) {
      case Intrinsic::cttz:
        Op = RISCV::VCTZ_V;
        break;
      case Intrinsic::ctlz:
        Op = RISCV::VCLZ_V;
        break;
      case Intrinsic::ctpop:
        Op = RISCV::VCPOP_V;
        break;
      }
      return LT.first * getRISCVInstructionCost(Op, LT.second, CostKind);
    }
    break;
  }
  case Intrinsic::abs: {
    auto LT = getTypeLegalizationCost(RetTy);
    if (ST->hasVInstructions() && LT.second.isVector()) {
      // vrsub.vi v10, v8, 0
      // vmax.vv v8, v8, v10
      return LT.first *
             getRISCVInstructionCost({RISCV::VRSUB_VI, RISCV::VMAX_VV},
                                     LT.second, CostKind);
    }
    break;
  }
  case Intrinsic::get_active_lane_mask: {
    if (ST->hasVInstructions()) {
      Type *ExpRetTy = VectorType::get(
          ICA.getArgTypes()[0], cast<VectorType>(RetTy)->getElementCount());
      auto LT = getTypeLegalizationCost(ExpRetTy);

      // vid.v   v8  // considered hoisted
      // vsaddu.vx   v8, v8, a0
      // vmsltu.vx   v0, v8, a1
      return LT.first *
             getRISCVInstructionCost({RISCV::VSADDU_VX, RISCV::VMSLTU_VX},
                                     LT.second, CostKind);
    }
    break;
  }
  // TODO: add more intrinsic
  case Intrinsic::stepvector: {
    auto LT = getTypeLegalizationCost(RetTy);
    // Legalisation of illegal types involves an `index' instruction plus
    // (LT.first - 1) vector adds.
    if (ST->hasVInstructions())
      return getRISCVInstructionCost(RISCV::VID_V, LT.second, CostKind) +
             (LT.first - 1) *
                 getRISCVInstructionCost(RISCV::VADD_VX, LT.second, CostKind);
    return 1 + (LT.first - 1);
  }
  case Intrinsic::experimental_cttz_elts: {
    Type *ArgTy = ICA.getArgTypes()[0];
    EVT ArgType = TLI->getValueType(DL, ArgTy, true);
    if (getTLI()->shouldExpandCttzElements(ArgType))
      break;
    InstructionCost Cost = getRISCVInstructionCost(
        RISCV::VFIRST_M, getTypeLegalizationCost(ArgTy).second, CostKind);

    // If zero_is_poison is false, then we will generate additional
    // cmp + select instructions to convert -1 to EVL.
    Type *BoolTy = Type::getInt1Ty(RetTy->getContext());
    if (ICA.getArgs().size() > 1 &&
        cast<ConstantInt>(ICA.getArgs()[1])->isZero())
      Cost += getCmpSelInstrCost(Instruction::ICmp, BoolTy, RetTy,
                                 CmpInst::ICMP_SLT, CostKind) +
              getCmpSelInstrCost(Instruction::Select, RetTy, BoolTy,
                                 CmpInst::BAD_ICMP_PREDICATE, CostKind);

    return Cost;
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
  case Intrinsic::vp_select: {
    Intrinsic::ID IID = ICA.getID();
    std::optional<unsigned> FOp = VPIntrinsic::getFunctionalOpcodeForVP(IID);
    assert(FOp.has_value());
    return getCmpSelInstrCost(*FOp, ICA.getReturnType(), ICA.getArgTypes()[0],
                              CmpInst::BAD_ICMP_PREDICATE, CostKind);
  }
  case Intrinsic::vp_merge:
    return getCmpSelInstrCost(Instruction::Select, ICA.getReturnType(),
                              ICA.getArgTypes()[0], CmpInst::BAD_ICMP_PREDICATE,
                              CostKind);
  case Intrinsic::experimental_vp_splat: {
    auto LT = getTypeLegalizationCost(RetTy);
    // TODO: Lower i1 experimental_vp_splat
    if (!ST->hasVInstructions() || LT.second.getScalarType() == MVT::i1)
      return InstructionCost::getInvalid();
    return LT.first * getRISCVInstructionCost(LT.second.isFloatingPoint()
                                                  ? RISCV::VFMV_V_F
                                                  : RISCV::VMV_V_X,
                                              LT.second, CostKind);
  }
  case Intrinsic::experimental_vp_splice: {
    // To support type-based query from vectorizer, set the index to 0.
    // Note that index only change the cost from vslide.vx to vslide.vi and in
    // current implementations they have same costs.
    return getShuffleCost(TTI::SK_Splice,
                          cast<VectorType>(ICA.getArgTypes()[0]), {}, CostKind,
                          0, cast<VectorType>(ICA.getReturnType()));
  }
  }

  if (ST->hasVInstructions() && RetTy->isVectorTy()) {
    if (auto LT = getTypeLegalizationCost(RetTy);
        LT.second.isVector()) {
      MVT EltTy = LT.second.getVectorElementType();
      if (const auto *Entry = CostTableLookup(VectorIntrinsicCostTable,
                                              ICA.getID(), EltTy))
        return LT.first * Entry->Cost;
    }
  }

  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

InstructionCost RISCVTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                               Type *Src,
                                               TTI::CastContextHint CCH,
                                               TTI::TargetCostKind CostKind,
                                               const Instruction *I) const {
  bool IsVectorType = isa<VectorType>(Dst) && isa<VectorType>(Src);
  if (!IsVectorType)
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

  // FIXME: Need to compute legalizing cost for illegal types.  The current
  // code handles only legal types and those which can be trivially
  // promoted to legal.
  if (!ST->hasVInstructions() || Src->getScalarSizeInBits() > ST->getELen() ||
      Dst->getScalarSizeInBits() > ST->getELen())
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");
  std::pair<InstructionCost, MVT> SrcLT = getTypeLegalizationCost(Src);
  std::pair<InstructionCost, MVT> DstLT = getTypeLegalizationCost(Dst);

  // Handle i1 source and dest cases *before* calling logic in BasicTTI.
  // The shared implementation doesn't model vector widening during legalization
  // and instead assumes scalarization.  In order to scalarize an <N x i1>
  // vector, we need to extend/trunc to/from i8.  If we don't special case
  // this, we can get an infinite recursion cycle.
  switch (ISD) {
  default:
    break;
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
    if (Src->getScalarSizeInBits() == 1) {
      // We do not use vsext/vzext to extend from mask vector.
      // Instead we use the following instructions to extend from mask vector:
      // vmv.v.i v8, 0
      // vmerge.vim v8, v8, -1, v0 (repeated per split)
      return getRISCVInstructionCost(RISCV::VMV_V_I, DstLT.second, CostKind) +
             DstLT.first * getRISCVInstructionCost(RISCV::VMERGE_VIM,
                                                   DstLT.second, CostKind) +
             DstLT.first - 1;
    }
    break;
  case ISD::TRUNCATE:
    if (Dst->getScalarSizeInBits() == 1) {
      // We do not use several vncvt to truncate to mask vector. So we could
      // not use PowDiff to calculate it.
      // Instead we use the following instructions to truncate to mask vector:
      // vand.vi v8, v8, 1
      // vmsne.vi v0, v8, 0
      return SrcLT.first *
                 getRISCVInstructionCost({RISCV::VAND_VI, RISCV::VMSNE_VI},
                                         SrcLT.second, CostKind) +
             SrcLT.first - 1;
    }
    break;
  };

  // Our actual lowering for the case where a wider legal type is available
  // uses promotion to the wider type.  This is reflected in the result of
  // getTypeLegalizationCost, but BasicTTI assumes the widened cases are
  // scalarized if the legalized Src and Dst are not equal sized.
  const DataLayout &DL = this->getDataLayout();
  if (!SrcLT.second.isVector() || !DstLT.second.isVector() ||
      !TypeSize::isKnownLE(DL.getTypeSizeInBits(Src),
                           SrcLT.second.getSizeInBits()) ||
      !TypeSize::isKnownLE(DL.getTypeSizeInBits(Dst),
                           DstLT.second.getSizeInBits()))
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);

  // The split cost is handled by the base getCastInstrCost
  assert((SrcLT.first == 1) && (DstLT.first == 1) && "Illegal type");

  int PowDiff = (int)Log2_32(DstLT.second.getScalarSizeInBits()) -
                (int)Log2_32(SrcLT.second.getScalarSizeInBits());
  switch (ISD) {
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND: {
    if ((PowDiff < 1) || (PowDiff > 3))
      return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
    unsigned SExtOp[] = {RISCV::VSEXT_VF2, RISCV::VSEXT_VF4, RISCV::VSEXT_VF8};
    unsigned ZExtOp[] = {RISCV::VZEXT_VF2, RISCV::VZEXT_VF4, RISCV::VZEXT_VF8};
    unsigned Op =
        (ISD == ISD::SIGN_EXTEND) ? SExtOp[PowDiff - 1] : ZExtOp[PowDiff - 1];
    return getRISCVInstructionCost(Op, DstLT.second, CostKind);
  }
  case ISD::TRUNCATE:
  case ISD::FP_EXTEND:
  case ISD::FP_ROUND: {
    // Counts of narrow/widen instructions.
    unsigned SrcEltSize = SrcLT.second.getScalarSizeInBits();
    unsigned DstEltSize = DstLT.second.getScalarSizeInBits();

    unsigned Op = (ISD == ISD::TRUNCATE)    ? RISCV::VNSRL_WI
                  : (ISD == ISD::FP_EXTEND) ? RISCV::VFWCVT_F_F_V
                                            : RISCV::VFNCVT_F_F_W;
    InstructionCost Cost = 0;
    for (; SrcEltSize != DstEltSize;) {
      MVT ElementMVT = (ISD == ISD::TRUNCATE)
                           ? MVT::getIntegerVT(DstEltSize)
                           : MVT::getFloatingPointVT(DstEltSize);
      MVT DstMVT = DstLT.second.changeVectorElementType(ElementMVT);
      DstEltSize =
          (DstEltSize > SrcEltSize) ? DstEltSize >> 1 : DstEltSize << 1;
      Cost += getRISCVInstructionCost(Op, DstMVT, CostKind);
    }
    return Cost;
  }
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT: {
    unsigned IsSigned = ISD == ISD::FP_TO_SINT;
    unsigned FCVT = IsSigned ? RISCV::VFCVT_RTZ_X_F_V : RISCV::VFCVT_RTZ_XU_F_V;
    unsigned FWCVT =
        IsSigned ? RISCV::VFWCVT_RTZ_X_F_V : RISCV::VFWCVT_RTZ_XU_F_V;
    unsigned FNCVT =
        IsSigned ? RISCV::VFNCVT_RTZ_X_F_W : RISCV::VFNCVT_RTZ_XU_F_W;
    unsigned SrcEltSize = Src->getScalarSizeInBits();
    unsigned DstEltSize = Dst->getScalarSizeInBits();
    InstructionCost Cost = 0;
    if ((SrcEltSize == 16) &&
        (!ST->hasVInstructionsF16() || ((DstEltSize / 2) > SrcEltSize))) {
      // If the target only supports zvfhmin or it is fp16-to-i64 conversion
      // pre-widening to f32 and then convert f32 to integer
      VectorType *VecF32Ty =
          VectorType::get(Type::getFloatTy(Dst->getContext()),
                          cast<VectorType>(Dst)->getElementCount());
      std::pair<InstructionCost, MVT> VecF32LT =
          getTypeLegalizationCost(VecF32Ty);
      Cost +=
          VecF32LT.first * getRISCVInstructionCost(RISCV::VFWCVT_F_F_V,
                                                   VecF32LT.second, CostKind);
      Cost += getCastInstrCost(Opcode, Dst, VecF32Ty, CCH, CostKind, I);
      return Cost;
    }
    if (DstEltSize == SrcEltSize)
      Cost += getRISCVInstructionCost(FCVT, DstLT.second, CostKind);
    else if (DstEltSize > SrcEltSize)
      Cost += getRISCVInstructionCost(FWCVT, DstLT.second, CostKind);
    else { // (SrcEltSize > DstEltSize)
      // First do a narrowing conversion to an integer half the size, then
      // truncate if needed.
      MVT ElementVT = MVT::getIntegerVT(SrcEltSize / 2);
      MVT VecVT = DstLT.second.changeVectorElementType(ElementVT);
      Cost += getRISCVInstructionCost(FNCVT, VecVT, CostKind);
      if ((SrcEltSize / 2) > DstEltSize) {
        Type *VecTy = EVT(VecVT).getTypeForEVT(Dst->getContext());
        Cost +=
            getCastInstrCost(Instruction::Trunc, Dst, VecTy, CCH, CostKind, I);
      }
    }
    return Cost;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP: {
    unsigned IsSigned = ISD == ISD::SINT_TO_FP;
    unsigned FCVT = IsSigned ? RISCV::VFCVT_F_X_V : RISCV::VFCVT_F_XU_V;
    unsigned FWCVT = IsSigned ? RISCV::VFWCVT_F_X_V : RISCV::VFWCVT_F_XU_V;
    unsigned FNCVT = IsSigned ? RISCV::VFNCVT_F_X_W : RISCV::VFNCVT_F_XU_W;
    unsigned SrcEltSize = Src->getScalarSizeInBits();
    unsigned DstEltSize = Dst->getScalarSizeInBits();

    InstructionCost Cost = 0;
    if ((DstEltSize == 16) &&
        (!ST->hasVInstructionsF16() || ((SrcEltSize / 2) > DstEltSize))) {
      // If the target only supports zvfhmin or it is i64-to-fp16 conversion
      // it is converted to f32 and then converted to f16
      VectorType *VecF32Ty =
          VectorType::get(Type::getFloatTy(Dst->getContext()),
                          cast<VectorType>(Dst)->getElementCount());
      std::pair<InstructionCost, MVT> VecF32LT =
          getTypeLegalizationCost(VecF32Ty);
      Cost += getCastInstrCost(Opcode, VecF32Ty, Src, CCH, CostKind, I);
      Cost += VecF32LT.first * getRISCVInstructionCost(RISCV::VFNCVT_F_F_W,
                                                       DstLT.second, CostKind);
      return Cost;
    }

    if (DstEltSize == SrcEltSize)
      Cost += getRISCVInstructionCost(FCVT, DstLT.second, CostKind);
    else if (DstEltSize > SrcEltSize) {
      if ((DstEltSize / 2) > SrcEltSize) {
        VectorType *VecTy =
            VectorType::get(IntegerType::get(Dst->getContext(), DstEltSize / 2),
                            cast<VectorType>(Dst)->getElementCount());
        unsigned Op = IsSigned ? Instruction::SExt : Instruction::ZExt;
        Cost += getCastInstrCost(Op, VecTy, Src, CCH, CostKind, I);
      }
      Cost += getRISCVInstructionCost(FWCVT, DstLT.second, CostKind);
    } else
      Cost += getRISCVInstructionCost(FNCVT, DstLT.second, CostKind);
    return Cost;
  }
  }
  return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
}

unsigned RISCVTTIImpl::getEstimatedVLFor(VectorType *Ty) const {
  if (isa<ScalableVectorType>(Ty)) {
    const unsigned EltSize = DL.getTypeSizeInBits(Ty->getElementType());
    const unsigned MinSize = DL.getTypeSizeInBits(Ty).getKnownMinValue();
    const unsigned VectorBits = *getVScaleForTuning() * RISCV::RVVBitsPerBlock;
    return RISCVTargetLowering::computeVLMAX(VectorBits, EltSize, MinSize);
  }
  return cast<FixedVectorType>(Ty)->getNumElements();
}

InstructionCost
RISCVTTIImpl::getMinMaxReductionCost(Intrinsic::ID IID, VectorType *Ty,
                                     FastMathFlags FMF,
                                     TTI::TargetCostKind CostKind) const {
  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getMinMaxReductionCost(IID, Ty, FMF, CostKind);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (Ty->getScalarSizeInBits() > ST->getELen())
    return BaseT::getMinMaxReductionCost(IID, Ty, FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  if (Ty->getElementType()->isIntegerTy(1)) {
    // SelectionDAGBuilder does following transforms:
    //   vector_reduce_{smin,umax}(<n x i1>) --> vector_reduce_or(<n x i1>)
    //   vector_reduce_{smax,umin}(<n x i1>) --> vector_reduce_and(<n x i1>)
    if (IID == Intrinsic::umax || IID == Intrinsic::smin)
      return getArithmeticReductionCost(Instruction::Or, Ty, FMF, CostKind);
    else
      return getArithmeticReductionCost(Instruction::And, Ty, FMF, CostKind);
  }

  if (IID == Intrinsic::maximum || IID == Intrinsic::minimum) {
    SmallVector<unsigned, 3> Opcodes;
    InstructionCost ExtraCost = 0;
    switch (IID) {
    case Intrinsic::maximum:
      if (FMF.noNaNs()) {
        Opcodes = {RISCV::VFREDMAX_VS, RISCV::VFMV_F_S};
      } else {
        Opcodes = {RISCV::VMFNE_VV, RISCV::VCPOP_M, RISCV::VFREDMAX_VS,
                   RISCV::VFMV_F_S};
        // Cost of Canonical Nan + branch
        // lui a0, 523264
        // fmv.w.x fa0, a0
        Type *DstTy = Ty->getScalarType();
        const unsigned EltTyBits = DstTy->getScalarSizeInBits();
        Type *SrcTy = IntegerType::getIntNTy(DstTy->getContext(), EltTyBits);
        ExtraCost = 1 +
                    getCastInstrCost(Instruction::UIToFP, DstTy, SrcTy,
                                     TTI::CastContextHint::None, CostKind) +
                    getCFInstrCost(Instruction::Br, CostKind);
      }
      break;

    case Intrinsic::minimum:
      if (FMF.noNaNs()) {
        Opcodes = {RISCV::VFREDMIN_VS, RISCV::VFMV_F_S};
      } else {
        Opcodes = {RISCV::VMFNE_VV, RISCV::VCPOP_M, RISCV::VFREDMIN_VS,
                   RISCV::VFMV_F_S};
        // Cost of Canonical Nan + branch
        // lui a0, 523264
        // fmv.w.x fa0, a0
        Type *DstTy = Ty->getScalarType();
        const unsigned EltTyBits = DL.getTypeSizeInBits(DstTy);
        Type *SrcTy = IntegerType::getIntNTy(DstTy->getContext(), EltTyBits);
        ExtraCost = 1 +
                    getCastInstrCost(Instruction::UIToFP, DstTy, SrcTy,
                                     TTI::CastContextHint::None, CostKind) +
                    getCFInstrCost(Instruction::Br, CostKind);
      }
      break;
    }
    return ExtraCost + getRISCVInstructionCost(Opcodes, LT.second, CostKind);
  }

  // IR Reduction is composed by one rvv reduction instruction and vmv
  unsigned SplitOp;
  SmallVector<unsigned, 3> Opcodes;
  switch (IID) {
  default:
    llvm_unreachable("Unsupported intrinsic");
  case Intrinsic::smax:
    SplitOp = RISCV::VMAX_VV;
    Opcodes = {RISCV::VREDMAX_VS, RISCV::VMV_X_S};
    break;
  case Intrinsic::smin:
    SplitOp = RISCV::VMIN_VV;
    Opcodes = {RISCV::VREDMIN_VS, RISCV::VMV_X_S};
    break;
  case Intrinsic::umax:
    SplitOp = RISCV::VMAXU_VV;
    Opcodes = {RISCV::VREDMAXU_VS, RISCV::VMV_X_S};
    break;
  case Intrinsic::umin:
    SplitOp = RISCV::VMINU_VV;
    Opcodes = {RISCV::VREDMINU_VS, RISCV::VMV_X_S};
    break;
  case Intrinsic::maxnum:
    SplitOp = RISCV::VFMAX_VV;
    Opcodes = {RISCV::VFREDMAX_VS, RISCV::VFMV_F_S};
    break;
  case Intrinsic::minnum:
    SplitOp = RISCV::VFMIN_VV;
    Opcodes = {RISCV::VFREDMIN_VS, RISCV::VFMV_F_S};
    break;
  }
  // Add a cost for data larger than LMUL8
  InstructionCost SplitCost =
      (LT.first > 1) ? (LT.first - 1) *
                           getRISCVInstructionCost(SplitOp, LT.second, CostKind)
                     : 0;
  return SplitCost + getRISCVInstructionCost(Opcodes, LT.second, CostKind);
}

InstructionCost
RISCVTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                         std::optional<FastMathFlags> FMF,
                                         TTI::TargetCostKind CostKind) const {
  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (Ty->getScalarSizeInBits() > ST->getELen())
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  if (ISD != ISD::ADD && ISD != ISD::OR && ISD != ISD::XOR && ISD != ISD::AND &&
      ISD != ISD::FADD)
    return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  Type *ElementTy = Ty->getElementType();
  if (ElementTy->isIntegerTy(1)) {
    // Example sequences:
    //   vfirst.m a0, v0
    //   seqz a0, a0
    if (LT.second == MVT::v1i1)
      return getRISCVInstructionCost(RISCV::VFIRST_M, LT.second, CostKind) +
             getCmpSelInstrCost(Instruction::ICmp, ElementTy, ElementTy,
                                CmpInst::ICMP_EQ, CostKind);

    if (ISD == ISD::AND) {
      // Example sequences:
      //   vmand.mm v8, v9, v8 ; needed every time type is split
      //   vmnot.m v8, v0      ; alias for vmnand
      //   vcpop.m a0, v8
      //   seqz a0, a0

      // See the discussion: https://github.com/llvm/llvm-project/pull/119160
      // For LMUL <= 8, there is no splitting,
      //   the sequences are vmnot, vcpop and seqz.
      // When LMUL > 8 and split = 1,
      //   the sequences are vmnand, vcpop and seqz.
      // When LMUL > 8 and split > 1,
      //   the sequences are (LT.first-2) * vmand, vmnand, vcpop and seqz.
      return ((LT.first > 2) ? (LT.first - 2) : 0) *
                 getRISCVInstructionCost(RISCV::VMAND_MM, LT.second, CostKind) +
             getRISCVInstructionCost(RISCV::VMNAND_MM, LT.second, CostKind) +
             getRISCVInstructionCost(RISCV::VCPOP_M, LT.second, CostKind) +
             getCmpSelInstrCost(Instruction::ICmp, ElementTy, ElementTy,
                                CmpInst::ICMP_EQ, CostKind);
    } else if (ISD == ISD::XOR || ISD == ISD::ADD) {
      // Example sequences:
      //   vsetvli a0, zero, e8, mf8, ta, ma
      //   vmxor.mm v8, v0, v8 ; needed every time type is split
      //   vcpop.m a0, v8
      //   andi a0, a0, 1
      return (LT.first - 1) *
                 getRISCVInstructionCost(RISCV::VMXOR_MM, LT.second, CostKind) +
             getRISCVInstructionCost(RISCV::VCPOP_M, LT.second, CostKind) + 1;
    } else {
      assert(ISD == ISD::OR);
      // Example sequences:
      //   vsetvli a0, zero, e8, mf8, ta, ma
      //   vmor.mm v8, v9, v8 ; needed every time type is split
      //   vcpop.m a0, v0
      //   snez a0, a0
      return (LT.first - 1) *
                 getRISCVInstructionCost(RISCV::VMOR_MM, LT.second, CostKind) +
             getRISCVInstructionCost(RISCV::VCPOP_M, LT.second, CostKind) +
             getCmpSelInstrCost(Instruction::ICmp, ElementTy, ElementTy,
                                CmpInst::ICMP_NE, CostKind);
    }
  }

  // IR Reduction of or/and is composed by one vmv and one rvv reduction
  // instruction, and others is composed by two vmv and one rvv reduction
  // instruction
  unsigned SplitOp;
  SmallVector<unsigned, 3> Opcodes;
  switch (ISD) {
  case ISD::ADD:
    SplitOp = RISCV::VADD_VV;
    Opcodes = {RISCV::VMV_S_X, RISCV::VREDSUM_VS, RISCV::VMV_X_S};
    break;
  case ISD::OR:
    SplitOp = RISCV::VOR_VV;
    Opcodes = {RISCV::VREDOR_VS, RISCV::VMV_X_S};
    break;
  case ISD::XOR:
    SplitOp = RISCV::VXOR_VV;
    Opcodes = {RISCV::VMV_S_X, RISCV::VREDXOR_VS, RISCV::VMV_X_S};
    break;
  case ISD::AND:
    SplitOp = RISCV::VAND_VV;
    Opcodes = {RISCV::VREDAND_VS, RISCV::VMV_X_S};
    break;
  case ISD::FADD:
    // We can't promote f16/bf16 fadd reductions.
    if ((LT.second.getScalarType() == MVT::f16 && !ST->hasVInstructionsF16()) ||
        LT.second.getScalarType() == MVT::bf16)
      return BaseT::getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);
    if (TTI::requiresOrderedReduction(FMF)) {
      Opcodes.push_back(RISCV::VFMV_S_F);
      for (unsigned i = 0; i < LT.first.getValue(); i++)
        Opcodes.push_back(RISCV::VFREDOSUM_VS);
      Opcodes.push_back(RISCV::VFMV_F_S);
      return getRISCVInstructionCost(Opcodes, LT.second, CostKind);
    }
    SplitOp = RISCV::VFADD_VV;
    Opcodes = {RISCV::VFMV_S_F, RISCV::VFREDUSUM_VS, RISCV::VFMV_F_S};
    break;
  }
  // Add a cost for data larger than LMUL8
  InstructionCost SplitCost =
      (LT.first > 1) ? (LT.first - 1) *
                           getRISCVInstructionCost(SplitOp, LT.second, CostKind)
                     : 0;
  return SplitCost + getRISCVInstructionCost(Opcodes, LT.second, CostKind);
}

InstructionCost RISCVTTIImpl::getExtendedReductionCost(
    unsigned Opcode, bool IsUnsigned, Type *ResTy, VectorType *ValTy,
    std::optional<FastMathFlags> FMF, TTI::TargetCostKind CostKind) const {
  if (isa<FixedVectorType>(ValTy) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  // Skip if scalar size of ResTy is bigger than ELEN.
  if (ResTy->getScalarSizeInBits() > ST->getELen())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  if (Opcode != Instruction::Add && Opcode != Instruction::FAdd)
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);

  if (IsUnsigned && Opcode == Instruction::Add &&
      LT.second.isFixedLengthVector() && LT.second.getScalarType() == MVT::i1) {
    // Represent vector_reduce_add(ZExt(<n x i1>)) as
    // ZExtOrTrunc(ctpop(bitcast <n x i1> to in)).
    return LT.first *
           getRISCVInstructionCost(RISCV::VCPOP_M, LT.second, CostKind);
  }

  if (ResTy->getScalarSizeInBits() != 2 * LT.second.getScalarSizeInBits())
    return BaseT::getExtendedReductionCost(Opcode, IsUnsigned, ResTy, ValTy,
                                           FMF, CostKind);

  return (LT.first - 1) +
         getArithmeticReductionCost(Opcode, ValTy, FMF, CostKind);
}

InstructionCost
RISCVTTIImpl::getStoreImmCost(Type *Ty, TTI::OperandValueInfo OpInfo,
                              TTI::TargetCostKind CostKind) const {
  assert(OpInfo.isConstant() && "non constant operand?");
  if (!isa<VectorType>(Ty))
    // FIXME: We need to account for immediate materialization here, but doing
    // a decent job requires more knowledge about the immediate than we
    // currently have here.
    return 0;

  if (OpInfo.isUniform())
    // vmv.v.i, vmv.v.x, or vfmv.v.f
    // We ignore the cost of the scalar constant materialization to be consistent
    // with how we treat scalar constants themselves just above.
    return 1;

  return getConstantPoolLoadCost(Ty, CostKind);
}

InstructionCost RISCVTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                              Align Alignment,
                                              unsigned AddressSpace,
                                              TTI::TargetCostKind CostKind,
                                              TTI::OperandValueInfo OpInfo,
                                              const Instruction *I) const {
  EVT VT = TLI->getValueType(DL, Src, true);
  // Type legalization can't handle structs
  if (VT == MVT::Other)
    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind, OpInfo, I);

  InstructionCost Cost = 0;
  if (Opcode == Instruction::Store && OpInfo.isConstant())
    Cost += getStoreImmCost(Src, OpInfo, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Src);

  InstructionCost BaseCost = [&]() {
    InstructionCost Cost = LT.first;
    if (CostKind != TTI::TCK_RecipThroughput)
      return Cost;

    // Our actual lowering for the case where a wider legal type is available
    // uses the a VL predicated load on the wider type.  This is reflected in
    // the result of getTypeLegalizationCost, but BasicTTI assumes the
    // widened cases are scalarized.
    const DataLayout &DL = this->getDataLayout();
    if (Src->isVectorTy() && LT.second.isVector() &&
        TypeSize::isKnownLT(DL.getTypeStoreSizeInBits(Src),
                            LT.second.getSizeInBits()))
        return Cost;

    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind, OpInfo, I);
  }();

  // Assume memory ops cost scale with the number of vector registers
  // possible accessed by the instruction.  Note that BasicTTI already
  // handles the LT.first term for us.
  if (LT.second.isVector() && CostKind != TTI::TCK_CodeSize)
    BaseCost *= TLI->getLMULCost(LT.second);
  return Cost + BaseCost;
}

InstructionCost RISCVTTIImpl::getCmpSelInstrCost(
    unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
    TTI::TargetCostKind CostKind, TTI::OperandValueInfo Op1Info,
    TTI::OperandValueInfo Op2Info, const Instruction *I) const {
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     Op1Info, Op2Info, I);

  if (isa<FixedVectorType>(ValTy) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     Op1Info, Op2Info, I);

  // Skip if scalar size of ValTy is bigger than ELEN.
  if (ValTy->isVectorTy() && ValTy->getScalarSizeInBits() > ST->getELen())
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     Op1Info, Op2Info, I);

  auto GetConstantMatCost =
      [&](TTI::OperandValueInfo OpInfo) -> InstructionCost {
    if (OpInfo.isUniform())
      // We return 0 we currently ignore the cost of materializing scalar
      // constants in GPRs.
      return 0;

    return getConstantPoolLoadCost(ValTy, CostKind);
  };

  InstructionCost ConstantMatCost;
  if (Op1Info.isConstant())
    ConstantMatCost += GetConstantMatCost(Op1Info);
  if (Op2Info.isConstant())
    ConstantMatCost += GetConstantMatCost(Op2Info);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);
  if (Opcode == Instruction::Select && ValTy->isVectorTy()) {
    if (CondTy->isVectorTy()) {
      if (ValTy->getScalarSizeInBits() == 1) {
        // vmandn.mm v8, v8, v9
        // vmand.mm v9, v0, v9
        // vmor.mm v0, v9, v8
        return ConstantMatCost +
               LT.first *
                   getRISCVInstructionCost(
                       {RISCV::VMANDN_MM, RISCV::VMAND_MM, RISCV::VMOR_MM},
                       LT.second, CostKind);
      }
      // vselect and max/min are supported natively.
      return ConstantMatCost +
             LT.first * getRISCVInstructionCost(RISCV::VMERGE_VVM, LT.second,
                                                CostKind);
    }

    if (ValTy->getScalarSizeInBits() == 1) {
      //  vmv.v.x v9, a0
      //  vmsne.vi v9, v9, 0
      //  vmandn.mm v8, v8, v9
      //  vmand.mm v9, v0, v9
      //  vmor.mm v0, v9, v8
      MVT InterimVT = LT.second.changeVectorElementType(MVT::i8);
      return ConstantMatCost +
             LT.first *
                 getRISCVInstructionCost({RISCV::VMV_V_X, RISCV::VMSNE_VI},
                                         InterimVT, CostKind) +
             LT.first * getRISCVInstructionCost(
                            {RISCV::VMANDN_MM, RISCV::VMAND_MM, RISCV::VMOR_MM},
                            LT.second, CostKind);
    }

    // vmv.v.x v10, a0
    // vmsne.vi v0, v10, 0
    // vmerge.vvm v8, v9, v8, v0
    return ConstantMatCost +
           LT.first * getRISCVInstructionCost(
                          {RISCV::VMV_V_X, RISCV::VMSNE_VI, RISCV::VMERGE_VVM},
                          LT.second, CostKind);
  }

  if ((Opcode == Instruction::ICmp) && ValTy->isVectorTy() &&
      CmpInst::isIntPredicate(VecPred)) {
    // Use VMSLT_VV to represent VMSEQ, VMSNE, VMSLTU, VMSLEU, VMSLT, VMSLE
    // provided they incur the same cost across all implementations
    return ConstantMatCost + LT.first * getRISCVInstructionCost(RISCV::VMSLT_VV,
                                                                LT.second,
                                                                CostKind);
  }

  if ((Opcode == Instruction::FCmp) && ValTy->isVectorTy() &&
      CmpInst::isFPPredicate(VecPred)) {

    // Use VMXOR_MM and VMXNOR_MM to generate all true/false mask
    if ((VecPred == CmpInst::FCMP_FALSE) || (VecPred == CmpInst::FCMP_TRUE))
      return ConstantMatCost +
             getRISCVInstructionCost(RISCV::VMXOR_MM, LT.second, CostKind);

    // If we do not support the input floating point vector type, use the base
    // one which will calculate as:
    // ScalarizeCost + Num * Cost for fixed vector,
    // InvalidCost for scalable vector.
    if ((ValTy->getScalarSizeInBits() == 16 && !ST->hasVInstructionsF16()) ||
        (ValTy->getScalarSizeInBits() == 32 && !ST->hasVInstructionsF32()) ||
        (ValTy->getScalarSizeInBits() == 64 && !ST->hasVInstructionsF64()))
      return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                       Op1Info, Op2Info, I);

    // Assuming vector fp compare and mask instructions are all the same cost
    // until a need arises to differentiate them.
    switch (VecPred) {
    case CmpInst::FCMP_ONE: // vmflt.vv + vmflt.vv + vmor.mm
    case CmpInst::FCMP_ORD: // vmfeq.vv + vmfeq.vv + vmand.mm
    case CmpInst::FCMP_UNO: // vmfne.vv + vmfne.vv + vmor.mm
    case CmpInst::FCMP_UEQ: // vmflt.vv + vmflt.vv + vmnor.mm
      return ConstantMatCost +
             LT.first * getRISCVInstructionCost(
                            {RISCV::VMFLT_VV, RISCV::VMFLT_VV, RISCV::VMOR_MM},
                            LT.second, CostKind);

    case CmpInst::FCMP_UGT: // vmfle.vv + vmnot.m
    case CmpInst::FCMP_UGE: // vmflt.vv + vmnot.m
    case CmpInst::FCMP_ULT: // vmfle.vv + vmnot.m
    case CmpInst::FCMP_ULE: // vmflt.vv + vmnot.m
      return ConstantMatCost +
             LT.first *
                 getRISCVInstructionCost({RISCV::VMFLT_VV, RISCV::VMNAND_MM},
                                         LT.second, CostKind);

    case CmpInst::FCMP_OEQ: // vmfeq.vv
    case CmpInst::FCMP_OGT: // vmflt.vv
    case CmpInst::FCMP_OGE: // vmfle.vv
    case CmpInst::FCMP_OLT: // vmflt.vv
    case CmpInst::FCMP_OLE: // vmfle.vv
    case CmpInst::FCMP_UNE: // vmfne.vv
      return ConstantMatCost +
             LT.first *
                 getRISCVInstructionCost(RISCV::VMFLT_VV, LT.second, CostKind);
    default:
      break;
    }
  }

  // With ShortForwardBranchOpt or ConditionalMoveFusion, scalar icmp + select
  // instructions will lower to SELECT_CC and lower to PseudoCCMOVGPR which will
  // generate a conditional branch + mv. The cost of scalar (icmp + select) will
  // be (0 + select instr cost).
  if (ST->hasConditionalMoveFusion() && I && isa<ICmpInst>(I) &&
      ValTy->isIntegerTy() && !I->user_empty()) {
    if (all_of(I->users(), [&](const User *U) {
          return match(U, m_Select(m_Specific(I), m_Value(), m_Value())) &&
                 U->getType()->isIntegerTy() &&
                 !isa<ConstantData>(U->getOperand(1)) &&
                 !isa<ConstantData>(U->getOperand(2));
        }))
      return 0;
  }

  // TODO: Add cost for scalar type.

  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                   Op1Info, Op2Info, I);
}

InstructionCost RISCVTTIImpl::getCFInstrCost(unsigned Opcode,
                                             TTI::TargetCostKind CostKind,
                                             const Instruction *I) const {
  if (CostKind != TTI::TCK_RecipThroughput)
    return Opcode == Instruction::PHI ? 0 : 1;
  // Branches are assumed to be predicted.
  return 0;
}

InstructionCost RISCVTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                 TTI::TargetCostKind CostKind,
                                                 unsigned Index,
                                                 const Value *Op0,
                                                 const Value *Op1) const {
  assert(Val->isVectorTy() && "This must be a vector type");

  if (Opcode != Instruction::ExtractElement &&
      Opcode != Instruction::InsertElement)
    return BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1);

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Val);

  // This type is legalized to a scalar type.
  if (!LT.second.isVector()) {
    auto *FixedVecTy = cast<FixedVectorType>(Val);
    // If Index is a known constant, cost is zero.
    if (Index != -1U)
      return 0;
    // Extract/InsertElement with non-constant index is very costly when
    // scalarized; estimate cost of loads/stores sequence via the stack:
    // ExtractElement cost: store vector to stack, load scalar;
    // InsertElement cost: store vector to stack, store scalar, load vector.
    Type *ElemTy = FixedVecTy->getElementType();
    auto NumElems = FixedVecTy->getNumElements();
    auto Align = DL.getPrefTypeAlign(ElemTy);
    InstructionCost LoadCost =
        getMemoryOpCost(Instruction::Load, ElemTy, Align, 0, CostKind);
    InstructionCost StoreCost =
        getMemoryOpCost(Instruction::Store, ElemTy, Align, 0, CostKind);
    return Opcode == Instruction::ExtractElement
               ? StoreCost * NumElems + LoadCost
               : (StoreCost + LoadCost) * NumElems + StoreCost;
  }

  // For unsupported scalable vector.
  if (LT.second.isScalableVector() && !LT.first.isValid())
    return LT.first;

  // Mask vector extract/insert is expanded via e8.
  if (Val->getScalarSizeInBits() == 1) {
    VectorType *WideTy =
      VectorType::get(IntegerType::get(Val->getContext(), 8),
                      cast<VectorType>(Val)->getElementCount());
    if (Opcode == Instruction::ExtractElement) {
      InstructionCost ExtendCost
        = getCastInstrCost(Instruction::ZExt, WideTy, Val,
                           TTI::CastContextHint::None, CostKind);
      InstructionCost ExtractCost
        = getVectorInstrCost(Opcode, WideTy, CostKind, Index, nullptr, nullptr);
      return ExtendCost + ExtractCost;
    }
    InstructionCost ExtendCost
      = getCastInstrCost(Instruction::ZExt, WideTy, Val,
                         TTI::CastContextHint::None, CostKind);
    InstructionCost InsertCost
      = getVectorInstrCost(Opcode, WideTy, CostKind, Index, nullptr, nullptr);
    InstructionCost TruncCost
      = getCastInstrCost(Instruction::Trunc, Val, WideTy,
                         TTI::CastContextHint::None, CostKind);
    return ExtendCost + InsertCost + TruncCost;
  }


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

    // If exact VLEN is known, we will insert/extract into the appropriate
    // subvector with no additional subvector insert/extract cost.
    if (auto VLEN = ST->getRealVLen()) {
      unsigned EltSize = LT.second.getScalarSizeInBits();
      unsigned M1Max = *VLEN / EltSize;
      Index = Index % M1Max;
    }

    // We could extract/insert the first element without vslidedown/vslideup.
    if (Index == 0)
      SlideCost = 0;
    else if (Opcode == Instruction::InsertElement)
      SlideCost = 1; // With a constant index, we do not need to use addi.
  }

  // When the vector needs to split into multiple register groups and the index
  // exceeds single vector register group, we need to insert/extract the element
  // via stack.
  if (LT.first > 1 &&
      ((Index == -1U) || (Index >= LT.second.getVectorMinNumElements() &&
                          LT.second.isScalableVector()))) {
    Type *ScalarType = Val->getScalarType();
    Align VecAlign = DL.getPrefTypeAlign(Val);
    Align SclAlign = DL.getPrefTypeAlign(ScalarType);
    // Extra addi for unknown index.
    InstructionCost IdxCost = Index == -1U ? 1 : 0;

    // Store all split vectors into stack and load the target element.
    if (Opcode == Instruction::ExtractElement)
      return getMemoryOpCost(Instruction::Store, Val, VecAlign, 0, CostKind) +
             getMemoryOpCost(Instruction::Load, ScalarType, SclAlign, 0,
                             CostKind) +
             IdxCost;

    // Store all split vectors into stack and store the target element and load
    // vectors back.
    return getMemoryOpCost(Instruction::Store, Val, VecAlign, 0, CostKind) +
           getMemoryOpCost(Instruction::Load, Val, VecAlign, 0, CostKind) +
           getMemoryOpCost(Instruction::Store, ScalarType, SclAlign, 0,
                           CostKind) +
           IdxCost;
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
    ArrayRef<const Value *> Args, const Instruction *CxtI) const {

  // TODO: Handle more cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  if (isa<FixedVectorType>(Ty) && !ST->useRVVForFixedLengthVectors())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  // Skip if scalar size of Ty is bigger than ELEN.
  if (isa<VectorType>(Ty) && Ty->getScalarSizeInBits() > ST->getELen())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);

  // TODO: Handle scalar type.
  if (!LT.second.isVector())
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);

  // f16 with zvfhmin and bf16 will be promoted to f32.
  // FIXME: nxv32[b]f16 will be custom lowered and split.
  unsigned ISDOpcode = TLI->InstructionOpcodeToISD(Opcode);
  InstructionCost CastCost = 0;
  if ((LT.second.getVectorElementType() == MVT::f16 ||
       LT.second.getVectorElementType() == MVT::bf16) &&
      TLI->getOperationAction(ISDOpcode, LT.second) ==
          TargetLoweringBase::LegalizeAction::Promote) {
    MVT PromotedVT = TLI->getTypeToPromoteTo(ISDOpcode, LT.second);
    Type *PromotedTy = EVT(PromotedVT).getTypeForEVT(Ty->getContext());
    Type *LegalTy = EVT(LT.second).getTypeForEVT(Ty->getContext());
    // Add cost of extending arguments
    CastCost += LT.first * Args.size() *
                getCastInstrCost(Instruction::FPExt, PromotedTy, LegalTy,
                                 TTI::CastContextHint::None, CostKind);
    // Add cost of truncating result
    CastCost +=
        LT.first * getCastInstrCost(Instruction::FPTrunc, LegalTy, PromotedTy,
                                    TTI::CastContextHint::None, CostKind);
    // Compute cost of op in promoted type
    LT.second = PromotedVT;
  }

  auto getConstantMatCost =
      [&](unsigned Operand, TTI::OperandValueInfo OpInfo) -> InstructionCost {
    if (OpInfo.isUniform() && canSplatOperand(Opcode, Operand))
      // Two sub-cases:
      // * Has a 5 bit immediate operand which can be splatted.
      // * Has a larger immediate which must be materialized in scalar register
      // We return 0 for both as we currently ignore the cost of materializing
      // scalar constants in GPRs.
      return 0;

    return getConstantPoolLoadCost(Ty, CostKind);
  };

  // Add the cost of materializing any constant vectors required.
  InstructionCost ConstantMatCost = 0;
  if (Op1Info.isConstant())
    ConstantMatCost += getConstantMatCost(0, Op1Info);
  if (Op2Info.isConstant())
    ConstantMatCost += getConstantMatCost(1, Op2Info);

  unsigned Op;
  switch (ISDOpcode) {
  case ISD::ADD:
  case ISD::SUB:
    Op = RISCV::VADD_VV;
    break;
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    Op = RISCV::VSLL_VV;
    break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    Op = (Ty->getScalarSizeInBits() == 1) ? RISCV::VMAND_MM : RISCV::VAND_VV;
    break;
  case ISD::MUL:
  case ISD::MULHS:
  case ISD::MULHU:
    Op = RISCV::VMUL_VV;
    break;
  case ISD::SDIV:
  case ISD::UDIV:
    Op = RISCV::VDIV_VV;
    break;
  case ISD::SREM:
  case ISD::UREM:
    Op = RISCV::VREM_VV;
    break;
  case ISD::FADD:
  case ISD::FSUB:
    Op = RISCV::VFADD_VV;
    break;
  case ISD::FMUL:
    Op = RISCV::VFMUL_VV;
    break;
  case ISD::FDIV:
    Op = RISCV::VFDIV_VV;
    break;
  case ISD::FNEG:
    Op = RISCV::VFSGNJN_VV;
    break;
  default:
    // Assuming all other instructions have the same cost until a need arises to
    // differentiate them.
    return CastCost + ConstantMatCost +
           BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info,
                                         Args, CxtI);
  }

  InstructionCost InstrCost = getRISCVInstructionCost(Op, LT.second, CostKind);
  // We use BasicTTIImpl to calculate scalar costs, which assumes floating point
  // ops are twice as expensive as integer ops. Do the same for vectors so
  // scalar floating point ops aren't cheaper than their vector equivalents.
  if (Ty->isFPOrFPVectorTy())
    InstrCost *= 2;
  return CastCost + ConstantMatCost + LT.first * InstrCost;
}

// TODO: Deduplicate from TargetTransformInfoImplCRTPBase.
InstructionCost RISCVTTIImpl::getPointersChainCost(
    ArrayRef<const Value *> Ptrs, const Value *Base,
    const TTI::PointersChainInfo &Info, Type *AccessTy,
    TTI::TargetCostKind CostKind) const {
  InstructionCost Cost = TTI::TCC_Free;
  // In the basic model we take into account GEP instructions only
  // (although here can come alloca instruction, a value, constants and/or
  // constant expressions, PHIs, bitcasts ... whatever allowed to be used as a
  // pointer). Typically, if Base is a not a GEP-instruction and all the
  // pointers are relative to the same base address, all the rest are
  // either GEP instructions, PHIs, bitcasts or constants. When we have same
  // base, we just calculate cost of each non-Base GEP as an ADD operation if
  // any their index is a non-const.
  // If no known dependencies between the pointers cost is calculated as a sum
  // of costs of GEP instructions.
  for (auto [I, V] : enumerate(Ptrs)) {
    const auto *GEP = dyn_cast<GetElementPtrInst>(V);
    if (!GEP)
      continue;
    if (Info.isSameBase() && V != Base) {
      if (GEP->hasAllConstantIndices())
        continue;
      // If the chain is unit-stride and BaseReg + stride*i is a legal
      // addressing mode, then presume the base GEP is sitting around in a
      // register somewhere and check if we can fold the offset relative to
      // it.
      unsigned Stride = DL.getTypeStoreSize(AccessTy);
      if (Info.isUnitStride() &&
          isLegalAddressingMode(AccessTy,
                                /* BaseGV */ nullptr,
                                /* BaseOffset */ Stride * I,
                                /* HasBaseReg */ true,
                                /* Scale */ 0,
                                GEP->getType()->getPointerAddressSpace()))
        continue;
      Cost += getArithmeticInstrCost(Instruction::Add, GEP->getType(), CostKind,
                                     {TTI::OK_AnyValue, TTI::OP_None},
                                     {TTI::OK_AnyValue, TTI::OP_None}, {});
    } else {
      SmallVector<const Value *> Indices(GEP->indices());
      Cost += getGEPCost(GEP->getSourceElementType(), GEP->getPointerOperand(),
                         Indices, AccessTy, CostKind);
    }
  }
  return Cost;
}

void RISCVTTIImpl::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, TTI::UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  // TODO: More tuning on benchmarks and metrics with changes as needed
  //       would apply to all settings below to enable performance.


  if (ST->enableDefaultUnroll())
    return BasicTTIImplBase::getUnrollingPreferences(L, SE, UP, ORE);

  // Enable Upper bound unrolling universally, not dependent upon the conditions
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

  // Force unrolling small loops can be very useful because of the branch
  // taken cost of the backedge.
  if (Cost < 12)
    UP.Force = true;
}

void RISCVTTIImpl::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::PeelingPreferences &PP) const {
  BaseT::getPeelingPreferences(L, SE, PP);
}

unsigned RISCVTTIImpl::getRegUsageForType(Type *Ty) const {
  if (Ty->isVectorTy()) {
    // f16 with only zvfhmin and bf16 will be promoted to f32
    Type *EltTy = cast<VectorType>(Ty)->getElementType();
    if ((EltTy->isHalfTy() && !ST->hasVInstructionsF16()) ||
        EltTy->isBFloatTy())
      Ty = VectorType::get(Type::getFloatTy(Ty->getContext()),
                           cast<VectorType>(Ty));

    TypeSize Size = DL.getTypeSizeInBits(Ty);
    if (Size.isScalable() && ST->hasVInstructions())
      return divideCeil(Size.getKnownMinValue(), RISCV::RVVBitsPerBlock);

    if (ST->useRVVForFixedLengthVectors())
      return divideCeil(Size, ST->getRealMinVLen());
  }

  return BaseT::getRegUsageForType(Ty);
}

unsigned RISCVTTIImpl::getMaximumVF(unsigned ElemWidth, unsigned Opcode) const {
  if (SLPMaxVF.getNumOccurrences())
    return SLPMaxVF;

  // Return how many elements can fit in getRegisterBitwidth.  This is the
  // same routine as used in LoopVectorizer.  We should probably be
  // accounting for whether we actually have instructions with the right
  // lane type, but we don't have enough information to do that without
  // some additional plumbing which hasn't been justified yet.
  TypeSize RegWidth =
    getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector);
  // If no vector registers, or absurd element widths, disable
  // vectorization by returning 1.
  return std::max<unsigned>(1U, RegWidth.getFixedValue() / ElemWidth);
}

unsigned RISCVTTIImpl::getMinTripCountTailFoldingThreshold() const {
  return RVVMinTripCount;
}

TTI::AddressingModeKind
RISCVTTIImpl::getPreferredAddressingMode(const Loop *L,
                                         ScalarEvolution *SE) const {
  if (ST->hasVendorXCVmem() && !ST->is64Bit())
    return TTI::AMK_PostIndexed;

  return BasicTTIImplBase::getPreferredAddressingMode(L, SE);
}

bool RISCVTTIImpl::isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                                 const TargetTransformInfo::LSRCost &C2) const {
  // RISC-V specific here are "instruction number 1st priority".
  // If we need to emit adds inside the loop to add up base registers, then
  // we need at least one extra temporary register.
  unsigned C1NumRegs = C1.NumRegs + (C1.NumBaseAdds != 0);
  unsigned C2NumRegs = C2.NumRegs + (C2.NumBaseAdds != 0);
  return std::tie(C1.Insns, C1NumRegs, C1.AddRecCost,
                  C1.NumIVMuls, C1.NumBaseAdds,
                  C1.ScaleCost, C1.ImmCost, C1.SetupCost) <
         std::tie(C2.Insns, C2NumRegs, C2.AddRecCost,
                  C2.NumIVMuls, C2.NumBaseAdds,
                  C2.ScaleCost, C2.ImmCost, C2.SetupCost);
}

bool RISCVTTIImpl::isLegalMaskedExpandLoad(Type *DataTy,
                                           Align Alignment) const {
  auto *VTy = dyn_cast<VectorType>(DataTy);
  if (!VTy || VTy->isScalableTy())
    return false;

  if (!isLegalMaskedLoadStore(DataTy, Alignment))
    return false;

  // FIXME: If it is an i8 vector and the element count exceeds 256, we should
  // scalarize these types with LMUL >= maximum fixed-length LMUL.
  if (VTy->getElementType()->isIntegerTy(8))
    if (VTy->getElementCount().getFixedValue() > 256)
      return VTy->getPrimitiveSizeInBits() / ST->getRealMinVLen() <
             ST->getMaxLMULForFixedLengthVectors();
  return true;
}

bool RISCVTTIImpl::isLegalMaskedCompressStore(Type *DataTy,
                                              Align Alignment) const {
  auto *VTy = dyn_cast<VectorType>(DataTy);
  if (!VTy || VTy->isScalableTy())
    return false;

  if (!isLegalMaskedLoadStore(DataTy, Alignment))
    return false;
  return true;
}

/// See if \p I should be considered for address type promotion. We check if \p
/// I is a sext with right type and used in memory accesses. If it used in a
/// "complex" getelementptr, we allow it to be promoted without finding other
/// sext instructions that sign extended the same initial value. A getelementptr
/// is considered as "complex" if it has more than 2 operands.
bool RISCVTTIImpl::shouldConsiderAddressTypePromotion(
    const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const {
  bool Considerable = false;
  AllowPromotionWithoutCommonHeader = false;
  if (!isa<SExtInst>(&I))
    return false;
  Type *ConsideredSExtType =
      Type::getInt64Ty(I.getParent()->getParent()->getContext());
  if (I.getType() != ConsideredSExtType)
    return false;
  // See if the sext is the one with the right type and used in at least one
  // GetElementPtrInst.
  for (const User *U : I.users()) {
    if (const GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(U)) {
      Considerable = true;
      // A getelementptr is considered as "complex" if it has more than 2
      // operands. We will promote a SExt used in such complex GEP as we
      // expect some computation to be merged if they are done on 64 bits.
      if (GEPInst->getNumOperands() > 2) {
        AllowPromotionWithoutCommonHeader = true;
        break;
      }
    }
  }
  return Considerable;
}

bool RISCVTTIImpl::canSplatOperand(unsigned Opcode, int Operand) const {
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::ICmp:
  case Instruction::FCmp:
    return true;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Select:
    return Operand == 1;
  default:
    return false;
  }
}

bool RISCVTTIImpl::canSplatOperand(Instruction *I, int Operand) const {
  if (!I->getType()->isVectorTy() || !ST->hasVInstructions())
    return false;

  if (canSplatOperand(I->getOpcode(), Operand))
    return true;

  auto *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;

  switch (II->getIntrinsicID()) {
  case Intrinsic::fma:
  case Intrinsic::vp_fma:
  case Intrinsic::fmuladd:
  case Intrinsic::vp_fmuladd:
    return Operand == 0 || Operand == 1;
  case Intrinsic::vp_shl:
  case Intrinsic::vp_lshr:
  case Intrinsic::vp_ashr:
  case Intrinsic::vp_udiv:
  case Intrinsic::vp_sdiv:
  case Intrinsic::vp_urem:
  case Intrinsic::vp_srem:
  case Intrinsic::ssub_sat:
  case Intrinsic::vp_ssub_sat:
  case Intrinsic::usub_sat:
  case Intrinsic::vp_usub_sat:
  case Intrinsic::vp_select:
    return Operand == 1;
    // These intrinsics are commutative.
  case Intrinsic::vp_add:
  case Intrinsic::vp_mul:
  case Intrinsic::vp_and:
  case Intrinsic::vp_or:
  case Intrinsic::vp_xor:
  case Intrinsic::vp_fadd:
  case Intrinsic::vp_fmul:
  case Intrinsic::vp_icmp:
  case Intrinsic::vp_fcmp:
  case Intrinsic::smin:
  case Intrinsic::vp_smin:
  case Intrinsic::umin:
  case Intrinsic::vp_umin:
  case Intrinsic::smax:
  case Intrinsic::vp_smax:
  case Intrinsic::umax:
  case Intrinsic::vp_umax:
  case Intrinsic::sadd_sat:
  case Intrinsic::vp_sadd_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::vp_uadd_sat:
    // These intrinsics have 'vr' versions.
  case Intrinsic::vp_sub:
  case Intrinsic::vp_fsub:
  case Intrinsic::vp_fdiv:
    return Operand == 0 || Operand == 1;
  default:
    return false;
  }
}

/// Check if sinking \p I's operands to I's basic block is profitable, because
/// the operands can be folded into a target instruction, e.g.
/// splats of scalars can fold into vector instructions.
bool RISCVTTIImpl::isProfitableToSinkOperands(
    Instruction *I, SmallVectorImpl<Use *> &Ops) const {
  using namespace llvm::PatternMatch;

  if (I->isBitwiseLogicOp()) {
    if (!I->getType()->isVectorTy()) {
      if (ST->hasStdExtZbb() || ST->hasStdExtZbkb()) {
        for (auto &Op : I->operands()) {
          // (and/or/xor X, (not Y)) -> (andn/orn/xnor X, Y)
          if (match(Op.get(), m_Not(m_Value()))) {
            Ops.push_back(&Op);
            return true;
          }
        }
      }
    } else if (I->getOpcode() == Instruction::And && ST->hasStdExtZvkb()) {
      for (auto &Op : I->operands()) {
        // (and X, (not Y)) -> (vandn.vv X, Y)
        if (match(Op.get(), m_Not(m_Value()))) {
          Ops.push_back(&Op);
          return true;
        }
        // (and X, (splat (not Y))) -> (vandn.vx X, Y)
        if (match(Op.get(), m_Shuffle(m_InsertElt(m_Value(), m_Not(m_Value()),
                                                  m_ZeroInt()),
                                      m_Value(), m_ZeroMask()))) {
          Use &InsertElt = cast<Instruction>(Op)->getOperandUse(0);
          Use &Not = cast<Instruction>(InsertElt)->getOperandUse(1);
          Ops.push_back(&Not);
          Ops.push_back(&InsertElt);
          Ops.push_back(&Op);
          return true;
        }
      }
    }
  }

  if (!I->getType()->isVectorTy() || !ST->hasVInstructions())
    return false;

  // Don't sink splat operands if the target prefers it. Some targets requires
  // S2V transfer buffers and we can run out of them copying the same value
  // repeatedly.
  // FIXME: It could still be worth doing if it would improve vector register
  // pressure and prevent a vector spill.
  if (!ST->sinkSplatOperands())
    return false;

  for (auto OpIdx : enumerate(I->operands())) {
    if (!canSplatOperand(I, OpIdx.index()))
      continue;

    Instruction *Op = dyn_cast<Instruction>(OpIdx.value().get());
    // Make sure we are not already sinking this operand
    if (!Op || any_of(Ops, [&](Use *U) { return U->get() == Op; }))
      continue;

    // We are looking for a splat/vp.splat that can be sunk.
    bool IsVPSplat = match(Op, m_Intrinsic<Intrinsic::experimental_vp_splat>(
                                   m_Value(), m_Value(), m_Value()));
    if (!IsVPSplat &&
        !match(Op, m_Shuffle(m_InsertElt(m_Undef(), m_Value(), m_ZeroInt()),
                             m_Undef(), m_ZeroMask())))
      continue;

    // Don't sink i1 splats.
    if (cast<VectorType>(Op->getType())->getElementType()->isIntegerTy(1))
      continue;

    // All uses of the shuffle should be sunk to avoid duplicating it across gpr
    // and vector registers
    for (Use &U : Op->uses()) {
      Instruction *Insn = cast<Instruction>(U.getUser());
      if (!canSplatOperand(Insn, U.getOperandNo()))
        return false;
    }

    // Sink any fpexts since they might be used in a widening fp pattern.
    if (IsVPSplat) {
      if (isa<FPExtInst>(Op->getOperand(0)))
        Ops.push_back(&Op->getOperandUse(0));
    } else {
      Use *InsertEltUse = &Op->getOperandUse(0);
      auto *InsertElt = cast<InsertElementInst>(InsertEltUse);
      if (isa<FPExtInst>(InsertElt->getOperand(1)))
        Ops.push_back(&InsertElt->getOperandUse(1));
      Ops.push_back(InsertEltUse);
    }
    Ops.push_back(&OpIdx.value());
  }
  return true;
}

RISCVTTIImpl::TTI::MemCmpExpansionOptions
RISCVTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;
  // TODO: Enable expansion when unaligned access is not supported after we fix
  // issues in ExpandMemcmp.
  if (!ST->enableUnalignedScalarMem())
    return Options;

  if (!ST->hasStdExtZbb() && !ST->hasStdExtZbkb() && !IsZeroCmp)
    return Options;

  Options.AllowOverlappingLoads = true;
  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = Options.MaxNumLoads;
  if (ST->is64Bit()) {
    Options.LoadSizes = {8, 4, 2, 1};
    Options.AllowedTailExpansions = {3, 5, 6};
  } else {
    Options.LoadSizes = {4, 2, 1};
    Options.AllowedTailExpansions = {3};
  }
  return Options;
}
