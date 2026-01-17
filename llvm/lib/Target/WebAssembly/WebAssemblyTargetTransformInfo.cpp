//===-- WebAssemblyTargetTransformInfo.cpp - WebAssembly-specific TTI -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the WebAssembly-specific TargetTransformInfo
/// implementation.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

#include "llvm/CodeGen/CostTable.h"
using namespace llvm;

#define DEBUG_TYPE "wasmtti"

TargetTransformInfo::PopcntSupportKind
WebAssemblyTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return TargetTransformInfo::PSK_FastHardware;
}

unsigned WebAssemblyTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  unsigned Result = BaseT::getNumberOfRegisters(ClassID);

  // For SIMD, use at least 16 registers, as a rough guess.
  bool Vector = (ClassID == 1);
  if (Vector)
    Result = std::max(Result, 16u);

  return Result;
}

TypeSize WebAssemblyTTIImpl::getRegisterBitWidth(
    TargetTransformInfo::RegisterKind K) const {
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(64);
  case TargetTransformInfo::RGK_FixedWidthVector:
    return TypeSize::getFixed(getST()->hasSIMD128() ? 128 : 64);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(0);
  }

  llvm_unreachable("Unsupported register kind");
}

InstructionCost WebAssemblyTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args, const Instruction *CxtI) const {

  if (ST->hasSIMD128()) {
    static const CostTblEntry ArithCostTbl[]{
        // extmul + (maybe awkward) shuffle
        {ISD::MUL, MVT::v8i8, 4},
        // 2x extmul + (okay) shuffle
        {ISD::MUL, MVT::v16i8, 4},
        // extmul
        {ISD::MUL, MVT::v4i16, 1},
        // extmul
        {ISD::MUL, MVT::v2i32, 1},
    };
    EVT DstVT = TLI->getValueType(DL, Ty);
    if (DstVT.isSimple()) {
      int ISD = TLI->InstructionOpcodeToISD(Opcode);
      if (const auto *Entry =
              CostTableLookup(ArithCostTbl, ISD, DstVT.getSimpleVT()))
        return Entry->Cost;
    }
  }

  InstructionCost Cost =
      BasicTTIImplBase<WebAssemblyTTIImpl>::getArithmeticInstrCost(
          Opcode, Ty, CostKind, Op1Info, Op2Info);

  if (auto *VTy = dyn_cast<VectorType>(Ty)) {
    switch (Opcode) {
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Shl:
      // SIMD128's shifts currently only accept a scalar shift count. For each
      // element, we'll need to extract, op, insert. The following is a rough
      // approximation.
      if (!Op2Info.isUniform())
        Cost =
            cast<FixedVectorType>(VTy)->getNumElements() *
            (TargetTransformInfo::TCC_Basic +
             getArithmeticInstrCost(Opcode, VTy->getElementType(), CostKind) +
             TargetTransformInfo::TCC_Basic);
      break;
    }
  }
  return Cost;
}

InstructionCost WebAssemblyTTIImpl::getCastInstrCost(
    unsigned Opcode, Type *Dst, Type *Src, TTI::CastContextHint CCH,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  auto SrcTy = TLI->getValueType(DL, Src);
  auto DstTy = TLI->getValueType(DL, Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple()) {
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  }

  if (!ST->hasSIMD128()) {
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  }

  auto DstVT = DstTy.getSimpleVT();
  auto SrcVT = SrcTy.getSimpleVT();

  if (I && I->hasOneUser()) {
    auto *SingleUser = cast<Instruction>(*I->user_begin());
    int UserISD = TLI->InstructionOpcodeToISD(SingleUser->getOpcode());

    // extmul_low support
    if (UserISD == ISD::MUL &&
        (ISD == ISD::ZERO_EXTEND || ISD == ISD::SIGN_EXTEND)) {
      // Free low extensions.
      if ((SrcVT == MVT::v8i8 && DstVT == MVT::v8i16) ||
          (SrcVT == MVT::v4i16 && DstVT == MVT::v4i32) ||
          (SrcVT == MVT::v2i32 && DstVT == MVT::v2i64)) {
        return 0;
      }
      // Will require an additional extlow operation for the intermediate
      // i16/i32 value.
      if ((SrcVT == MVT::v4i8 && DstVT == MVT::v4i32) ||
          (SrcVT == MVT::v2i16 && DstVT == MVT::v2i64)) {
        return 1;
      }
    }
  }

  static constexpr TypeConversionCostTblEntry ConversionTbl[] = {
      // extend_low
      {ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i32, 1},
      {ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i32, 1},
      {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 1},
      {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 1},
      {ISD::SIGN_EXTEND, MVT::v8i16, MVT::v8i8, 1},
      {ISD::ZERO_EXTEND, MVT::v8i16, MVT::v8i8, 1},
      // 2 x extend_low
      {ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i16, 2},
      {ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i16, 2},
      {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i8, 2},
      {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i8, 2},
      // extend_low, extend_high
      {ISD::SIGN_EXTEND, MVT::v4i64, MVT::v4i32, 2},
      {ISD::ZERO_EXTEND, MVT::v4i64, MVT::v4i32, 2},
      {ISD::SIGN_EXTEND, MVT::v8i32, MVT::v8i16, 2},
      {ISD::ZERO_EXTEND, MVT::v8i32, MVT::v8i16, 2},
      {ISD::SIGN_EXTEND, MVT::v16i16, MVT::v16i8, 2},
      {ISD::ZERO_EXTEND, MVT::v16i16, MVT::v16i8, 2},
      // 2x extend_low, extend_high
      {ISD::SIGN_EXTEND, MVT::v8i64, MVT::v8i32, 4},
      {ISD::ZERO_EXTEND, MVT::v8i64, MVT::v8i32, 4},
      {ISD::SIGN_EXTEND, MVT::v16i32, MVT::v16i16, 4},
      {ISD::ZERO_EXTEND, MVT::v16i32, MVT::v16i16, 4},
      // shuffle
      {ISD::TRUNCATE, MVT::v2i16, MVT::v2i32, 2},
      {ISD::TRUNCATE, MVT::v2i8, MVT::v2i32, 4},
      {ISD::TRUNCATE, MVT::v4i16, MVT::v4i32, 2},
      {ISD::TRUNCATE, MVT::v4i8, MVT::v4i32, 4},
      // narrow, and
      {ISD::TRUNCATE, MVT::v8i16, MVT::v8i32, 2},
      {ISD::TRUNCATE, MVT::v8i8, MVT::v8i16, 2},
      // narrow, 2x and
      {ISD::TRUNCATE, MVT::v16i8, MVT::v16i16, 3},
      // 3x narrow, 4x and
      {ISD::TRUNCATE, MVT::v8i16, MVT::v8i64, 7},
      {ISD::TRUNCATE, MVT::v16i8, MVT::v16i32, 7},
      // 7x narrow, 8x and
      {ISD::TRUNCATE, MVT::v16i8, MVT::v16i64, 15},
      // convert_i32x4
      {ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i32, 1},
      {ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i32, 1},
      {ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i32, 1},
      {ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i32, 1},
      // extend_low, convert
      {ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i16, 2},
      {ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i16, 2},
      {ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i16, 2},
      {ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i16, 2},
      // extend_low x 2, convert
      {ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i8, 3},
      {ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i8, 3},
      {ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i8, 3},
      {ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i8, 3},
      // several shuffles
      {ISD::SINT_TO_FP, MVT::v8f32, MVT::v8i8, 10},
      {ISD::UINT_TO_FP, MVT::v8f32, MVT::v8i8, 10},
      {ISD::SINT_TO_FP, MVT::v8f32, MVT::v8i16, 10},
      {ISD::UINT_TO_FP, MVT::v8f32, MVT::v8i8, 10},
      /// trunc_sat, const, and, 3x narrow
      {ISD::FP_TO_SINT, MVT::v2i8, MVT::v2f32, 6},
      {ISD::FP_TO_UINT, MVT::v2i8, MVT::v2f32, 6},
      {ISD::FP_TO_SINT, MVT::v4i8, MVT::v4f32, 6},
      {ISD::FP_TO_UINT, MVT::v4i8, MVT::v4f32, 6},
      /// trunc_sat, const, and, narrow
      {ISD::FP_TO_UINT, MVT::v2i16, MVT::v2f32, 4},
      {ISD::FP_TO_SINT, MVT::v2i16, MVT::v2f32, 4},
      {ISD::FP_TO_SINT, MVT::v4i16, MVT::v4f32, 4},
      {ISD::FP_TO_UINT, MVT::v4i16, MVT::v4f32, 4},
      // 2x trunc_sat, const, 2x and, 3x narrow
      {ISD::FP_TO_SINT, MVT::v8i8, MVT::v8f32, 8},
      {ISD::FP_TO_UINT, MVT::v8i8, MVT::v8f32, 8},
      // 2x trunc_sat, const, 2x and, narrow
      {ISD::FP_TO_SINT, MVT::v8i16, MVT::v8f32, 6},
      {ISD::FP_TO_UINT, MVT::v8i16, MVT::v8f32, 6},
  };

  if (const auto *Entry =
          ConvertCostTableLookup(ConversionTbl, ISD, DstVT, SrcVT)) {
    return Entry->Cost;
  }

  return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
}

WebAssemblyTTIImpl::TTI::MemCmpExpansionOptions
WebAssemblyTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;

  Options.AllowOverlappingLoads = true;

  if (ST->hasSIMD128())
    Options.LoadSizes.push_back(16);

  Options.LoadSizes.append({8, 4, 2, 1});
  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = Options.MaxNumLoads;

  return Options;
}

InstructionCost WebAssemblyTTIImpl::getMemoryOpCost(
    unsigned Opcode, Type *Ty, Align Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind, TTI::OperandValueInfo OpInfo,
    const Instruction *I) const {
  if (!ST->hasSIMD128() || !isa<FixedVectorType>(Ty)) {
    return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace,
                                  CostKind);
  }

  EVT VT = TLI->getValueType(DL, Ty, true);
  // Type legalization can't handle structs
  if (VT == MVT::Other)
    return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace,
                                  CostKind);

  auto LT = getTypeLegalizationCost(Ty);
  if (!LT.first.isValid())
    return InstructionCost::getInvalid();

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  unsigned width = VT.getSizeInBits();
  if (ISD == ISD::LOAD) {
    // 128-bit loads are a single instruction. 32-bit and 64-bit vector loads
    // can be lowered to load32_zero and load64_zero respectively. Assume SIMD
    // loads are twice as expensive as scalar.
    switch (width) {
    default:
      break;
    case 32:
    case 64:
    case 128:
      return 2;
    }
  } else if (ISD == ISD::STORE) {
    // For stores, we can use store lane operations.
    switch (width) {
    default:
      break;
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      return 2;
    }
  }

  return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace, CostKind);
}

InstructionCost WebAssemblyTTIImpl::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *Ty, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) const {
  assert(Factor >= 2 && "Invalid interleave factor");

  auto *VecTy = cast<VectorType>(Ty);
  if (!ST->hasSIMD128() || !isa<FixedVectorType>(VecTy)) {
    return InstructionCost::getInvalid();
  }

  if (UseMaskForCond || UseMaskForGaps)
    return BaseT::getInterleavedMemoryOpCost(Opcode, Ty, Factor, Indices,
                                             Alignment, AddressSpace, CostKind,
                                             UseMaskForCond, UseMaskForGaps);

  constexpr unsigned MaxInterleaveFactor = 4;
  if (Factor <= MaxInterleaveFactor) {
    unsigned MinElts = VecTy->getElementCount().getKnownMinValue();
    // Ensure the number of vector elements is greater than 1.
    if (MinElts < 2 || MinElts % Factor != 0)
      return InstructionCost::getInvalid();

    unsigned ElSize = DL.getTypeSizeInBits(VecTy->getElementType());
    // Ensure the element type is legal.
    if (ElSize != 8 && ElSize != 16 && ElSize != 32 && ElSize != 64)
      return InstructionCost::getInvalid();

    if (Factor != 2 && Factor != 4)
      return InstructionCost::getInvalid();

    auto *SubVecTy =
        VectorType::get(VecTy->getElementType(),
                        VecTy->getElementCount().divideCoefficientBy(Factor));
    InstructionCost MemCost =
        getMemoryOpCost(Opcode, SubVecTy, Alignment, AddressSpace, CostKind);

    unsigned VecSize = DL.getTypeSizeInBits(SubVecTy);
    unsigned MaxVecSize = 128;
    unsigned NumAccesses =
        std::max<unsigned>(1, (MinElts * ElSize + MaxVecSize - 1) / VecSize);

    // A stride of two is commonly supported via dedicated instructions, so it
    // should be relatively cheap for all element sizes. A stride of four is
    // more expensive as it will likely require more shuffles. Using two
    // simd128 inputs is considered more expensive and we mainly account for
    // shuffling two inputs (32 bytes), but we do model 4 x v4i32 to enable
    // arithmetic kernels.
    static const CostTblEntry ShuffleCostTbl[] = {
        // One reg.
        {2, MVT::v2i8, 1},  // interleave 2 x 2i8 into 4i8
        {2, MVT::v4i8, 1},  // interleave 2 x 4i8 into 8i8
        {2, MVT::v8i8, 1},  // interleave 2 x 8i8 into 16i8
        {2, MVT::v2i16, 1}, // interleave 2 x 2i16 into 4i16
        {2, MVT::v4i16, 1}, // interleave 2 x 4i16 into 8i16
        {2, MVT::v2i32, 1}, // interleave 2 x 2i32 into 4i32

        // Two regs.
        {2, MVT::v16i8, 2}, // interleave 2 x 16i8 into 32i8
        {2, MVT::v8i16, 2}, // interleave 2 x 8i16 into 16i16
        {2, MVT::v4i32, 2}, // interleave 2 x 4i32 into 8i32

        // One reg.
        {4, MVT::v2i8, 4},  // interleave 4 x 2i8 into 8i8
        {4, MVT::v4i8, 4},  // interleave 4 x 4i8 into 16i8
        {4, MVT::v2i16, 4}, // interleave 4 x 2i16 into 8i16

        // Two regs.
        {4, MVT::v8i8, 16}, // interleave 4 x 8i8 into 32i8
        {4, MVT::v4i16, 8}, // interleave 4 x 4i16 into 16i16
        {4, MVT::v2i32, 4}, // interleave 4 x 2i32 into 8i32

        // Four regs.
        {4, MVT::v4i32, 16}, // interleave 4 x 4i32 into 16i32
    };

    EVT ETy = TLI->getValueType(DL, SubVecTy);
    if (const auto *Entry =
            CostTableLookup(ShuffleCostTbl, Factor, ETy.getSimpleVT()))
      return Entry->Cost + (NumAccesses * MemCost);
  }

  return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace, CostKind,
                                           UseMaskForCond, UseMaskForGaps);
}

InstructionCost WebAssemblyTTIImpl::getVectorInstrCost(
    unsigned Opcode, Type *Val, TTI::TargetCostKind CostKind, unsigned Index,
    const Value *Op0, const Value *Op1) const {
  InstructionCost Cost = BasicTTIImplBase::getVectorInstrCost(
      Opcode, Val, CostKind, Index, Op0, Op1);

  // SIMD128's insert/extract currently only take constant indices.
  if (Index == -1u)
    return Cost + 25 * TargetTransformInfo::TCC_Expensive;

  return Cost;
}

InstructionCost WebAssemblyTTIImpl::getPartialReductionCost(
    unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
    ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
    TTI::PartialReductionExtendKind OpBExtend, std::optional<unsigned> BinOp,
    TTI::TargetCostKind CostKind) const {
  InstructionCost Invalid = InstructionCost::getInvalid();
  if (!VF.isFixed() || !ST->hasSIMD128())
    return Invalid;

  if (CostKind != TTI::TCK_RecipThroughput)
    return Invalid;

  if (Opcode != Instruction::Add)
    return Invalid;

  EVT AccumEVT = EVT::getEVT(AccumType);
  // TODO: Add i64 accumulator.
  if (AccumEVT != MVT::i32)
    return Invalid;

  // Possible options:
  // - i16x8.extadd_pairwise_i8x16_sx
  // - i32x4.extadd_pairwise_i16x8_sx
  // - i32x4.dot_i16x8_s
  // Only try to support dot, for now.

  EVT InputEVT = EVT::getEVT(InputTypeA);
  if (!((InputEVT == MVT::i16 && VF.getFixedValue() == 8) ||
        (InputEVT == MVT::i8 && VF.getFixedValue() == 16))) {
    return Invalid;
  }

  if (OpAExtend == TTI::PR_None)
    return Invalid;

  InstructionCost Cost(TTI::TCC_Basic);
  if (!BinOp)
    return Cost;

  if (OpAExtend != OpBExtend)
    return Invalid;

  if (*BinOp != Instruction::Mul)
    return Invalid;

  if (InputTypeA != InputTypeB)
    return Invalid;

  // Signed inputs can lower to dot
  if (InputEVT == MVT::i16 && VF.getFixedValue() == 8)
    return OpAExtend == TTI::PR_SignExtend ? Cost : Cost * 2;

  // Double the size of the lowered sequence.
  if (InputEVT == MVT::i8 && VF.getFixedValue() == 16)
    return OpAExtend == TTI::PR_SignExtend ? Cost * 2 : Cost * 4;

  return Invalid;
}

InstructionCost
WebAssemblyTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                          TTI::TargetCostKind CostKind) const {
  switch (ICA.getID()) {
  case Intrinsic::experimental_vector_extract_last_active:
    // TODO: Remove once the intrinsic can be lowered without crashes.
    return InstructionCost::getInvalid();
  default:
    break;
  }
  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

TTI::ReductionShuffle WebAssemblyTTIImpl::getPreferredExpandedReductionShuffle(
    const IntrinsicInst *II) const {

  switch (II->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::vector_reduce_fadd:
    return TTI::ReductionShuffle::Pairwise;
  }
  return TTI::ReductionShuffle::SplitHalf;
}

void WebAssemblyTTIImpl::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, TTI::UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  // Scan the loop: don't unroll loops with calls. This is a standard approach
  // for most (all?) targets.
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      if (isa<CallInst>(I) || isa<InvokeInst>(I))
        if (const Function *F = cast<CallBase>(I).getCalledFunction())
          if (isLoweredToCall(F))
            return;

  // The chosen threshold is within the range of 'LoopMicroOpBufferSize' of
  // the various microarchitectures that use the BasicTTI implementation and
  // has been selected through heuristics across multiple cores and runtimes.
  UP.Partial = UP.Runtime = UP.UpperBound = true;
  UP.PartialThreshold = 30;

  // Avoid unrolling when optimizing for size.
  UP.OptSizeThreshold = 0;
  UP.PartialOptSizeThreshold = 0;

  // Set number of instructions optimized when "back edge"
  // becomes "fall through" to default value of 2.
  UP.BEInsns = 2;
}

bool WebAssemblyTTIImpl::supportsTailCalls() const {
  return getST()->hasTailCall();
}

bool WebAssemblyTTIImpl::isProfitableToSinkOperands(
    Instruction *I, SmallVectorImpl<Use *> &Ops) const {
  using namespace llvm::PatternMatch;

  if (!I->getType()->isVectorTy() || !I->isShift())
    return false;

  Value *V = I->getOperand(1);
  // We dont need to sink constant splat.
  if (isa<Constant>(V))
    return false;

  if (match(V, m_Shuffle(m_InsertElt(m_Value(), m_Value(), m_ZeroInt()),
                         m_Value(), m_ZeroMask()))) {
    // Sink insert
    Ops.push_back(&cast<Instruction>(V)->getOperandUse(0));
    // Sink shuffle
    Ops.push_back(&I->getOperandUse(1));
    return true;
  }

  return false;
}

/// Attempt to convert [relaxed_]swizzle to shufflevector if the mask is
/// constant.
static Value *simplifyWasmSwizzle(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder,
                                  bool IsRelaxed) {
  auto *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<FixedVectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  assert(NumElts == 16);

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[16];
  bool AnyOutOfBounds = false;

  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    if (IsRelaxed && cast<ConstantInt>(COp)->getSExtValue() >= NumElts) {
      // The relaxed_swizzle operation always returns 0 if the lane index is
      // less than 0 when interpreted as a signed value. For lane indices above
      // 15, however, it can choose between returning 0 or the lane at `Index %
      // 16`. However, the choice must be made consistently. As the WebAssembly
      // spec states:
      //
      // "The result of relaxed operators are implementation-dependent, because
      // the set of possible results may depend on properties of the host
      // environment, such as its hardware. Technically, their behaviour is
      // controlled by a set of global parameters to the semantics that an
      // implementation can instantiate in different ways. These choices are
      // fixed, that is, parameters are constant during the execution of any
      // given program."
      //
      // The WebAssembly runtime may choose differently from us, so we can't
      // optimize a relaxed swizzle with lane indices above 15.
      return nullptr;
    }

    uint64_t Index = cast<ConstantInt>(COp)->getZExtValue();
    if (Index >= NumElts) {
      AnyOutOfBounds = true;
      // If there are out-of-bounds indices, the swizzle instruction returns
      // zeroes in those lanes. We'll provide an all-zeroes vector as the
      // second argument to shufflevector and read the first element from it.
      Indexes[I] = NumElts;
      continue;
    }

    Indexes[I] = Index;
  }

  auto *V1 = II.getArgOperand(0);
  auto *V2 =
      AnyOutOfBounds ? Constant::getNullValue(VecTy) : PoisonValue::get(VecTy);

  return Builder.CreateShuffleVector(V1, V2, ArrayRef(Indexes, NumElts));
}

std::optional<Instruction *>
WebAssemblyTTIImpl::instCombineIntrinsic(InstCombiner &IC,
                                         IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::wasm_swizzle:
  case Intrinsic::wasm_relaxed_swizzle:
    if (Value *V = simplifyWasmSwizzle(
            II, IC.Builder, IID == Intrinsic::wasm_relaxed_swizzle)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;
  }

  return std::nullopt;
}
