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
    ArrayRef<const Value *> Args, const Instruction *CxtI) {

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
    TTI::TargetCostKind CostKind, const Instruction *I) {
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

  // extend_low
  static constexpr TypeConversionCostTblEntry ConversionTbl[] = {
      {ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i32, 1},
      {ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i32, 1},
      {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i16, 1},
      {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i16, 1},
      {ISD::SIGN_EXTEND, MVT::v8i16, MVT::v8i8, 1},
      {ISD::ZERO_EXTEND, MVT::v8i16, MVT::v8i8, 1},
      {ISD::SIGN_EXTEND, MVT::v2i64, MVT::v2i16, 2},
      {ISD::ZERO_EXTEND, MVT::v2i64, MVT::v2i16, 2},
      {ISD::SIGN_EXTEND, MVT::v4i32, MVT::v4i8, 2},
      {ISD::ZERO_EXTEND, MVT::v4i32, MVT::v4i8, 2},
  };

  if (const auto *Entry =
          ConvertCostTableLookup(ConversionTbl, ISD, DstVT, SrcVT)) {
    return Entry->Cost;
  }

  return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
}

InstructionCost WebAssemblyTTIImpl::getMemoryOpCost(
    unsigned Opcode, Type *Ty, MaybeAlign Alignment, unsigned AddressSpace,
    TTI::TargetCostKind CostKind, TTI::OperandValueInfo OpInfo,
    const Instruction *I) {
  if (!ST->hasSIMD128() || !isa<FixedVectorType>(Ty)) {
    return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace,
                                  CostKind);
  }

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  if (ISD != ISD::LOAD) {
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

  // 128-bit loads are a single instruction. 32-bit and 64-bit vector loads can
  // be lowered to load32_zero and load64_zero respectively. Assume SIMD loads
  // are twice as expensive as scalar.
  unsigned width = VT.getSizeInBits();
  switch (width) {
  default:
    break;
  case 32:
  case 64:
  case 128:
    return 2;
  }

  return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace, CostKind);
}

InstructionCost
WebAssemblyTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                       TTI::TargetCostKind CostKind,
                                       unsigned Index, Value *Op0, Value *Op1) {
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
    TTI::PartialReductionExtendKind OpBExtend,
    std::optional<unsigned> BinOp) const {
  InstructionCost Invalid = InstructionCost::getInvalid();
  if (!VF.isFixed() || !ST->hasSIMD128())
    return Invalid;

  InstructionCost Cost(TTI::TCC_Basic);

  // Possible options:
  // - i16x8.extadd_pairwise_i8x16_sx
  // - i32x4.extadd_pairwise_i16x8_sx
  // - i32x4.dot_i16x8_s
  // Only try to support dot, for now.

  if (Opcode != Instruction::Add)
    return Invalid;

  if (!BinOp || *BinOp != Instruction::Mul)
    return Invalid;

  if (InputTypeA != InputTypeB)
    return Invalid;

  if (OpAExtend != OpBExtend)
    return Invalid;

  EVT InputEVT = EVT::getEVT(InputTypeA);
  EVT AccumEVT = EVT::getEVT(AccumType);

  // TODO: Add i64 accumulator.
  if (AccumEVT != MVT::i32)
    return Invalid;

  // Signed inputs can lower to dot
  if (InputEVT == MVT::i16 && VF.getFixedValue() == 8)
    return OpAExtend == TTI::PR_SignExtend ? Cost : Cost * 2;

  // Double the size of the lowered sequence.
  if (InputEVT == MVT::i8 && VF.getFixedValue() == 16)
    return OpAExtend == TTI::PR_SignExtend ? Cost * 2 : Cost * 4;

  return Invalid;
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
  if (dyn_cast<Constant>(V))
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
