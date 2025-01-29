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
    ArrayRef<const Value *> Args,
    const Instruction *CxtI) {

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
