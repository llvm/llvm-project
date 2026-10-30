//===- AIRTargetTransformInfo.h - AIR TTI -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H

#include "AIRSubtarget.h"
#include "AIRTargetMachine.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
class AIRTTIImpl final : public BasicTTIImplBase<AIRTTIImpl> {
  using BaseT = BasicTTIImplBase<AIRTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const AIRSubtarget *ST;
  const AIRTargetLowering *TLI;

  const AIRSubtarget *getST() const { return ST; }
  const AIRTargetLowering *getTLI() const { return TLI; }

public:
  explicit AIRTTIImpl(const AIRTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  // What every GPU that runs AIR shares, for the optimizations of portable
  // AIR: threads of a SIMD-group that branch apart, values that are the
  // same in all of them (the uniform system values, SIMD-group reductions
  // and votes), scalar registers (vectors are taken apart: the SLP and loop
  // vectorizers only cost compile time), and device and threadgroup accesses
  // of up to 16 bytes as one, which the load/store vectorizer makes of
  // neighbouring ones.
  bool hasBranchDivergence(const Function *F = nullptr) const override {
    return true;
  }
  ValueUniformity getValueUniformity(const Value *V) const override {
    if (const auto *CI = dyn_cast<CallInst>(V)) {
      const Function *F = CI->getCalledFunction();
      StringRef N = F ? F->getName() : "";
      if (N.starts_with("air.threadgroup_position_in_grid") ||
          N.starts_with("air.threads_per_threadgroup") ||
          N.starts_with("air.threadgroups_per_grid") ||
          N.starts_with("air.simd_ballot") || N.starts_with("air.simd_vote") ||
          N.starts_with("air.simd_sum") || N.starts_with("air.simd_product") ||
          N.starts_with("air.simd_min") || N.starts_with("air.simd_max") ||
          N.starts_with("air.simd_and") || N.starts_with("air.simd_or") ||
          N.starts_with("air.simd_xor") || N.starts_with("air.simd_broadcast"))
        return ValueUniformity::AlwaysUniform;
      // intrinsics are as the default says; other calls (thread positions,
      // shuffles, library functions) differ per thread
      if (F && F->isIntrinsic())
        return ValueUniformity::Default;
      return ValueUniformity::NeverUniform;
    }
    // thread memory is per thread, and atomics return per thread values
    if (const auto *LI = dyn_cast<LoadInst>(V))
      return LI->getPointerAddressSpace() == 0 ? ValueUniformity::NeverUniform
                                               : ValueUniformity::Default;
    if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V))
      return ValueUniformity::NeverUniform;
    return ValueUniformity::Default;
  }
  unsigned getNumberOfRegisters(unsigned ClassID) const override {
    return ClassID == 1 ? 0 : 128;
  }
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override {
    return TypeSize::getFixed(32);
  }
  // Portable AIR leaves inlining big functions to its consumers (the AGX
  // backend inlines everything, the Metal pipeline keeps large shared
  // functions as calls): the default threshold here.
  unsigned getLoadStoreVecRegBitWidth(unsigned AS) const override {
    return 128;
  }
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AS) const override {
    return vectorizableChain(ChainSizeInBytes, Alignment, AS);
  }
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AS) const override {
    return vectorizableChain(ChainSizeInBytes, Alignment, AS);
  }
  // Scalar casts between integers and pointers of any width are a move or
  // nothing (registers are 32 bits, addresses 64): not the type legalization
  // AIR's lowering, which has no registers, would cost them.
  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   TTI::CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I = nullptr) const override {
    if (!Dst->isVectorTy() && !Src->isVectorTy()) {
      switch (Opcode) {
      case Instruction::Trunc:
      case Instruction::IntToPtr:
      case Instruction::PtrToInt:
      case Instruction::BitCast:
      case Instruction::AddrSpaceCast:
        return 0;
      case Instruction::ZExt:
        // to a 64-bit register pair: the high half is zero
        if (Dst->isIntegerTy(64) && Src->getScalarSizeInBits() == 32)
          return 0;
        return 1;
      case Instruction::SExt:
        return 1;
      default:
        break;
      }
    }
    return BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  }

  static bool vectorizableChain(unsigned Bytes, Align Alignment, unsigned AS) {
    return Alignment >= Align(4) && Bytes % 4 == 0 && Bytes <= 16 &&
           (AS == 1 || AS == 3);
  }
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRTARGETTRANSFORMINFO_H
