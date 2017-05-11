//===-- AMDGPUTargetTransformInfo.h - AMDGPU specific TTI -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// AMDGPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {
class AMDGPUTargetLowering;

class AMDGPUTTIImpl final : public BasicTTIImplBase<AMDGPUTTIImpl> {
  typedef BasicTTIImplBase<AMDGPUTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const AMDGPUSubtarget *ST;
  const AMDGPUTargetLowering *TLI;
  bool IsGraphicsShader;

  const AMDGPUSubtarget *getST() const { return ST; }
  const AMDGPUTargetLowering *getTLI() const { return TLI; }


  static inline int getFullRateInstrCost() {
    return TargetTransformInfo::TCC_Basic;
  }

  static inline int getHalfRateInstrCost() {
    return 2 * TargetTransformInfo::TCC_Basic;
  }

  // TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe
  // should be 2 or 4.
  static inline int getQuarterRateInstrCost() {
    return 3 * TargetTransformInfo::TCC_Basic;
  }

   // On some parts, normal fp64 operations are half rate, and others
   // quarter. This also applies to some integer operations.
  inline int get64BitInstrCost() const {
    return ST->hasHalfRate64Ops() ?
      getHalfRateInstrCost() : getQuarterRateInstrCost();
  }

public:
  explicit AMDGPUTTIImpl(const AMDGPUTargetMachine *TM, const Function &F)
    : BaseT(TM, F.getParent()->getDataLayout()),
      ST(TM->getSubtargetImpl(F)),
      TLI(ST->getTargetLowering()),
      IsGraphicsShader(AMDGPU::isShader(F.getCallingConv())) {}

  bool hasBranchDivergence() { return true; }

  void getUnrollingPreferences(Loop *L, TTI::UnrollingPreferences &UP);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return TTI::PSK_FastHardware;
  }

  unsigned getNumberOfRegisters(bool Vector);
  unsigned getRegisterBitWidth(bool Vector);
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;

  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes,
                                  unsigned Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                   unsigned Alignment,
                                   unsigned AddrSpace) const;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                    unsigned Alignment,
                                    unsigned AddrSpace) const;

  unsigned getMaxInterleaveFactor(unsigned VF);

  int getArithmeticInstrCost(
    unsigned Opcode, Type *Ty,
    TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
    TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
    TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
    TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
    ArrayRef<const Value *> Args = ArrayRef<const Value *>());

  unsigned getCFInstrCost(unsigned Opcode);

  int getVectorInstrCost(unsigned Opcode, Type *ValTy, unsigned Index);
  bool isSourceOfDivergence(const Value *V) const;

  unsigned getFlatAddressSpace() const {
    // Don't bother running InferAddressSpaces pass on graphics shaders which
    // don't use flat addressing.
    if (IsGraphicsShader)
      return -1;
    return ST->hasFlatAddressSpace() ?
      ST->getAMDGPUAS().FLAT_ADDRESS : ST->getAMDGPUAS().UNKNOWN_ADDRESS_SPACE;
  }

  unsigned getVectorSplitCost() { return 0; }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp);
};

} // end namespace llvm

#endif
