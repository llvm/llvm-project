//===-- SystemZTargetTransformInfo.h - SystemZ-specific TTI ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETTRANSFORMINFO_H

#include "SystemZTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class SystemZTTIImpl : public BasicTTIImplBase<SystemZTTIImpl> {
  typedef BasicTTIImplBase<SystemZTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const SystemZSubtarget *ST;
  const SystemZTargetLowering *TLI;

  const SystemZSubtarget *getST() const { return ST; }
  const SystemZTargetLowering *getTLI() const { return TLI; }

  unsigned const LIBCALL_COST = 30;

public:
  explicit SystemZTTIImpl(const SystemZTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  int getIntImmCost(const APInt &Imm, Type *Ty);

  int getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty);
  int getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                    Type *Ty);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector);
  unsigned getRegisterBitWidth(bool Vector) const;

  bool prefersVectorizedAddressing() { return false; }
  bool supportsEfficientVectorElementLoadStore() { return true; }
  bool enableInterleavedAccessVectorization() { return true; }

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>());
  int getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index, Type *SubTp);
  unsigned getVectorTruncCost(Type *SrcTy, Type *DstTy);
  unsigned getVectorBitmaskConversionCost(Type *SrcTy, Type *DstTy);
  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       const Instruction *I = nullptr);
  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         const Instruction *I = nullptr);
  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  int getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                      unsigned AddressSpace, const Instruction *I = nullptr);

  int getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                 unsigned Factor,
                                 ArrayRef<unsigned> Indices,
                                 unsigned Alignment,
                                 unsigned AddressSpace);
  /// @}
};

} // end namespace llvm

#endif
