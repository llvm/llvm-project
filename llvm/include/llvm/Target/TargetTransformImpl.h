//=- llvm/Target/TargetTransformImpl.h - Target Loop Trans Info----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the target-specific implementations of the
// TargetTransform interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGET_TRANSFORMATION_IMPL_H
#define LLVM_TARGET_TARGET_TRANSFORMATION_IMPL_H

#include "llvm/TargetTransformInfo.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

class TargetLowering;

/// ScalarTargetTransformInfo - This is a default implementation for the
/// ScalarTargetTransformInfo interface. Different targets can implement
/// this interface differently.
class ScalarTargetTransformImpl : public ScalarTargetTransformInfo {
private:
  const TargetLowering *TLI;

public:
  /// Ctor
  explicit ScalarTargetTransformImpl(const TargetLowering *TL) : TLI(TL) {}

  virtual bool isLegalAddImmediate(int64_t imm) const;

  virtual bool isLegalICmpImmediate(int64_t imm) const;

  virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;

  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;

  virtual bool isTypeLegal(Type *Ty) const;

  virtual unsigned getJumpBufAlignment() const;

  virtual unsigned getJumpBufSize() const;

  virtual bool shouldBuildLookupTables() const;
};

class VectorTargetTransformImpl : public VectorTargetTransformInfo {
protected:
  const TargetLowering *TLI;

  /// Estimate the cost of type-legalization and the legalized type.
  std::pair<unsigned, MVT> getTypeLegalizationCost(Type *Ty) const;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

  // Get the ISD node that corresponds to the Instruction class opcode.
  int InstructionOpcodeToISD(unsigned Opcode) const;

public:
  explicit VectorTargetTransformImpl(const TargetLowering *TL) : TLI(TL) {}

  virtual ~VectorTargetTransformImpl() {}

  virtual unsigned getInstrCost(unsigned Opcode, Type *Ty1, Type *Ty2) const;

  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const;

  virtual unsigned getBroadcastCost(Type *Tp) const;

  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const;

  virtual unsigned getCFInstrCost(unsigned Opcode) const;

  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const;

  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const;

  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const;

  virtual unsigned getNumberOfParts(Type *Tp) const;
};

} // end llvm namespace

#endif
