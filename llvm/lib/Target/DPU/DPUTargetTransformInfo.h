//===-- DPUTargetTransformInfo.h - DPU specific TTI ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file a TargetTransformInfo::Concept conforming object specific to the
// DPU target machine. It uses the target's detailed information to
// provide more precise answers to certain TTI queries, while letting the
// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_DPU_DPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_DPU_DPUTARGETTRANSFORMINFO_H

#include "DPUTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
class DPUTTIImpl : public BasicTTIImplBase<DPUTTIImpl> {
  typedef BasicTTIImplBase<DPUTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const DPUSubtarget *ST;
  const DPUTargetLowering *TLI;

  const DPUSubtarget *getST() const { return ST; }
  const DPUTargetLowering *getTLI() const { return TLI; }

public:
  explicit DPUTTIImpl(const DPUTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool hasDivRemOp(Type *DataType, bool IsSigned) const { return true; }

};
} // namespace llvm

#endif /* LLVM_LIB_TARGET_DPU_DPUTARGETTRANSFORMINFO_H */
