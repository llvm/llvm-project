//===-- LanaiTargetTransformInfo.h - Lanai specific TTI ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file a TargetTransformInfo::Concept conforming object specific to the
// Lanai target machine. It uses the target's detailed information to
// provide more precise answers to certain TTI queries, while letting the
// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAITARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_LANAI_LANAITARGETTRANSFORMINFO_H

#include "Lanai.h"
#include "LanaiSubtarget.h"
#include "LanaiTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
class LanaiTTIImpl : public BasicTTIImplBase<LanaiTTIImpl> {
  typedef BasicTTIImplBase<LanaiTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const LanaiSubtarget *ST;
  const LanaiTargetLowering *TLI;

  const LanaiSubtarget *getST() const { return ST; }
  const LanaiTargetLowering *getTLI() const { return TLI; }

public:
  explicit LanaiTTIImpl(const LanaiTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  LanaiTTIImpl(const LanaiTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST), TLI(Arg.TLI) {}
  LanaiTTIImpl(LanaiTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(Arg.ST), TLI(Arg.TLI) {}

  bool shouldBuildLookupTables() const { return false; }

  TargetTransformInfo::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    if (TyWidth == 32)
      return TTI::PSK_FastHardware;
    return TTI::PSK_Software;
  }

  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None) {
    int ISD = TLI->InstructionOpcodeToISD(Opcode);

    switch (ISD) {
    default:
      return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                           Opd1PropInfo, Opd2PropInfo);
    case ISD::MUL:
    case ISD::SDIV:
    case ISD::UDIV:
    case ISD::UREM:
      // This increases the cost associated with multiplication and division
      // to 64 times what the baseline arithmetic cost is. The arithmetic
      // instruction cost was arbitrarily chosen to reduce the desirability
      // of emitting arithmetic instructions that are emulated in software.
      // TODO: Investigate the performance impact given specialized lowerings.
      return 64 * BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                                Opd1PropInfo, Opd2PropInfo);
    }
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAITARGETTRANSFORMINFO_H
