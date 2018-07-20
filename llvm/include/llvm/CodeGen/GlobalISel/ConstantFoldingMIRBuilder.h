//===-- llvm/CodeGen/GlobalISel/ConstantFoldingMIRBuilder.h  --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a version of MachineIRBuilder which does trivial
/// constant folding.
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"

namespace llvm {

static Optional<APInt> ConstantFoldBinOp(unsigned Opcode, const unsigned Op1,
                                         const unsigned Op2,
                                         const MachineRegisterInfo &MRI) {
  auto MaybeOp1Cst = getConstantVRegVal(Op1, MRI);
  auto MaybeOp2Cst = getConstantVRegVal(Op2, MRI);
  if (MaybeOp1Cst && MaybeOp2Cst) {
    LLT Ty = MRI.getType(Op1);
    APInt C1(Ty.getSizeInBits(), *MaybeOp1Cst, true);
    APInt C2(Ty.getSizeInBits(), *MaybeOp2Cst, true);
    switch (Opcode) {
    default:
      break;
    case TargetOpcode::G_ADD:
      return C1 + C2;
    case TargetOpcode::G_AND:
      return C1 & C2;
    case TargetOpcode::G_ASHR:
      return C1.ashr(C2);
    case TargetOpcode::G_LSHR:
      return C1.lshr(C2);
    case TargetOpcode::G_MUL:
      return C1 * C2;
    case TargetOpcode::G_OR:
      return C1 | C2;
    case TargetOpcode::G_SHL:
      return C1 << C2;
    case TargetOpcode::G_SUB:
      return C1 - C2;
    case TargetOpcode::G_XOR:
      return C1 ^ C2;
    case TargetOpcode::G_UDIV:
      if (!C2.getBoolValue())
        break;
      return C1.udiv(C2);
    case TargetOpcode::G_SDIV:
      if (!C2.getBoolValue())
        break;
      return C1.sdiv(C2);
    case TargetOpcode::G_UREM:
      if (!C2.getBoolValue())
        break;
      return C1.urem(C2);
    case TargetOpcode::G_SREM:
      if (!C2.getBoolValue())
        break;
      return C1.srem(C2);
    }
  }
  return None;
}

/// An MIRBuilder which does trivial constant folding of binary ops.
/// Calls to buildInstr will also try to constant fold binary ops.
class ConstantFoldingMIRBuilder
    : public FoldableInstructionsBuilder<ConstantFoldingMIRBuilder> {
public:
  // Pull in base class constructors.
  using FoldableInstructionsBuilder<
      ConstantFoldingMIRBuilder>::FoldableInstructionsBuilder;
  // Unhide buildInstr
  using FoldableInstructionsBuilder<ConstantFoldingMIRBuilder>::buildInstr;

  // Implement buildBinaryOp required by FoldableInstructionsBuilder which
  // tries to constant fold.
  MachineInstrBuilder buildBinaryOp(unsigned Opcode, unsigned Dst,
                                    unsigned Src0, unsigned Src1) {
    validateBinaryOp(Dst, Src0, Src1);
    auto MaybeCst = ConstantFoldBinOp(Opcode, Src0, Src1, getMF().getRegInfo());
    if (MaybeCst)
      return buildConstant(Dst, MaybeCst->getSExtValue());
    return buildInstr(Opcode).addDef(Dst).addUse(Src0).addUse(Src1);
  }

  template <typename DstTy, typename UseArg1Ty, typename UseArg2Ty>
  MachineInstrBuilder buildInstr(unsigned Opc, DstTy &&Ty, UseArg1Ty &&Arg1,
                                 UseArg2Ty &&Arg2) {
    unsigned Dst = getDestFromArg(Ty);
    return buildInstr(Opc, Dst, getRegFromArg(std::forward<UseArg1Ty>(Arg1)),
                      getRegFromArg(std::forward<UseArg2Ty>(Arg2)));
  }

  // Try to provide an overload for buildInstr for binary ops in order to
  // constant fold.
  MachineInstrBuilder buildInstr(unsigned Opc, unsigned Dst, unsigned Src0,
                                 unsigned Src1) {
    switch (Opc) {
    default:
      break;
    case TargetOpcode::G_ADD:
    case TargetOpcode::G_AND:
    case TargetOpcode::G_ASHR:
    case TargetOpcode::G_LSHR:
    case TargetOpcode::G_MUL:
    case TargetOpcode::G_OR:
    case TargetOpcode::G_SHL:
    case TargetOpcode::G_SUB:
    case TargetOpcode::G_XOR:
    case TargetOpcode::G_UDIV:
    case TargetOpcode::G_SDIV:
    case TargetOpcode::G_UREM:
    case TargetOpcode::G_SREM: {
      return buildBinaryOp(Opc, Dst, Src0, Src1);
    }
    }
    return buildInstr(Opc).addDef(Dst).addUse(Src0).addUse(Src1);
  }

  // Fallback implementation of buildInstr.
  template <typename DstTy, typename... UseArgsTy>
  MachineInstrBuilder buildInstr(unsigned Opc, DstTy &&Ty,
                                 UseArgsTy &&... Args) {
    auto MIB = buildInstr(Opc).addDef(getDestFromArg(Ty));
    addUsesFromArgs(MIB, std::forward<UseArgsTy>(Args)...);
    return MIB;
  }
};
} // namespace llvm
