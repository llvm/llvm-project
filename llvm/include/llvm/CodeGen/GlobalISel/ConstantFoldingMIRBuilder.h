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
class ConstantFoldingMIRBuilder : public MachineIRBuilder {
public:
  // Pull in base class constructors.
  using MachineIRBuilder::MachineIRBuilder;

  virtual ~ConstantFoldingMIRBuilder() = default;

  // Try to provide an overload for buildInstr for binary ops in order to
  // constant fold.
  MachineInstrBuilder buildInstr(unsigned Opc, ArrayRef<DstOp> DstOps,
                                 ArrayRef<SrcOp> SrcOps,
                                 Optional<unsigned> Flags = None) override {
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
      assert(DstOps.size() == 1 && "Invalid dst ops");
      assert(SrcOps.size() == 2 && "Invalid src ops");
      const DstOp &Dst = DstOps[0];
      const SrcOp &Src0 = SrcOps[0];
      const SrcOp &Src1 = SrcOps[1];
      if (auto MaybeCst =
              ConstantFoldBinOp(Opc, Src0.getReg(), Src1.getReg(), *getMRI()))
        return buildConstant(Dst, MaybeCst->getSExtValue());
      break;
    }
    }
    return MachineIRBuilder::buildInstr(Opc, DstOps, SrcOps);
  }
};
} // namespace llvm
