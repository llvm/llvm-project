//===-- X86EncodingOptimization.cpp - X86 Encoding optimization -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the X86 encoding optimization
//
//===----------------------------------------------------------------------===//

#include "X86EncodingOptimization.h"
#include "X86BaseInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

static bool shouldExchange(const MCInst &MI, unsigned OpIdx1, unsigned OpIdx2) {
  return !X86II::isX86_64ExtendedReg(MI.getOperand(OpIdx1).getReg()) &&
         X86II::isX86_64ExtendedReg(MI.getOperand(OpIdx2).getReg());
}

bool X86::optimizeInstFromVEX3ToVEX2(MCInst &MI, const MCInstrDesc &Desc) {
  unsigned OpIdx1, OpIdx2;
  unsigned NewOpc;
  unsigned Opcode = MI.getOpcode();
#define FROM_TO(FROM, TO, IDX1, IDX2)                                          \
  case X86::FROM:                                                              \
    NewOpc = X86::TO;                                                          \
    OpIdx1 = IDX1;                                                             \
    OpIdx2 = IDX2;                                                             \
    break;
#define TO_REV(FROM) FROM_TO(FROM, FROM##_REV, 0, 1)
  switch (MI.getOpcode()) {
  default: {
    // If the instruction is a commutable arithmetic instruction we might be
    // able to commute the operands to get a 2 byte VEX prefix.
    uint64_t TSFlags = Desc.TSFlags;
    if (!Desc.isCommutable() || (TSFlags & X86II::EncodingMask) != X86II::VEX ||
        (TSFlags & X86II::OpMapMask) != X86II::TB ||
        (TSFlags & X86II::FormMask) != X86II::MRMSrcReg ||
        (TSFlags & X86II::REX_W) || !(TSFlags & X86II::VEX_4V) ||
        MI.getNumOperands() != 3)
      return false;
    // These two are not truly commutable.
    if (Opcode == X86::VMOVHLPSrr || Opcode == X86::VUNPCKHPDrr)
      return false;
    OpIdx1 = 1;
    OpIdx2 = 2;
    if (!shouldExchange(MI, OpIdx1, OpIdx2))
      return false;
    std::swap(MI.getOperand(OpIdx1), MI.getOperand(OpIdx2));
    return true;
  }
    // Commute operands to get a smaller encoding by using VEX.R instead of
    // VEX.B if one of the registers is extended, but other isn't.
    FROM_TO(VMOVZPQILo2PQIrr, VMOVPQI2QIrr, 0, 1)
    TO_REV(VMOVAPDrr)
    TO_REV(VMOVAPDYrr)
    TO_REV(VMOVAPSrr)
    TO_REV(VMOVAPSYrr)
    TO_REV(VMOVDQArr)
    TO_REV(VMOVDQAYrr)
    TO_REV(VMOVDQUrr)
    TO_REV(VMOVDQUYrr)
    TO_REV(VMOVUPDrr)
    TO_REV(VMOVUPDYrr)
    TO_REV(VMOVUPSrr)
    TO_REV(VMOVUPSYrr)
#undef TO_REV
#define TO_REV(FROM) FROM_TO(FROM, FROM##_REV, 0, 2)
    TO_REV(VMOVSDrr)
    TO_REV(VMOVSSrr)
#undef TO_REV
#undef FROM_TO
  }
  if (!shouldExchange(MI, OpIdx1, OpIdx2))
    return false;
  MI.setOpcode(NewOpc);
  return true;
}

// NOTE: We may write this as an InstAlias if it's only used by AsmParser. See
// validateTargetOperandClass.
bool X86::optimizeShiftRotateWithImmediateOne(MCInst &MI) {
  unsigned NewOpc;
#define TO_IMM1(FROM)                                                          \
  case X86::FROM##i:                                                           \
    NewOpc = X86::FROM##1;                                                     \
    break;
  switch (MI.getOpcode()) {
  default:
    return false;
    TO_IMM1(RCR8r)
    TO_IMM1(RCR16r)
    TO_IMM1(RCR32r)
    TO_IMM1(RCR64r)
    TO_IMM1(RCL8r)
    TO_IMM1(RCL16r)
    TO_IMM1(RCL32r)
    TO_IMM1(RCL64r)
    TO_IMM1(ROR8r)
    TO_IMM1(ROR16r)
    TO_IMM1(ROR32r)
    TO_IMM1(ROR64r)
    TO_IMM1(ROL8r)
    TO_IMM1(ROL16r)
    TO_IMM1(ROL32r)
    TO_IMM1(ROL64r)
    TO_IMM1(SAR8r)
    TO_IMM1(SAR16r)
    TO_IMM1(SAR32r)
    TO_IMM1(SAR64r)
    TO_IMM1(SHR8r)
    TO_IMM1(SHR16r)
    TO_IMM1(SHR32r)
    TO_IMM1(SHR64r)
    TO_IMM1(SHL8r)
    TO_IMM1(SHL16r)
    TO_IMM1(SHL32r)
    TO_IMM1(SHL64r)
    TO_IMM1(RCR8m)
    TO_IMM1(RCR16m)
    TO_IMM1(RCR32m)
    TO_IMM1(RCR64m)
    TO_IMM1(RCL8m)
    TO_IMM1(RCL16m)
    TO_IMM1(RCL32m)
    TO_IMM1(RCL64m)
    TO_IMM1(ROR8m)
    TO_IMM1(ROR16m)
    TO_IMM1(ROR32m)
    TO_IMM1(ROR64m)
    TO_IMM1(ROL8m)
    TO_IMM1(ROL16m)
    TO_IMM1(ROL32m)
    TO_IMM1(ROL64m)
    TO_IMM1(SAR8m)
    TO_IMM1(SAR16m)
    TO_IMM1(SAR32m)
    TO_IMM1(SAR64m)
    TO_IMM1(SHR8m)
    TO_IMM1(SHR16m)
    TO_IMM1(SHR32m)
    TO_IMM1(SHR64m)
    TO_IMM1(SHL8m)
    TO_IMM1(SHL16m)
    TO_IMM1(SHL32m)
    TO_IMM1(SHL64m)
#undef TO_IMM1
  }
  MCOperand &LastOp = MI.getOperand(MI.getNumOperands() - 1);
  if (!LastOp.isImm() || LastOp.getImm() != 1)
    return false;
  MI.setOpcode(NewOpc);
  MI.erase(&LastOp);
  return true;
}

bool X86::optimizeVPCMPWithImmediateOneOrSix(MCInst &MI) {
  unsigned Opc1;
  unsigned Opc2;
#define FROM_TO(FROM, TO1, TO2)                                                \
  case X86::FROM:                                                              \
    Opc1 = X86::TO1;                                                           \
    Opc2 = X86::TO2;                                                           \
    break;
  switch (MI.getOpcode()) {
  default:
    return false;
    FROM_TO(VPCMPBZ128rmi, VPCMPEQBZ128rm, VPCMPGTBZ128rm)
    FROM_TO(VPCMPBZ128rmik, VPCMPEQBZ128rmk, VPCMPGTBZ128rmk)
    FROM_TO(VPCMPBZ128rri, VPCMPEQBZ128rr, VPCMPGTBZ128rr)
    FROM_TO(VPCMPBZ128rrik, VPCMPEQBZ128rrk, VPCMPGTBZ128rrk)
    FROM_TO(VPCMPBZ256rmi, VPCMPEQBZ256rm, VPCMPGTBZ256rm)
    FROM_TO(VPCMPBZ256rmik, VPCMPEQBZ256rmk, VPCMPGTBZ256rmk)
    FROM_TO(VPCMPBZ256rri, VPCMPEQBZ256rr, VPCMPGTBZ256rr)
    FROM_TO(VPCMPBZ256rrik, VPCMPEQBZ256rrk, VPCMPGTBZ256rrk)
    FROM_TO(VPCMPBZrmi, VPCMPEQBZrm, VPCMPGTBZrm)
    FROM_TO(VPCMPBZrmik, VPCMPEQBZrmk, VPCMPGTBZrmk)
    FROM_TO(VPCMPBZrri, VPCMPEQBZrr, VPCMPGTBZrr)
    FROM_TO(VPCMPBZrrik, VPCMPEQBZrrk, VPCMPGTBZrrk)
    FROM_TO(VPCMPDZ128rmi, VPCMPEQDZ128rm, VPCMPGTDZ128rm)
    FROM_TO(VPCMPDZ128rmib, VPCMPEQDZ128rmb, VPCMPGTDZ128rmb)
    FROM_TO(VPCMPDZ128rmibk, VPCMPEQDZ128rmbk, VPCMPGTDZ128rmbk)
    FROM_TO(VPCMPDZ128rmik, VPCMPEQDZ128rmk, VPCMPGTDZ128rmk)
    FROM_TO(VPCMPDZ128rri, VPCMPEQDZ128rr, VPCMPGTDZ128rr)
    FROM_TO(VPCMPDZ128rrik, VPCMPEQDZ128rrk, VPCMPGTDZ128rrk)
    FROM_TO(VPCMPDZ256rmi, VPCMPEQDZ256rm, VPCMPGTDZ256rm)
    FROM_TO(VPCMPDZ256rmib, VPCMPEQDZ256rmb, VPCMPGTDZ256rmb)
    FROM_TO(VPCMPDZ256rmibk, VPCMPEQDZ256rmbk, VPCMPGTDZ256rmbk)
    FROM_TO(VPCMPDZ256rmik, VPCMPEQDZ256rmk, VPCMPGTDZ256rmk)
    FROM_TO(VPCMPDZ256rri, VPCMPEQDZ256rr, VPCMPGTDZ256rr)
    FROM_TO(VPCMPDZ256rrik, VPCMPEQDZ256rrk, VPCMPGTDZ256rrk)
    FROM_TO(VPCMPDZrmi, VPCMPEQDZrm, VPCMPGTDZrm)
    FROM_TO(VPCMPDZrmib, VPCMPEQDZrmb, VPCMPGTDZrmb)
    FROM_TO(VPCMPDZrmibk, VPCMPEQDZrmbk, VPCMPGTDZrmbk)
    FROM_TO(VPCMPDZrmik, VPCMPEQDZrmk, VPCMPGTDZrmk)
    FROM_TO(VPCMPDZrri, VPCMPEQDZrr, VPCMPGTDZrr)
    FROM_TO(VPCMPDZrrik, VPCMPEQDZrrk, VPCMPGTDZrrk)
    FROM_TO(VPCMPQZ128rmi, VPCMPEQQZ128rm, VPCMPGTQZ128rm)
    FROM_TO(VPCMPQZ128rmib, VPCMPEQQZ128rmb, VPCMPGTQZ128rmb)
    FROM_TO(VPCMPQZ128rmibk, VPCMPEQQZ128rmbk, VPCMPGTQZ128rmbk)
    FROM_TO(VPCMPQZ128rmik, VPCMPEQQZ128rmk, VPCMPGTQZ128rmk)
    FROM_TO(VPCMPQZ128rri, VPCMPEQQZ128rr, VPCMPGTQZ128rr)
    FROM_TO(VPCMPQZ128rrik, VPCMPEQQZ128rrk, VPCMPGTQZ128rrk)
    FROM_TO(VPCMPQZ256rmi, VPCMPEQQZ256rm, VPCMPGTQZ256rm)
    FROM_TO(VPCMPQZ256rmib, VPCMPEQQZ256rmb, VPCMPGTQZ256rmb)
    FROM_TO(VPCMPQZ256rmibk, VPCMPEQQZ256rmbk, VPCMPGTQZ256rmbk)
    FROM_TO(VPCMPQZ256rmik, VPCMPEQQZ256rmk, VPCMPGTQZ256rmk)
    FROM_TO(VPCMPQZ256rri, VPCMPEQQZ256rr, VPCMPGTQZ256rr)
    FROM_TO(VPCMPQZ256rrik, VPCMPEQQZ256rrk, VPCMPGTQZ256rrk)
    FROM_TO(VPCMPQZrmi, VPCMPEQQZrm, VPCMPGTQZrm)
    FROM_TO(VPCMPQZrmib, VPCMPEQQZrmb, VPCMPGTQZrmb)
    FROM_TO(VPCMPQZrmibk, VPCMPEQQZrmbk, VPCMPGTQZrmbk)
    FROM_TO(VPCMPQZrmik, VPCMPEQQZrmk, VPCMPGTQZrmk)
    FROM_TO(VPCMPQZrri, VPCMPEQQZrr, VPCMPGTQZrr)
    FROM_TO(VPCMPQZrrik, VPCMPEQQZrrk, VPCMPGTQZrrk)
    FROM_TO(VPCMPWZ128rmi, VPCMPEQWZ128rm, VPCMPGTWZ128rm)
    FROM_TO(VPCMPWZ128rmik, VPCMPEQWZ128rmk, VPCMPGTWZ128rmk)
    FROM_TO(VPCMPWZ128rri, VPCMPEQWZ128rr, VPCMPGTWZ128rr)
    FROM_TO(VPCMPWZ128rrik, VPCMPEQWZ128rrk, VPCMPGTWZ128rrk)
    FROM_TO(VPCMPWZ256rmi, VPCMPEQWZ256rm, VPCMPGTWZ256rm)
    FROM_TO(VPCMPWZ256rmik, VPCMPEQWZ256rmk, VPCMPGTWZ256rmk)
    FROM_TO(VPCMPWZ256rri, VPCMPEQWZ256rr, VPCMPGTWZ256rr)
    FROM_TO(VPCMPWZ256rrik, VPCMPEQWZ256rrk, VPCMPGTWZ256rrk)
    FROM_TO(VPCMPWZrmi, VPCMPEQWZrm, VPCMPGTWZrm)
    FROM_TO(VPCMPWZrmik, VPCMPEQWZrmk, VPCMPGTWZrmk)
    FROM_TO(VPCMPWZrri, VPCMPEQWZrr, VPCMPGTWZrr)
    FROM_TO(VPCMPWZrrik, VPCMPEQWZrrk, VPCMPGTWZrrk)
#undef FROM_TO
  }
  MCOperand &LastOp = MI.getOperand(MI.getNumOperands() - 1);
  int64_t Imm = LastOp.getImm();
  unsigned NewOpc;
  if (Imm == 0)
    NewOpc = Opc1;
  else if(Imm == 6)
    NewOpc = Opc2;
  else
    return false;
  MI.setOpcode(NewOpc);
  MI.erase(&LastOp);
  return true;
}

bool X86::optimizeMOVSX(MCInst &MI) {
  unsigned NewOpc;
#define FROM_TO(FROM, TO, R0, R1)                                              \
  case X86::FROM:                                                              \
    if (MI.getOperand(0).getReg() != X86::R0 ||                                \
        MI.getOperand(1).getReg() != X86::R1)                                  \
      return false;                                                            \
    NewOpc = X86::TO;                                                          \
    break;
  switch (MI.getOpcode()) {
  default:
    return false;
    FROM_TO(MOVSX16rr8, CBW, AX, AL)     // movsbw %al, %ax   --> cbtw
    FROM_TO(MOVSX32rr16, CWDE, EAX, AX)  // movswl %ax, %eax  --> cwtl
    FROM_TO(MOVSX64rr32, CDQE, RAX, EAX) // movslq %eax, %rax --> cltq
  }
  MI.clear();
  MI.setOpcode(NewOpc);
  return true;
}
