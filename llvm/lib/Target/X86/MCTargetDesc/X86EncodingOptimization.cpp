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
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

bool X86::optimizeInstFromVEX3ToVEX2(MCInst &MI, const MCInstrDesc &Desc) {
  unsigned OpIdx1, OpIdx2;
  unsigned Opcode = MI.getOpcode();
  unsigned NewOpc = 0;
#define FROM_TO(FROM, TO, IDX1, IDX2)                                          \
  case X86::FROM:                                                              \
    NewOpc = X86::TO;                                                          \
    OpIdx1 = IDX1;                                                             \
    OpIdx2 = IDX2;                                                             \
    break;
#define TO_REV(FROM) FROM_TO(FROM, FROM##_REV, 0, 1)
  switch (Opcode) {
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
    break;
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
  if (X86II::isX86_64ExtendedReg(MI.getOperand(OpIdx1).getReg()) ||
      !X86II::isX86_64ExtendedReg(MI.getOperand(OpIdx2).getReg()))
    return false;
  if (NewOpc)
    MI.setOpcode(NewOpc);
  else
    std::swap(MI.getOperand(OpIdx1), MI.getOperand(OpIdx2));
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
#undef FROM_TO
  }
  MI.clear();
  MI.setOpcode(NewOpc);
  return true;
}

bool X86::optimizeINCDEC(MCInst &MI, bool In64BitMode) {
  if (In64BitMode)
    return false;
  unsigned NewOpc;
  // If we aren't in 64-bit mode we can use the 1-byte inc/dec instructions.
#define FROM_TO(FROM, TO)                                                      \
  case X86::FROM:                                                              \
    NewOpc = X86::TO;                                                          \
    break;
  switch (MI.getOpcode()) {
  default:
    return false;
    FROM_TO(DEC16r, DEC16r_alt)
    FROM_TO(DEC32r, DEC32r_alt)
    FROM_TO(INC16r, INC16r_alt)
    FROM_TO(INC32r, INC32r_alt)
  }
  MI.setOpcode(NewOpc);
  return true;
}

/// Simplify things like MOV32rm to MOV32o32a.
bool X86::optimizeMOV(MCInst &MI, bool In64BitMode) {
  // Don't make these simplifications in 64-bit mode; other assemblers don't
  // perform them because they make the code larger.
  if (In64BitMode)
    return false;
  unsigned NewOpc;
  // We don't currently select the correct instruction form for instructions
  // which have a short %eax, etc. form. Handle this by custom lowering, for
  // now.
  //
  // Note, we are currently not handling the following instructions:
  // MOV64ao8, MOV64o8a
  // XCHG16ar, XCHG32ar, XCHG64ar
  switch (MI.getOpcode()) {
  default:
    return false;
    FROM_TO(MOV8mr_NOREX, MOV8o32a)
    FROM_TO(MOV8mr, MOV8o32a)
    FROM_TO(MOV8rm_NOREX, MOV8ao32)
    FROM_TO(MOV8rm, MOV8ao32)
    FROM_TO(MOV16mr, MOV16o32a)
    FROM_TO(MOV16rm, MOV16ao32)
    FROM_TO(MOV32mr, MOV32o32a)
    FROM_TO(MOV32rm, MOV32ao32)
  }
  bool IsStore = MI.getOperand(0).isReg() && MI.getOperand(1).isReg();
  unsigned AddrBase = IsStore;
  unsigned RegOp = IsStore ? 0 : 5;
  unsigned AddrOp = AddrBase + 3;
  // Check whether the destination register can be fixed.
  unsigned Reg = MI.getOperand(RegOp).getReg();
  if (Reg != X86::AL && Reg != X86::AX && Reg != X86::EAX && Reg != X86::RAX)
    return false;
  // Check whether this is an absolute address.
  // FIXME: We know TLVP symbol refs aren't, but there should be a better way
  // to do this here.
  bool Absolute = true;
  if (MI.getOperand(AddrOp).isExpr()) {
    const MCExpr *MCE = MI.getOperand(AddrOp).getExpr();
    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(MCE))
      if (SRE->getKind() == MCSymbolRefExpr::VK_TLVP)
        Absolute = false;
  }
  if (Absolute && (MI.getOperand(AddrBase + X86::AddrBaseReg).getReg() != 0 ||
                   MI.getOperand(AddrBase + X86::AddrScaleAmt).getImm() != 1 ||
                   MI.getOperand(AddrBase + X86::AddrIndexReg).getReg() != 0))
    return false;
  // If so, rewrite the instruction.
  MCOperand Saved = MI.getOperand(AddrOp);
  MCOperand Seg = MI.getOperand(AddrBase + X86::AddrSegmentReg);
  MI.clear();
  MI.setOpcode(NewOpc);
  MI.addOperand(Saved);
  MI.addOperand(Seg);
  return true;
}
