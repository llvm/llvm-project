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

using namespace llvm;

bool X86::optimizeInstFromVEX3ToVEX2(MCInst &MI) {
  unsigned OpIdx1, OpIdx2;
  unsigned NewOpc;
#define FROM_TO(FROM, TO, IDX1, IDX2)                                          \
  case X86::FROM:                                                              \
    NewOpc = X86::TO;                                                          \
    OpIdx1 = IDX1;                                                             \
    OpIdx2 = IDX2;                                                             \
    break;
#define TO_REV(FROM) FROM_TO(FROM, FROM##_REV, 0, 1)
  switch (MI.getOpcode()) {
  default:
    return false;
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
  }
  MCOperand &LastOp = MI.getOperand(MI.getNumOperands() - 1);
  if (!LastOp.isImm() || LastOp.getImm() != 1)
    return false;
  MI.setOpcode(NewOpc);
  MI.erase(&LastOp);
  return true;
}
