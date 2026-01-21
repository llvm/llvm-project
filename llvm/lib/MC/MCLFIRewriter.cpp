//===- MCLFIRewriter.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file was written by the LFI and Native Client authors.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MCLFIRewriter class. This is a base
// class that encapsulates the rewriting logic for MCInsts.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {

void MCLFIRewriter::error(const MCInst &Inst, const char Msg[]) {
  Ctx.reportError(Inst.getLoc(), Msg);
}

void MCLFIRewriter::disable() { Enabled = false; }

void MCLFIRewriter::enable() { Enabled = true; }

bool MCLFIRewriter::isEnabled() { return Enabled; }

bool MCLFIRewriter::isCall(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).isCall();
}

bool MCLFIRewriter::isBranch(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).isBranch();
}

bool MCLFIRewriter::isIndirectBranch(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).isIndirectBranch();
}

bool MCLFIRewriter::isReturn(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).isReturn();
}

bool MCLFIRewriter::mayLoad(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).mayLoad();
}

bool MCLFIRewriter::mayStore(const MCInst &Inst) const {
  return InstInfo->get(Inst.getOpcode()).mayStore();
}

bool MCLFIRewriter::mayModifyRegister(const MCInst &Inst,
                                      MCRegister Reg) const {
  return InstInfo->get(Inst.getOpcode()).hasDefOfPhysReg(Inst, Reg, *RegInfo);
}
} // namespace llvm
