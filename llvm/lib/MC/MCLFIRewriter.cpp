//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the MCLFIRewriter class, a base class that
/// encapsulates the rewriting logic for MCInsts.
///
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

void MCLFIRewriter::error(const MCInst &Inst, const Twine &Msg) {
  Ctx.reportError(Inst.getLoc(), Msg);
}

void MCLFIRewriter::warning(const MCInst &Inst, const Twine &Msg) {
  Ctx.reportWarning(Inst.getLoc(), Msg);
}

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

bool MCLFIRewriter::explicitlyModifiesRegister(const MCInst &Inst,
                                               MCRegister Reg) const {
  return InstInfo->get(Inst.getOpcode())
      .hasExplicitDefOfPhysReg(Inst, Reg, *RegInfo);
}
