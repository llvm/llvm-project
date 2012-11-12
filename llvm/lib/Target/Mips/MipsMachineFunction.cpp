//===-- MipsMachineFunctionInfo.cpp - Private data used for Mips ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsMachineFunction.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool>
FixGlobalBaseReg("mips-fix-global-base-reg", cl::Hidden, cl::init(true),
                 cl::desc("Always use $gp as the global base register."));

bool MipsFunctionInfo::globalBaseRegSet() const {
  return GlobalBaseReg;
}

unsigned MipsFunctionInfo::getGlobalBaseReg() {
  // Return if it has already been initialized.
  if (GlobalBaseReg)
    return GlobalBaseReg;

  const MipsSubtarget &ST = MF.getTarget().getSubtarget<MipsSubtarget>();

  const TargetRegisterClass *RC;
  if (ST.inMips16Mode())
    RC=(const TargetRegisterClass*)&Mips::CPU16RegsRegClass;
  else
    RC = ST.isABI_N64() ?
      (const TargetRegisterClass*)&Mips::CPU64RegsRegClass :
      (const TargetRegisterClass*)&Mips::CPURegsRegClass;
  return GlobalBaseReg = MF.getRegInfo().createVirtualRegister(RC);
}

bool MipsFunctionInfo::mips16SPAliasRegSet() const {
  return Mips16SPAliasReg;
}
unsigned MipsFunctionInfo::getMips16SPAliasReg() {
  // Return if it has already been initialized.
  if (Mips16SPAliasReg)
    return Mips16SPAliasReg;

  const TargetRegisterClass *RC;
  RC=(const TargetRegisterClass*)&Mips::CPU16RegsRegClass;
  return Mips16SPAliasReg = MF.getRegInfo().createVirtualRegister(RC);
}

void MipsFunctionInfo::anchor() { }
