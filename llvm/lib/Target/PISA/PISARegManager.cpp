//===-- PISARegManager.cpp - Manage PISA virtual registers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISARegManager.h"
#include "PISAInstrInfo.h"
#include "PISASubtarget.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;
using namespace PISA;

void RegManager::computeMapping() {
  std::array<unsigned, RegType::NUM_TYPE> Count{};
  const PISAInstrInfo *TII = MF.getSubtarget<PISASubtarget>().getInstrInfo();
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || (!MO.isDef() && !MO.isUndef()))
          continue;

        Register CurReg = MO.getReg();

        if (CurReg.isPhysical())
          continue;

        if (Mapping.count(CurReg) != 0)
          continue;

        unsigned Flags = Usage::None;
        if (TII->isNoEmissionInstr(MI))
          Flags |= Usage::NoEmissionDef;

        RegType Type = getRegType(MRI.getRegClass(MO.getReg()));
        RegInfo Info{Type, Count[Type]++, static_cast<Usage>(Flags)};
        Mapping[CurReg] = Info;
      }
    }
  }
}

unsigned RegManager::getRegIdx(Register Reg) const {
  MappingTy::const_iterator I = Mapping.find(Reg);
  assert(I != Mapping.end() && "missing?");
  return I->second.Idx;
}

unsigned RegManager::encodeVirtualRegister(RegBank Bank, Register Reg) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned Idx = getRegIdx(Reg);
  RegType Type = getRegType(MRI.getRegClass(Reg));
  return RegEncoder::encodeVirtualRegister(Idx, Bank, Type);
}

RegManager::RegManager(const MachineFunction &MF)
    : MF(MF), MRI(MF.getRegInfo()) {
  computeMapping();
}
