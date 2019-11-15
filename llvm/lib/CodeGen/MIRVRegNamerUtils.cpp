//===---------- MIRVRegNamerUtils.cpp - MIR VReg Renaming Utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MIRVRegNamerUtils.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "mir-vregnamer-utils"

bool VRegRenamer::doVRegRenaming(
    const std::map<unsigned, unsigned> &VRegRenameMap) {
  bool Changed = false;
  for (auto I = VRegRenameMap.begin(), E = VRegRenameMap.end(); I != E; ++I) {

    auto VReg = I->first;
    auto Rename = I->second;

    std::vector<MachineOperand *> RenameMOs;
    for (auto &MO : MRI.reg_operands(VReg)) {
      RenameMOs.push_back(&MO);
    }

    for (auto *MO : RenameMOs) {
      Changed = true;
      MO->setReg(Rename);

      if (!MO->isDef())
        MO->setIsKill(false);
    }
  }

  return Changed;
}

std::map<unsigned, unsigned>
VRegRenamer::getVRegRenameMap(const std::vector<NamedVReg> &VRegs) {
  std::map<unsigned, unsigned> VRegRenameMap;

  std::map<std::string, unsigned> VRegNameCollisionMap;

  auto GetUniqueVRegName =
      [&VRegNameCollisionMap](const NamedVReg &Reg) -> std::string {
    auto It = VRegNameCollisionMap.find(Reg.getName());
    unsigned Counter = 0;
    if (It != VRegNameCollisionMap.end()) {
      Counter = It->second;
    }
    ++Counter;
    VRegNameCollisionMap[Reg.getName()] = Counter;
    return Reg.getName() + "__" + std::to_string(Counter);
  };

  for (auto &Vreg : VRegs) {
    auto Reg = Vreg.getReg();
    assert(Register::isVirtualRegister(Reg) &&
           "Expecting Virtual Registers Only");
    auto NewNameForReg = GetUniqueVRegName(Vreg);
    auto Rename = createVirtualRegisterWithName(Reg, NewNameForReg);

    VRegRenameMap.insert(std::pair<unsigned, unsigned>(Reg, Rename));
  }
  return VRegRenameMap;
}

std::string VRegRenamer::getInstructionOpcodeHash(MachineInstr &MI) {
  std::string S;
  raw_string_ostream OS(S);
  auto HashOperand = [this](const MachineOperand &MO) -> unsigned {
    if (MO.isImm())
      return MO.getImm();
    if (MO.isTargetIndex())
      return MO.getOffset() | (MO.getTargetFlags() << 16);
    if (MO.isReg()) {
      return Register::isVirtualRegister(MO.getReg())
                 ? MRI.getVRegDef(MO.getReg())->getOpcode()
                 : (unsigned)MO.getReg();
    }
    // We could explicitly handle all the types of the MachineOperand,
    // here but we can just return a common number until we find a
    // compelling test case where this is bad. The only side effect here
    // is contributing to a hash collission but there's enough information
    // (Opcodes,other registers etc) that this will likely not be a problem.
    return 0;
  };
  SmallVector<unsigned, 16> MIOperands;
  MIOperands.push_back(MI.getOpcode());
  for (auto &Op : MI.uses()) {
    MIOperands.push_back(HashOperand(Op));
  }
  auto HashMI = hash_combine_range(MIOperands.begin(), MIOperands.end());
  return std::to_string(HashMI).substr(0, 5);
}

unsigned VRegRenamer::createVirtualRegister(unsigned VReg) {
  return createVirtualRegisterWithName(
      VReg, getInstructionOpcodeHash(*MRI.getVRegDef(VReg)));
}

bool VRegRenamer::renameInstsInMBB(MachineBasicBlock *MBB) {
  std::vector<NamedVReg> VRegs;
  std::string Prefix = "bb" + std::to_string(getCurrentBBNumber()) + "_";
  for (auto &MII : *MBB) {
    MachineInstr &Candidate = MII;
    // Don't rename stores/branches.
    if (Candidate.mayStore() || Candidate.isBranch())
      continue;
    if (!Candidate.getNumOperands())
      continue;
    // Look for instructions that define VRegs in operand 0.
    MachineOperand &MO = Candidate.getOperand(0);
    // Avoid non regs, instructions defining physical regs.
    if (!MO.isReg() || !Register::isVirtualRegister(MO.getReg()))
      continue;
    VRegs.push_back(
        NamedVReg(MO.getReg(), Prefix + getInstructionOpcodeHash(Candidate)));
  }

  // If we have populated no vregs to rename then bail.
  // The rest of this function does the vreg remaping.
  if (VRegs.size() == 0)
    return false;

  auto VRegRenameMap = getVRegRenameMap(VRegs);
  return doVRegRenaming(VRegRenameMap);
}

bool VRegRenamer::renameVRegs(MachineBasicBlock *MBB, unsigned BBNum) {
  CurrentBBNumber = BBNum;
  return renameInstsInMBB(MBB);
}

unsigned VRegRenamer::createVirtualRegisterWithName(unsigned VReg,
                                                    const std::string &Name) {
  std::string Temp(Name);
  std::transform(Temp.begin(), Temp.end(), Temp.begin(), ::tolower);
  if (auto RC = MRI.getRegClassOrNull(VReg))
    return MRI.createVirtualRegister(RC, Temp);
  return MRI.createGenericVirtualRegister(MRI.getType(VReg), Name);
}
