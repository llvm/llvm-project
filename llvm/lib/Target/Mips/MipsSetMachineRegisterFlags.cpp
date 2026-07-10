//===- MipsSetMachineRegisterFlags.cpp - Set Machine Register Flags -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass sets machine register flags for MIPS backend.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mips-set-machine-register-flags"

using namespace llvm;

namespace {

class MipsSetMachineRegisterFlags : public MachineFunctionPass {
public:
  MipsSetMachineRegisterFlags() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Mips Set Machine Register Flags";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;

private:
  bool processBasicBlock(MachineBasicBlock &MBB, const MipsInstrInfo &MipsII,
                         const MachineRegisterInfo &RegInfo);
};

} // namespace

INITIALIZE_PASS(MipsSetMachineRegisterFlags, DEBUG_TYPE,
                "Mips Set Machine Register Flags", false, false)

char MipsSetMachineRegisterFlags::ID = 0;

bool MipsSetMachineRegisterFlags::runOnMachineFunction(MachineFunction &MF) {
  const MipsInstrInfo &MipsII =
      *static_cast<const MipsInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const MachineRegisterInfo &RegInfo = MF.getRegInfo();

  bool Modified = false;

  for (auto &MBB : MF)
    Modified |= processBasicBlock(MBB, MipsII, RegInfo);

  return Modified;
}

bool MipsSetMachineRegisterFlags::processBasicBlock(
    MachineBasicBlock &MBB, const MipsInstrInfo &MipsII,
    const MachineRegisterInfo &RegInfo) {
  bool Modified = false;

  // Iterate through the instructions in the basic block
  for (MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end(); MII != E;
       ++MII) {
    MachineInstr &MI = *MII;

    LLVM_DEBUG(dbgs() << "Processing instruction: " << MI << "\n");

    unsigned Opcode = MI.getOpcode();
    if (Opcode >= Mips::CMP_AF_D_MMR6 && Opcode <= Mips::CMP_UN_S_MMR6) {
      MachineOperand &DestOperand = MI.getOperand(0);
      assert(DestOperand.isReg());
      Register Dest = DestOperand.getReg();
      if (Dest.isVirtual() &&
          RegInfo.getRegClassOrNull(Dest) == &Mips::FGR64CCRegClass) {
        MI.setFlag(MachineInstr::MIFlag::NoSWrap);
      }
    } else if (Opcode == Mips::COPY) {
      MachineOperand &SrcOperand = MI.getOperand(1);
      assert(SrcOperand.isReg());
      Register Src = SrcOperand.getReg();
      if (Src.isVirtual() &&
          RegInfo.getRegClassOrNull(Src) == &Mips::FGR64CCRegClass) {
        MI.setFlag(MachineInstr::MIFlag::NoSWrap);
      }
    } else if (Opcode == Mips::INSERT_SUBREG) {
      MachineOperand &SrcOperand = MI.getOperand(2);
      assert(SrcOperand.isReg());
      Register Src = SrcOperand.getReg();
      if (Src.isVirtual() &&
          RegInfo.getRegClassOrNull(Src) == &Mips::FGR64CCRegClass) {
        MI.setFlag(MachineInstr::MIFlag::NoSWrap);
      }
    }
  }

  return Modified;
}

FunctionPass *llvm::createMipsSetMachineRegisterFlagsPass() {
  return new MipsSetMachineRegisterFlags();
}
