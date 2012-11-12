//=======- NVPTXFrameLowering.cpp - NVPTX Frame Information ---*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXFrameLowering.h"
#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

bool NVPTXFrameLowering::hasFP(const MachineFunction &MF) const {
  return true;
}

void NVPTXFrameLowering::emitPrologue(MachineFunction &MF) const {
  if (MF.getFrameInfo()->hasStackObjects()) {
    MachineBasicBlock &MBB = MF.front();
    // Insert "mov.u32 %SP, %Depot"
    MachineBasicBlock::iterator MBBI = MBB.begin();
    // This instruction really occurs before first instruction
    // in the BB, so giving it no debug location.
    DebugLoc dl = DebugLoc();

    if (tm.getSubtargetImpl()->hasGenericLdSt()) {
      // mov %SPL, %depot;
      // cvta.local %SP, %SPL;
      if (is64bit) {
        MachineInstr *MI = BuildMI(MBB, MBBI, dl,
                               tm.getInstrInfo()->get(NVPTX::cvta_local_yes_64),
                                   NVPTX::VRFrame).addReg(NVPTX::VRFrameLocal);
        BuildMI(MBB, MI, dl,
                tm.getInstrInfo()->get(NVPTX::IMOV64rr), NVPTX::VRFrameLocal)
        .addReg(NVPTX::VRDepot);
      } else {
        MachineInstr *MI = BuildMI(MBB, MBBI, dl,
                                  tm.getInstrInfo()->get(NVPTX::cvta_local_yes),
                                   NVPTX::VRFrame).addReg(NVPTX::VRFrameLocal);
        BuildMI(MBB, MI, dl,
                tm.getInstrInfo()->get(NVPTX::IMOV32rr), NVPTX::VRFrameLocal)
        .addReg(NVPTX::VRDepot);
      }
    }
    else {
      // mov %SP, %depot;
      if (is64bit)
        BuildMI(MBB, MBBI, dl,
                tm.getInstrInfo()->get(NVPTX::IMOV64rr), NVPTX::VRFrame)
                .addReg(NVPTX::VRDepot);
      else
        BuildMI(MBB, MBBI, dl,
                tm.getInstrInfo()->get(NVPTX::IMOV32rr), NVPTX::VRFrame)
                .addReg(NVPTX::VRDepot);
    }
  }
}

void NVPTXFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
}
