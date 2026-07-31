//=======- NVPTXFrameLowering.cpp - NVPTX Frame Information ---*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "NVPTXFrameLowering.h"
#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

NVPTXFrameLowering::NVPTXFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, Align(8), 0) {}

bool NVPTXFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return true;
}

void NVPTXFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  if (MF.getFrameInfo().hasStackObjects()) {
    assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
    MachineBasicBlock::iterator MBBI = MBB.begin();
    MachineRegisterInfo &MR = MF.getRegInfo();

    const NVPTXRegisterInfo *NRI =
        MF.getSubtarget<NVPTXSubtarget>().getRegisterInfo();

    // This instruction really occurs before first instruction
    // in the BB, so giving it no debug location.
    DebugLoc dl = DebugLoc();

    // Emits the %SPL depot move and, when %SP is used, cvta.local.
    // Mixed-width targets widen %SPL before cvta.local.
    const Register FrameReg = NRI->getFrameRegister(MF);
    const Register FrameLocalReg = NRI->getFrameLocalRegister(MF);
    const bool Is64Bit = FrameReg == NVPTX::VRFrame64;
    const bool IsLocal64 = FrameLocalReg == NVPTX::VRFrameLocal64;
    unsigned CvtaLocalOpcode =
        (Is64Bit ? NVPTX::cvta_local_64 : NVPTX::cvta_local_32);
    unsigned MovDepotOpcode =
        (IsLocal64 ? NVPTX::MOV_DEPOT_ADDR_64 : NVPTX::MOV_DEPOT_ADDR);
    if (!MR.use_empty(FrameReg)) {
      // If %SP is not used, do not bother emitting "cvta.local %SP, %SPL".
      MBBI = BuildMI(MBB, MBBI, dl,
                     MF.getSubtarget().getInstrInfo()->get(CvtaLocalOpcode),
                     FrameReg)
                 .addReg(Is64Bit == IsLocal64 ? FrameLocalReg : FrameReg);
      if (Is64Bit && !IsLocal64) {
        // Widen %SPL before converting it to a generic pointer.
        MBBI =
            BuildMI(MBB, MBBI, dl,
                    MF.getSubtarget().getInstrInfo()->get(NVPTX::CVT_u64_u32),
                    FrameReg)
                .addReg(FrameLocalReg)
                .addImm(NVPTX::PTXCvtMode::NONE);
      }
    }
    if (!MR.use_empty(FrameLocalReg)) {
      BuildMI(MBB, MBBI, dl,
              MF.getSubtarget().getInstrInfo()->get(MovDepotOpcode),
              FrameLocalReg)
          .addImm(MF.getFunctionNumber());
    }
  }
}

StackOffset
NVPTXFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  FrameReg = NVPTX::VRDepot;
  return StackOffset::getFixed(MFI.getObjectOffset(FI) -
                               getOffsetOfLocalArea());
}

void NVPTXFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {}

// This function eliminates ADJCALLSTACKDOWN,
// ADJCALLSTACKUP pseudo instructions
MachineBasicBlock::iterator NVPTXFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN,
  // ADJCALLSTACKUP instructions.
  return MBB.erase(I);
}

TargetFrameLowering::DwarfFrameBase
NVPTXFrameLowering::getDwarfFrameBase(const MachineFunction &MF) const {
  DwarfFrameBase FrameBase;
  FrameBase.Kind = DwarfFrameBase::CFA;
  FrameBase.Location.Offset = 0;
  return FrameBase;
}
