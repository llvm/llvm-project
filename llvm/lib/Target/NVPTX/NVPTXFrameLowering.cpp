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
#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
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

    // Emits
    //   mov %SPL, %depot;
    //   cvta.local %SP, %SPL;
    // for local address accesses in MF.
    // if the generic and local address spaces are different,
    // it emits:
    //   mov %SPL, %depot;
    //   cvt.u64.u32 %SP, %SPL;
    //   cvta.local %SP, %SP;

    if (MR.use_empty(NRI->getFrameLocalRegister(MF)))
      // If %SPL is not used, do not bother emitting anything
      return;
    bool IsLocal64Bit =
        MF.getTarget().getPointerSize(NVPTXAS::ADDRESS_SPACE_LOCAL) == 8;
    bool IsGeneric64Bit =
        MF.getTarget().getPointerSize(NVPTXAS::ADDRESS_SPACE_GENERIC) == 8;
    bool NeedsCast = IsGeneric64Bit != IsLocal64Bit;
    Register SourceReg = NRI->getFrameLocalRegister(MF);
    if (NeedsCast)
      SourceReg = NRI->getFrameRegister(MF);

    unsigned CvtaLocalOpcode =
        (IsGeneric64Bit ? NVPTX::cvta_local_64 : NVPTX::cvta_local);

    MBBI = BuildMI(MBB, MBBI, dl,
                   MF.getSubtarget().getInstrInfo()->get(CvtaLocalOpcode),
                   NRI->getFrameRegister(MF))
               .addReg(SourceReg);

    if (NeedsCast)
      MBBI = BuildMI(MBB, MBBI, dl,
                     MF.getSubtarget().getInstrInfo()->get(NVPTX::CVT_u64_u32),
                     NRI->getFrameRegister(MF))
                 .addReg(NRI->getFrameLocalRegister(MF))
                 .addImm(NVPTX::PTXCvtMode::NONE);

    unsigned MovDepotOpcode =
        (IsLocal64Bit ? NVPTX::MOV_DEPOT_ADDR_64 : NVPTX::MOV_DEPOT_ADDR);
    BuildMI(MBB, MBBI, dl,
            MF.getSubtarget().getInstrInfo()->get(MovDepotOpcode),
            NRI->getFrameLocalRegister(MF))
        .addImm(MF.getFunctionNumber());
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
