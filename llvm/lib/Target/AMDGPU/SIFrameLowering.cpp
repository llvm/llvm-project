//===----------------------- SIFrameLowering.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

#include "SIFrameLowering.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool MayNeedScavengingEmergencySlot = MFI->hasStackObjects();

  assert((RS || !MayNeedScavengingEmergencySlot) &&
         "RegScavenger required if spilling");

  if (MayNeedScavengingEmergencySlot) {
    int ScavengeFI = MFI->CreateSpillStackObject(
      AMDGPU::SGPR_32RegClass.getSize(),
      AMDGPU::SGPR_32RegClass.getAlignment());
    RS->addScavengingFrameIndex(ScavengeFI);
  }
}
