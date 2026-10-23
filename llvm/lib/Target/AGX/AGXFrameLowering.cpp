//===-- AGXFrameLowering.cpp - AGX frame lowering -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXFrameLowering.h"
#include "AGX.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

unsigned AGXFrameLowering::getStackAdjustUnits(const MachineFunction &MF) {
  uint64_t Bytes = MF.getFrameInfo().getStackSize();
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() == CallingConv::AGX_KERNEL) {
    bool HasCall = false;
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        HasCall |= MI.getOpcode() == AGX::AGX_CALL;
    if (HasCall) {
      if (Bytes > CallFrameRegionBytes)
        report_fatal_error("experimental AGX caller frame exceeds 4096 bytes");
      Bytes = 2 * CallFrameRegionBytes;
    }
  }
  if (!Bytes)
    return 0;
  // Like the vendor compiler, allocate 40 more bytes and round up to 16
  // bytes, in 32-bit units.
  return alignTo(divideCeil(Bytes, 4) + 10, 4);
}

void AGXFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  unsigned Units = getStackAdjustUnits(MF);
  if (Units && agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    if (MF.getFrameInfo().getStackSize() > CallFrameRegionBytes)
      report_fatal_error("experimental AGX callee frame exceeds 4096 bytes");
    return; // The kernel owns the stack_adjust for both regions.
  }
  if (!Units)
    return;
  if (Units > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(AGX::STACK_ADJUST))
      .addImm(Units);
}
