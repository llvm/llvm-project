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
#include "llvm/IR/Function.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

unsigned AGXFrameLowering::getCallDepth(const MachineFunction &MF) {
  unsigned Depth = 0;
  if (auto Attr = MF.getFunction().getFnAttribute("agx-call-depth");
      Attr.isStringAttribute())
    if (Attr.getValueAsString().getAsInteger(10, Depth))
      report_fatal_error("invalid AGX call depth");
  return Depth;
}

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
      unsigned MaxDepth = 1;
      auto Attr = MF.getFunction().getFnAttribute("agx-max-call-depth");
      if (Attr.isStringAttribute() &&
          Attr.getValueAsString().getAsInteger(10, MaxDepth))
        report_fatal_error("invalid AGX maximum call depth");
      Bytes = (MaxDepth + 1) * CallFrameRegionBytes;
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
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    if (MF.getFrameInfo().getStackSize() > CallFrameRegionBytes)
      report_fatal_error("experimental AGX callee frame exceeds 4096 bytes");
    // The kernel owns stack_adjust. Save the incoming link only in functions
    // that make another call; a leaf can return directly through r1.
    bool HasCall = false;
    for (const MachineBasicBlock &Block : MF)
      for (const MachineInstr &MI : Block)
        HasCall |= MI.getOpcode() == AGX::AGX_CALL;
    if (HasCall) {
      unsigned Depth = getCallDepth(MF);
      if (Depth == 0 || Depth > 7)
        report_fatal_error("unsupported AGX nested call depth");
      const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
      BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(TargetOpcode::COPY),
              AGX::GPR32RegClass.getRegister(29 + Depth))
          .addReg(AGX::R1);
    }
    return;
  }
  if (!Units)
    return;
  if (Units > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(AGX::STACK_ADJUST))
      .addImm(Units);
}

void AGXFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  if (!agxExperimentalLeafCallsEnabled() ||
      MF.getFunction().getCallingConv() == CallingConv::AGX_KERNEL)
    return;
  bool HasCall = false;
  for (const MachineBasicBlock &Block : MF)
    for (const MachineInstr &MI : Block)
      HasCall |= MI.getOpcode() == AGX::AGX_CALL;
  if (!HasCall)
    return;
  unsigned Depth = getCallDepth(MF);
  if (Depth == 0 || Depth > 7)
    report_fatal_error("unsupported AGX nested call depth");
  const AGXInstrInfo *TII = MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  BuildMI(MBB, MBB.getFirstTerminator(), DebugLoc(),
          TII->get(TargetOpcode::COPY), AGX::R1)
      .addReg(AGX::GPR32RegClass.getRegister(29 + Depth));
}
