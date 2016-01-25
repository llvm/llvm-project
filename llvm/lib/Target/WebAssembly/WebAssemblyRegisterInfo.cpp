//===-- WebAssemblyRegisterInfo.cpp - WebAssembly Register Information ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of the
/// TargetRegisterInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyRegisterInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyFrameLowering.h"
#include "WebAssemblyInstrInfo.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "WebAssemblyGenRegisterInfo.inc"

WebAssemblyRegisterInfo::WebAssemblyRegisterInfo(const Triple &TT)
    : WebAssemblyGenRegisterInfo(0), TT(TT) {}

const MCPhysReg *
WebAssemblyRegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  static const MCPhysReg CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector
WebAssemblyRegisterInfo::getReservedRegs(const MachineFunction & /*MF*/) const {
  BitVector Reserved(getNumRegs());
  for (auto Reg : {WebAssembly::SP32, WebAssembly::SP64, WebAssembly::FP32,
                   WebAssembly::FP64})
    Reserved.set(Reg);
  return Reserved;
}

void WebAssemblyRegisterInfo::eliminateFrameIndex(
    MachineBasicBlock::iterator II, int SPAdj, unsigned FIOperandNum,
    RegScavenger * /*RS*/) const {
  assert(SPAdj == 0);
  MachineInstr &MI = *II;

  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  int64_t FrameOffset = MFI.getStackSize() + MFI.getObjectOffset(FrameIndex);

  if (MI.mayLoadOrStore()) {
    // If this is a load or store, make it relative to SP and fold the frame
    // offset directly in.
    assert(FrameOffset >= 0 && MI.getOperand(1).getImm() >= 0);
    int64_t Offset = MI.getOperand(1).getImm() + FrameOffset;

    if (static_cast<uint64_t>(Offset) > std::numeric_limits<uint32_t>::max()) {
      // If this happens the program is invalid, but better to error here than
      // generate broken code.
      report_fatal_error("Memory offset field overflow");
    }
    MI.getOperand(1).setImm(Offset);
    MI.getOperand(2).ChangeToRegister(WebAssembly::SP32, /*IsDef=*/false);
  } else {
    // Otherwise create an i32.add SP, offset and make it the operand.
    auto &MRI = MF.getRegInfo();
    const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

    unsigned OffsetReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(WebAssembly::CONST_I32),
            OffsetReg)
        .addImm(FrameOffset);
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(WebAssembly::ADD_I32),
            OffsetReg)
        .addReg(WebAssembly::SP32)
        .addReg(OffsetReg);
    MI.getOperand(FIOperandNum).ChangeToRegister(OffsetReg, /*IsDef=*/false);
  }
}

unsigned
WebAssemblyRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  static const unsigned Regs[2][2] = {
      /*            !isArch64Bit       isArch64Bit      */
      /* !hasFP */ {WebAssembly::SP32, WebAssembly::SP64},
      /*  hasFP */ {WebAssembly::FP32, WebAssembly::FP64}};
  const WebAssemblyFrameLowering *TFI = getFrameLowering(MF);
  return Regs[TFI->hasFP(MF)][TT.isArch64Bit()];
}

const TargetRegisterClass *
WebAssemblyRegisterInfo::getPointerRegClass(const MachineFunction &MF,
                                            unsigned Kind) const {
  assert(Kind == 0 && "Only one kind of pointer on WebAssembly");
  if (MF.getSubtarget<WebAssemblySubtarget>().hasAddr64())
    return &WebAssembly::I64RegClass;
  return &WebAssembly::I32RegClass;
}
