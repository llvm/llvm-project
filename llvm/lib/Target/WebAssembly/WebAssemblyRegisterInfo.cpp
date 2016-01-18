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
WebAssemblyRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  for (auto Reg : {WebAssembly::SP32, WebAssembly::SP64, WebAssembly::FP32,
                   WebAssembly::FP64})
    Reserved.set(Reg);
  return Reserved;
}

void WebAssemblyRegisterInfo::eliminateFrameIndex(
    MachineBasicBlock::iterator II, int SPAdj, unsigned FIOperandNum,
    RegScavenger *RS) const {
  llvm_unreachable("WebAssemblyRegisterInfo::eliminateFrameIndex"); // FIXME
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
