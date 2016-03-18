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

static bool isStackifiedVReg(const WebAssemblyFunctionInfo *WFI,
                             const MachineOperand& Op) {
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    return TargetRegisterInfo::isVirtualRegister(Reg) &&
        WFI->isVRegStackified(Reg);
  }
  return false;
}

static bool canStackifyOperand(const MachineInstr& Inst) {
  unsigned Op = Inst.getOpcode();
  return Op != TargetOpcode::PHI &&
      Op != TargetOpcode::INLINEASM &&
      Op != TargetOpcode::DBG_VALUE;
}

// Determine if the FI sequence can be stackified, and if so, where the code can
// be inserted. If stackification is possible, returns true and ajusts II to
// point to the insertion point.
bool findInsertPt(const WebAssemblyFunctionInfo *WFI, MachineBasicBlock &MBB,
                  unsigned OperandNum, MachineBasicBlock::iterator &II) {
  if (!canStackifyOperand(*II)) return false;

  MachineBasicBlock::iterator InsertPt(II);
  int StackCount = 0;
  // Operands are popped in reverse order, so any operands after FIOperand
  // impose a constraint
  for (unsigned i = OperandNum; i < II->getNumOperands(); i++) {
    if (isStackifiedVReg(WFI, II->getOperand(i))) ++StackCount;
  }
  // Walk backwards, tracking stack depth. When it reaches 0 we have reached the
  // top of the subtree.
  while (StackCount) {
    if (InsertPt == MBB.begin()) return false;
    --InsertPt;
    for (const auto &def : InsertPt->defs())
      if (isStackifiedVReg(WFI, def)) --StackCount;
    for (const auto &use : InsertPt->explicit_uses())
      if (isStackifiedVReg(WFI, use)) ++StackCount;
  }
  II = InsertPt;
  return true;
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

  if (MI.mayLoadOrStore() && FIOperandNum == WebAssembly::MemOpAddressOperandNo) {
    // If this is the address operand of a load or store, make it relative to SP
    // and fold the frame offset directly in.
    assert(FrameOffset >= 0 && MI.getOperand(1).getImm() >= 0);
    int64_t Offset = MI.getOperand(1).getImm() + FrameOffset;

    if (static_cast<uint64_t>(Offset) > std::numeric_limits<uint32_t>::max()) {
      // If this happens the program is invalid, but better to error here than
      // generate broken code.
      report_fatal_error("Memory offset field overflow");
    }
    MI.getOperand(FIOperandNum - 1).setImm(Offset);
    MI.getOperand(FIOperandNum)
        .ChangeToRegister(WebAssembly::SP32, /*IsDef=*/false);
  } else {
    // Otherwise calculate the address
    auto &MRI = MF.getRegInfo();
    const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

    unsigned FIRegOperand = WebAssembly::SP32;
    if (FrameOffset) {
      // Create i32.add SP, offset and make it the operand. We want to stackify
      // this sequence, but we need to preserve the LIFO expr stack ordering
      // (i.e. we can't insert our code in between MI and any operands it
      // pops before FIOperand).
      auto *WFI = MF.getInfo<WebAssemblyFunctionInfo>();
      bool CanStackifyFI = findInsertPt(WFI, MBB, FIOperandNum, II);

      unsigned OffsetOp = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
      BuildMI(MBB, *II, II->getDebugLoc(), TII->get(WebAssembly::CONST_I32),
              OffsetOp)
          .addImm(FrameOffset);
      if (CanStackifyFI) {
        WFI->stackifyVReg(OffsetOp);
        FIRegOperand = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
        WFI->stackifyVReg(FIRegOperand);
      } else {
        FIRegOperand = OffsetOp;
      }
      BuildMI(MBB, *II, II->getDebugLoc(), TII->get(WebAssembly::ADD_I32),
              FIRegOperand)
          .addReg(WebAssembly::SP32)
          .addReg(OffsetOp);
    }
    MI.getOperand(FIOperandNum).ChangeToRegister(FIRegOperand, /*IsDef=*/false);
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
