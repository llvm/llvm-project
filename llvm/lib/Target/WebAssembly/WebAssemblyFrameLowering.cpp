//===-- WebAssemblyFrameLowering.cpp - WebAssembly Frame Lowering ----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of
/// TargetFrameLowering class.
///
/// On WebAssembly, there aren't a lot of things to do here. There are no
/// callee-saved registers to save, and no spill slots.
///
/// The stack grows downward.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyFrameLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyInstrInfo.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-frame-info"

// TODO: Implement a red zone?
// TODO: wasm64
// TODO: Prolog/epilog should be stackified too. This pass runs after register
//       stackification, so we'll have to do it manually.
// TODO: Emit TargetOpcode::CFI_INSTRUCTION instructions

/// Return true if the specified function should have a dedicated frame pointer
/// register.
bool WebAssemblyFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const auto *RegInfo =
      MF.getSubtarget<WebAssemblySubtarget>().getRegisterInfo();
  return MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken() ||
         MFI->hasStackMap() || MFI->hasPatchPoint() ||
         RegInfo->needsStackRealignment(MF);
}

/// Under normal circumstances, when a frame pointer is not required, we reserve
/// argument space for call sites in the function immediately on entry to the
/// current function. This eliminates the need for add/sub sp brackets around
/// call sites. Returns true if the call frame is included as part of the stack
/// frame.
bool WebAssemblyFrameLowering::hasReservedCallFrame(
    const MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}


/// Adjust the stack pointer by a constant amount.
static void adjustStackPointer(unsigned StackSize,
                               bool AdjustUp,
                               MachineFunction& MF,
                               MachineBasicBlock& MBB,
                               const TargetInstrInfo* TII,
                               MachineBasicBlock::iterator InsertPt,
                               const DebugLoc& DL) {
  auto &MRI = MF.getRegInfo();
  unsigned SPReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  auto *SPSymbol = MF.createExternalSymbolName("__stack_pointer");
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::CONST_I32), SPReg)
      .addExternalSymbol(SPSymbol);
  // This MachinePointerInfo should reference __stack_pointer as well but
  // doesn't because MachinePointerInfo() takes a GV which we don't have for
  // __stack_pointer. TODO: check if PseudoSourceValue::ExternalSymbolCallEntry
  // is appropriate instead. (likewise for EmitEpologue below)
  auto *LoadMMO = new MachineMemOperand(MachinePointerInfo(),
                                        MachineMemOperand::MOLoad, 4, 4);
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::LOAD_I32), SPReg)
      .addImm(0)
      .addReg(SPReg)
      .addMemOperand(LoadMMO);
  // Add/Subtract the frame size
  unsigned OffsetReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::CONST_I32), OffsetReg)
      .addImm(StackSize);
  BuildMI(MBB, InsertPt, DL,
          TII->get(AdjustUp ? WebAssembly::ADD_I32 : WebAssembly::SUB_I32),
          WebAssembly::SP32)
      .addReg(SPReg)
      .addReg(OffsetReg);
  // The SP32 register now has the new stacktop. Also write it back to memory.
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::CONST_I32), OffsetReg)
      .addExternalSymbol(SPSymbol);
  auto *MMO = new MachineMemOperand(MachinePointerInfo(),
                                    MachineMemOperand::MOStore, 4, 4);
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::STORE_I32), WebAssembly::SP32)
      .addImm(0)
      .addReg(OffsetReg)
      .addReg(WebAssembly::SP32)
      .addMemOperand(MMO);
}

void WebAssemblyFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const auto *TII =
      static_cast<const WebAssemblyInstrInfo*>(MF.getSubtarget().getInstrInfo());
  DebugLoc DL = I->getDebugLoc();
  unsigned Opc = I->getOpcode();
  bool IsDestroy = Opc == TII->getCallFrameDestroyOpcode();
  unsigned Amount = I->getOperand(0).getImm();
  if (Amount)
    adjustStackPointer(Amount, IsDestroy, MF, MBB,
                       TII, I, DL);
  MBB.erase(I);
}

void WebAssemblyFrameLowering::emitPrologue(MachineFunction &MF,
                                            MachineBasicBlock &MBB) const {
  // TODO: Do ".setMIFlag(MachineInstr::FrameSetup)" on emitted instructions
  auto *MFI = MF.getFrameInfo();
  assert(MFI->getCalleeSavedInfo().empty() &&
         "WebAssembly should not have callee-saved registers");
  assert(!hasFP(MF) && "Functions needing frame pointers not yet supported");
  uint64_t StackSize = MFI->getStackSize();
  if (!StackSize && (!MFI->adjustsStack() || MFI->getMaxCallFrameSize() == 0))
    return;

  const auto *TII = MF.getSubtarget().getInstrInfo();

  auto InsertPt = MBB.begin();
  DebugLoc DL;

  adjustStackPointer(StackSize, false, MF, MBB, TII, InsertPt, DL);
}

void WebAssemblyFrameLowering::emitEpilogue(MachineFunction &MF,
                                            MachineBasicBlock &MBB) const {
  uint64_t StackSize = MF.getFrameInfo()->getStackSize();
  if (!StackSize)
    return;
  const auto *TII = MF.getSubtarget().getInstrInfo();
  auto &MRI = MF.getRegInfo();
  unsigned OffsetReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  auto InsertPt = MBB.getFirstTerminator();
  DebugLoc DL;

  if (InsertPt != MBB.end()) {
    DL = InsertPt->getDebugLoc();
  }

  // Restore the stack pointer. Without FP its value is just SP32 - stacksize
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::CONST_I32), OffsetReg)
      .addImm(StackSize);
  auto *SPSymbol = MF.createExternalSymbolName("__stack_pointer");
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::ADD_I32), WebAssembly::SP32)
      .addReg(WebAssembly::SP32)
      .addReg(OffsetReg);
  // Re-use OffsetReg to hold the address of the stacktop
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::CONST_I32), OffsetReg)
      .addExternalSymbol(SPSymbol);
  auto *MMO = new MachineMemOperand(MachinePointerInfo(),
                                    MachineMemOperand::MOStore, 4, 4);
  BuildMI(MBB, InsertPt, DL, TII->get(WebAssembly::STORE_I32), WebAssembly::SP32)
      .addImm(0)
      .addReg(OffsetReg)
      .addReg(WebAssembly::SP32)
      .addMemOperand(MMO);
}
