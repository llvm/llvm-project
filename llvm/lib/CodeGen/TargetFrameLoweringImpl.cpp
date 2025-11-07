//===- TargetFrameLoweringImpl.cpp - Implement target frame interface ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

TargetFrameLowering::~TargetFrameLowering() = default;

bool TargetFrameLowering::enableCalleeSaveSkip(const MachineFunction &MF) const {
  assert(MF.getFunction().hasFnAttribute(Attribute::NoReturn) &&
         MF.getFunction().hasFnAttribute(Attribute::NoUnwind) &&
         !MF.getFunction().hasFnAttribute(Attribute::UWTable));
  return false;
}

bool TargetFrameLowering::enableCFIFixup(const MachineFunction &MF) const {
  return MF.needsFrameMoves() &&
         !MF.getTarget().getMCAsmInfo()->usesWindowsCFI();
}

/// Returns the displacement from the frame register to the stack
/// frame of the specified index, along with the frame register used
/// (in output arg FrameReg). This is the default implementation which
/// is overridden for some targets.
StackOffset
TargetFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                            Register &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();

  // By default, assume all frame indices are referenced via whatever
  // getFrameRegister() says. The target can override this if it's doing
  // something different.
  FrameReg = RI->getFrameRegister(MF);

  return StackOffset::getFixed(MFI.getObjectOffset(FI) + MFI.getStackSize() -
                               getOffsetOfLocalArea() +
                               MFI.getOffsetAdjustment());
}

/// Returns the offset from the stack pointer to the slot of the specified
/// index. This function serves to provide a comparable offset from a single
/// reference point (the value of the stack-pointer at function entry) that can
/// be used for analysis. This is the default implementation using
/// MachineFrameInfo offsets.
StackOffset
TargetFrameLowering::getFrameIndexReferenceFromSP(const MachineFunction &MF,
                                                  int FI) const {
  // To display the true offset from SP, we need to subtract the offset to the
  // local area from MFI's ObjectOffset.
  return StackOffset::getFixed(MF.getFrameInfo().getObjectOffset(FI) -
                               getOffsetOfLocalArea());
}

bool TargetFrameLowering::needsFrameIndexResolution(
    const MachineFunction &MF) const {
  return MF.getFrameInfo().hasStackObjects();
}

void TargetFrameLowering::getCalleeSaves(const MachineFunction &MF,
                                         BitVector &CalleeSaves) const {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  CalleeSaves.resize(TRI.getNumRegs());

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.isCalleeSavedInfoValid())
    return;

  for (const CalleeSavedInfo &Info : MFI.getCalleeSavedInfo())
    CalleeSaves.set(Info.getReg());
}

// LLVM_DEPRECATED("Use determinePrologCalleeSaves instead",
//                 "determinePrologCalleeSaves")
void TargetFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  determinePrologCalleeSaves(MF, SavedRegs, RS);
}

const MCPhysReg *
TargetFrameLowering::getMustPreserveRegisters(const MachineFunction &MF) const {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  // In Naked functions we aren't going to save any registers.
  if (MF.getFunction().hasFnAttribute(Attribute::Naked))
    return nullptr;

  // Noreturn+nounwind functions never restore CSR, so no saves are needed.
  // Purely noreturn functions may still return through throws, so those must
  // save CSR for caller exception handlers.
  //
  // If the function uses longjmp to break out of its current path of
  // execution we do not need the CSR spills either: setjmp stores all CSRs
  // it was called with into the jmp_buf, which longjmp then restores.
  if (MF.getFunction().hasFnAttribute(Attribute::NoReturn) &&
        MF.getFunction().hasFnAttribute(Attribute::NoUnwind) &&
        !MF.getFunction().hasFnAttribute(Attribute::UWTable) &&
        enableCalleeSaveSkip(MF))
    return nullptr;

  // When interprocedural register allocation is enabled, callee saved register
  // list should be empty, since caller saved registers are preferred over
  // callee saved registers. Unless it has some risked CSR to be optimized out.
  if (MF.getTarget().Options.EnableIPRA &&
      isSafeForNoCSROpt(MF.getFunction()) &&
      isProfitableForNoCSROpt(MF.getFunction()))
    return TRI.getIPRACSRegs(&MF);
  return MF.getRegInfo().getCalleeSavedRegs();
}

void TargetFrameLowering::determineUncondPrologCalleeSaves(
    MachineFunction &MF, const MCPhysReg *CSRegs,
    BitVector &UncondPrologCSRs) const {
  // Functions which call __builtin_unwind_init get all their registers saved.
  if (MF.callsUnwindInit()) {
    for (unsigned i = 0; CSRegs[i]; ++i) {
      unsigned Reg = CSRegs[i];
      UncondPrologCSRs.set(Reg);
    }
  }
  return;
}

void TargetFrameLowering::determineEarlyCalleeSaves(
    MachineFunction &MF, BitVector &EarlyCSRs) const {
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  if (!ST.savesCSRsEarly())
    return;

  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  // Get the callee saved register list...
  const MCPhysReg *CSRegs = getMustPreserveRegisters(MF);
  // Early exit if there are no callee saved registers.
  if (!CSRegs || CSRegs[0] == 0)
    return;

  BitVector UncondPrologCSRs(TRI.getNumRegs(), false);
  determineUncondPrologCalleeSaves(MF, CSRegs, UncondPrologCSRs);

  EarlyCSRs.resize(TRI.getNumRegs());
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (!UncondPrologCSRs[Reg])
      EarlyCSRs.set(Reg);
  }
}

void TargetFrameLowering::determinePrologCalleeSaves(MachineFunction &MF,
                                                     BitVector &PrologCSRs,
                                                     RegScavenger *RS) const {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  // Resize before the early returns. Some backends expect that
  // SavedRegs.size() == TRI.getNumRegs() after this call even if there are no
  // saved registers.
  PrologCSRs.resize(TRI.getNumRegs());

  // Get the callee saved register list...
  const MCPhysReg *CSRegs = getMustPreserveRegisters(MF);
  // Early exit if there are no callee saved registers.
  if (!CSRegs || CSRegs[0] == 0)
    return;

  determineUncondPrologCalleeSaves(MF, CSRegs, PrologCSRs);

  BitVector EarlyCSRs(TRI.getNumRegs(), false);
  determineEarlyCalleeSaves(MF, EarlyCSRs);

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (MRI.isPhysRegModified(Reg) && !EarlyCSRs[Reg])
      PrologCSRs.set(Reg);
  }
}

bool TargetFrameLowering::allocateScavengingFrameIndexesNearIncomingSP(
  const MachineFunction &MF) const {
  if (!hasFP(MF))
    return false;

  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  return RegInfo->useFPForScavengingIndex(MF) &&
         !RegInfo->hasStackRealignment(MF);
}

bool TargetFrameLowering::isSafeForNoCSROpt(const Function &F) {
  if (!F.hasLocalLinkage() || F.hasAddressTaken() ||
      !F.hasFnAttribute(Attribute::NoRecurse))
    return false;
  // Function should not be optimized as tail call.
  for (const User *U : F.users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->isTailCall())
        return false;
  return true;
}

int TargetFrameLowering::getInitialCFAOffset(const MachineFunction &MF) const {
  llvm_unreachable("getInitialCFAOffset() not implemented!");
}

Register
TargetFrameLowering::getInitialCFARegister(const MachineFunction &MF) const {
  llvm_unreachable("getInitialCFARegister() not implemented!");
}

TargetFrameLowering::DwarfFrameBase
TargetFrameLowering::getDwarfFrameBase(const MachineFunction &MF) const {
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  return DwarfFrameBase{DwarfFrameBase::Register, {RI->getFrameRegister(MF).id()}};
}

void TargetFrameLowering::spillCalleeSavedRegister(
    MachineBasicBlock &SaveBlock, MachineBasicBlock::iterator MI,
    const CalleeSavedInfo &CS, const TargetInstrInfo *TII,
    const TargetRegisterInfo *TRI) const {
  // Insert the spill to the stack frame.
  MCRegister Reg = CS.getReg();

  if (CS.isSpilledToReg()) {
    BuildMI(SaveBlock, MI, DebugLoc(), TII->get(TargetOpcode::COPY),
            CS.getDstReg())
        .addReg(Reg, getKillRegState(true));
  } else {
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII->storeRegToStackSlot(SaveBlock, MI, Reg, true, CS.getFrameIdx(), RC,
                             Register());
  }
}

void TargetFrameLowering::restoreCalleeSavedRegister(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    const CalleeSavedInfo &CS, const TargetInstrInfo *TII,
    const TargetRegisterInfo *TRI) const {
  MCRegister Reg = CS.getReg();
  if (CS.isSpilledToReg()) {
    BuildMI(MBB, MI, DebugLoc(), TII->get(TargetOpcode::COPY), Reg)
        .addReg(CS.getDstReg(), getKillRegState(true));
  } else {
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII->loadRegFromStackSlot(MBB, MI, Reg, CS.getFrameIdx(), RC, Register());
    assert(MI != MBB.begin() && "loadRegFromStackSlot didn't insert any code!");
  }
}
