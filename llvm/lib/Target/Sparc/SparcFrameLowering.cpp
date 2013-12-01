//===-- SparcFrameLowering.cpp - Sparc Frame Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "SparcFrameLowering.h"
#include "SparcInstrInfo.h"
#include "SparcMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

static cl::opt<bool>
DisableLeafProc("disable-sparc-leaf-proc",
                cl::init(false),
                cl::desc("Disable Sparc leaf procedure optimization."),
                cl::Hidden);


void SparcFrameLowering::emitSPAdjustment(MachineFunction &MF,
                                          MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          int NumBytes,
                                          unsigned ADDrr,
                                          unsigned ADDri) const {

  DebugLoc dl = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());

  if (NumBytes >= -4096 && NumBytes < 4096) {
    BuildMI(MBB, MBBI, dl, TII.get(ADDri), SP::O6)
      .addReg(SP::O6).addImm(NumBytes);
    return;
  }

  // Emit this the hard way.  This clobbers G1 which we always know is
  // available here.
  if (NumBytes >= 0) {
    // Emit nonnegative numbers with sethi + or.
    // sethi %hi(NumBytes), %g1
    // or %g1, %lo(NumBytes), %g1
    // add %sp, %g1, %sp
    BuildMI(MBB, MBBI, dl, TII.get(SP::SETHIi), SP::G1)
      .addImm(HI22(NumBytes));
    BuildMI(MBB, MBBI, dl, TII.get(SP::ORri), SP::G1)
      .addReg(SP::G1).addImm(LO10(NumBytes));
    BuildMI(MBB, MBBI, dl, TII.get(ADDrr), SP::O6)
      .addReg(SP::O6).addReg(SP::G1);
    return ;
  }

  // Emit negative numbers with sethi + xor.
  // sethi %hix(NumBytes), %g1
  // xor %g1, %lox(NumBytes), %g1
  // add %sp, %g1, %sp
  BuildMI(MBB, MBBI, dl, TII.get(SP::SETHIi), SP::G1)
    .addImm(HIX22(NumBytes));
  BuildMI(MBB, MBBI, dl, TII.get(SP::XORri), SP::G1)
    .addReg(SP::G1).addImm(LOX10(NumBytes));
  BuildMI(MBB, MBBI, dl, TII.get(ADDrr), SP::O6)
    .addReg(SP::O6).addReg(SP::G1);
}

void SparcFrameLowering::emitPrologue(MachineFunction &MF) const {
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();

  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  unsigned SAVEri = SP::SAVEri;
  unsigned SAVErr = SP::SAVErr;
  if (FuncInfo->isLeafProc()) {
    if (NumBytes == 0)
      return;
    SAVEri = SP::ADDri;
    SAVErr = SP::ADDrr;
  }
  NumBytes = - SubTarget.getAdjustedFrameSize(NumBytes);
  emitSPAdjustment(MF, MBB, MBBI, NumBytes, SAVErr, SAVEri);

  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  MCSymbol *FrameLabel = MMI.getContext().CreateTempSymbol();
  BuildMI(MBB, MBBI, dl, TII.get(SP::PROLOG_LABEL)).addSym(FrameLabel);

  unsigned regFP = MRI->getDwarfRegNum(SP::I6, true);

  // Emit ".cfi_def_cfa_register 30".
  MMI.addFrameInst(MCCFIInstruction::createDefCfaRegister(FrameLabel,
                                                          regFP));
  // Emit ".cfi_window_save".
  MMI.addFrameInst(MCCFIInstruction::createWindowSave(FrameLabel));

  unsigned regInRA = MRI->getDwarfRegNum(SP::I7, true);
  unsigned regOutRA = MRI->getDwarfRegNum(SP::O7, true);
  // Emit ".cfi_register 15, 31".
  MMI.addFrameInst(MCCFIInstruction::createRegister(FrameLabel,
                                                    regOutRA,
                                                    regInRA));
}

void SparcFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    MachineInstr &MI = *I;
    int Size = MI.getOperand(0).getImm();
    if (MI.getOpcode() == SP::ADJCALLSTACKDOWN)
      Size = -Size;

    if (Size)
      emitSPAdjustment(MF, MBB, I, Size, SP::ADDrr, SP::ADDri);
  }
  MBB.erase(I);
}


void SparcFrameLowering::emitEpilogue(MachineFunction &MF,
                                  MachineBasicBlock &MBB) const {
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SparcInstrInfo &TII =
    *static_cast<const SparcInstrInfo*>(MF.getTarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  assert(MBBI->getOpcode() == SP::RETL &&
         "Can only put epilog before 'retl' instruction!");
  if (!FuncInfo->isLeafProc()) {
    BuildMI(MBB, MBBI, dl, TII.get(SP::RESTORErr), SP::G0).addReg(SP::G0)
      .addReg(SP::G0);
    return;
  }
  MachineFrameInfo *MFI = MF.getFrameInfo();

  int NumBytes = (int) MFI->getStackSize();
  if (NumBytes == 0)
    return;

  NumBytes = SubTarget.getAdjustedFrameSize(NumBytes);
  emitSPAdjustment(MF, MBB, MBBI, NumBytes, SP::ADDrr, SP::ADDri);
}

bool SparcFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // Reserve call frame if there are no variable sized objects on the stack.
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool SparcFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
    MFI->hasVarSizedObjects() || MFI->isFrameAddressTaken();
}


static bool LLVM_ATTRIBUTE_UNUSED verifyLeafProcRegUse(MachineRegisterInfo *MRI)
{

  for (unsigned reg = SP::I0; reg <= SP::I7; ++reg)
    if (MRI->isPhysRegUsed(reg))
      return false;

  for (unsigned reg = SP::L0; reg <= SP::L7; ++reg)
    if (MRI->isPhysRegUsed(reg))
      return false;

  return true;
}

bool SparcFrameLowering::isLeafProc(MachineFunction &MF) const
{

  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo    *MFI = MF.getFrameInfo();

  return !(MFI->hasCalls()              // has calls
           || MRI.isPhysRegUsed(SP::L0) // Too many registers needed
           || MRI.isPhysRegUsed(SP::O6) // %SP is used
           || hasFP(MF));               // need %FP
}

void SparcFrameLowering::remapRegsForLeafProc(MachineFunction &MF) const {

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Remap %i[0-7] to %o[0-7].
  for (unsigned reg = SP::I0; reg <= SP::I7; ++reg) {
    if (!MRI.isPhysRegUsed(reg))
      continue;
    unsigned mapped_reg = (reg - SP::I0 + SP::O0);
    assert(!MRI.isPhysRegUsed(mapped_reg));

    // Replace I register with O register.
    MRI.replaceRegWith(reg, mapped_reg);

    // Mark the reg unused.
    MRI.setPhysRegUnused(reg);
  }

  // Rewrite MBB's Live-ins.
  for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
       MBB != E; ++MBB) {
    for (unsigned reg = SP::I0; reg <= SP::I7; ++reg) {
      if (!MBB->isLiveIn(reg))
        continue;
      MBB->removeLiveIn(reg);
      MBB->addLiveIn(reg - SP::I0 + SP::O0);
    }
  }

  assert(verifyLeafProcRegUse(&MRI));
#ifdef XDEBUG
  MF.verify(0, "After LeafProc Remapping");
#endif
}

void SparcFrameLowering::processFunctionBeforeCalleeSavedScan
                  (MachineFunction &MF, RegScavenger *RS) const {

  if (!DisableLeafProc && isLeafProc(MF)) {
    SparcMachineFunctionInfo *MFI = MF.getInfo<SparcMachineFunctionInfo>();
    MFI->setLeafProc(true);

    remapRegsForLeafProc(MF);
  }

}
