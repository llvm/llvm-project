//===-- SystemZFrameLowering.cpp - Frame lowering for SystemZ -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZFrameLowering.h"
#include "SystemZCallingConv.h"
#include "SystemZInstrBuilder.h"
#include "SystemZInstrInfo.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZRegisterInfo.h"
#include "SystemZSubtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace {
// The ABI-defined register save slots, relative to the CFA (i.e.
// incoming stack pointer + SystemZMC::CallFrameSize).
static const TargetFrameLowering::SpillSlot SpillOffsetTable[] = {
  { SystemZ::R2D,  0x10 },
  { SystemZ::R3D,  0x18 },
  { SystemZ::R4D,  0x20 },
  { SystemZ::R5D,  0x28 },
  { SystemZ::R6D,  0x30 },
  { SystemZ::R7D,  0x38 },
  { SystemZ::R8D,  0x40 },
  { SystemZ::R9D,  0x48 },
  { SystemZ::R10D, 0x50 },
  { SystemZ::R11D, 0x58 },
  { SystemZ::R12D, 0x60 },
  { SystemZ::R13D, 0x68 },
  { SystemZ::R14D, 0x70 },
  { SystemZ::R15D, 0x78 },
  { SystemZ::F0D,  0x80 },
  { SystemZ::F2D,  0x88 },
  { SystemZ::F4D,  0x90 },
  { SystemZ::F6D,  0x98 }
};
} // end anonymous namespace

SystemZFrameLowering::SystemZFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8),
                          0, Align(8), false /* StackRealignable */),
      RegSpillOffsets(0) {
  // Due to the SystemZ ABI, the DWARF CFA (Canonical Frame Address) is not
  // equal to the incoming stack pointer, but to incoming stack pointer plus
  // 160.  Instead of using a Local Area Offset, the Register save area will
  // be occupied by fixed frame objects, and all offsets are actually
  // relative to CFA.

  // Create a mapping from register number to save slot offset.
  // These offsets are relative to the start of the register save area.
  RegSpillOffsets.grow(SystemZ::NUM_TARGET_REGS);
  for (unsigned I = 0, E = array_lengthof(SpillOffsetTable); I != E; ++I)
    RegSpillOffsets[SpillOffsetTable[I].Reg] = SpillOffsetTable[I].Offset;
}

bool SystemZFrameLowering::
assignCalleeSavedSpillSlots(MachineFunction &MF,
                            const TargetRegisterInfo *TRI,
                            std::vector<CalleeSavedInfo> &CSI) const {
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  bool IsVarArg = MF.getFunction().isVarArg();
  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  unsigned LowGPR = 0;
  unsigned HighGPR = SystemZ::R15D;
  int StartSPOffset = SystemZMC::CallFrameSize;
  for (auto &CS : CSI) {
    unsigned Reg = CS.getReg();
    int Offset = getRegSpillOffset(MF, Reg);
    if (Offset) {
      if (SystemZ::GR64BitRegClass.contains(Reg) && StartSPOffset > Offset) {
        LowGPR = Reg;
        StartSPOffset = Offset;
      }
      Offset -= SystemZMC::CallFrameSize;
      int FrameIdx = MFFrame.CreateFixedSpillStackObject(8, Offset);
      CS.setFrameIdx(FrameIdx);
    } else
      CS.setFrameIdx(INT32_MAX);
  }

  // Save the range of call-saved registers, for use by the
  // prologue/epilogue inserters.
  ZFI->setRestoreGPRRegs(LowGPR, HighGPR, StartSPOffset);
  if (IsVarArg) {
    // Also save the GPR varargs, if any.  R6D is call-saved, so would
    // already be included, but we also need to handle the call-clobbered
    // argument registers.
    unsigned FirstGPR = ZFI->getVarArgsFirstGPR();
    if (FirstGPR < SystemZ::NumArgGPRs) {
      unsigned Reg = SystemZ::ArgGPRs[FirstGPR];
      int Offset = getRegSpillOffset(MF, Reg);
      if (StartSPOffset > Offset) {
        LowGPR = Reg; StartSPOffset = Offset;
      }
    }
  }
  ZFI->setSpillGPRRegs(LowGPR, HighGPR, StartSPOffset);

  // Create fixed stack objects for the remaining registers.
  int CurrOffset = -SystemZMC::CallFrameSize;
  if (usePackedStack(MF))
    CurrOffset += StartSPOffset;

  for (auto &CS : CSI) {
    if (CS.getFrameIdx() != INT32_MAX)
      continue;
    unsigned Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    unsigned Size = TRI->getSpillSize(*RC);
    CurrOffset -= Size;
    assert(CurrOffset % 8 == 0 &&
           "8-byte alignment required for for all register save slots");
    int FrameIdx = MFFrame.CreateFixedSpillStackObject(Size, CurrOffset);
    CS.setFrameIdx(FrameIdx);
  }

  return true;
}

void SystemZFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                BitVector &SavedRegs,
                                                RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  bool HasFP = hasFP(MF);
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  bool IsVarArg = MF.getFunction().isVarArg();

  // va_start stores incoming FPR varargs in the normal way, but delegates
  // the saving of incoming GPR varargs to spillCalleeSavedRegisters().
  // Record these pending uses, which typically include the call-saved
  // argument register R6D.
  if (IsVarArg)
    for (unsigned I = MFI->getVarArgsFirstGPR(); I < SystemZ::NumArgGPRs; ++I)
      SavedRegs.set(SystemZ::ArgGPRs[I]);

  // If there are any landing pads, entering them will modify r6/r7.
  if (!MF.getLandingPads().empty()) {
    SavedRegs.set(SystemZ::R6D);
    SavedRegs.set(SystemZ::R7D);
  }

  // If the function requires a frame pointer, record that the hard
  // frame pointer will be clobbered.
  if (HasFP)
    SavedRegs.set(SystemZ::R11D);

  // If the function calls other functions, record that the return
  // address register will be clobbered.
  if (MFFrame.hasCalls())
    SavedRegs.set(SystemZ::R14D);

  // If we are saving GPRs other than the stack pointer, we might as well
  // save and restore the stack pointer at the same time, via STMG and LMG.
  // This allows the deallocation to be done by the LMG, rather than needing
  // a separate %r15 addition.
  const MCPhysReg *CSRegs = TRI->getCalleeSavedRegs(&MF);
  for (unsigned I = 0; CSRegs[I]; ++I) {
    unsigned Reg = CSRegs[I];
    if (SystemZ::GR64BitRegClass.contains(Reg) && SavedRegs.test(Reg)) {
      SavedRegs.set(SystemZ::R15D);
      break;
    }
  }
}

// Add GPR64 to the save instruction being built by MIB, which is in basic
// block MBB.  IsImplicit says whether this is an explicit operand to the
// instruction, or an implicit one that comes between the explicit start
// and end registers.
static void addSavedGPR(MachineBasicBlock &MBB, MachineInstrBuilder &MIB,
                        unsigned GPR64, bool IsImplicit) {
  const TargetRegisterInfo *RI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  Register GPR32 = RI->getSubReg(GPR64, SystemZ::subreg_l32);
  bool IsLive = MBB.isLiveIn(GPR64) || MBB.isLiveIn(GPR32);
  if (!IsLive || !IsImplicit) {
    MIB.addReg(GPR64, getImplRegState(IsImplicit) | getKillRegState(!IsLive));
    if (!IsLive)
      MBB.addLiveIn(GPR64);
  }
}

bool SystemZFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  bool IsVarArg = MF.getFunction().isVarArg();
  DebugLoc DL;

  // Save GPRs
  SystemZ::GPRRegs SpillGPRs = ZFI->getSpillGPRRegs();
  if (SpillGPRs.LowGPR) {
    assert(SpillGPRs.LowGPR != SpillGPRs.HighGPR &&
           "Should be saving %r15 and something else");

    // Build an STMG instruction.
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(SystemZ::STMG));

    // Add the explicit register operands.
    addSavedGPR(MBB, MIB, SpillGPRs.LowGPR, false);
    addSavedGPR(MBB, MIB, SpillGPRs.HighGPR, false);

    // Add the address.
    MIB.addReg(SystemZ::R15D).addImm(SpillGPRs.GPROffset);

    // Make sure all call-saved GPRs are included as operands and are
    // marked as live on entry.
    for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
      unsigned Reg = CSI[I].getReg();
      if (SystemZ::GR64BitRegClass.contains(Reg))
        addSavedGPR(MBB, MIB, Reg, true);
    }

    // ...likewise GPR varargs.
    if (IsVarArg)
      for (unsigned I = ZFI->getVarArgsFirstGPR(); I < SystemZ::NumArgGPRs; ++I)
        addSavedGPR(MBB, MIB, SystemZ::ArgGPRs[I], true);
  }

  // Save FPRs/VRs in the normal TargetInstrInfo way.
  for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
    unsigned Reg = CSI[I].getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, CSI[I].getFrameIdx(),
                               &SystemZ::FP64BitRegClass, TRI);
    }
    if (SystemZ::VR128BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, CSI[I].getFrameIdx(),
                               &SystemZ::VR128BitRegClass, TRI);
    }
  }

  return true;
}

bool SystemZFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  bool HasFP = hasFP(MF);
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Restore FPRs/VRs in the normal TargetInstrInfo way.
  for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
    unsigned Reg = CSI[I].getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg))
      TII->loadRegFromStackSlot(MBB, MBBI, Reg, CSI[I].getFrameIdx(),
                                &SystemZ::FP64BitRegClass, TRI);
    if (SystemZ::VR128BitRegClass.contains(Reg))
      TII->loadRegFromStackSlot(MBB, MBBI, Reg, CSI[I].getFrameIdx(),
                                &SystemZ::VR128BitRegClass, TRI);
  }

  // Restore call-saved GPRs (but not call-clobbered varargs, which at
  // this point might hold return values).
  SystemZ::GPRRegs RestoreGPRs = ZFI->getRestoreGPRRegs();
  if (RestoreGPRs.LowGPR) {
    // If we saved any of %r2-%r5 as varargs, we should also be saving
    // and restoring %r6.  If we're saving %r6 or above, we should be
    // restoring it too.
    assert(RestoreGPRs.LowGPR != RestoreGPRs.HighGPR &&
           "Should be loading %r15 and something else");

    // Build an LMG instruction.
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(SystemZ::LMG));

    // Add the explicit register operands.
    MIB.addReg(RestoreGPRs.LowGPR, RegState::Define);
    MIB.addReg(RestoreGPRs.HighGPR, RegState::Define);

    // Add the address.
    MIB.addReg(HasFP ? SystemZ::R11D : SystemZ::R15D);
    MIB.addImm(RestoreGPRs.GPROffset);

    // Do a second scan adding regs as being defined by instruction
    for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
      unsigned Reg = CSI[I].getReg();
      if (Reg != RestoreGPRs.LowGPR && Reg != RestoreGPRs.HighGPR &&
          SystemZ::GR64BitRegClass.contains(Reg))
        MIB.addReg(Reg, RegState::ImplicitDefine);
    }
  }

  return true;
}

void SystemZFrameLowering::
processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                    RegScavenger *RS) const {
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  bool BackChain = MF.getFunction().hasFnAttribute("backchain");

  if (!usePackedStack(MF) || BackChain)
    // Create the incoming register save area.
    getOrCreateFramePointerSaveIndex(MF);

  // Get the size of our stack frame to be allocated ...
  uint64_t StackSize = (MFFrame.estimateStackSize(MF) +
                        SystemZMC::CallFrameSize);
  // ... and the maximum offset we may need to reach into the
  // caller's frame to access the save area or stack arguments.
  int64_t MaxArgOffset = 0;
  for (int I = MFFrame.getObjectIndexBegin(); I != 0; ++I)
    if (MFFrame.getObjectOffset(I) >= 0) {
      int64_t ArgOffset = MFFrame.getObjectOffset(I) +
                          MFFrame.getObjectSize(I);
      MaxArgOffset = std::max(MaxArgOffset, ArgOffset);
    }

  uint64_t MaxReach = StackSize + MaxArgOffset;
  if (!isUInt<12>(MaxReach)) {
    // We may need register scavenging slots if some parts of the frame
    // are outside the reach of an unsigned 12-bit displacement.
    // Create 2 for the case where both addresses in an MVC are
    // out of range.
    RS->addScavengingFrameIndex(MFFrame.CreateStackObject(8, 8, false));
    RS->addScavengingFrameIndex(MFFrame.CreateStackObject(8, 8, false));
  }
}

// Emit instructions before MBBI (in MBB) to add NumBytes to Reg.
static void emitIncrement(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &MBBI,
                          const DebugLoc &DL,
                          unsigned Reg, int64_t NumBytes,
                          const TargetInstrInfo *TII) {
  while (NumBytes) {
    unsigned Opcode;
    int64_t ThisVal = NumBytes;
    if (isInt<16>(NumBytes))
      Opcode = SystemZ::AGHI;
    else {
      Opcode = SystemZ::AGFI;
      // Make sure we maintain 8-byte stack alignment.
      int64_t MinVal = -uint64_t(1) << 31;
      int64_t MaxVal = (int64_t(1) << 31) - 8;
      if (ThisVal < MinVal)
        ThisVal = MinVal;
      else if (ThisVal > MaxVal)
        ThisVal = MaxVal;
    }
    MachineInstr *MI = BuildMI(MBB, MBBI, DL, TII->get(Opcode), Reg)
      .addReg(Reg).addImm(ThisVal);
    // The CC implicit def is dead.
    MI->getOperand(3).setIsDead();
    NumBytes -= ThisVal;
  }
}

void SystemZFrameLowering::emitPrologue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  auto *ZII =
      static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFFrame.getCalleeSavedInfo();
  bool HasFP = hasFP(MF);

  // In GHC calling convention C stack space, including the ABI-defined
  // 160-byte base area, is (de)allocated by GHC itself.  This stack space may
  // be used by LLVM as spill slots for the tail recursive GHC functions.  Thus
  // do not allocate stack space here, too.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC) {
    if (MFFrame.getStackSize() > 2048 * sizeof(long)) {
      report_fatal_error(
          "Pre allocated stack space for GHC function is too small");
    }
    if (HasFP) {
      report_fatal_error(
          "In GHC calling convention a frame pointer is not supported");
    }
    MFFrame.setStackSize(MFFrame.getStackSize() + SystemZMC::CallFrameSize);
    return;
  }

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // The current offset of the stack pointer from the CFA.
  int64_t SPOffsetFromCFA = -SystemZMC::CFAOffsetFromInitialSP;

  if (ZFI->getSpillGPRRegs().LowGPR) {
    // Skip over the GPR saves.
    if (MBBI != MBB.end() && MBBI->getOpcode() == SystemZ::STMG)
      ++MBBI;
    else
      llvm_unreachable("Couldn't skip over GPR saves");

    // Add CFI for the GPR saves.
    for (auto &Save : CSI) {
      unsigned Reg = Save.getReg();
      if (SystemZ::GR64BitRegClass.contains(Reg)) {
        int FI = Save.getFrameIdx();
        int64_t Offset = MFFrame.getObjectOffset(FI);
        unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, MRI->getDwarfRegNum(Reg, true), Offset));
        BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
    }
  }

  uint64_t StackSize = MFFrame.getStackSize();
  // We need to allocate the ABI-defined 160-byte base area whenever
  // we allocate stack space for our own use and whenever we call another
  // function.
  bool HasStackObject = false;
  for (unsigned i = 0, e = MFFrame.getObjectIndexEnd(); i != e; ++i)
    if (!MFFrame.isDeadObjectIndex(i)) {
      HasStackObject = true;
      break;
    }
  if (HasStackObject || MFFrame.hasCalls())
    StackSize += SystemZMC::CallFrameSize;
  // Don't allocate the incoming reg save area.
  StackSize = StackSize > SystemZMC::CallFrameSize
                  ? StackSize - SystemZMC::CallFrameSize
                  : 0;
  MFFrame.setStackSize(StackSize);

  if (StackSize) {
    // Determine if we want to store a backchain.
    bool StoreBackchain = MF.getFunction().hasFnAttribute("backchain");

    // If we need backchain, save current stack pointer.  R1 is free at this
    // point.
    if (StoreBackchain)
      BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::LGR))
        .addReg(SystemZ::R1D, RegState::Define).addReg(SystemZ::R15D);

    // Allocate StackSize bytes.
    int64_t Delta = -int64_t(StackSize);
    emitIncrement(MBB, MBBI, DL, SystemZ::R15D, Delta, ZII);

    // Add CFI for the allocation.
    unsigned CFIIndex = MF.addFrameInst(
        MCCFIInstruction::createDefCfaOffset(nullptr, SPOffsetFromCFA + Delta));
    BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);
    SPOffsetFromCFA += Delta;

    if (StoreBackchain) {
      // The back chain is stored topmost with packed-stack.
      int Offset = usePackedStack(MF) ? SystemZMC::CallFrameSize - 8 : 0;
      BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::STG))
        .addReg(SystemZ::R1D, RegState::Kill).addReg(SystemZ::R15D)
        .addImm(Offset).addReg(0);
    }
  }

  if (HasFP) {
    // Copy the base of the frame to R11.
    BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::LGR), SystemZ::R11D)
      .addReg(SystemZ::R15D);

    // Add CFI for the new frame location.
    unsigned HardFP = MRI->getDwarfRegNum(SystemZ::R11D, true);
    unsigned CFIIndex = MF.addFrameInst(
        MCCFIInstruction::createDefCfaRegister(nullptr, HardFP));
    BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);

    // Mark the FramePtr as live at the beginning of every block except
    // the entry block.  (We'll have marked R11 as live on entry when
    // saving the GPRs.)
    for (auto I = std::next(MF.begin()), E = MF.end(); I != E; ++I)
      I->addLiveIn(SystemZ::R11D);
  }

  // Skip over the FPR/VR saves.
  SmallVector<unsigned, 8> CFIIndexes;
  for (auto &Save : CSI) {
    unsigned Reg = Save.getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      if (MBBI != MBB.end() &&
          (MBBI->getOpcode() == SystemZ::STD ||
           MBBI->getOpcode() == SystemZ::STDY))
        ++MBBI;
      else
        llvm_unreachable("Couldn't skip over FPR save");
    } else if (SystemZ::VR128BitRegClass.contains(Reg)) {
      if (MBBI != MBB.end() &&
          MBBI->getOpcode() == SystemZ::VST)
        ++MBBI;
      else
        llvm_unreachable("Couldn't skip over VR save");
    } else
      continue;

    // Add CFI for the this save.
    unsigned DwarfReg = MRI->getDwarfRegNum(Reg, true);
    unsigned IgnoredFrameReg;
    int64_t Offset =
        getFrameIndexReference(MF, Save.getFrameIdx(), IgnoredFrameReg);

    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
          nullptr, DwarfReg, SPOffsetFromCFA + Offset));
    CFIIndexes.push_back(CFIIndex);
  }
  // Complete the CFI for the FPR/VR saves, modelling them as taking effect
  // after the last save.
  for (auto CFIIndex : CFIIndexes) {
    BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);
  }
}

void SystemZFrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  auto *ZII =
      static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();

  // See SystemZFrameLowering::emitPrologue
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // Skip the return instruction.
  assert(MBBI->isReturn() && "Can only insert epilogue into returning blocks");

  uint64_t StackSize = MFFrame.getStackSize();
  if (ZFI->getRestoreGPRRegs().LowGPR) {
    --MBBI;
    unsigned Opcode = MBBI->getOpcode();
    if (Opcode != SystemZ::LMG)
      llvm_unreachable("Expected to see callee-save register restore code");

    unsigned AddrOpNo = 2;
    DebugLoc DL = MBBI->getDebugLoc();
    uint64_t Offset = StackSize + MBBI->getOperand(AddrOpNo + 1).getImm();
    unsigned NewOpcode = ZII->getOpcodeForOffset(Opcode, Offset);

    // If the offset is too large, use the largest stack-aligned offset
    // and add the rest to the base register (the stack or frame pointer).
    if (!NewOpcode) {
      uint64_t NumBytes = Offset - 0x7fff8;
      emitIncrement(MBB, MBBI, DL, MBBI->getOperand(AddrOpNo).getReg(),
                    NumBytes, ZII);
      Offset -= NumBytes;
      NewOpcode = ZII->getOpcodeForOffset(Opcode, Offset);
      assert(NewOpcode && "No restore instruction available");
    }

    MBBI->setDesc(ZII->get(NewOpcode));
    MBBI->getOperand(AddrOpNo + 1).ChangeToImmediate(Offset);
  } else if (StackSize) {
    DebugLoc DL = MBBI->getDebugLoc();
    emitIncrement(MBB, MBBI, DL, SystemZ::R15D, StackSize, ZII);
  }
}

bool SystemZFrameLowering::hasFP(const MachineFunction &MF) const {
  return (MF.getTarget().Options.DisableFramePointerElim(MF) ||
          MF.getFrameInfo().hasVarSizedObjects() ||
          MF.getInfo<SystemZMachineFunctionInfo>()->getManipulatesSP());
}

bool
SystemZFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // The ABI requires us to allocate 160 bytes of stack space for the callee,
  // with any outgoing stack arguments being placed above that.  It seems
  // better to make that area a permanent feature of the frame even if
  // we're using a frame pointer.
  return true;
}

int SystemZFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                 int FI,
                                                 unsigned &FrameReg) const {
  // Our incoming SP is actually SystemZMC::CallFrameSize below the CFA, so
  // add that difference here.
  int64_t Offset =
    TargetFrameLowering::getFrameIndexReference(MF, FI, FrameReg);
  return Offset + SystemZMC::CallFrameSize;
}

MachineBasicBlock::iterator SystemZFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF,
                              MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI) const {
  switch (MI->getOpcode()) {
  case SystemZ::ADJCALLSTACKDOWN:
  case SystemZ::ADJCALLSTACKUP:
    assert(hasReservedCallFrame(MF) &&
           "ADJSTACKDOWN and ADJSTACKUP should be no-ops");
    return MBB.erase(MI);
    break;

  default:
    llvm_unreachable("Unexpected call frame instruction");
  }
}

unsigned SystemZFrameLowering::getRegSpillOffset(MachineFunction &MF,
                                                 unsigned Reg) const {
  bool IsVarArg = MF.getFunction().isVarArg();
  bool BackChain = MF.getFunction().hasFnAttribute("backchain");
  bool SoftFloat = MF.getSubtarget<SystemZSubtarget>().hasSoftFloat();
  unsigned Offset = RegSpillOffsets[Reg];
  if (usePackedStack(MF) && !(IsVarArg && !SoftFloat)) {
    if (SystemZ::GR64BitRegClass.contains(Reg))
      // Put all GPRs at the top of the Register save area with packed
      // stack. Make room for the backchain if needed.
      Offset += BackChain ? 24 : 32;
    else
      Offset = 0;
  }
  return Offset;
}

int SystemZFrameLowering::
getOrCreateFramePointerSaveIndex(MachineFunction &MF) const {
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  int FI = ZFI->getFramePointerSaveIndex();
  if (!FI) {
    MachineFrameInfo &MFFrame = MF.getFrameInfo();
    // The back chain is stored topmost with packed-stack.
    int Offset = usePackedStack(MF) ? -8 : -SystemZMC::CallFrameSize;
    FI = MFFrame.CreateFixedObject(8, Offset, false);
    ZFI->setFramePointerSaveIndex(FI);
  }
  return FI;
}

bool SystemZFrameLowering::usePackedStack(MachineFunction &MF) const {
  bool HasPackedStackAttr = MF.getFunction().hasFnAttribute("packed-stack");
  bool BackChain = MF.getFunction().hasFnAttribute("backchain");
  bool SoftFloat = MF.getSubtarget<SystemZSubtarget>().hasSoftFloat();
  if (HasPackedStackAttr && BackChain && !SoftFloat)
    report_fatal_error("packed-stack + backchain + hard-float is unsupported.");
  bool CallConv = MF.getFunction().getCallingConv() != CallingConv::GHC;
  return HasPackedStackAttr && CallConv;
}
