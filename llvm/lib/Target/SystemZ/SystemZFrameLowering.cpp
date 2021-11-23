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
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
// The ABI-defined register save slots, relative to the CFA (i.e.
// incoming stack pointer + SystemZMC::ELFCallFrameSize).
static const TargetFrameLowering::SpillSlot ELFSpillOffsetTable[] = {
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

static const TargetFrameLowering::SpillSlot XPLINKSpillOffsetTable[] = {
    {SystemZ::R4D, 0x00},  {SystemZ::R5D, 0x08},  {SystemZ::R6D, 0x10},
    {SystemZ::R7D, 0x18},  {SystemZ::R8D, 0x20},  {SystemZ::R9D, 0x28},
    {SystemZ::R10D, 0x30}, {SystemZ::R11D, 0x38}, {SystemZ::R12D, 0x40},
    {SystemZ::R13D, 0x48}, {SystemZ::R14D, 0x50}, {SystemZ::R15D, 0x58}};
} // end anonymous namespace

SystemZFrameLowering::SystemZFrameLowering(StackDirection D, Align StackAl,
                                           int LAO, Align TransAl,
                                           bool StackReal)
    : TargetFrameLowering(D, StackAl, LAO, TransAl, StackReal) {}

std::unique_ptr<SystemZFrameLowering>
SystemZFrameLowering::create(const SystemZSubtarget &STI) {
  if (STI.isTargetXPLINK64())
    return std::make_unique<SystemZXPLINKFrameLowering>();
  return std::make_unique<SystemZELFFrameLowering>();
}

MachineBasicBlock::iterator SystemZFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
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

bool SystemZFrameLowering::hasReservedCallFrame(
    const MachineFunction &MF) const {
  // The ELF ABI requires us to allocate 160 bytes of stack space for the
  // callee, with any outgoing stack arguments being placed above that. It
  // seems better to make that area a permanent feature of the frame even if
  // we're using a frame pointer. Similarly, 64-bit XPLINK requires 96 bytes
  // of stack space for the register save area.
  return true;
}

bool SystemZELFFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  bool IsVarArg = MF.getFunction().isVarArg();
  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  unsigned LowGPR = 0;
  unsigned HighGPR = SystemZ::R15D;
  int StartSPOffset = SystemZMC::ELFCallFrameSize;
  for (auto &CS : CSI) {
    unsigned Reg = CS.getReg();
    int Offset = getRegSpillOffset(MF, Reg);
    if (Offset) {
      if (SystemZ::GR64BitRegClass.contains(Reg) && StartSPOffset > Offset) {
        LowGPR = Reg;
        StartSPOffset = Offset;
      }
      Offset -= SystemZMC::ELFCallFrameSize;
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
    if (FirstGPR < SystemZ::ELFNumArgGPRs) {
      unsigned Reg = SystemZ::ELFArgGPRs[FirstGPR];
      int Offset = getRegSpillOffset(MF, Reg);
      if (StartSPOffset > Offset) {
        LowGPR = Reg; StartSPOffset = Offset;
      }
    }
  }
  ZFI->setSpillGPRRegs(LowGPR, HighGPR, StartSPOffset);

  // Create fixed stack objects for the remaining registers.
  int CurrOffset = -SystemZMC::ELFCallFrameSize;
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

void SystemZELFFrameLowering::determineCalleeSaves(MachineFunction &MF,
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
    for (unsigned I = MFI->getVarArgsFirstGPR(); I < SystemZ::ELFNumArgGPRs; ++I)
      SavedRegs.set(SystemZ::ELFArgGPRs[I]);

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

SystemZELFFrameLowering::SystemZELFFrameLowering()
    : SystemZFrameLowering(TargetFrameLowering::StackGrowsDown, Align(8), 0,
                           Align(8), /* StackRealignable */ false),
      RegSpillOffsets(0) {

  // Due to the SystemZ ABI, the DWARF CFA (Canonical Frame Address) is not
  // equal to the incoming stack pointer, but to incoming stack pointer plus
  // 160.  Instead of using a Local Area Offset, the Register save area will
  // be occupied by fixed frame objects, and all offsets are actually
  // relative to CFA.

  // Create a mapping from register number to save slot offset.
  // These offsets are relative to the start of the register save area.
  RegSpillOffsets.grow(SystemZ::NUM_TARGET_REGS);
  for (unsigned I = 0, E = array_lengthof(ELFSpillOffsetTable); I != E; ++I)
    RegSpillOffsets[ELFSpillOffsetTable[I].Reg] = ELFSpillOffsetTable[I].Offset;
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

bool SystemZELFFrameLowering::spillCalleeSavedRegisters(
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
    for (const CalleeSavedInfo &I : CSI) {
      unsigned Reg = I.getReg();
      if (SystemZ::GR64BitRegClass.contains(Reg))
        addSavedGPR(MBB, MIB, Reg, true);
    }

    // ...likewise GPR varargs.
    if (IsVarArg)
      for (unsigned I = ZFI->getVarArgsFirstGPR(); I < SystemZ::ELFNumArgGPRs; ++I)
        addSavedGPR(MBB, MIB, SystemZ::ELFArgGPRs[I], true);
  }

  // Save FPRs/VRs in the normal TargetInstrInfo way.
  for (const CalleeSavedInfo &I : CSI) {
    unsigned Reg = I.getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, I.getFrameIdx(),
                               &SystemZ::FP64BitRegClass, TRI);
    }
    if (SystemZ::VR128BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, I.getFrameIdx(),
                               &SystemZ::VR128BitRegClass, TRI);
    }
  }

  return true;
}

bool SystemZELFFrameLowering::restoreCalleeSavedRegisters(
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
  for (const CalleeSavedInfo &I : CSI) {
    unsigned Reg = I.getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg))
      TII->loadRegFromStackSlot(MBB, MBBI, Reg, I.getFrameIdx(),
                                &SystemZ::FP64BitRegClass, TRI);
    if (SystemZ::VR128BitRegClass.contains(Reg))
      TII->loadRegFromStackSlot(MBB, MBBI, Reg, I.getFrameIdx(),
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
    for (const CalleeSavedInfo &I : CSI) {
      unsigned Reg = I.getReg();
      if (Reg != RestoreGPRs.LowGPR && Reg != RestoreGPRs.HighGPR &&
          SystemZ::GR64BitRegClass.contains(Reg))
        MIB.addReg(Reg, RegState::ImplicitDefine);
    }
  }

  return true;
}

void SystemZELFFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  bool BackChain = MF.getFunction().hasFnAttribute("backchain");

  if (!usePackedStack(MF) || BackChain)
    // Create the incoming register save area.
    getOrCreateFramePointerSaveIndex(MF);

  // Get the size of our stack frame to be allocated ...
  uint64_t StackSize = (MFFrame.estimateStackSize(MF) +
                        SystemZMC::ELFCallFrameSize);
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
    RS->addScavengingFrameIndex(MFFrame.CreateStackObject(8, Align(8), false));
    RS->addScavengingFrameIndex(MFFrame.CreateStackObject(8, Align(8), false));
  }

  // If R6 is used as an argument register it is still callee saved. If it in
  // this case is not clobbered (and restored) it should never be marked as
  // killed.
  if (MF.front().isLiveIn(SystemZ::R6D) &&
      ZFI->getRestoreGPRRegs().LowGPR != SystemZ::R6D)
    for (auto &MO : MRI->use_nodbg_operands(SystemZ::R6D))
      MO.setIsKill(false);
}

// Emit instructions before MBBI (in MBB) to add NumBytes to Reg.
static void emitIncrement(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &MBBI, const DebugLoc &DL,
                          Register Reg, int64_t NumBytes,
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

// Add CFI for the new CFA offset.
static void buildCFAOffs(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         const DebugLoc &DL, int Offset,
                         const SystemZInstrInfo *ZII) {
  unsigned CFIIndex = MBB.getParent()->addFrameInst(
    MCCFIInstruction::cfiDefCfaOffset(nullptr, -Offset));
  BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
    .addCFIIndex(CFIIndex);
}

// Add CFI for the new frame location.
static void buildDefCFAReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           const DebugLoc &DL, unsigned Reg,
                           const SystemZInstrInfo *ZII) {
  MachineFunction &MF = *MBB.getParent();
  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  unsigned RegNum = MRI->getDwarfRegNum(Reg, true);
  unsigned CFIIndex = MF.addFrameInst(
                        MCCFIInstruction::createDefCfaRegister(nullptr, RegNum));
  BuildMI(MBB, MBBI, DL, ZII->get(TargetOpcode::CFI_INSTRUCTION))
    .addCFIIndex(CFIIndex);
}

void SystemZELFFrameLowering::emitPrologue(MachineFunction &MF,
                                           MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  const SystemZSubtarget &STI = MF.getSubtarget<SystemZSubtarget>();
  const SystemZTargetLowering &TLI = *STI.getTargetLowering();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  auto *ZII = static_cast<const SystemZInstrInfo *>(STI.getInstrInfo());
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
    MFFrame.setStackSize(MFFrame.getStackSize() + SystemZMC::ELFCallFrameSize);
    return;
  }

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // The current offset of the stack pointer from the CFA.
  int64_t SPOffsetFromCFA = -SystemZMC::ELFCFAOffsetFromInitialSP;

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
    StackSize += SystemZMC::ELFCallFrameSize;
  // Don't allocate the incoming reg save area.
  StackSize = StackSize > SystemZMC::ELFCallFrameSize
                  ? StackSize - SystemZMC::ELFCallFrameSize
                  : 0;
  MFFrame.setStackSize(StackSize);

  if (StackSize) {
    // Allocate StackSize bytes.
    int64_t Delta = -int64_t(StackSize);
    const unsigned ProbeSize = TLI.getStackProbeSize(MF);
    bool FreeProbe = (ZFI->getSpillGPRRegs().GPROffset &&
           (ZFI->getSpillGPRRegs().GPROffset + StackSize) < ProbeSize);
    if (!FreeProbe &&
        MF.getSubtarget().getTargetLowering()->hasInlineStackProbe(MF)) {
      // Stack probing may involve looping, but splitting the prologue block
      // is not possible at this point since it would invalidate the
      // SaveBlocks / RestoreBlocks sets of PEI in the single block function
      // case. Build a pseudo to be handled later by inlineStackProbe().
      BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::PROBED_STACKALLOC))
        .addImm(StackSize);
    }
    else {
      bool StoreBackchain = MF.getFunction().hasFnAttribute("backchain");
      // If we need backchain, save current stack pointer.  R1 is free at
      // this point.
      if (StoreBackchain)
        BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::LGR))
          .addReg(SystemZ::R1D, RegState::Define).addReg(SystemZ::R15D);
      emitIncrement(MBB, MBBI, DL, SystemZ::R15D, Delta, ZII);
      buildCFAOffs(MBB, MBBI, DL, SPOffsetFromCFA + Delta, ZII);
      if (StoreBackchain)
        BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::STG))
          .addReg(SystemZ::R1D, RegState::Kill).addReg(SystemZ::R15D)
          .addImm(getBackchainOffset(MF)).addReg(0);
    }
    SPOffsetFromCFA += Delta;
  }

  if (HasFP) {
    // Copy the base of the frame to R11.
    BuildMI(MBB, MBBI, DL, ZII->get(SystemZ::LGR), SystemZ::R11D)
      .addReg(SystemZ::R15D);

    // Add CFI for the new frame location.
    buildDefCFAReg(MBB, MBBI, DL, SystemZ::R11D, ZII);

    // Mark the FramePtr as live at the beginning of every block except
    // the entry block.  (We'll have marked R11 as live on entry when
    // saving the GPRs.)
    for (MachineBasicBlock &MBBJ : llvm::drop_begin(MF))
      MBBJ.addLiveIn(SystemZ::R11D);
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
    Register IgnoredFrameReg;
    int64_t Offset =
        getFrameIndexReference(MF, Save.getFrameIdx(), IgnoredFrameReg)
            .getFixed();

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

void SystemZELFFrameLowering::emitEpilogue(MachineFunction &MF,
                                           MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  auto *ZII =
      static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();

  // See SystemZELFFrameLowering::emitPrologue
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

void SystemZELFFrameLowering::inlineStackProbe(
    MachineFunction &MF, MachineBasicBlock &PrologMBB) const {
  auto *ZII =
    static_cast<const SystemZInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SystemZSubtarget &STI = MF.getSubtarget<SystemZSubtarget>();
  const SystemZTargetLowering &TLI = *STI.getTargetLowering();

  MachineInstr *StackAllocMI = nullptr;
  for (MachineInstr &MI : PrologMBB)
    if (MI.getOpcode() == SystemZ::PROBED_STACKALLOC) {
      StackAllocMI = &MI;
      break;
    }
  if (StackAllocMI == nullptr)
    return;
  uint64_t StackSize = StackAllocMI->getOperand(0).getImm();
  const unsigned ProbeSize = TLI.getStackProbeSize(MF);
  uint64_t NumFullBlocks = StackSize / ProbeSize;
  uint64_t Residual = StackSize % ProbeSize;
  int64_t SPOffsetFromCFA = -SystemZMC::ELFCFAOffsetFromInitialSP;
  MachineBasicBlock *MBB = &PrologMBB;
  MachineBasicBlock::iterator MBBI = StackAllocMI;
  const DebugLoc DL = StackAllocMI->getDebugLoc();

  // Allocate a block of Size bytes on the stack and probe it.
  auto allocateAndProbe = [&](MachineBasicBlock &InsMBB,
                              MachineBasicBlock::iterator InsPt, unsigned Size,
                              bool EmitCFI) -> void {
    emitIncrement(InsMBB, InsPt, DL, SystemZ::R15D, -int64_t(Size), ZII);
    if (EmitCFI) {
      SPOffsetFromCFA -= Size;
      buildCFAOffs(InsMBB, InsPt, DL, SPOffsetFromCFA, ZII);
    }
    // Probe by means of a volatile compare.
    MachineMemOperand *MMO = MF.getMachineMemOperand(MachinePointerInfo(),
      MachineMemOperand::MOVolatile | MachineMemOperand::MOLoad, 8, Align(1));
    BuildMI(InsMBB, InsPt, DL, ZII->get(SystemZ::CG))
      .addReg(SystemZ::R0D, RegState::Undef)
      .addReg(SystemZ::R15D).addImm(Size - 8).addReg(0)
      .addMemOperand(MMO);
  };

  bool StoreBackchain = MF.getFunction().hasFnAttribute("backchain");
  if (StoreBackchain)
    BuildMI(*MBB, MBBI, DL, ZII->get(SystemZ::LGR))
      .addReg(SystemZ::R1D, RegState::Define).addReg(SystemZ::R15D);

  MachineBasicBlock *DoneMBB = nullptr;
  MachineBasicBlock *LoopMBB = nullptr;
  if (NumFullBlocks < 3) {
    // Emit unrolled probe statements.
    for (unsigned int i = 0; i < NumFullBlocks; i++)
      allocateAndProbe(*MBB, MBBI, ProbeSize, true/*EmitCFI*/);
  } else {
    // Emit a loop probing the pages.
    uint64_t LoopAlloc = ProbeSize * NumFullBlocks;
    SPOffsetFromCFA -= LoopAlloc;

    // Use R0D to hold the exit value.
    BuildMI(*MBB, MBBI, DL, ZII->get(SystemZ::LGR), SystemZ::R0D)
      .addReg(SystemZ::R15D);
    buildDefCFAReg(*MBB, MBBI, DL, SystemZ::R0D, ZII);
    emitIncrement(*MBB, MBBI, DL, SystemZ::R0D, -int64_t(LoopAlloc), ZII);
    buildCFAOffs(*MBB, MBBI, DL, -int64_t(SystemZMC::ELFCallFrameSize + LoopAlloc),
                 ZII);

    DoneMBB = SystemZ::splitBlockBefore(MBBI, MBB);
    LoopMBB = SystemZ::emitBlockAfter(MBB);
    MBB->addSuccessor(LoopMBB);
    LoopMBB->addSuccessor(LoopMBB);
    LoopMBB->addSuccessor(DoneMBB);

    MBB = LoopMBB;
    allocateAndProbe(*MBB, MBB->end(), ProbeSize, false/*EmitCFI*/);
    BuildMI(*MBB, MBB->end(), DL, ZII->get(SystemZ::CLGR))
      .addReg(SystemZ::R15D).addReg(SystemZ::R0D);
    BuildMI(*MBB, MBB->end(), DL, ZII->get(SystemZ::BRC))
      .addImm(SystemZ::CCMASK_ICMP).addImm(SystemZ::CCMASK_CMP_GT).addMBB(MBB);

    MBB = DoneMBB;
    MBBI = DoneMBB->begin();
    buildDefCFAReg(*MBB, MBBI, DL, SystemZ::R15D, ZII);
  }

  if (Residual)
    allocateAndProbe(*MBB, MBBI, Residual, true/*EmitCFI*/);

  if (StoreBackchain)
    BuildMI(*MBB, MBBI, DL, ZII->get(SystemZ::STG))
      .addReg(SystemZ::R1D, RegState::Kill).addReg(SystemZ::R15D)
      .addImm(getBackchainOffset(MF)).addReg(0);

  StackAllocMI->eraseFromParent();
  if (DoneMBB != nullptr) {
    // Compute the live-in lists for the new blocks.
    recomputeLiveIns(*DoneMBB);
    recomputeLiveIns(*LoopMBB);
  }
}

bool SystemZELFFrameLowering::hasFP(const MachineFunction &MF) const {
  return (MF.getTarget().Options.DisableFramePointerElim(MF) ||
          MF.getFrameInfo().hasVarSizedObjects() ||
          MF.getInfo<SystemZMachineFunctionInfo>()->getManipulatesSP());
}

StackOffset SystemZELFFrameLowering::getFrameIndexReference(
    const MachineFunction &MF, int FI, Register &FrameReg) const {
  // Our incoming SP is actually SystemZMC::ELFCallFrameSize below the CFA, so
  // add that difference here.
  StackOffset Offset =
      TargetFrameLowering::getFrameIndexReference(MF, FI, FrameReg);
  return Offset + StackOffset::getFixed(SystemZMC::ELFCallFrameSize);
}

unsigned SystemZELFFrameLowering::getRegSpillOffset(MachineFunction &MF,
                                                    Register Reg) const {
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

int SystemZELFFrameLowering::getOrCreateFramePointerSaveIndex(
    MachineFunction &MF) const {
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  int FI = ZFI->getFramePointerSaveIndex();
  if (!FI) {
    MachineFrameInfo &MFFrame = MF.getFrameInfo();
    int Offset = getBackchainOffset(MF) - SystemZMC::ELFCallFrameSize;
    FI = MFFrame.CreateFixedObject(8, Offset, false);
    ZFI->setFramePointerSaveIndex(FI);
  }
  return FI;
}

bool SystemZELFFrameLowering::usePackedStack(MachineFunction &MF) const {
  bool HasPackedStackAttr = MF.getFunction().hasFnAttribute("packed-stack");
  bool BackChain = MF.getFunction().hasFnAttribute("backchain");
  bool SoftFloat = MF.getSubtarget<SystemZSubtarget>().hasSoftFloat();
  if (HasPackedStackAttr && BackChain && !SoftFloat)
    report_fatal_error("packed-stack + backchain + hard-float is unsupported.");
  bool CallConv = MF.getFunction().getCallingConv() != CallingConv::GHC;
  return HasPackedStackAttr && CallConv;
}

SystemZXPLINKFrameLowering::SystemZXPLINKFrameLowering()
    : SystemZFrameLowering(TargetFrameLowering::StackGrowsUp, Align(32), 128,
                           Align(32), /* StackRealignable */ false),
      RegSpillOffsets(-1) {

  // Create a mapping from register number to save slot offset.
  // These offsets are relative to the start of the local are area.
  RegSpillOffsets.grow(SystemZ::NUM_TARGET_REGS);
  for (unsigned I = 0, E = array_lengthof(XPLINKSpillOffsetTable); I != E; ++I)
    RegSpillOffsets[XPLINKSpillOffsetTable[I].Reg] =
        XPLINKSpillOffsetTable[I].Offset;
}

bool SystemZXPLINKFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  const SystemZSubtarget &Subtarget = MF.getSubtarget<SystemZSubtarget>();
  auto &Regs = Subtarget.getSpecialRegisters<SystemZXPLINK64Registers>();

  // Scan the call-saved GPRs and find the bounds of the register spill area.
  unsigned LowGPR = 0;
  int LowOffset = INT32_MAX;
  unsigned HighGPR = LowGPR;
  int HighOffset = -1;

  unsigned RegSP = Regs.getStackPointerRegister();
  auto &GRRegClass = SystemZ::GR64BitRegClass;
  const unsigned RegSize = 8;

  auto ProcessCSI = [&](std::vector<CalleeSavedInfo> &CSIList) {
    for (auto &CS : CSIList) {
      unsigned Reg = CS.getReg();
      int Offset = RegSpillOffsets[Reg];
      if (Offset >= 0) {
        if (GRRegClass.contains(Reg)) {
          if (LowOffset > Offset) {
            LowOffset = Offset;
            LowGPR = Reg;
          }

          if (Offset > HighOffset) {
            HighOffset = Offset;
            HighGPR = Reg;
          }
        }
        int FrameIdx = MFFrame.CreateFixedSpillStackObject(RegSize, Offset);
        CS.setFrameIdx(FrameIdx);
      } else
        CS.setFrameIdx(INT32_MAX);
    }
  };

  std::vector<CalleeSavedInfo> Spills;

  // For non-leaf functions:
  // - the address of callee (entry point) register R6 must be saved
  Spills.push_back(CalleeSavedInfo(Regs.getAddressOfCalleeRegister()));

  // If the function needs a frame pointer, or if the backchain pointer should
  // be stored, then save the stack pointer register R4.
  if (hasFP(MF) || MF.getFunction().hasFnAttribute("backchain"))
    Spills.push_back(CalleeSavedInfo(RegSP));

  // Save the range of call-saved registers, for use by the
  // prologue/epilogue inserters.
  ProcessCSI(CSI);
  MFI->setRestoreGPRRegs(LowGPR, HighGPR, LowOffset);

  // Save the range of call-saved registers, for use by the epilogue inserter.
  ProcessCSI(Spills);
  MFI->setSpillGPRRegs(LowGPR, HighGPR, LowOffset);

  // Create spill slots for the remaining registers.
  for (auto &CS : CSI) {
    if (CS.getFrameIdx() != INT32_MAX)
      continue;
    unsigned Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    Align Alignment = TRI->getSpillAlign(*RC);
    unsigned Size = TRI->getSpillSize(*RC);
    Alignment = std::min(Alignment, getStackAlign());
    int FrameIdx = MFFrame.CreateStackObject(Size, Alignment, true);
    CS.setFrameIdx(FrameIdx);
  }

  return true;
}

void SystemZXPLINKFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                                      BitVector &SavedRegs,
                                                      RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  bool HasFP = hasFP(MF);
  const SystemZSubtarget &Subtarget = MF.getSubtarget<SystemZSubtarget>();
  auto &Regs = Subtarget.getSpecialRegisters<SystemZXPLINK64Registers>();

  // If the function requires a frame pointer, record that the hard
  // frame pointer will be clobbered.
  if (HasFP)
    SavedRegs.set(Regs.getFramePointerRegister());

  // If the function is not an XPLeaf function, we need to save the
  // return address register. We also always use that register for
  // the return instruction, so it needs to be restored in the
  // epilogue even though that register is considered to be volatile.
  // #TODO: Implement leaf detection.
  SavedRegs.set(Regs.getReturnFunctionAddressRegister());
}

bool SystemZXPLINKFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction &MF = *MBB.getParent();
  SystemZMachineFunctionInfo *ZFI = MF.getInfo<SystemZMachineFunctionInfo>();
  const SystemZSubtarget &Subtarget = MF.getSubtarget<SystemZSubtarget>();
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  auto &Regs = Subtarget.getSpecialRegisters<SystemZXPLINK64Registers>();
  SystemZ::GPRRegs SpillGPRs = ZFI->getSpillGPRRegs();
  DebugLoc DL;

  // Save GPRs
  if (SpillGPRs.LowGPR) {
    assert(SpillGPRs.LowGPR != SpillGPRs.HighGPR &&
           "Should be saving multiple registers");

    // Build an STM/STMG instruction.
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(SystemZ::STMG));

    // Add the explicit register operands.
    addSavedGPR(MBB, MIB, SpillGPRs.LowGPR, false);
    addSavedGPR(MBB, MIB, SpillGPRs.HighGPR, false);

    // Add the address r4
    MIB.addReg(Regs.getStackPointerRegister());

    // Add the partial offset
    // We cannot add the actual offset as, at the stack is not finalized
    MIB.addImm(SpillGPRs.GPROffset);

    // Make sure all call-saved GPRs are included as operands and are
    // marked as live on entry.
    auto &GRRegClass = SystemZ::GR64BitRegClass;
    for (const CalleeSavedInfo &I : CSI) {
      unsigned Reg = I.getReg();
      if (GRRegClass.contains(Reg))
        addSavedGPR(MBB, MIB, Reg, true);
    }
  }

  // Spill FPRs to the stack in the normal TargetInstrInfo way
  for (const CalleeSavedInfo &I : CSI) {
    unsigned Reg = I.getReg();
    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, I.getFrameIdx(),
                               &SystemZ::FP64BitRegClass, TRI);
    }
    if (SystemZ::VR128BitRegClass.contains(Reg)) {
      MBB.addLiveIn(Reg);
      TII->storeRegToStackSlot(MBB, MBBI, Reg, true, I.getFrameIdx(),
                               &SystemZ::VR128BitRegClass, TRI);
    }
  }

  return true;
}

void SystemZXPLINKFrameLowering::emitPrologue(MachineFunction &MF,
                                              MachineBasicBlock &MBB) const {}

void SystemZXPLINKFrameLowering::emitEpilogue(MachineFunction &MF,
                                              MachineBasicBlock &MBB) const {}

bool SystemZXPLINKFrameLowering::hasFP(const MachineFunction &MF) const {
  return false;
}
