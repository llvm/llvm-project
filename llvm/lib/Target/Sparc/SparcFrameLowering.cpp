//===-- SparcFrameLowering.cpp - Sparc Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Sparc implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "SparcFrameLowering.h"
#include "SparcInstrInfo.h"
#include "SparcMachineFunctionInfo.h"
#include "SparcSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

static cl::opt<bool>
DisableLeafProc("disable-sparc-leaf-proc",
                cl::init(false),
                cl::desc("Disable Sparc leaf procedure optimization."),
                cl::Hidden);

SparcFrameLowering::SparcFrameLowering(const SparcSubtarget &ST)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          ST.is64Bit() ? Align(16) : Align(8), 0,
                          ST.is64Bit() ? Align(16) : Align(8),
                          /*StackRealignable=*/false) {}

void SparcFrameLowering::emitSPAdjustment(MachineFunction &MF,
                                          MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          int NumBytes,
                                          unsigned ADDrr,
                                          unsigned ADDri) const {

  DebugLoc dl;
  const SparcInstrInfo &TII =
      *static_cast<const SparcInstrInfo *>(MF.getSubtarget().getInstrInfo());

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

void SparcFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();

  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const SparcSubtarget &Subtarget = MF.getSubtarget<SparcSubtarget>();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI.getStackSize();

  unsigned SAVEri = SP::SAVEri;
  unsigned SAVErr = SP::SAVErr;
  if (FuncInfo->isLeafProc()) {
    if (NumBytes == 0)
      return;
    SAVEri = SP::ADDri;
    SAVErr = SP::ADDrr;
  }

  // The SPARC ABI is a bit odd in that it requires a reserved 92-byte
  // (128 in v9) area in the user's stack, starting at %sp. Thus, the
  // first part of the stack that can actually be used is located at
  // %sp + 92.
  //
  // We therefore need to add that offset to the total stack size
  // after all the stack objects are placed by
  // PrologEpilogInserter calculateFrameObjectOffsets. However, since the stack needs to be
  // aligned *after* the extra size is added, we need to disable
  // calculateFrameObjectOffsets's built-in stack alignment, by having
  // targetHandlesStackFrameRounding return true.


  // Add the extra call frame stack size, if needed. (This is the same
  // code as in PrologEpilogInserter, but also gets disabled by
  // targetHandlesStackFrameRounding)
  if (MFI.adjustsStack() && hasReservedCallFrame(MF))
    NumBytes += MFI.getMaxCallFrameSize();

  // Adds the SPARC subtarget-specific spill area to the stack
  // size. Also ensures target-required alignment.
  NumBytes = Subtarget.getAdjustedFrameSize(NumBytes);

  // Finally, ensure that the size is sufficiently aligned for the
  // data on the stack.
  NumBytes = alignTo(NumBytes, MFI.getMaxAlign());

  // Update stack size with corrected value.
  MFI.setStackSize(NumBytes);

  emitSPAdjustment(MF, MBB, MBBI, -NumBytes, SAVErr, SAVEri);

  if (MF.needsFrameMoves()) {
    CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::NoFlags);
    if (FuncInfo->isLeafProc()) {
      CFIBuilder.buildDefCFAOffset(NumBytes + Subtarget.getStackPointerBias());
    } else {
      CFIBuilder.buildDefCFARegister(SP::I6);
      CFIBuilder.buildWindowSave();
      CFIBuilder.buildRegister(SP::O7, SP::I7);
    }
  }
}

MachineBasicBlock::iterator SparcFrameLowering::
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
  return MBB.erase(I);
}


void SparcFrameLowering::emitEpilogue(MachineFunction &MF,
                                  MachineBasicBlock &MBB) const {
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const SparcInstrInfo &TII =
      *static_cast<const SparcInstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc dl = MBBI->getDebugLoc();
  assert((MBBI->getOpcode() == SP::RETL || MBBI->getOpcode() == SP::TAIL_CALL ||
          MBBI->getOpcode() == SP::TAIL_CALLri) &&
         "Can only put epilog before 'retl' or 'tail_call' instruction!");
  if (!FuncInfo->isLeafProc()) {
    BuildMI(MBB, MBBI, dl, TII.get(SP::RESTORErr), SP::G0).addReg(SP::G0)
      .addReg(SP::G0);
    return;
  }
  MachineFrameInfo &MFI = MF.getFrameInfo();

  int NumBytes = (int) MFI.getStackSize();
  if (NumBytes != 0)
    emitSPAdjustment(MF, MBB, MBBI, NumBytes, SP::ADDrr, SP::ADDri);

  // Preserve return address in %o7
  if (MBBI->getOpcode() == SP::TAIL_CALL) {
    MBB.addLiveIn(SP::O7);
    BuildMI(MBB, MBBI, dl, TII.get(SP::ORrr), SP::G1)
        .addReg(SP::G0)
        .addReg(SP::O7);
    BuildMI(MBB, MBBI, dl, TII.get(SP::ORrr), SP::O7)
        .addReg(SP::G0)
        .addReg(SP::G1);
  }
}

bool SparcFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // Reserve call frame if there are no variable sized objects on the stack.
  return !MF.getFrameInfo().hasVarSizedObjects();
}

// hasFPImpl - Return true if the specified function should have a dedicated
// frame pointer register.  This is true if the function has variable sized
// allocas or if frame pointer elimination is disabled.
bool SparcFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken();
}

StackOffset
SparcFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {
  const SparcSubtarget &Subtarget = MF.getSubtarget<SparcSubtarget>();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SparcRegisterInfo *RegInfo = Subtarget.getRegisterInfo();
  const SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();
  bool isFixed = MFI.isFixedObjectIndex(FI);

  // Addressable stack objects are accessed using neg. offsets from
  // %fp, or positive offsets from %sp.
  bool UseFP;

  // Sparc uses FP-based references in general, even when "hasFP" is
  // false. That function is rather a misnomer, because %fp is
  // actually always available, unless isLeafProc.
  if (FuncInfo->isLeafProc()) {
    // If there's a leaf proc, all offsets need to be %sp-based,
    // because we haven't caused %fp to actually point to our frame.
    UseFP = false;
  } else if (isFixed) {
    // Otherwise, argument access should always use %fp.
    UseFP = true;
  } else {
    // Finally, default to using %fp.
    UseFP = true;
  }

  int64_t FrameOffset = MF.getFrameInfo().getObjectOffset(FI) +
      Subtarget.getStackPointerBias();

  if (UseFP) {
    FrameReg = RegInfo->getFrameRegister(MF);
    return StackOffset::getFixed(FrameOffset);
  } else {
    FrameReg = SP::O6; // %sp
    return StackOffset::getFixed(FrameOffset + MF.getFrameInfo().getStackSize());
  }
}

[[maybe_unused]] static bool verifyLeafProcRegUse(MachineRegisterInfo *MRI) {

  for (unsigned reg = SP::I0; reg <= SP::I7; ++reg)
    if (MRI->isPhysRegUsed(reg))
      return false;

  for (unsigned reg = SP::L0; reg <= SP::L7; ++reg)
    if (MRI->isPhysRegUsed(reg))
      return false;

  return true;
}

bool SparcFrameLowering::isLeafProc(MachineFunction &MF) const {
  return !MF.getFrameInfo().hasCalls() &&
         !MF.getRegInfo().isPhysRegUsed(SP::O6) && !hasFP(MF) &&
         !MF.hasInlineAsm() &&
         !MF.getSubtarget<SparcSubtarget>().hasReservedLocalRegister();
}

void SparcFrameLowering::remapRegsForLeafProc(MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SparcRegisterInfo *TRI =
      MF.getSubtarget<SparcSubtarget>().getRegisterInfo();
  DenseMap<MCRegister, MCRegister> RegMap;
  auto RemapReg = [&](MCRegister Reg, MCRegister MappedReg) {
    auto [It, Inserted] = RegMap.try_emplace(Reg, MappedReg);
    assert((Inserted || It->second == MappedReg) &&
           "inconsistent leaf register mapping");
    if (!Inserted)
      return;
    MRI.replaceRegWith(Reg, MappedReg);
  };

  // Remap %i[0-7] to %o[0-7].
  for (unsigned Reg = SP::I0; Reg <= SP::I7; ++Reg) {
    MCRegister MappedReg = Reg - SP::I0 + SP::O0;
    RemapReg(Reg, MappedReg);
    for (MCRegAliasIterator AI(Reg, TRI, false); AI.isValid(); ++AI) {
      unsigned SubRegIndex = TRI->getSubRegIndex(*AI, Reg);
      if (!SubRegIndex)
        continue;
      MCRegister MappedAlias = TRI->getMatchingSuperReg(MappedReg, SubRegIndex,
                                                        &SP::IntPairRegClass);
      assert(MappedAlias);
      RemapReg(*AI, MappedAlias);
    }
  }

  // Rewrite MBB's Live-ins.
  for (MachineBasicBlock &MBB : MF) {
    std::vector<MachineBasicBlock::RegisterMaskPair> LiveIns;
    MBB.clearLiveIns(LiveIns);
    for (const auto &LiveIn : LiveIns) {
      auto It = RegMap.find(LiveIn.PhysReg);
      MBB.addLiveIn(It == RegMap.end() ? LiveIn.PhysReg : It->second,
                    LiveIn.LaneMask);
    }
    MBB.sortUniqueLiveIns();
  }

  assert(verifyLeafProcRegUse(&MRI));
#ifdef EXPENSIVE_CHECKS
  MF.verify(0, "After LeafProc Remapping");
#endif
}

void SparcFrameLowering::finalizeLeafProc(MachineFunction &MF) const {
  bool IsLeafProc = !DisableLeafProc && isLeafProc(MF);
  MF.getInfo<SparcMachineFunctionInfo>()->setLeafProc(IsLeafProc);
  if (IsLeafProc)
    remapRegsForLeafProc(MF);
}
