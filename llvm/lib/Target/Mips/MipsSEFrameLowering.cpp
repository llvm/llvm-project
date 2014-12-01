//===-- MipsSEFrameLowering.cpp - Mips32/64 Frame Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips32/64 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MipsSEFrameLowering.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsMachineFunction.h"
#include "MipsSEInstrInfo.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

namespace {
typedef MachineBasicBlock::iterator Iter;

static std::pair<unsigned, unsigned> getMFHiLoOpc(unsigned Src) {
  if (Mips::ACC64RegClass.contains(Src))
    return std::make_pair((unsigned)Mips::PseudoMFHI,
                          (unsigned)Mips::PseudoMFLO);

  if (Mips::ACC64DSPRegClass.contains(Src))
    return std::make_pair((unsigned)Mips::MFHI_DSP, (unsigned)Mips::MFLO_DSP);

  if (Mips::ACC128RegClass.contains(Src))
    return std::make_pair((unsigned)Mips::PseudoMFHI64,
                          (unsigned)Mips::PseudoMFLO64);

  return std::make_pair(0, 0);
}

/// Helper class to expand pseudos.
class ExpandPseudo {
public:
  ExpandPseudo(MachineFunction &MF);
  bool expand();

private:
  bool expandInstr(MachineBasicBlock &MBB, Iter I);
  void expandLoadCCond(MachineBasicBlock &MBB, Iter I);
  void expandStoreCCond(MachineBasicBlock &MBB, Iter I);
  void expandLoadACC(MachineBasicBlock &MBB, Iter I, unsigned RegSize);
  void expandStoreACC(MachineBasicBlock &MBB, Iter I, unsigned MFHiOpc,
                      unsigned MFLoOpc, unsigned RegSize);
  bool expandCopy(MachineBasicBlock &MBB, Iter I);
  bool expandCopyACC(MachineBasicBlock &MBB, Iter I, unsigned MFHiOpc,
                     unsigned MFLoOpc);
  bool expandBuildPairF64(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator I, bool FP64) const;
  bool expandExtractElementF64(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, bool FP64) const;

  MachineFunction &MF;
  MachineRegisterInfo &MRI;
};
}

ExpandPseudo::ExpandPseudo(MachineFunction &MF_)
  : MF(MF_), MRI(MF.getRegInfo()) {}

bool ExpandPseudo::expand() {
  bool Expanded = false;

  for (MachineFunction::iterator BB = MF.begin(), BBEnd = MF.end();
       BB != BBEnd; ++BB)
    for (Iter I = BB->begin(), End = BB->end(); I != End;)
      Expanded |= expandInstr(*BB, I++);

  return Expanded;
}

bool ExpandPseudo::expandInstr(MachineBasicBlock &MBB, Iter I) {
  switch(I->getOpcode()) {
  case Mips::LOAD_CCOND_DSP:
    expandLoadCCond(MBB, I);
    break;
  case Mips::STORE_CCOND_DSP:
    expandStoreCCond(MBB, I);
    break;
  case Mips::LOAD_ACC64:
  case Mips::LOAD_ACC64DSP:
    expandLoadACC(MBB, I, 4);
    break;
  case Mips::LOAD_ACC128:
    expandLoadACC(MBB, I, 8);
    break;
  case Mips::STORE_ACC64:
    expandStoreACC(MBB, I, Mips::PseudoMFHI, Mips::PseudoMFLO, 4);
    break;
  case Mips::STORE_ACC64DSP:
    expandStoreACC(MBB, I, Mips::MFHI_DSP, Mips::MFLO_DSP, 4);
    break;
  case Mips::STORE_ACC128:
    expandStoreACC(MBB, I, Mips::PseudoMFHI64, Mips::PseudoMFLO64, 8);
    break;
  case Mips::BuildPairF64:
    if (expandBuildPairF64(MBB, I, false))
      MBB.erase(I);
    return false;
  case Mips::BuildPairF64_64:
    if (expandBuildPairF64(MBB, I, true))
      MBB.erase(I);
    return false;
  case Mips::ExtractElementF64:
    if (expandExtractElementF64(MBB, I, false))
      MBB.erase(I);
    return false;
  case Mips::ExtractElementF64_64:
    if (expandExtractElementF64(MBB, I, true))
      MBB.erase(I);
    return false;
  case TargetOpcode::COPY:
    if (!expandCopy(MBB, I))
      return false;
    break;
  default:
    return false;
  }

  MBB.erase(I);
  return true;
}

void ExpandPseudo::expandLoadCCond(MachineBasicBlock &MBB, Iter I) {
  //  load $vr, FI
  //  copy ccond, $vr

  assert(I->getOperand(0).isReg() && I->getOperand(1).isFI());

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  const TargetRegisterClass *RC = RegInfo.intRegClass(4);
  unsigned VR = MRI.createVirtualRegister(RC);
  unsigned Dst = I->getOperand(0).getReg(), FI = I->getOperand(1).getIndex();

  TII.loadRegFromStack(MBB, I, VR, FI, RC, &RegInfo, 0);
  BuildMI(MBB, I, I->getDebugLoc(), TII.get(TargetOpcode::COPY), Dst)
    .addReg(VR, RegState::Kill);
}

void ExpandPseudo::expandStoreCCond(MachineBasicBlock &MBB, Iter I) {
  //  copy $vr, ccond
  //  store $vr, FI

  assert(I->getOperand(0).isReg() && I->getOperand(1).isFI());

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  const TargetRegisterClass *RC = RegInfo.intRegClass(4);
  unsigned VR = MRI.createVirtualRegister(RC);
  unsigned Src = I->getOperand(0).getReg(), FI = I->getOperand(1).getIndex();

  BuildMI(MBB, I, I->getDebugLoc(), TII.get(TargetOpcode::COPY), VR)
    .addReg(Src, getKillRegState(I->getOperand(0).isKill()));
  TII.storeRegToStack(MBB, I, VR, true, FI, RC, &RegInfo, 0);
}

void ExpandPseudo::expandLoadACC(MachineBasicBlock &MBB, Iter I,
                                 unsigned RegSize) {
  //  load $vr0, FI
  //  copy lo, $vr0
  //  load $vr1, FI + 4
  //  copy hi, $vr1

  assert(I->getOperand(0).isReg() && I->getOperand(1).isFI());

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  const TargetRegisterClass *RC = RegInfo.intRegClass(RegSize);
  unsigned VR0 = MRI.createVirtualRegister(RC);
  unsigned VR1 = MRI.createVirtualRegister(RC);
  unsigned Dst = I->getOperand(0).getReg(), FI = I->getOperand(1).getIndex();
  unsigned Lo = RegInfo.getSubReg(Dst, Mips::sub_lo);
  unsigned Hi = RegInfo.getSubReg(Dst, Mips::sub_hi);
  DebugLoc DL = I->getDebugLoc();
  const MCInstrDesc &Desc = TII.get(TargetOpcode::COPY);

  TII.loadRegFromStack(MBB, I, VR0, FI, RC, &RegInfo, 0);
  BuildMI(MBB, I, DL, Desc, Lo).addReg(VR0, RegState::Kill);
  TII.loadRegFromStack(MBB, I, VR1, FI, RC, &RegInfo, RegSize);
  BuildMI(MBB, I, DL, Desc, Hi).addReg(VR1, RegState::Kill);
}

void ExpandPseudo::expandStoreACC(MachineBasicBlock &MBB, Iter I,
                                  unsigned MFHiOpc, unsigned MFLoOpc,
                                  unsigned RegSize) {
  //  mflo $vr0, src
  //  store $vr0, FI
  //  mfhi $vr1, src
  //  store $vr1, FI + 4

  assert(I->getOperand(0).isReg() && I->getOperand(1).isFI());

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  const TargetRegisterClass *RC = RegInfo.intRegClass(RegSize);
  unsigned VR0 = MRI.createVirtualRegister(RC);
  unsigned VR1 = MRI.createVirtualRegister(RC);
  unsigned Src = I->getOperand(0).getReg(), FI = I->getOperand(1).getIndex();
  unsigned SrcKill = getKillRegState(I->getOperand(0).isKill());
  DebugLoc DL = I->getDebugLoc();

  BuildMI(MBB, I, DL, TII.get(MFLoOpc), VR0).addReg(Src);
  TII.storeRegToStack(MBB, I, VR0, true, FI, RC, &RegInfo, 0);
  BuildMI(MBB, I, DL, TII.get(MFHiOpc), VR1).addReg(Src, SrcKill);
  TII.storeRegToStack(MBB, I, VR1, true, FI, RC, &RegInfo, RegSize);
}

bool ExpandPseudo::expandCopy(MachineBasicBlock &MBB, Iter I) {
  unsigned Src = I->getOperand(1).getReg();
  std::pair<unsigned, unsigned> Opcodes = getMFHiLoOpc(Src);

  if (!Opcodes.first)
    return false;

  return expandCopyACC(MBB, I, Opcodes.first, Opcodes.second);
}

bool ExpandPseudo::expandCopyACC(MachineBasicBlock &MBB, Iter I,
                                 unsigned MFHiOpc, unsigned MFLoOpc) {
  //  mflo $vr0, src
  //  copy dst_lo, $vr0
  //  mfhi $vr1, src
  //  copy dst_hi, $vr1

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  unsigned Dst = I->getOperand(0).getReg(), Src = I->getOperand(1).getReg();
  unsigned VRegSize = RegInfo.getMinimalPhysRegClass(Dst)->getSize() / 2;
  const TargetRegisterClass *RC = RegInfo.intRegClass(VRegSize);
  unsigned VR0 = MRI.createVirtualRegister(RC);
  unsigned VR1 = MRI.createVirtualRegister(RC);
  unsigned SrcKill = getKillRegState(I->getOperand(1).isKill());
  unsigned DstLo = RegInfo.getSubReg(Dst, Mips::sub_lo);
  unsigned DstHi = RegInfo.getSubReg(Dst, Mips::sub_hi);
  DebugLoc DL = I->getDebugLoc();

  BuildMI(MBB, I, DL, TII.get(MFLoOpc), VR0).addReg(Src);
  BuildMI(MBB, I, DL, TII.get(TargetOpcode::COPY), DstLo)
    .addReg(VR0, RegState::Kill);
  BuildMI(MBB, I, DL, TII.get(MFHiOpc), VR1).addReg(Src, SrcKill);
  BuildMI(MBB, I, DL, TII.get(TargetOpcode::COPY), DstHi)
    .addReg(VR1, RegState::Kill);
  return true;
}

/// This method expands the same instruction that MipsSEInstrInfo::
/// expandBuildPairF64 does, for the case when ABI is fpxx and mthc1 is not
/// available and the case where the ABI is FP64A. It is implemented here
/// because frame indexes are eliminated before MipsSEInstrInfo::
/// expandBuildPairF64 is called.
bool ExpandPseudo::expandBuildPairF64(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      bool FP64) const {
  // For fpxx and when mthc1 is not available, use:
  //   spill + reload via ldc1
  //
  // The case where dmtc1 is available doesn't need to be handled here
  // because it never creates a BuildPairF64 node.
  //
  // The FP64A ABI (fp64 with nooddspreg) must also use a spill/reload sequence
  // for odd-numbered double precision values (because the lower 32-bits is
  // transferred with mtc1 which is redirected to the upper half of the even
  // register). Unfortunately, we have to make this decision before register
  // allocation so for now we use a spill/reload sequence for all
  // double-precision values in regardless of being an odd/even register.

  const TargetMachine &TM = MF.getTarget();
  const MipsSubtarget &Subtarget = TM.getSubtarget<MipsSubtarget>();
  if ((Subtarget.isABI_FPXX() && !Subtarget.hasMTHC1()) ||
      (FP64 && !Subtarget.useOddSPReg())) {
    const MipsSEInstrInfo &TII =
      *static_cast<const MipsSEInstrInfo*>(TM.getInstrInfo());
    const MipsRegisterInfo &TRI =
      *static_cast<const MipsRegisterInfo*>(TM.getRegisterInfo());

    unsigned DstReg = I->getOperand(0).getReg();
    unsigned LoReg = I->getOperand(1).getReg();
    unsigned HiReg = I->getOperand(2).getReg();

    // It should be impossible to have FGR64 on MIPS-II or MIPS32r1 (which are
    // the cases where mthc1 is not available). 64-bit architectures and
    // MIPS32r2 or later can use FGR64 though.
    assert(Subtarget.isGP64bit() || Subtarget.hasMTHC1() ||
           !Subtarget.isFP64bit());

    const TargetRegisterClass *RC = &Mips::GPR32RegClass;
    const TargetRegisterClass *RC2 =
        FP64 ? &Mips::FGR64RegClass : &Mips::AFGR64RegClass;

    // We re-use the same spill slot each time so that the stack frame doesn't
    // grow too much in functions with a large number of moves.
    int FI = MF.getInfo<MipsFunctionInfo>()->getMoveF64ViaSpillFI(RC2);
    if (!Subtarget.isLittle())
      std::swap(LoReg, HiReg);
    TII.storeRegToStack(MBB, I, LoReg, I->getOperand(1).isKill(), FI, RC, &TRI,
                        0);
    TII.storeRegToStack(MBB, I, HiReg, I->getOperand(2).isKill(), FI, RC, &TRI,
                        4);
    TII.loadRegFromStack(MBB, I, DstReg, FI, RC2, &TRI, 0);
    return true;
  }

  return false;
}

/// This method expands the same instruction that MipsSEInstrInfo::
/// expandExtractElementF64 does, for the case when ABI is fpxx and mfhc1 is not
/// available and the case where the ABI is FP64A. It is implemented here
/// because frame indexes are eliminated before MipsSEInstrInfo::
/// expandExtractElementF64 is called.
bool ExpandPseudo::expandExtractElementF64(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I,
                                           bool FP64) const {
  // For fpxx and when mfhc1 is not available, use:
  //   spill + reload via ldc1
  //
  // The case where dmfc1 is available doesn't need to be handled here
  // because it never creates a ExtractElementF64 node.
  //
  // The FP64A ABI (fp64 with nooddspreg) must also use a spill/reload sequence
  // for odd-numbered double precision values (because the lower 32-bits is
  // transferred with mfc1 which is redirected to the upper half of the even
  // register). Unfortunately, we have to make this decision before register
  // allocation so for now we use a spill/reload sequence for all
  // double-precision values in regardless of being an odd/even register.

  const TargetMachine &TM = MF.getTarget();
  const MipsSubtarget &Subtarget = TM.getSubtarget<MipsSubtarget>();
  if ((Subtarget.isABI_FPXX() && !Subtarget.hasMTHC1()) ||
      (FP64 && !Subtarget.useOddSPReg())) {
    const MipsSEInstrInfo &TII =
        *static_cast<const MipsSEInstrInfo *>(TM.getInstrInfo());
    const MipsRegisterInfo &TRI =
        *static_cast<const MipsRegisterInfo *>(TM.getRegisterInfo());

    unsigned DstReg = I->getOperand(0).getReg();
    unsigned SrcReg = I->getOperand(1).getReg();
    unsigned N = I->getOperand(2).getImm();
    int64_t Offset = 4 * (Subtarget.isLittle() ? N : (1 - N));

    // It should be impossible to have FGR64 on MIPS-II or MIPS32r1 (which are
    // the cases where mfhc1 is not available). 64-bit architectures and
    // MIPS32r2 or later can use FGR64 though.
    assert(Subtarget.isGP64bit() || Subtarget.hasMTHC1() ||
           !Subtarget.isFP64bit());

    const TargetRegisterClass *RC =
        FP64 ? &Mips::FGR64RegClass : &Mips::AFGR64RegClass;
    const TargetRegisterClass *RC2 = &Mips::GPR32RegClass;

    // We re-use the same spill slot each time so that the stack frame doesn't
    // grow too much in functions with a large number of moves.
    int FI = MF.getInfo<MipsFunctionInfo>()->getMoveF64ViaSpillFI(RC);
    TII.storeRegToStack(MBB, I, SrcReg, I->getOperand(1).isKill(), FI, RC, &TRI,
                        0);
    TII.loadRegFromStack(MBB, I, DstReg, FI, RC2, &TRI, Offset);
    return true;
  }

  return false;
}

MipsSEFrameLowering::MipsSEFrameLowering(const MipsSubtarget &STI)
    : MipsFrameLowering(STI, STI.stackAlignment()) {}

unsigned MipsSEFrameLowering::ehDataReg(unsigned I) const {
  static const unsigned EhDataReg[] = {
    Mips::A0, Mips::A1, Mips::A2, Mips::A3
  };
  static const unsigned EhDataReg64[] = {
    Mips::A0_64, Mips::A1_64, Mips::A2_64, Mips::A3_64
  };

  return STI.isABI_N64() ? EhDataReg64[I] : EhDataReg[I];
}

void MipsSEFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned SP = STI.isABI_N64() ? Mips::SP_64 : Mips::SP;
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;
  unsigned ZERO = STI.isABI_N64() ? Mips::ZERO_64 : Mips::ZERO;
  unsigned ADDu = STI.isABI_N64() ? Mips::DADDu : Mips::ADDu;

  // First, compute final stack size.
  uint64_t StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->adjustsStack()) return;

  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  MachineLocation DstML, SrcML;

  // Adjust stack.
  TII.adjustStackPtr(SP, -StackSize, MBB, MBBI);

  // emit ".cfi_def_cfa_offset StackSize"
  unsigned CFIIndex = MMI.addFrameInst(
      MCCFIInstruction::createDefCfaOffset(nullptr, -StackSize));
  BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  if (CSI.size()) {
    // Find the instruction past the last instruction that saves a callee-saved
    // register to the stack.
    for (unsigned i = 0; i < CSI.size(); ++i)
      ++MBBI;

    // Iterate over list of callee-saved registers and emit .cfi_offset
    // directives.
    for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
           E = CSI.end(); I != E; ++I) {
      int64_t Offset = MFI->getObjectOffset(I->getFrameIdx());
      unsigned Reg = I->getReg();

      // If Reg is a double precision register, emit two cfa_offsets,
      // one for each of the paired single precision registers.
      if (Mips::AFGR64RegClass.contains(Reg)) {
        unsigned Reg0 =
            MRI->getDwarfRegNum(RegInfo.getSubReg(Reg, Mips::sub_lo), true);
        unsigned Reg1 =
            MRI->getDwarfRegNum(RegInfo.getSubReg(Reg, Mips::sub_hi), true);

        if (!STI.isLittle())
          std::swap(Reg0, Reg1);

        unsigned CFIIndex = MMI.addFrameInst(
            MCCFIInstruction::createOffset(nullptr, Reg0, Offset));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);

        CFIIndex = MMI.addFrameInst(
            MCCFIInstruction::createOffset(nullptr, Reg1, Offset + 4));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      } else if (Mips::FGR64RegClass.contains(Reg)) {
        unsigned Reg0 = MRI->getDwarfRegNum(Reg, true);
        unsigned Reg1 = MRI->getDwarfRegNum(Reg, true) + 1;

        if (!STI.isLittle())
          std::swap(Reg0, Reg1);

        unsigned CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, Reg0, Offset));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);

        CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, Reg1, Offset + 4));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      } else {
        // Reg is either in GPR32 or FGR32.
        unsigned CFIIndex = MMI.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, MRI->getDwarfRegNum(Reg, 1), Offset));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
    }
  }

  if (MipsFI->callsEhReturn()) {
    const TargetRegisterClass *RC = STI.isABI_N64() ?
        &Mips::GPR64RegClass : &Mips::GPR32RegClass;

    // Insert instructions that spill eh data registers.
    for (int I = 0; I < 4; ++I) {
      if (!MBB.isLiveIn(ehDataReg(I)))
        MBB.addLiveIn(ehDataReg(I));
      TII.storeRegToStackSlot(MBB, MBBI, ehDataReg(I), false,
                              MipsFI->getEhDataRegFI(I), RC, &RegInfo);
    }

    // Emit .cfi_offset directives for eh data registers.
    for (int I = 0; I < 4; ++I) {
      int64_t Offset = MFI->getObjectOffset(MipsFI->getEhDataRegFI(I));
      unsigned Reg = MRI->getDwarfRegNum(ehDataReg(I), true);
      unsigned CFIIndex = MMI.addFrameInst(
          MCCFIInstruction::createOffset(nullptr, Reg, Offset));
      BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    }
  }

  // if framepointer enabled, set it to point to the stack pointer.
  if (hasFP(MF)) {
    // Insert instruction "move $fp, $sp" at this location.
    BuildMI(MBB, MBBI, dl, TII.get(ADDu), FP).addReg(SP).addReg(ZERO)
      .setMIFlag(MachineInstr::FrameSetup);

    // emit ".cfi_def_cfa_register $fp"
    unsigned CFIIndex = MMI.addFrameInst(MCCFIInstruction::createDefCfaRegister(
        nullptr, MRI->getDwarfRegNum(FP, true)));
    BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);
  }
}

void MipsSEFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());
  const MipsRegisterInfo &RegInfo =
    *static_cast<const MipsRegisterInfo*>(MF.getTarget().getRegisterInfo());

  DebugLoc dl = MBBI->getDebugLoc();
  unsigned SP = STI.isABI_N64() ? Mips::SP_64 : Mips::SP;
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;
  unsigned ZERO = STI.isABI_N64() ? Mips::ZERO_64 : Mips::ZERO;
  unsigned ADDu = STI.isABI_N64() ? Mips::DADDu : Mips::ADDu;

  // if framepointer enabled, restore the stack pointer.
  if (hasFP(MF)) {
    // Find the first instruction that restores a callee-saved register.
    MachineBasicBlock::iterator I = MBBI;

    for (unsigned i = 0; i < MFI->getCalleeSavedInfo().size(); ++i)
      --I;

    // Insert instruction "move $sp, $fp" at this location.
    BuildMI(MBB, I, dl, TII.get(ADDu), SP).addReg(FP).addReg(ZERO);
  }

  if (MipsFI->callsEhReturn()) {
    const TargetRegisterClass *RC = STI.isABI_N64() ?
        &Mips::GPR64RegClass : &Mips::GPR32RegClass;

    // Find first instruction that restores a callee-saved register.
    MachineBasicBlock::iterator I = MBBI;
    for (unsigned i = 0; i < MFI->getCalleeSavedInfo().size(); ++i)
      --I;

    // Insert instructions that restore eh data registers.
    for (int J = 0; J < 4; ++J) {
      TII.loadRegFromStackSlot(MBB, I, ehDataReg(J), MipsFI->getEhDataRegFI(J),
                               RC, &RegInfo);
    }
  }

  // Get the number of bytes from FrameInfo
  uint64_t StackSize = MFI->getStackSize();

  if (!StackSize)
    return;

  // Adjust stack.
  TII.adjustStackPtr(SP, StackSize, MBB, MBBI);
}

bool MipsSEFrameLowering::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI,
                          const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  MachineBasicBlock *EntryBlock = MF->begin();
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    // Add the callee-saved register as live-in. Do not add if the register is
    // RA and return address is taken, because it has already been added in
    // method MipsTargetLowering::LowerRETURNADDR.
    // It's killed at the spill, unless the register is RA and return address
    // is taken.
    unsigned Reg = CSI[i].getReg();
    bool IsRAAndRetAddrIsTaken = (Reg == Mips::RA || Reg == Mips::RA_64)
        && MF->getFrameInfo()->isReturnAddressTaken();
    if (!IsRAAndRetAddrIsTaken)
      EntryBlock->addLiveIn(Reg);

    // Insert the spill to the stack frame.
    bool IsKill = !IsRAAndRetAddrIsTaken;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(*EntryBlock, MI, Reg, IsKill,
                            CSI[i].getFrameIdx(), RC, TRI);
  }

  return true;
}

bool
MipsSEFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Reserve call frame if the size of the maximum call frame fits into 16-bit
  // immediate field and there are no variable sized objects on the stack.
  // Make sure the second register scavenger spill slot can be accessed with one
  // instruction.
  return isInt<16>(MFI->getMaxCallFrameSize() + getStackAlignment()) &&
    !MFI->hasVarSizedObjects();
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions
void MipsSEFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const MipsSEInstrInfo &TII =
    *static_cast<const MipsSEInstrInfo*>(MF.getTarget().getInstrInfo());

  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();

    if (I->getOpcode() == Mips::ADJCALLSTACKDOWN)
      Amount = -Amount;

    unsigned SP = STI.isABI_N64() ? Mips::SP_64 : Mips::SP;
    TII.adjustStackPtr(SP, Amount, MBB, I);
  }

  MBB.erase(I);
}

void MipsSEFrameLowering::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  unsigned FP = STI.isABI_N64() ? Mips::FP_64 : Mips::FP;

  // Mark $fp as used if function has dedicated frame pointer.
  if (hasFP(MF))
    MRI.setPhysRegUsed(FP);

  // Create spill slots for eh data registers if function calls eh_return.
  if (MipsFI->callsEhReturn())
    MipsFI->createEhDataRegsFI();

  // Expand pseudo instructions which load, store or copy accumulators.
  // Add an emergency spill slot if a pseudo was expanded.
  if (ExpandPseudo(MF).expand()) {
    // The spill slot should be half the size of the accumulator. If target is
    // mips64, it should be 64-bit, otherwise it should be 32-bt.
    const TargetRegisterClass *RC = STI.hasMips64() ?
      &Mips::GPR64RegClass : &Mips::GPR32RegClass;
    int FI = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
                                                  RC->getAlignment(), false);
    RS->addScavengingFrameIndex(FI);
  }

  // Set scavenging frame index if necessary.
  uint64_t MaxSPOffset = MF.getInfo<MipsFunctionInfo>()->getIncomingArgSize() +
    estimateStackSize(MF);

  if (isInt<16>(MaxSPOffset))
    return;

  const TargetRegisterClass *RC = STI.isABI_N64() ?
    &Mips::GPR64RegClass : &Mips::GPR32RegClass;
  int FI = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
                                                RC->getAlignment(), false);
  RS->addScavengingFrameIndex(FI);
}

const MipsFrameLowering *
llvm::createMipsSEFrameLowering(const MipsSubtarget &ST) {
  return new MipsSEFrameLowering(ST);
}
