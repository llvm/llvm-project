//===-- PPCRegisterInfo.cpp - PowerPC Register Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "PPCRegisterInfo.h"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCFrameLowering.h"
#include "PPCSubtarget.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>

#define GET_REGINFO_TARGET_DESC
#include "PPCGenRegisterInfo.inc"

namespace llvm {
cl::opt<bool> DisablePPC32RS("disable-ppc32-regscavenger",
                                   cl::init(false),
                                   cl::desc("Disable PPC32 register scavenger"),
                                   cl::Hidden);
cl::opt<bool> DisablePPC64RS("disable-ppc64-regscavenger",
                                   cl::init(false),
                                   cl::desc("Disable PPC64 register scavenger"),
                                   cl::Hidden);
}

using namespace llvm;

// FIXME (64-bit): Should be inlined.
bool
PPCRegisterInfo::requiresRegisterScavenging(const MachineFunction &) const {
  return ((!DisablePPC32RS && !Subtarget.isPPC64()) ||
          (!DisablePPC64RS && Subtarget.isPPC64()));
}

PPCRegisterInfo::PPCRegisterInfo(const PPCSubtarget &ST,
                                 const TargetInstrInfo &tii)
  : PPCGenRegisterInfo(ST.isPPC64() ? PPC::LR8 : PPC::LR,
                       ST.isPPC64() ? 0 : 1,
                       ST.isPPC64() ? 0 : 1),
    Subtarget(ST), TII(tii), CRSpillFrameIdx(0) {
  ImmToIdxMap[PPC::LD]   = PPC::LDX;    ImmToIdxMap[PPC::STD]  = PPC::STDX;
  ImmToIdxMap[PPC::LBZ]  = PPC::LBZX;   ImmToIdxMap[PPC::STB]  = PPC::STBX;
  ImmToIdxMap[PPC::LHZ]  = PPC::LHZX;   ImmToIdxMap[PPC::LHA]  = PPC::LHAX;
  ImmToIdxMap[PPC::LWZ]  = PPC::LWZX;   ImmToIdxMap[PPC::LWA]  = PPC::LWAX;
  ImmToIdxMap[PPC::LFS]  = PPC::LFSX;   ImmToIdxMap[PPC::LFD]  = PPC::LFDX;
  ImmToIdxMap[PPC::STH]  = PPC::STHX;   ImmToIdxMap[PPC::STW]  = PPC::STWX;
  ImmToIdxMap[PPC::STFS] = PPC::STFSX;  ImmToIdxMap[PPC::STFD] = PPC::STFDX;
  ImmToIdxMap[PPC::ADDI] = PPC::ADD4;

  // 64-bit
  ImmToIdxMap[PPC::LHA8] = PPC::LHAX8; ImmToIdxMap[PPC::LBZ8] = PPC::LBZX8;
  ImmToIdxMap[PPC::LHZ8] = PPC::LHZX8; ImmToIdxMap[PPC::LWZ8] = PPC::LWZX8;
  ImmToIdxMap[PPC::STB8] = PPC::STBX8; ImmToIdxMap[PPC::STH8] = PPC::STHX8;
  ImmToIdxMap[PPC::STW8] = PPC::STWX8; ImmToIdxMap[PPC::STDU] = PPC::STDUX;
  ImmToIdxMap[PPC::ADDI8] = PPC::ADD8; ImmToIdxMap[PPC::STD_32] = PPC::STDX_32;
}

bool
PPCRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  return requiresRegisterScavenging(MF);
}


/// getPointerRegClass - Return the register class to use to hold pointers.
/// This is used for addressing modes.
const TargetRegisterClass *
PPCRegisterInfo::getPointerRegClass(const MachineFunction &MF, unsigned Kind)
                                                                       const {
  if (Subtarget.isPPC64())
    return &PPC::G8RCRegClass;
  return &PPC::GPRCRegClass;
}

const uint16_t*
PPCRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  if (Subtarget.isDarwinABI())
    return Subtarget.isPPC64() ? CSR_Darwin64_SaveList :
                                 CSR_Darwin32_SaveList;

  // For 32-bit SVR4, also initialize the frame index associated with
  // the CR spill slot.
  if (!Subtarget.isPPC64())
    CRSpillFrameIdx = 0;

  return Subtarget.isPPC64() ? CSR_SVR464_SaveList : CSR_SVR432_SaveList;
}

const uint32_t*
PPCRegisterInfo::getCallPreservedMask(CallingConv::ID CC) const {
  if (Subtarget.isDarwinABI())
    return Subtarget.isPPC64() ? CSR_Darwin64_RegMask :
                                 CSR_Darwin32_RegMask;

  return Subtarget.isPPC64() ? CSR_SVR464_RegMask : CSR_SVR432_RegMask;
}

BitVector PPCRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const PPCFrameLowering *PPCFI =
    static_cast<const PPCFrameLowering*>(MF.getTarget().getFrameLowering());

  Reserved.set(PPC::R0);
  Reserved.set(PPC::R1);
  Reserved.set(PPC::LR);
  Reserved.set(PPC::LR8);
  Reserved.set(PPC::RM);

  // The SVR4 ABI reserves r2 and r13
  if (Subtarget.isSVR4ABI()) {
    Reserved.set(PPC::R2);  // System-reserved register
    Reserved.set(PPC::R13); // Small Data Area pointer register
  }
  // Reserve R2 on Darwin to hack around the problem of save/restore of CR
  // when the stack frame is too big to address directly; we need two regs.
  // This is a hack.
  if (Subtarget.isDarwinABI()) {
    Reserved.set(PPC::R2);
  }
  
  // On PPC64, r13 is the thread pointer. Never allocate this register.
  // Note that this is over conservative, as it also prevents allocation of R31
  // when the FP is not needed.
  if (Subtarget.isPPC64()) {
    Reserved.set(PPC::R13);
    Reserved.set(PPC::R31);

    Reserved.set(PPC::X0);
    Reserved.set(PPC::X1);
    Reserved.set(PPC::X13);
    Reserved.set(PPC::X31);

    // The 64-bit SVR4 ABI reserves r2 for the TOC pointer.
    if (Subtarget.isSVR4ABI()) {
      Reserved.set(PPC::X2);
    }
    // Reserve X2 on Darwin to hack around the problem of save/restore of CR
    // when the stack frame is too big to address directly; we need two regs.
    // This is a hack.
    if (Subtarget.isDarwinABI()) {
      Reserved.set(PPC::X2);
    }
  }

  if (PPCFI->needsFP(MF))
    Reserved.set(PPC::R31);

  return Reserved;
}

unsigned
PPCRegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                         MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  const unsigned DefaultSafety = 1;

  switch (RC->getID()) {
  default:
    return 0;
  case PPC::G8RCRegClassID:
  case PPC::GPRCRegClassID: {
    unsigned FP = TFI->hasFP(MF) ? 1 : 0;
    return 32 - FP - DefaultSafety;
  }
  case PPC::F8RCRegClassID:
  case PPC::F4RCRegClassID:
  case PPC::VRRCRegClassID:
    return 32 - DefaultSafety;
  case PPC::CRRCRegClassID:
    return 8 - DefaultSafety;
  }
}

bool
PPCRegisterInfo::avoidWriteAfterWrite(const TargetRegisterClass *RC) const {
  switch (RC->getID()) {
  case PPC::G8RCRegClassID:
  case PPC::GPRCRegClassID:
  case PPC::F8RCRegClassID:
  case PPC::F4RCRegClassID:
  case PPC::VRRCRegClassID:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

void PPCRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (MF.getTarget().Options.GuaranteedTailCallOpt &&
      I->getOpcode() == PPC::ADJCALLSTACKUP) {
    // Add (actually subtract) back the amount the callee popped on return.
    if (int CalleeAmt =  I->getOperand(1).getImm()) {
      bool is64Bit = Subtarget.isPPC64();
      CalleeAmt *= -1;
      unsigned StackReg = is64Bit ? PPC::X1 : PPC::R1;
      unsigned TmpReg = is64Bit ? PPC::X0 : PPC::R0;
      unsigned ADDIInstr = is64Bit ? PPC::ADDI8 : PPC::ADDI;
      unsigned ADDInstr = is64Bit ? PPC::ADD8 : PPC::ADD4;
      unsigned LISInstr = is64Bit ? PPC::LIS8 : PPC::LIS;
      unsigned ORIInstr = is64Bit ? PPC::ORI8 : PPC::ORI;
      MachineInstr *MI = I;
      DebugLoc dl = MI->getDebugLoc();

      if (isInt<16>(CalleeAmt)) {
        BuildMI(MBB, I, dl, TII.get(ADDIInstr), StackReg)
          .addReg(StackReg, RegState::Kill)
          .addImm(CalleeAmt);
      } else {
        MachineBasicBlock::iterator MBBI = I;
        BuildMI(MBB, MBBI, dl, TII.get(LISInstr), TmpReg)
          .addImm(CalleeAmt >> 16);
        BuildMI(MBB, MBBI, dl, TII.get(ORIInstr), TmpReg)
          .addReg(TmpReg, RegState::Kill)
          .addImm(CalleeAmt & 0xFFFF);
        BuildMI(MBB, MBBI, dl, TII.get(ADDInstr), StackReg)
          .addReg(StackReg, RegState::Kill)
          .addReg(TmpReg);
      }
    }
  }
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

/// findScratchRegister - Find a 'free' PPC register. Try for a call-clobbered
/// register first and then a spilled callee-saved register if that fails.
static
unsigned findScratchRegister(MachineBasicBlock::iterator II, RegScavenger *RS,
                             const TargetRegisterClass *RC, int SPAdj) {
  assert(RS && "Register scavenging must be on");
  unsigned Reg = RS->FindUnusedReg(RC);
  // FIXME: move ARM callee-saved reg scan to target independent code, then 
  // search for already spilled CS register here.
  if (Reg == 0)
    Reg = RS->scavengeRegister(RC, II, SPAdj);
  return Reg;
}

/// lowerDynamicAlloc - Generate the code for allocating an object in the
/// current frame.  The sequence of code with be in the general form
///
///   addi   R0, SP, \#frameSize ; get the address of the previous frame
///   stwxu  R0, SP, Rnegsize   ; add and update the SP with the negated size
///   addi   Rnew, SP, \#maxCalFrameSize ; get the top of the allocation
///
void PPCRegisterInfo::lowerDynamicAlloc(MachineBasicBlock::iterator II,
                                        int SPAdj, RegScavenger *RS) const {
  // Get the instruction.
  MachineInstr &MI = *II;
  // Get the instruction's basic block.
  MachineBasicBlock &MBB = *MI.getParent();
  // Get the basic block's function.
  MachineFunction &MF = *MBB.getParent();
  // Get the frame info.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  // Determine whether 64-bit pointers are used.
  bool LP64 = Subtarget.isPPC64();
  DebugLoc dl = MI.getDebugLoc();

  // Get the maximum call stack size.
  unsigned maxCallFrameSize = MFI->getMaxCallFrameSize();
  // Get the total frame size.
  unsigned FrameSize = MFI->getStackSize();
  
  // Get stack alignments.
  unsigned TargetAlign = MF.getTarget().getFrameLowering()->getStackAlignment();
  unsigned MaxAlign = MFI->getMaxAlignment();
  if (MaxAlign > TargetAlign)
    report_fatal_error("Dynamic alloca with large aligns not supported");

  // Determine the previous frame's address.  If FrameSize can't be
  // represented as 16 bits or we need special alignment, then we load the
  // previous frame's address from 0(SP).  Why not do an addis of the hi? 
  // Because R0 is our only safe tmp register and addi/addis treat R0 as zero. 
  // Constructing the constant and adding would take 3 instructions. 
  // Fortunately, a frame greater than 32K is rare.
  const TargetRegisterClass *G8RC = &PPC::G8RCRegClass;
  const TargetRegisterClass *GPRC = &PPC::GPRCRegClass;
  const TargetRegisterClass *RC = LP64 ? G8RC : GPRC;

  // FIXME (64-bit): Use "findScratchRegister"
  unsigned Reg;
  if (requiresRegisterScavenging(MF))
    Reg = findScratchRegister(II, RS, RC, SPAdj);
  else
    Reg = PPC::R0;
  
  if (MaxAlign < TargetAlign && isInt<16>(FrameSize)) {
    BuildMI(MBB, II, dl, TII.get(PPC::ADDI), Reg)
      .addReg(PPC::R31)
      .addImm(FrameSize);
  } else if (LP64) {
    if (requiresRegisterScavenging(MF)) // FIXME (64-bit): Use "true" part.
      BuildMI(MBB, II, dl, TII.get(PPC::LD), Reg)
        .addImm(0)
        .addReg(PPC::X1);
    else
      BuildMI(MBB, II, dl, TII.get(PPC::LD), PPC::X0)
        .addImm(0)
        .addReg(PPC::X1);
  } else {
    BuildMI(MBB, II, dl, TII.get(PPC::LWZ), Reg)
      .addImm(0)
      .addReg(PPC::R1);
  }
  
  // Grow the stack and update the stack pointer link, then determine the
  // address of new allocated space.
  if (LP64) {
    if (requiresRegisterScavenging(MF)) // FIXME (64-bit): Use "true" part.
      BuildMI(MBB, II, dl, TII.get(PPC::STDUX), PPC::X1)
        .addReg(Reg, RegState::Kill)
        .addReg(PPC::X1)
        .addReg(MI.getOperand(1).getReg());
    else
      BuildMI(MBB, II, dl, TII.get(PPC::STDUX), PPC::X1)
        .addReg(PPC::X0, RegState::Kill)
        .addReg(PPC::X1)
        .addReg(MI.getOperand(1).getReg());

    if (!MI.getOperand(1).isKill())
      BuildMI(MBB, II, dl, TII.get(PPC::ADDI8), MI.getOperand(0).getReg())
        .addReg(PPC::X1)
        .addImm(maxCallFrameSize);
    else
      // Implicitly kill the register.
      BuildMI(MBB, II, dl, TII.get(PPC::ADDI8), MI.getOperand(0).getReg())
        .addReg(PPC::X1)
        .addImm(maxCallFrameSize)
        .addReg(MI.getOperand(1).getReg(), RegState::ImplicitKill);
  } else {
    BuildMI(MBB, II, dl, TII.get(PPC::STWUX), PPC::R1)
      .addReg(Reg, RegState::Kill)
      .addReg(PPC::R1)
      .addReg(MI.getOperand(1).getReg());

    if (!MI.getOperand(1).isKill())
      BuildMI(MBB, II, dl, TII.get(PPC::ADDI), MI.getOperand(0).getReg())
        .addReg(PPC::R1)
        .addImm(maxCallFrameSize);
    else
      // Implicitly kill the register.
      BuildMI(MBB, II, dl, TII.get(PPC::ADDI), MI.getOperand(0).getReg())
        .addReg(PPC::R1)
        .addImm(maxCallFrameSize)
        .addReg(MI.getOperand(1).getReg(), RegState::ImplicitKill);
  }
  
  // Discard the DYNALLOC instruction.
  MBB.erase(II);
}

/// lowerCRSpilling - Generate the code for spilling a CR register. Instead of
/// reserving a whole register (R0), we scrounge for one here. This generates
/// code like this:
///
///   mfcr rA                  ; Move the conditional register into GPR rA.
///   rlwinm rA, rA, SB, 0, 31 ; Shift the bits left so they are in CR0's slot.
///   stw rA, FI               ; Store rA to the frame.
///
void PPCRegisterInfo::lowerCRSpilling(MachineBasicBlock::iterator II,
                                      unsigned FrameIndex, int SPAdj,
                                      RegScavenger *RS) const {
  // Get the instruction.
  MachineInstr &MI = *II;       // ; SPILL_CR <SrcReg>, <offset>
  // Get the instruction's basic block.
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc dl = MI.getDebugLoc();

  // FIXME: Once LLVM supports creating virtual registers here, or the register
  // scavenger can return multiple registers, stop using reserved registers
  // here.
  (void) SPAdj;
  (void) RS;

  bool LP64 = Subtarget.isPPC64();
  unsigned Reg = Subtarget.isDarwinABI() ?  (LP64 ? PPC::X2 : PPC::R2) :
                                            (LP64 ? PPC::X0 : PPC::R0);
  unsigned SrcReg = MI.getOperand(0).getReg();

  // We need to store the CR in the low 4-bits of the saved value. First, issue
  // an MFCRpsued to save all of the CRBits and, if needed, kill the SrcReg.
  BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::MFCR8pseud : PPC::MFCRpseud), Reg)
          .addReg(SrcReg, getKillRegState(MI.getOperand(0).isKill()));
    
  // If the saved register wasn't CR0, shift the bits left so that they are in
  // CR0's slot.
  if (SrcReg != PPC::CR0)
    // rlwinm rA, rA, ShiftBits, 0, 31.
    BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::RLWINM8 : PPC::RLWINM), Reg)
      .addReg(Reg, RegState::Kill)
      .addImm(getPPCRegisterNumbering(SrcReg) * 4)
      .addImm(0)
      .addImm(31);

  addFrameReference(BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::STW8 : PPC::STW))
                    .addReg(Reg, getKillRegState(MI.getOperand(1).getImm())),
                    FrameIndex);

  // Discard the pseudo instruction.
  MBB.erase(II);
}

void PPCRegisterInfo::lowerCRRestore(MachineBasicBlock::iterator II,
                                      unsigned FrameIndex, int SPAdj,
                                      RegScavenger *RS) const {
  // Get the instruction.
  MachineInstr &MI = *II;       // ; <DestReg> = RESTORE_CR <offset>
  // Get the instruction's basic block.
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc dl = MI.getDebugLoc();

  // FIXME: Once LLVM supports creating virtual registers here, or the register
  // scavenger can return multiple registers, stop using reserved registers
  // here.
  (void) SPAdj;
  (void) RS;

  bool LP64 = Subtarget.isPPC64();
  unsigned Reg = Subtarget.isDarwinABI() ?  (LP64 ? PPC::X2 : PPC::R2) :
                                            (LP64 ? PPC::X0 : PPC::R0);
  unsigned DestReg = MI.getOperand(0).getReg();
  assert(MI.definesRegister(DestReg) &&
    "RESTORE_CR does not define its destination");

  addFrameReference(BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::LWZ8 : PPC::LWZ),
                              Reg), FrameIndex);

  // If the reloaded register isn't CR0, shift the bits right so that they are
  // in the right CR's slot.
  if (DestReg != PPC::CR0) {
    unsigned ShiftBits = getPPCRegisterNumbering(DestReg)*4;
    // rlwinm r11, r11, 32-ShiftBits, 0, 31.
    BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::RLWINM8 : PPC::RLWINM), Reg)
             .addReg(Reg).addImm(32-ShiftBits).addImm(0)
             .addImm(31);
  }

  BuildMI(MBB, II, dl, TII.get(LP64 ? PPC::MTCRF8 : PPC::MTCRF), DestReg)
             .addReg(Reg);

  // Discard the pseudo instruction.
  MBB.erase(II);
}

bool
PPCRegisterInfo::hasReservedSpillSlot(const MachineFunction &MF,
				      unsigned Reg, int &FrameIdx) const {

  // For the nonvolatile condition registers (CR2, CR3, CR4) in an SVR4
  // ABI, return true to prevent allocating an additional frame slot.
  // For 64-bit, the CR save area is at SP+8; the value of FrameIdx = 0
  // is arbitrary and will be subsequently ignored.  For 32-bit, we must
  // create exactly one stack slot and return its FrameIdx for all
  // nonvolatiles.
  if (Subtarget.isSVR4ABI() && PPC::CR2 <= Reg && Reg <= PPC::CR4) {
    if (Subtarget.isPPC64()) {
      FrameIdx = 0;
    } else if (CRSpillFrameIdx) {
      FrameIdx = CRSpillFrameIdx;
    } else {
      MachineFrameInfo *MFI = ((MachineFunction &)MF).getFrameInfo();
      FrameIdx = MFI->CreateFixedObject((uint64_t)4, (int64_t)-4, true);
      CRSpillFrameIdx = FrameIdx;
    }
    return true;
  }
  return false;
}

void
PPCRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                     int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  // Get the instruction.
  MachineInstr &MI = *II;
  // Get the instruction's basic block.
  MachineBasicBlock &MBB = *MI.getParent();
  // Get the basic block's function.
  MachineFunction &MF = *MBB.getParent();
  // Get the frame info.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  DebugLoc dl = MI.getDebugLoc();

  // Find out which operand is the frame index.
  unsigned FIOperandNo = 0;
  while (!MI.getOperand(FIOperandNo).isFI()) {
    ++FIOperandNo;
    assert(FIOperandNo != MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }
  // Take into account whether it's an add or mem instruction
  unsigned OffsetOperandNo = (FIOperandNo == 2) ? 1 : 2;
  if (MI.isInlineAsm())
    OffsetOperandNo = FIOperandNo-1;

  // Get the frame index.
  int FrameIndex = MI.getOperand(FIOperandNo).getIndex();

  // Get the frame pointer save index.  Users of this index are primarily
  // DYNALLOC instructions.
  PPCFunctionInfo *FI = MF.getInfo<PPCFunctionInfo>();
  int FPSI = FI->getFramePointerSaveIndex();
  // Get the instruction opcode.
  unsigned OpC = MI.getOpcode();
  
  // Special case for dynamic alloca.
  if (FPSI && FrameIndex == FPSI &&
      (OpC == PPC::DYNALLOC || OpC == PPC::DYNALLOC8)) {
    lowerDynamicAlloc(II, SPAdj, RS);
    return;
  }

  // Special case for pseudo-ops SPILL_CR and RESTORE_CR.
  if (requiresRegisterScavenging(MF)) {
    if (OpC == PPC::SPILL_CR) {
      lowerCRSpilling(II, FrameIndex, SPAdj, RS);
      return;
    } else if (OpC == PPC::RESTORE_CR) {
      lowerCRRestore(II, FrameIndex, SPAdj, RS);
      return;
    }
  }

  // Replace the FrameIndex with base register with GPR1 (SP) or GPR31 (FP).

  bool is64Bit = Subtarget.isPPC64();
  MI.getOperand(FIOperandNo).ChangeToRegister(TFI->hasFP(MF) ?
                                              (is64Bit ? PPC::X31 : PPC::R31) :
                                                (is64Bit ? PPC::X1 : PPC::R1),
                                              false);

  // Figure out if the offset in the instruction is shifted right two bits. This
  // is true for instructions like "STD", which the machine implicitly adds two
  // low zeros to.
  bool isIXAddr = false;
  switch (OpC) {
  case PPC::LWA:
  case PPC::LD:
  case PPC::STD:
  case PPC::STD_32:
    isIXAddr = true;
    break;
  }
  
  // Now add the frame object offset to the offset from r1.
  int Offset = MFI->getObjectOffset(FrameIndex);
  if (!isIXAddr)
    Offset += MI.getOperand(OffsetOperandNo).getImm();
  else
    Offset += MI.getOperand(OffsetOperandNo).getImm() << 2;

  // If we're not using a Frame Pointer that has been set to the value of the
  // SP before having the stack size subtracted from it, then add the stack size
  // to Offset to get the correct offset.
  // Naked functions have stack size 0, although getStackSize may not reflect that
  // because we didn't call all the pieces that compute it for naked functions.
  if (!MF.getFunction()->getFnAttributes().hasAttribute(Attributes::Naked))
    Offset += MFI->getStackSize();

  // If we can, encode the offset directly into the instruction.  If this is a
  // normal PPC "ri" instruction, any 16-bit value can be safely encoded.  If
  // this is a PPC64 "ix" instruction, only a 16-bit value with the low two bits
  // clear can be encoded.  This is extremely uncommon, because normally you
  // only "std" to a stack slot that is at least 4-byte aligned, but it can
  // happen in invalid code.
  if (OpC == PPC::DBG_VALUE || // DBG_VALUE is always Reg+Imm
      (isInt<16>(Offset) && (!isIXAddr || (Offset & 3) == 0))) {
    if (isIXAddr)
      Offset >>= 2;    // The actual encoded value has the low two bits zero.
    MI.getOperand(OffsetOperandNo).ChangeToImmediate(Offset);
    return;
  }

  // The offset doesn't fit into a single register, scavenge one to build the
  // offset in.

  unsigned SReg;
  if (requiresRegisterScavenging(MF)) {
    const TargetRegisterClass *G8RC = &PPC::G8RCRegClass;
    const TargetRegisterClass *GPRC = &PPC::GPRCRegClass;
    SReg = findScratchRegister(II, RS, is64Bit ? G8RC : GPRC, SPAdj);
  } else
    SReg = is64Bit ? PPC::X0 : PPC::R0;

  // Insert a set of rA with the full offset value before the ld, st, or add
  BuildMI(MBB, II, dl, TII.get(is64Bit ? PPC::LIS8 : PPC::LIS), SReg)
    .addImm(Offset >> 16);
  BuildMI(MBB, II, dl, TII.get(is64Bit ? PPC::ORI8 : PPC::ORI), SReg)
    .addReg(SReg, RegState::Kill)
    .addImm(Offset);

  // Convert into indexed form of the instruction:
  // 
  //   sth 0:rA, 1:imm 2:(rB) ==> sthx 0:rA, 2:rB, 1:r0
  //   addi 0:rA 1:rB, 2, imm ==> add 0:rA, 1:rB, 2:r0
  unsigned OperandBase;

  if (OpC != TargetOpcode::INLINEASM) {
    assert(ImmToIdxMap.count(OpC) &&
           "No indexed form of load or store available!");
    unsigned NewOpcode = ImmToIdxMap.find(OpC)->second;
    MI.setDesc(TII.get(NewOpcode));
    OperandBase = 1;
  } else {
    OperandBase = OffsetOperandNo;
  }

  unsigned StackReg = MI.getOperand(FIOperandNo).getReg();
  MI.getOperand(OperandBase).ChangeToRegister(StackReg, false);
  MI.getOperand(OperandBase + 1).ChangeToRegister(SReg, false, false, true);
}

unsigned PPCRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (!Subtarget.isPPC64())
    return TFI->hasFP(MF) ? PPC::R31 : PPC::R1;
  else
    return TFI->hasFP(MF) ? PPC::X31 : PPC::X1;
}

unsigned PPCRegisterInfo::getEHExceptionRegister() const {
  return !Subtarget.isPPC64() ? PPC::R3 : PPC::X3;
}

unsigned PPCRegisterInfo::getEHHandlerRegister() const {
  return !Subtarget.isPPC64() ? PPC::R4 : PPC::X4;
}
