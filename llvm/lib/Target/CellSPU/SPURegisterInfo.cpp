//===- SPURegisterInfo.cpp - Cell SPU Register Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cell implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "SPU.h"
#include "SPURegisterInfo.h"
#include "SPURegisterNames.h"
#include "SPUInstrBuilder.h"
#include "SPUSubtarget.h"
#include "SPUMachineFunction.h"
#include "SPUFrameInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>

using namespace llvm;

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// PPC::F14, return the number that it corresponds to (e.g. 14).
unsigned SPURegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  using namespace SPU;
  switch (RegEnum) {
  case SPU::R0: return 0;
  case SPU::R1: return 1;
  case SPU::R2: return 2;
  case SPU::R3: return 3;
  case SPU::R4: return 4;
  case SPU::R5: return 5;
  case SPU::R6: return 6;
  case SPU::R7: return 7;
  case SPU::R8: return 8;
  case SPU::R9: return 9;
  case SPU::R10: return 10;
  case SPU::R11: return 11;
  case SPU::R12: return 12;
  case SPU::R13: return 13;
  case SPU::R14: return 14;
  case SPU::R15: return 15;
  case SPU::R16: return 16;
  case SPU::R17: return 17;
  case SPU::R18: return 18;
  case SPU::R19: return 19;
  case SPU::R20: return 20;
  case SPU::R21: return 21;
  case SPU::R22: return 22;
  case SPU::R23: return 23;
  case SPU::R24: return 24;
  case SPU::R25: return 25;
  case SPU::R26: return 26;
  case SPU::R27: return 27;
  case SPU::R28: return 28;
  case SPU::R29: return 29;
  case SPU::R30: return 30;
  case SPU::R31: return 31;
  case SPU::R32: return 32;
  case SPU::R33: return 33;
  case SPU::R34: return 34;
  case SPU::R35: return 35;
  case SPU::R36: return 36;
  case SPU::R37: return 37;
  case SPU::R38: return 38;
  case SPU::R39: return 39;
  case SPU::R40: return 40;
  case SPU::R41: return 41;
  case SPU::R42: return 42;
  case SPU::R43: return 43;
  case SPU::R44: return 44;
  case SPU::R45: return 45;
  case SPU::R46: return 46;
  case SPU::R47: return 47;
  case SPU::R48: return 48;
  case SPU::R49: return 49;
  case SPU::R50: return 50;
  case SPU::R51: return 51;
  case SPU::R52: return 52;
  case SPU::R53: return 53;
  case SPU::R54: return 54;
  case SPU::R55: return 55;
  case SPU::R56: return 56;
  case SPU::R57: return 57;
  case SPU::R58: return 58;
  case SPU::R59: return 59;
  case SPU::R60: return 60;
  case SPU::R61: return 61;
  case SPU::R62: return 62;
  case SPU::R63: return 63;
  case SPU::R64: return 64;
  case SPU::R65: return 65;
  case SPU::R66: return 66;
  case SPU::R67: return 67;
  case SPU::R68: return 68;
  case SPU::R69: return 69;
  case SPU::R70: return 70;
  case SPU::R71: return 71;
  case SPU::R72: return 72;
  case SPU::R73: return 73;
  case SPU::R74: return 74;
  case SPU::R75: return 75;
  case SPU::R76: return 76;
  case SPU::R77: return 77;
  case SPU::R78: return 78;
  case SPU::R79: return 79;
  case SPU::R80: return 80;
  case SPU::R81: return 81;
  case SPU::R82: return 82;
  case SPU::R83: return 83;
  case SPU::R84: return 84;
  case SPU::R85: return 85;
  case SPU::R86: return 86;
  case SPU::R87: return 87;
  case SPU::R88: return 88;
  case SPU::R89: return 89;
  case SPU::R90: return 90;
  case SPU::R91: return 91;
  case SPU::R92: return 92;
  case SPU::R93: return 93;
  case SPU::R94: return 94;
  case SPU::R95: return 95;
  case SPU::R96: return 96;
  case SPU::R97: return 97;
  case SPU::R98: return 98;
  case SPU::R99: return 99;
  case SPU::R100: return 100;
  case SPU::R101: return 101;
  case SPU::R102: return 102;
  case SPU::R103: return 103;
  case SPU::R104: return 104;
  case SPU::R105: return 105;
  case SPU::R106: return 106;
  case SPU::R107: return 107;
  case SPU::R108: return 108;
  case SPU::R109: return 109;
  case SPU::R110: return 110;
  case SPU::R111: return 111;
  case SPU::R112: return 112;
  case SPU::R113: return 113;
  case SPU::R114: return 114;
  case SPU::R115: return 115;
  case SPU::R116: return 116;
  case SPU::R117: return 117;
  case SPU::R118: return 118;
  case SPU::R119: return 119;
  case SPU::R120: return 120;
  case SPU::R121: return 121;
  case SPU::R122: return 122;
  case SPU::R123: return 123;
  case SPU::R124: return 124;
  case SPU::R125: return 125;
  case SPU::R126: return 126;
  case SPU::R127: return 127;
  default:
    cerr << "Unhandled reg in SPURegisterInfo::getRegisterNumbering!\n";
    abort();
  }
}

SPURegisterInfo::SPURegisterInfo(const SPUSubtarget &subtarget,
                                 const TargetInstrInfo &tii) :
  SPUGenRegisterInfo(SPU::ADJCALLSTACKDOWN, SPU::ADJCALLSTACKUP),
  Subtarget(subtarget),
  TII(tii)
{
}

// SPU's 128-bit registers used for argument passing:
static const unsigned SPU_ArgRegs[] = {
  SPU::R3,  SPU::R4,  SPU::R5,  SPU::R6,  SPU::R7,  SPU::R8,  SPU::R9,
  SPU::R10, SPU::R11, SPU::R12, SPU::R13, SPU::R14, SPU::R15, SPU::R16,
  SPU::R17, SPU::R18, SPU::R19, SPU::R20, SPU::R21, SPU::R22, SPU::R23,
  SPU::R24, SPU::R25, SPU::R26, SPU::R27, SPU::R28, SPU::R29, SPU::R30,
  SPU::R31, SPU::R32, SPU::R33, SPU::R34, SPU::R35, SPU::R36, SPU::R37,
  SPU::R38, SPU::R39, SPU::R40, SPU::R41, SPU::R42, SPU::R43, SPU::R44,
  SPU::R45, SPU::R46, SPU::R47, SPU::R48, SPU::R49, SPU::R50, SPU::R51,
  SPU::R52, SPU::R53, SPU::R54, SPU::R55, SPU::R56, SPU::R57, SPU::R58,
  SPU::R59, SPU::R60, SPU::R61, SPU::R62, SPU::R63, SPU::R64, SPU::R65,
  SPU::R66, SPU::R67, SPU::R68, SPU::R69, SPU::R70, SPU::R71, SPU::R72,
  SPU::R73, SPU::R74, SPU::R75, SPU::R76, SPU::R77, SPU::R78, SPU::R79
};

const unsigned *
SPURegisterInfo::getArgRegs()
{
  return SPU_ArgRegs;
}

unsigned
SPURegisterInfo::getNumArgRegs()
{
  return sizeof(SPU_ArgRegs) / sizeof(SPU_ArgRegs[0]);
}

const unsigned *
SPURegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const
{
  // Cell ABI calling convention
  static const unsigned SPU_CalleeSaveRegs[] = {
    SPU::R80, SPU::R81, SPU::R82, SPU::R83,
    SPU::R84, SPU::R85, SPU::R86, SPU::R87,
    SPU::R88, SPU::R89, SPU::R90, SPU::R91,
    SPU::R92, SPU::R93, SPU::R94, SPU::R95,
    SPU::R96, SPU::R97, SPU::R98, SPU::R99,
    SPU::R100, SPU::R101, SPU::R102, SPU::R103,
    SPU::R104, SPU::R105, SPU::R106, SPU::R107,
    SPU::R108, SPU::R109, SPU::R110, SPU::R111,
    SPU::R112, SPU::R113, SPU::R114, SPU::R115,
    SPU::R116, SPU::R117, SPU::R118, SPU::R119,
    SPU::R120, SPU::R121, SPU::R122, SPU::R123,
    SPU::R124, SPU::R125, SPU::R126, SPU::R127,
    SPU::R2,    /* environment pointer */
    SPU::R1,    /* stack pointer */
    SPU::R0,    /* link register */
    0 /* end */
  };

  return SPU_CalleeSaveRegs;
}

const TargetRegisterClass* const*
SPURegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const
{
  // Cell ABI Calling Convention
  static const TargetRegisterClass * const SPU_CalleeSaveRegClasses[] = {
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, &SPU::GPRCRegClass, &SPU::GPRCRegClass,
    &SPU::GPRCRegClass, /* environment pointer */
    &SPU::GPRCRegClass, /* stack pointer */
    &SPU::GPRCRegClass, /* link register */
    0 /* end */
  };

  return SPU_CalleeSaveRegClasses;
}

/*!
 R0 (link register), R1 (stack pointer) and R2 (environment pointer -- this is
 generally unused) are the Cell's reserved registers
 */
BitVector SPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(SPU::R0);                // LR
  Reserved.set(SPU::R1);                // SP
  Reserved.set(SPU::R2);                // environment pointer
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// needsFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool needsFP(const MachineFunction &MF) {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

//--------------------------------------------------------------------------
// hasFP - Return true if the specified function actually has a dedicated frame
// pointer register.  This is true if the function needs a frame pointer and has
// a non-zero stack size.
bool
SPURegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getStackSize() && needsFP(MF);
}

//--------------------------------------------------------------------------
void
SPURegisterInfo::eliminateCallFramePseudoInstr(MachineFunction &MF,
                                               MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator I)
  const
{
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

void
SPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                                     RegScavenger *RS) const
{
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  MachineOperand &SPOp = MI.getOperand(i);
  int FrameIndex = SPOp.getIndex();

  // Now add the frame object offset to the offset from r1.
  int Offset = MFI->getObjectOffset(FrameIndex);

  // Most instructions, except for generated FrameIndex additions using AIr32
  // and ILAr32, have the immediate in operand 1. AIr32 and ILAr32 have the
  // immediate in operand 2.
  unsigned OpNo = 1;
  if (MI.getOpcode() == SPU::AIr32 || MI.getOpcode() == SPU::ILAr32)
    OpNo = 2;

  MachineOperand &MO = MI.getOperand(OpNo);

  // Offset is biased by $lr's slot at the bottom.
  Offset += MO.getImm() + MFI->getStackSize() + SPUFrameInfo::minStackSize();
  assert((Offset & 0xf) == 0
         && "16-byte alignment violated in eliminateFrameIndex");

  // Replace the FrameIndex with base register with $sp (aka $r1)
  SPOp.ChangeToRegister(SPU::R1, false);
  if (Offset > SPUFrameInfo::maxFrameOffset()
      || Offset < SPUFrameInfo::minFrameOffset()) {
    cerr << "Large stack adjustment ("
         << Offset
         << ") in SPURegisterInfo::eliminateFrameIndex.";
  } else {
    MO.ChangeToImmediate(Offset);
  }
}

/// determineFrameLayout - Determine the size of the frame and maximum call
/// frame size.
void
SPURegisterInfo::determineFrameLayout(MachineFunction &MF) const
{
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo
  unsigned FrameSize = MFI->getStackSize();

  // Get the alignments provided by the target, and the maximum alignment
  // (if any) of the fixed frame objects.
  unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned Align = std::max(TargetAlign, MFI->getMaxAlignment());
  assert(isPowerOf2_32(Align) && "Alignment is not power of 2");
  unsigned AlignMask = Align - 1;

  // Get the maximum call frame size of all the calls.
  unsigned maxCallFrameSize = MFI->getMaxCallFrameSize();

  // If we have dynamic alloca then maxCallFrameSize needs to be aligned so
  // that allocations will be aligned.
  if (MFI->hasVarSizedObjects())
    maxCallFrameSize = (maxCallFrameSize + AlignMask) & ~AlignMask;

  // Update maximum call frame size.
  MFI->setMaxCallFrameSize(maxCallFrameSize);

  // Include call frame size in total.
  FrameSize += maxCallFrameSize;

  // Make sure the frame is aligned.
  FrameSize = (FrameSize + AlignMask) & ~AlignMask;

  // Update frame info.
  MFI->setStackSize(FrameSize);
}

void SPURegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                           RegScavenger *RS)
  const {
  // Mark LR and SP unused, since the prolog spills them to stack and
  // we don't want anyone else to spill them for us.
  //
  // Also, unless R2 is really used someday, don't spill it automatically.
  MF.getRegInfo().setPhysRegUnused(SPU::R0);
  MF.getRegInfo().setPhysRegUnused(SPU::R1);
  MF.getRegInfo().setPhysRegUnused(SPU::R2);
}

void SPURegisterInfo::emitPrologue(MachineFunction &MF) const
{
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();

  // Prepare for debug frame info.
  bool hasDebugInfo = MMI && MMI->hasDebugInfo();
  unsigned FrameLabelId = 0;

  // Move MBBI back to the beginning of the function.
  MBBI = MBB.begin();

  // Work out frame sizes.
  determineFrameLayout(MF);
  int FrameSize = MFI->getStackSize();

  assert((FrameSize & 0xf) == 0
         && "SPURegisterInfo::emitPrologue: FrameSize not aligned");

  if (FrameSize > 0 || MFI->hasCalls()) {
    FrameSize = -(FrameSize + SPUFrameInfo::minStackSize());
    if (hasDebugInfo) {
      // Mark effective beginning of when frame pointer becomes valid.
      FrameLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, TII.get(SPU::DBG_LABEL)).addImm(FrameLabelId);
    }

    // Adjust stack pointer, spilling $lr -> 16($sp) and $sp -> -FrameSize($sp)
    // for the ABI
    BuildMI(MBB, MBBI, TII.get(SPU::STQDr32), SPU::R0).addImm(16)
      .addReg(SPU::R1);
    if (isS10Constant(FrameSize)) {
      // Spill $sp to adjusted $sp
      BuildMI(MBB, MBBI, TII.get(SPU::STQDr32), SPU::R1).addImm(FrameSize)
        .addReg(SPU::R1);
      // Adjust $sp by required amout
      BuildMI(MBB, MBBI, TII.get(SPU::AIr32), SPU::R1).addReg(SPU::R1)
        .addImm(FrameSize);
    } else if (FrameSize <= (1 << 16) - 1 && FrameSize >= -(1 << 16)) {
      // Frame size can be loaded into ILr32n, so temporarily spill $r2 and use
      // $r2 to adjust $sp:
      BuildMI(MBB, MBBI, TII.get(SPU::STQDr128), SPU::R2)
        .addImm(-16)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, TII.get(SPU::ILr32), SPU::R2)
        .addImm(FrameSize);
      BuildMI(MBB, MBBI, TII.get(SPU::STQDr32), SPU::R1)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, TII.get(SPU::Ar32), SPU::R1)
        .addReg(SPU::R1)
        .addReg(SPU::R2);
      BuildMI(MBB, MBBI, TII.get(SPU::SFIr32), SPU::R2)
        .addReg(SPU::R2)
        .addImm(16);
      BuildMI(MBB, MBBI, TII.get(SPU::LQXr128), SPU::R2)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
    } else {
      cerr << "Unhandled frame size: " << FrameSize << "\n";
      abort();
    }

    if (hasDebugInfo) {
      std::vector<MachineMove> &Moves = MMI->getFrameMoves();

      // Show update of SP.
      MachineLocation SPDst(MachineLocation::VirtualFP);
      MachineLocation SPSrc(MachineLocation::VirtualFP, -FrameSize);
      Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));

      // Add callee saved registers to move list.
      const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
      for (unsigned I = 0, E = CSI.size(); I != E; ++I) {
        int Offset = MFI->getObjectOffset(CSI[I].getFrameIdx());
        unsigned Reg = CSI[I].getReg();
        if (Reg == SPU::R0) continue;
        MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
        MachineLocation CSSrc(Reg);
        Moves.push_back(MachineMove(FrameLabelId, CSDst, CSSrc));
      }

      // Mark effective beginning of when frame pointer is ready.
      unsigned ReadyLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, TII.get(SPU::DBG_LABEL)).addImm(ReadyLabelId);

      MachineLocation FPDst(SPU::R1);
      MachineLocation FPSrc(MachineLocation::VirtualFP);
      Moves.push_back(MachineMove(ReadyLabelId, FPDst, FPSrc));
    }
  } else {
    // This is a leaf function -- insert a branch hint iff there are
    // sufficient number instructions in the basic block. Note that
    // this is just a best guess based on the basic block's size.
    if (MBB.size() >= (unsigned) SPUFrameInfo::branchHintPenalty()) {
      MachineBasicBlock::iterator MBBI = prior(MBB.end());
      // Insert terminator label
      unsigned BranchLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, TII.get(SPU::DBG_LABEL)).addImm(BranchLabelId);
    }
  }
}

void
SPURegisterInfo::emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const
{
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int FrameSize = MFI->getStackSize();
  int LinkSlotOffset = SPUFrameInfo::stackSlotSize();

  assert(MBBI->getOpcode() == SPU::RET &&
         "Can only insert epilog into returning blocks");
  assert((FrameSize & 0xf) == 0
         && "SPURegisterInfo::emitEpilogue: FrameSize not aligned");
  if (FrameSize > 0 || MFI->hasCalls()) {
    FrameSize = FrameSize + SPUFrameInfo::minStackSize();
    if (isS10Constant(FrameSize + LinkSlotOffset)) {
      // Reload $lr, adjust $sp by required amount
      // Note: We do this to slightly improve dual issue -- not by much, but it
      // is an opportunity for dual issue.
      BuildMI(MBB, MBBI, TII.get(SPU::LQDr128), SPU::R0)
        .addImm(FrameSize + LinkSlotOffset)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, TII.get(SPU::AIr32), SPU::R1)
        .addReg(SPU::R1)
        .addImm(FrameSize);
    } else if (FrameSize <= (1 << 16) - 1 && FrameSize >= -(1 << 16)) {
      // Frame size can be loaded into ILr32n, so temporarily spill $r2 and use
      // $r2 to adjust $sp:
      BuildMI(MBB, MBBI, TII.get(SPU::STQDr128), SPU::R2)
        .addImm(16)
        .addReg(SPU::R1);
      BuildMI(MBB, MBBI, TII.get(SPU::ILr32), SPU::R2)
        .addImm(FrameSize);
      BuildMI(MBB, MBBI, TII.get(SPU::Ar32), SPU::R1)
        .addReg(SPU::R1)
        .addReg(SPU::R2);
      BuildMI(MBB, MBBI, TII.get(SPU::LQDr128), SPU::R0)
        .addImm(16)
        .addReg(SPU::R2);
      BuildMI(MBB, MBBI, TII.get(SPU::SFIr32), SPU::R2).
        addReg(SPU::R2)
        .addImm(16);
      BuildMI(MBB, MBBI, TII.get(SPU::LQXr128), SPU::R2)
        .addReg(SPU::R2)
        .addReg(SPU::R1);
    } else {
      cerr << "Unhandled frame size: " << FrameSize << "\n";
      abort();
    }
   }
}

unsigned
SPURegisterInfo::getRARegister() const
{
  return SPU::R0;
}

unsigned
SPURegisterInfo::getFrameRegister(MachineFunction &MF) const
{
  return SPU::R1;
}

void
SPURegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves) const
{
  // Initial state of the frame pointer is R1.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(SPU::R1, 0);
  Moves.push_back(MachineMove(0, Dst, Src));
}


int
SPURegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  // FIXME: Most probably dwarf numbers differs for Linux and Darwin
  return SPUGenRegisterInfo::getDwarfRegNumFull(RegNum, 0);
}

#include "SPUGenRegisterInfo.inc"
