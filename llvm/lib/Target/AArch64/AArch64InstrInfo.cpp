//===- AArch64InstrInfo.cpp - AArch64 Instruction Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "AArch64GenInstrInfo.inc"

static cl::opt<unsigned> TBZDisplacementBits(
    "aarch64-tbz-offset-bits", cl::Hidden, cl::init(14),
    cl::desc("Restrict range of TB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned> CBZDisplacementBits(
    "aarch64-cbz-offset-bits", cl::Hidden, cl::init(19),
    cl::desc("Restrict range of CB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned>
    BCCDisplacementBits("aarch64-bcc-offset-bits", cl::Hidden, cl::init(19),
                        cl::desc("Restrict range of Bcc instructions (DEBUG)"));

AArch64InstrInfo::AArch64InstrInfo(const AArch64Subtarget &STI)
    : AArch64GenInstrInfo(AArch64::ADJCALLSTACKDOWN, AArch64::ADJCALLSTACKUP),
      RI(STI.getTargetTriple()), Subtarget(STI) {}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
unsigned AArch64InstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  const MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction *MF = MBB.getParent();
  const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();

  if (MI.getOpcode() == AArch64::INLINEASM)
    return getInlineAsmLength(MI.getOperand(0).getSymbolName(), *MAI);

  // FIXME: We currently only handle pseudoinstructions that don't get expanded
  //        before the assembly printer.
  unsigned NumBytes = 0;
  const MCInstrDesc &Desc = MI.getDesc();
  switch (Desc.getOpcode()) {
  default:
    // Anything not explicitly designated otherwise is a normal 4-byte insn.
    NumBytes = 4;
    break;
  case TargetOpcode::DBG_VALUE:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
    NumBytes = 0;
    break;
  case TargetOpcode::STACKMAP:
    // The upper bound for a stackmap intrinsic is the full length of its shadow
    NumBytes = StackMapOpers(&MI).getNumPatchBytes();
    assert(NumBytes % 4 == 0 && "Invalid number of NOP bytes requested!");
    break;
  case TargetOpcode::PATCHPOINT:
    // The size of the patchpoint intrinsic is the number of bytes requested
    NumBytes = PatchPointOpers(&MI).getNumPatchBytes();
    assert(NumBytes % 4 == 0 && "Invalid number of NOP bytes requested!");
    break;
  case AArch64::TLSDESC_CALLSEQ:
    // This gets lowered to an instruction sequence which takes 16 bytes
    NumBytes = 16;
    break;
  }

  return NumBytes;
}

static void parseCondBranch(MachineInstr *LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  switch (LastInst->getOpcode()) {
  default:
    llvm_unreachable("Unknown branch instruction?");
  case AArch64::Bcc:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(LastInst->getOperand(0));
    break;
  case AArch64::CBZW:
  case AArch64::CBZX:
  case AArch64::CBNZW:
  case AArch64::CBNZX:
    Target = LastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    break;
  case AArch64::TBZW:
  case AArch64::TBZX:
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    Target = LastInst->getOperand(2).getMBB();
    Cond.push_back(MachineOperand::CreateImm(-1));
    Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
    Cond.push_back(LastInst->getOperand(0));
    Cond.push_back(LastInst->getOperand(1));
  }
}

static unsigned getBranchDisplacementBits(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("unexpected opcode!");
  case AArch64::B:
    return 64;
  case AArch64::TBNZW:
  case AArch64::TBZW:
  case AArch64::TBNZX:
  case AArch64::TBZX:
    return TBZDisplacementBits;
  case AArch64::CBNZW:
  case AArch64::CBZW:
  case AArch64::CBNZX:
  case AArch64::CBZX:
    return CBZDisplacementBits;
  case AArch64::Bcc:
    return BCCDisplacementBits;
  }
}

bool AArch64InstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                             int64_t BrOffset) const {
  unsigned Bits = getBranchDisplacementBits(BranchOp);
  assert(Bits >= 3 && "max branch displacement must be enough to jump"
                      "over conditional branch expansion");
  return isIntN(Bits, BrOffset / 4);
}

MachineBasicBlock *
AArch64InstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("unexpected opcode!");
  case AArch64::B:
    return MI.getOperand(0).getMBB();
  case AArch64::TBZW:
  case AArch64::TBNZW:
  case AArch64::TBZX:
  case AArch64::TBNZX:
    return MI.getOperand(2).getMBB();
  case AArch64::CBZW:
  case AArch64::CBNZW:
  case AArch64::CBZX:
  case AArch64::CBNZX:
  case AArch64::Bcc:
    return MI.getOperand(1).getMBB();
  }
}

// Branch analysis.
bool AArch64InstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                     MachineBasicBlock *&FBB,
                                     SmallVectorImpl<MachineOperand> &Cond,
                                     bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return false;

  if (!isUnpredicatedTerminator(*I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = &*I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
    if (isUncondBranchOpcode(LastOpc)) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isCondBranchOpcode(LastOpc)) {
      // Block ends with fall-through condbranch.
      parseCondBranch(LastInst, TBB, Cond);
      return false;
    }
    return true; // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = &*I;
  unsigned SecondLastOpc = SecondLastInst->getOpcode();

  // If AllowModify is true and the block ends with two or more unconditional
  // branches, delete all but the first unconditional branch.
  if (AllowModify && isUncondBranchOpcode(LastOpc)) {
    while (isUncondBranchOpcode(SecondLastOpc)) {
      LastInst->eraseFromParent();
      LastInst = SecondLastInst;
      LastOpc = LastInst->getOpcode();
      if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
        // Return now the only terminator is an unconditional branch.
        TBB = LastInst->getOperand(0).getMBB();
        return false;
      } else {
        SecondLastInst = &*I;
        SecondLastOpc = SecondLastInst->getOpcode();
      }
    }
  }

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(*--I))
    return true;

  // If the block ends with a B and a Bcc, handle it.
  if (isCondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    parseCondBranch(SecondLastInst, TBB, Cond);
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (isUncondBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // ...likewise if it ends with an indirect branch followed by an unconditional
  // branch.
  if (isIndirectBranchOpcode(SecondLastOpc) && isUncondBranchOpcode(LastOpc)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}

bool AArch64InstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    AArch64CC::CondCode CC = (AArch64CC::CondCode)(int)Cond[0].getImm();
    Cond[0].setImm(AArch64CC::getInvertedCondCode(CC));
  } else {
    // Folded compare-and-branch
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown conditional branch!");
    case AArch64::CBZW:
      Cond[1].setImm(AArch64::CBNZW);
      break;
    case AArch64::CBNZW:
      Cond[1].setImm(AArch64::CBZW);
      break;
    case AArch64::CBZX:
      Cond[1].setImm(AArch64::CBNZX);
      break;
    case AArch64::CBNZX:
      Cond[1].setImm(AArch64::CBZX);
      break;
    case AArch64::TBZW:
      Cond[1].setImm(AArch64::TBNZW);
      break;
    case AArch64::TBNZW:
      Cond[1].setImm(AArch64::TBZW);
      break;
    case AArch64::TBZX:
      Cond[1].setImm(AArch64::TBNZX);
      break;
    case AArch64::TBNZX:
      Cond[1].setImm(AArch64::TBZX);
      break;
    }
  }

  return false;
}

unsigned AArch64InstrInfo::removeBranch(MachineBasicBlock &MBB,
                                        int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!isUncondBranchOpcode(I->getOpcode()) &&
      !isCondBranchOpcode(I->getOpcode()))
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) {
    if (BytesRemoved)
      *BytesRemoved = 4;
    return 1;
  }
  --I;
  if (!isCondBranchOpcode(I->getOpcode())) {
    if (BytesRemoved)
      *BytesRemoved = 4;
    return 1;
  }

  // Remove the branch.
  I->eraseFromParent();
  if (BytesRemoved)
    *BytesRemoved = 8;

  return 2;
}

void AArch64InstrInfo::instantiateCondBranch(
    MachineBasicBlock &MBB, const DebugLoc &DL, MachineBasicBlock *TBB,
    ArrayRef<MachineOperand> Cond) const {
  if (Cond[0].getImm() != -1) {
    // Regular Bcc
    BuildMI(&MBB, DL, get(AArch64::Bcc)).addImm(Cond[0].getImm()).addMBB(TBB);
  } else {
    // Folded compare-and-branch
    // Note that we use addOperand instead of addReg to keep the flags.
    const MachineInstrBuilder MIB =
        BuildMI(&MBB, DL, get(Cond[1].getImm())).add(Cond[2]);
    if (Cond.size() > 3)
      MIB.addImm(Cond[3].getImm());
    MIB.addMBB(TBB);
  }
}

unsigned AArch64InstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");

  if (!FBB) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, DL, get(AArch64::B)).addMBB(TBB);
    else
      instantiateCondBranch(MBB, DL, TBB, Cond);

    if (BytesAdded)
      *BytesAdded = 4;

    return 1;
  }

  // Two-way conditional branch.
  instantiateCondBranch(MBB, DL, TBB, Cond);
  BuildMI(&MBB, DL, get(AArch64::B)).addMBB(FBB);

  if (BytesAdded)
    *BytesAdded = 8;

  return 2;
}

// Find the original register that VReg is copied from.
static unsigned removeCopies(const MachineRegisterInfo &MRI, unsigned VReg) {
  while (TargetRegisterInfo::isVirtualRegister(VReg)) {
    const MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (!DefMI->isFullCopy())
      return VReg;
    VReg = DefMI->getOperand(1).getReg();
  }
  return VReg;
}

// Determine if VReg is defined by an instruction that can be folded into a
// csel instruction. If so, return the folded opcode, and the replacement
// register.
static unsigned canFoldIntoCSel(const MachineRegisterInfo &MRI, unsigned VReg,
                                unsigned *NewVReg = nullptr) {
  VReg = removeCopies(MRI, VReg);
  if (!TargetRegisterInfo::isVirtualRegister(VReg))
    return 0;

  bool Is64Bit = AArch64::GPR64allRegClass.hasSubClassEq(MRI.getRegClass(VReg));
  const MachineInstr *DefMI = MRI.getVRegDef(VReg);
  unsigned Opc = 0;
  unsigned SrcOpNum = 0;
  switch (DefMI->getOpcode()) {
  case AArch64::ADDSXri:
  case AArch64::ADDSWri:
    // if NZCV is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(AArch64::NZCV, true) == -1)
      return 0;
    // fall-through to ADDXri and ADDWri.
    LLVM_FALLTHROUGH;
  case AArch64::ADDXri:
  case AArch64::ADDWri:
    // add x, 1 -> csinc.
    if (!DefMI->getOperand(2).isImm() || DefMI->getOperand(2).getImm() != 1 ||
        DefMI->getOperand(3).getImm() != 0)
      return 0;
    SrcOpNum = 1;
    Opc = Is64Bit ? AArch64::CSINCXr : AArch64::CSINCWr;
    break;

  case AArch64::ORNXrr:
  case AArch64::ORNWrr: {
    // not x -> csinv, represented as orn dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != AArch64::XZR && ZReg != AArch64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? AArch64::CSINVXr : AArch64::CSINVWr;
    break;
  }

  case AArch64::SUBSXrr:
  case AArch64::SUBSWrr:
    // if NZCV is used, do not fold.
    if (DefMI->findRegisterDefOperandIdx(AArch64::NZCV, true) == -1)
      return 0;
    // fall-through to SUBXrr and SUBWrr.
    LLVM_FALLTHROUGH;
  case AArch64::SUBXrr:
  case AArch64::SUBWrr: {
    // neg x -> csneg, represented as sub dst, xzr, src.
    unsigned ZReg = removeCopies(MRI, DefMI->getOperand(1).getReg());
    if (ZReg != AArch64::XZR && ZReg != AArch64::WZR)
      return 0;
    SrcOpNum = 2;
    Opc = Is64Bit ? AArch64::CSNEGXr : AArch64::CSNEGWr;
    break;
  }
  default:
    return 0;
  }
  assert(Opc && SrcOpNum && "Missing parameters");

  if (NewVReg)
    *NewVReg = DefMI->getOperand(SrcOpNum).getReg();
  return Opc;
}

bool AArch64InstrInfo::canInsertSelect(const MachineBasicBlock &MBB,
                                       ArrayRef<MachineOperand> Cond,
                                       unsigned TrueReg, unsigned FalseReg,
                                       int &CondCycles, int &TrueCycles,
                                       int &FalseCycles) const {
  // Check register classes.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
      RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  if (!RC)
    return false;

  // Expanding cbz/tbz requires an extra cycle of latency on the condition.
  unsigned ExtraCondLat = Cond.size() != 1;

  // GPRs are handled by csel.
  // FIXME: Fold in x+1, -x, and ~x when applicable.
  if (AArch64::GPR64allRegClass.hasSubClassEq(RC) ||
      AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
    // Single-cycle csel, csinc, csinv, and csneg.
    CondCycles = 1 + ExtraCondLat;
    TrueCycles = FalseCycles = 1;
    if (canFoldIntoCSel(MRI, TrueReg))
      TrueCycles = 0;
    else if (canFoldIntoCSel(MRI, FalseReg))
      FalseCycles = 0;
    return true;
  }

  // Scalar floating point is handled by fcsel.
  // FIXME: Form fabs, fmin, and fmax when applicable.
  if (AArch64::FPR64RegClass.hasSubClassEq(RC) ||
      AArch64::FPR32RegClass.hasSubClassEq(RC)) {
    CondCycles = 5 + ExtraCondLat;
    TrueCycles = FalseCycles = 2;
    return true;
  }

  // Can't do vectors.
  return false;
}

void AArch64InstrInfo::insertSelect(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    const DebugLoc &DL, unsigned DstReg,
                                    ArrayRef<MachineOperand> Cond,
                                    unsigned TrueReg, unsigned FalseReg) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // Parse the condition code, see parseCondBranch() above.
  AArch64CC::CondCode CC;
  switch (Cond.size()) {
  default:
    llvm_unreachable("Unknown condition opcode in Cond");
  case 1: // b.cc
    CC = AArch64CC::CondCode(Cond[0].getImm());
    break;
  case 3: { // cbz/cbnz
    // We must insert a compare against 0.
    bool Is64Bit;
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown branch opcode in Cond");
    case AArch64::CBZW:
      Is64Bit = false;
      CC = AArch64CC::EQ;
      break;
    case AArch64::CBZX:
      Is64Bit = true;
      CC = AArch64CC::EQ;
      break;
    case AArch64::CBNZW:
      Is64Bit = false;
      CC = AArch64CC::NE;
      break;
    case AArch64::CBNZX:
      Is64Bit = true;
      CC = AArch64CC::NE;
      break;
    }
    unsigned SrcReg = Cond[2].getReg();
    if (Is64Bit) {
      // cmp reg, #0 is actually subs xzr, reg, #0.
      MRI.constrainRegClass(SrcReg, &AArch64::GPR64spRegClass);
      BuildMI(MBB, I, DL, get(AArch64::SUBSXri), AArch64::XZR)
          .addReg(SrcReg)
          .addImm(0)
          .addImm(0);
    } else {
      MRI.constrainRegClass(SrcReg, &AArch64::GPR32spRegClass);
      BuildMI(MBB, I, DL, get(AArch64::SUBSWri), AArch64::WZR)
          .addReg(SrcReg)
          .addImm(0)
          .addImm(0);
    }
    break;
  }
  case 4: { // tbz/tbnz
    // We must insert a tst instruction.
    switch (Cond[1].getImm()) {
    default:
      llvm_unreachable("Unknown branch opcode in Cond");
    case AArch64::TBZW:
    case AArch64::TBZX:
      CC = AArch64CC::EQ;
      break;
    case AArch64::TBNZW:
    case AArch64::TBNZX:
      CC = AArch64CC::NE;
      break;
    }
    // cmp reg, #foo is actually ands xzr, reg, #1<<foo.
    if (Cond[1].getImm() == AArch64::TBZW || Cond[1].getImm() == AArch64::TBNZW)
      BuildMI(MBB, I, DL, get(AArch64::ANDSWri), AArch64::WZR)
          .addReg(Cond[2].getReg())
          .addImm(
              AArch64_AM::encodeLogicalImmediate(1ull << Cond[3].getImm(), 32));
    else
      BuildMI(MBB, I, DL, get(AArch64::ANDSXri), AArch64::XZR)
          .addReg(Cond[2].getReg())
          .addImm(
              AArch64_AM::encodeLogicalImmediate(1ull << Cond[3].getImm(), 64));
    break;
  }
  }

  unsigned Opc = 0;
  const TargetRegisterClass *RC = nullptr;
  bool TryFold = false;
  if (MRI.constrainRegClass(DstReg, &AArch64::GPR64RegClass)) {
    RC = &AArch64::GPR64RegClass;
    Opc = AArch64::CSELXr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::GPR32RegClass)) {
    RC = &AArch64::GPR32RegClass;
    Opc = AArch64::CSELWr;
    TryFold = true;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::FPR64RegClass)) {
    RC = &AArch64::FPR64RegClass;
    Opc = AArch64::FCSELDrrr;
  } else if (MRI.constrainRegClass(DstReg, &AArch64::FPR32RegClass)) {
    RC = &AArch64::FPR32RegClass;
    Opc = AArch64::FCSELSrrr;
  }
  assert(RC && "Unsupported regclass");

  // Try folding simple instructions into the csel.
  if (TryFold) {
    unsigned NewVReg = 0;
    unsigned FoldedOpc = canFoldIntoCSel(MRI, TrueReg, &NewVReg);
    if (FoldedOpc) {
      // The folded opcodes csinc, csinc and csneg apply the operation to
      // FalseReg, so we need to invert the condition.
      CC = AArch64CC::getInvertedCondCode(CC);
      TrueReg = FalseReg;
    } else
      FoldedOpc = canFoldIntoCSel(MRI, FalseReg, &NewVReg);

    // Fold the operation. Leave any dead instructions for DCE to clean up.
    if (FoldedOpc) {
      FalseReg = NewVReg;
      Opc = FoldedOpc;
      // The extends the live range of NewVReg.
      MRI.clearKillFlags(NewVReg);
    }
  }

  // Pull all virtual register into the appropriate class.
  MRI.constrainRegClass(TrueReg, RC);
  MRI.constrainRegClass(FalseReg, RC);

  // Insert the csel.
  BuildMI(MBB, I, DL, get(Opc), DstReg)
      .addReg(TrueReg)
      .addReg(FalseReg)
      .addImm(CC);
}

/// Returns true if a MOVi32imm or MOVi64imm can be expanded to an  ORRxx.
static bool canBeExpandedToORR(const MachineInstr &MI, unsigned BitSize) {
  uint64_t Imm = MI.getOperand(1).getImm();
  uint64_t UImm = Imm << (64 - BitSize) >> (64 - BitSize);
  uint64_t Encoding;
  return AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding);
}

// FIXME: this implementation should be micro-architecture dependent, so a
// micro-architecture target hook should be introduced here in future.
bool AArch64InstrInfo::isAsCheapAsAMove(const MachineInstr &MI) const {
  if (!Subtarget.hasCustomCheapAsMoveHandling())
    return MI.isAsCheapAsAMove();
  if (Subtarget.getProcFamily() == AArch64Subtarget::ExynosM1 &&
      isExynosShiftLeftFast(MI))
    return true;

  switch (MI.getOpcode()) {
  default:
    return false;

  // add/sub on register without shift
  case AArch64::ADDWri:
  case AArch64::ADDXri:
  case AArch64::SUBWri:
  case AArch64::SUBXri:
    return (MI.getOperand(3).getImm() == 0);

  // logical ops on immediate
  case AArch64::ANDWri:
  case AArch64::ANDXri:
  case AArch64::EORWri:
  case AArch64::EORXri:
  case AArch64::ORRWri:
  case AArch64::ORRXri:
    return true;

  // logical ops on register without shift
  case AArch64::ANDWrr:
  case AArch64::ANDXrr:
  case AArch64::BICWrr:
  case AArch64::BICXrr:
  case AArch64::EONWrr:
  case AArch64::EONXrr:
  case AArch64::EORWrr:
  case AArch64::EORXrr:
  case AArch64::ORNWrr:
  case AArch64::ORNXrr:
  case AArch64::ORRWrr:
  case AArch64::ORRXrr:
    return true;

  // If MOVi32imm or MOVi64imm can be expanded into ORRWri or
  // ORRXri, it is as cheap as MOV
  case AArch64::MOVi32imm:
    return canBeExpandedToORR(MI, 32);
  case AArch64::MOVi64imm:
    return canBeExpandedToORR(MI, 64);

  // It is cheap to zero out registers if the subtarget has ZeroCycleZeroing
  // feature.
  case AArch64::FMOVH0:
  case AArch64::FMOVS0:
  case AArch64::FMOVD0:
    return Subtarget.hasZeroCycleZeroing();
  case TargetOpcode::COPY:
    return (Subtarget.hasZeroCycleZeroing() &&
            (MI.getOperand(1).getReg() == AArch64::WZR ||
             MI.getOperand(1).getReg() == AArch64::XZR));
  }

  llvm_unreachable("Unknown opcode to check as cheap as a move!");
}

bool AArch64InstrInfo::isExynosShiftLeftFast(const MachineInstr &MI) const {
  unsigned Imm, Shift;
  AArch64_AM::ShiftExtendType Ext;

  switch (MI.getOpcode()) {
  default:
    return false;

  // WriteI
  case AArch64::ADDSWri:
  case AArch64::ADDSXri:
  case AArch64::ADDWri:
  case AArch64::ADDXri:
  case AArch64::SUBSWri:
  case AArch64::SUBSXri:
  case AArch64::SUBWri:
  case AArch64::SUBXri:
    return true;

  // WriteISReg
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::ANDWrs:
  case AArch64::ANDXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
  case AArch64::BICWrs:
  case AArch64::BICXrs:
  case AArch64::EONWrs:
  case AArch64::EONXrs:
  case AArch64::EORWrs:
  case AArch64::EORXrs:
  case AArch64::ORNWrs:
  case AArch64::ORNXrs:
  case AArch64::ORRWrs:
  case AArch64::ORRXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::SUBWrs:
  case AArch64::SUBXrs:
    Imm = MI.getOperand(3).getImm();
    Shift = AArch64_AM::getShiftValue(Imm);
    Ext = AArch64_AM::getShiftType(Imm);
    return (Shift == 0 || (Shift <= 3 && Ext == AArch64_AM::LSL));

  // WriteIEReg
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrx:
  case AArch64::ADDSXrx64:
  case AArch64::ADDWrx:
  case AArch64::ADDXrx:
  case AArch64::ADDXrx64:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrx:
  case AArch64::SUBSXrx64:
  case AArch64::SUBWrx:
  case AArch64::SUBXrx:
  case AArch64::SUBXrx64:
    Imm = MI.getOperand(3).getImm();
    Shift = AArch64_AM::getArithShiftValue(Imm);
    Ext = AArch64_AM::getArithExtendType(Imm);
    return (Shift == 0 || (Shift <= 3 && Ext == AArch64_AM::UXTX));

  case AArch64::PRFMroW:
  case AArch64::PRFMroX:

  // WriteLDIdx
  case AArch64::LDRBBroW:
  case AArch64::LDRBBroX:
  case AArch64::LDRHHroW:
  case AArch64::LDRHHroX:
  case AArch64::LDRSBWroW:
  case AArch64::LDRSBWroX:
  case AArch64::LDRSBXroW:
  case AArch64::LDRSBXroX:
  case AArch64::LDRSHWroW:
  case AArch64::LDRSHWroX:
  case AArch64::LDRSHXroW:
  case AArch64::LDRSHXroX:
  case AArch64::LDRSWroW:
  case AArch64::LDRSWroX:
  case AArch64::LDRWroW:
  case AArch64::LDRWroX:
  case AArch64::LDRXroW:
  case AArch64::LDRXroX:

  case AArch64::LDRBroW:
  case AArch64::LDRBroX:
  case AArch64::LDRDroW:
  case AArch64::LDRDroX:
  case AArch64::LDRHroW:
  case AArch64::LDRHroX:
  case AArch64::LDRSroW:
  case AArch64::LDRSroX:

  // WriteSTIdx
  case AArch64::STRBBroW:
  case AArch64::STRBBroX:
  case AArch64::STRHHroW:
  case AArch64::STRHHroX:
  case AArch64::STRWroW:
  case AArch64::STRWroX:
  case AArch64::STRXroW:
  case AArch64::STRXroX:

  case AArch64::STRBroW:
  case AArch64::STRBroX:
  case AArch64::STRDroW:
  case AArch64::STRDroX:
  case AArch64::STRHroW:
  case AArch64::STRHroX:
  case AArch64::STRSroW:
  case AArch64::STRSroX:
    Imm = MI.getOperand(3).getImm();
    Ext = AArch64_AM::getMemExtendType(Imm);
    return (Ext == AArch64_AM::SXTX || Ext == AArch64_AM::UXTX);
  }
}

bool AArch64InstrInfo::isFalkorShiftExtFast(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    return false;

  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs: {
    unsigned Imm = MI.getOperand(3).getImm();
    unsigned ShiftVal = AArch64_AM::getShiftValue(Imm);
    if (ShiftVal == 0)
      return true;
    return AArch64_AM::getShiftType(Imm) == AArch64_AM::LSL && ShiftVal <= 5;
  }

  case AArch64::ADDWrx:
  case AArch64::ADDXrx:
  case AArch64::ADDXrx64:
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrx:
  case AArch64::ADDSXrx64: {
    unsigned Imm = MI.getOperand(3).getImm();
    switch (AArch64_AM::getArithExtendType(Imm)) {
    default:
      return false;
    case AArch64_AM::UXTB:
    case AArch64_AM::UXTH:
    case AArch64_AM::UXTW:
    case AArch64_AM::UXTX:
      return AArch64_AM::getArithShiftValue(Imm) <= 4;
    }
  }

  case AArch64::SUBWrs:
  case AArch64::SUBSWrs: {
    unsigned Imm = MI.getOperand(3).getImm();
    unsigned ShiftVal = AArch64_AM::getShiftValue(Imm);
    return ShiftVal == 0 ||
           (AArch64_AM::getShiftType(Imm) == AArch64_AM::ASR && ShiftVal == 31);
  }

  case AArch64::SUBXrs:
  case AArch64::SUBSXrs: {
    unsigned Imm = MI.getOperand(3).getImm();
    unsigned ShiftVal = AArch64_AM::getShiftValue(Imm);
    return ShiftVal == 0 ||
           (AArch64_AM::getShiftType(Imm) == AArch64_AM::ASR && ShiftVal == 63);
  }

  case AArch64::SUBWrx:
  case AArch64::SUBXrx:
  case AArch64::SUBXrx64:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrx:
  case AArch64::SUBSXrx64: {
    unsigned Imm = MI.getOperand(3).getImm();
    switch (AArch64_AM::getArithExtendType(Imm)) {
    default:
      return false;
    case AArch64_AM::UXTB:
    case AArch64_AM::UXTH:
    case AArch64_AM::UXTW:
    case AArch64_AM::UXTX:
      return AArch64_AM::getArithShiftValue(Imm) == 0;
    }
  }

  case AArch64::LDRBBroW:
  case AArch64::LDRBBroX:
  case AArch64::LDRBroW:
  case AArch64::LDRBroX:
  case AArch64::LDRDroW:
  case AArch64::LDRDroX:
  case AArch64::LDRHHroW:
  case AArch64::LDRHHroX:
  case AArch64::LDRHroW:
  case AArch64::LDRHroX:
  case AArch64::LDRQroW:
  case AArch64::LDRQroX:
  case AArch64::LDRSBWroW:
  case AArch64::LDRSBWroX:
  case AArch64::LDRSBXroW:
  case AArch64::LDRSBXroX:
  case AArch64::LDRSHWroW:
  case AArch64::LDRSHWroX:
  case AArch64::LDRSHXroW:
  case AArch64::LDRSHXroX:
  case AArch64::LDRSWroW:
  case AArch64::LDRSWroX:
  case AArch64::LDRSroW:
  case AArch64::LDRSroX:
  case AArch64::LDRWroW:
  case AArch64::LDRWroX:
  case AArch64::LDRXroW:
  case AArch64::LDRXroX:
  case AArch64::PRFMroW:
  case AArch64::PRFMroX:
  case AArch64::STRBBroW:
  case AArch64::STRBBroX:
  case AArch64::STRBroW:
  case AArch64::STRBroX:
  case AArch64::STRDroW:
  case AArch64::STRDroX:
  case AArch64::STRHHroW:
  case AArch64::STRHHroX:
  case AArch64::STRHroW:
  case AArch64::STRHroX:
  case AArch64::STRQroW:
  case AArch64::STRQroX:
  case AArch64::STRSroW:
  case AArch64::STRSroX:
  case AArch64::STRWroW:
  case AArch64::STRWroX:
  case AArch64::STRXroW:
  case AArch64::STRXroX: {
    unsigned IsSigned = MI.getOperand(3).getImm();
    return !IsSigned;
  }
  }
}

bool AArch64InstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                             unsigned &SrcReg, unsigned &DstReg,
                                             unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::SBFMXri: // aka sxtw
  case AArch64::UBFMXri: // aka uxtw
    // Check for the 32 -> 64 bit extension case, these instructions can do
    // much more.
    if (MI.getOperand(2).getImm() != 0 || MI.getOperand(3).getImm() != 31)
      return false;
    // This is a signed or unsigned 32 -> 64 bit extension.
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SubIdx = AArch64::sub_32;
    return true;
  }
}

bool AArch64InstrInfo::areMemAccessesTriviallyDisjoint(
    MachineInstr &MIa, MachineInstr &MIb, AliasAnalysis *AA) const {
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  unsigned BaseRegA = 0, BaseRegB = 0;
  int64_t OffsetA = 0, OffsetB = 0;
  unsigned WidthA = 0, WidthB = 0;

  assert(MIa.mayLoadOrStore() && "MIa must be a load or store.");
  assert(MIb.mayLoadOrStore() && "MIb must be a load or store.");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects() ||
      MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  // Retrieve the base register, offset from the base register and width. Width
  // is the size of memory that is being loaded/stored (e.g. 1, 2, 4, 8).  If
  // base registers are identical, and the offset of a lower memory access +
  // the width doesn't overlap the offset of a higher memory access,
  // then the memory accesses are different.
  if (getMemOpBaseRegImmOfsWidth(MIa, BaseRegA, OffsetA, WidthA, TRI) &&
      getMemOpBaseRegImmOfsWidth(MIb, BaseRegB, OffsetB, WidthB, TRI)) {
    if (BaseRegA == BaseRegB) {
      int LowOffset = OffsetA < OffsetB ? OffsetA : OffsetB;
      int HighOffset = OffsetA < OffsetB ? OffsetB : OffsetA;
      int LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
      if (LowOffset + LowWidth <= HighOffset)
        return true;
    }
  }
  return false;
}

/// analyzeCompare - For a comparison instruction, return the source registers
/// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
/// Return true if the comparison instruction can be analyzed.
bool AArch64InstrInfo::analyzeCompare(const MachineInstr &MI, unsigned &SrcReg,
                                      unsigned &SrcReg2, int &CmpMask,
                                      int &CmpValue) const {
  // The first operand can be a frame index where we'd normally expect a
  // register.
  assert(MI.getNumOperands() >= 2 && "All AArch64 cmps should have 2 operands");
  if (!MI.getOperand(1).isReg())
    return false;

  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::SUBSWrr:
  case AArch64::SUBSWrs:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrr:
  case AArch64::SUBSXrs:
  case AArch64::SUBSXrx:
  case AArch64::ADDSWrr:
  case AArch64::ADDSWrs:
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXrs:
  case AArch64::ADDSXrx:
    // Replace SUBSWrr with SUBWrr if NZCV is not used.
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = MI.getOperand(2).getReg();
    CmpMask = ~0;
    CmpValue = 0;
    return true;
  case AArch64::SUBSWri:
  case AArch64::ADDSWri:
  case AArch64::SUBSXri:
  case AArch64::ADDSXri:
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    // FIXME: In order to convert CmpValue to 0 or 1
    CmpValue = MI.getOperand(2).getImm() != 0;
    return true;
  case AArch64::ANDSWri:
  case AArch64::ANDSXri:
    // ANDS does not use the same encoding scheme as the others xxxS
    // instructions.
    SrcReg = MI.getOperand(1).getReg();
    SrcReg2 = 0;
    CmpMask = ~0;
    // FIXME:The return val type of decodeLogicalImmediate is uint64_t,
    // while the type of CmpValue is int. When converting uint64_t to int,
    // the high 32 bits of uint64_t will be lost.
    // In fact it causes a bug in spec2006-483.xalancbmk
    // CmpValue is only used to compare with zero in OptimizeCompareInstr
    CmpValue = AArch64_AM::decodeLogicalImmediate(
                   MI.getOperand(2).getImm(),
                   MI.getOpcode() == AArch64::ANDSWri ? 32 : 64) != 0;
    return true;
  }

  return false;
}

static bool UpdateOperandRegClass(MachineInstr &Instr) {
  MachineBasicBlock *MBB = Instr.getParent();
  assert(MBB && "Can't get MachineBasicBlock here");
  MachineFunction *MF = MBB->getParent();
  assert(MF && "Can't get MachineFunction here");
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  MachineRegisterInfo *MRI = &MF->getRegInfo();

  for (unsigned OpIdx = 0, EndIdx = Instr.getNumOperands(); OpIdx < EndIdx;
       ++OpIdx) {
    MachineOperand &MO = Instr.getOperand(OpIdx);
    const TargetRegisterClass *OpRegCstraints =
        Instr.getRegClassConstraint(OpIdx, TII, TRI);

    // If there's no constraint, there's nothing to do.
    if (!OpRegCstraints)
      continue;
    // If the operand is a frame index, there's nothing to do here.
    // A frame index operand will resolve correctly during PEI.
    if (MO.isFI())
      continue;

    assert(MO.isReg() &&
           "Operand has register constraints without being a register!");

    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (!OpRegCstraints->contains(Reg))
        return false;
    } else if (!OpRegCstraints->hasSubClassEq(MRI->getRegClass(Reg)) &&
               !MRI->constrainRegClass(Reg, OpRegCstraints))
      return false;
  }

  return true;
}

/// \brief Return the opcode that does not set flags when possible - otherwise
/// return the original opcode. The caller is responsible to do the actual
/// substitution and legality checking.
static unsigned convertToNonFlagSettingOpc(const MachineInstr &MI) {
  // Don't convert all compare instructions, because for some the zero register
  // encoding becomes the sp register.
  bool MIDefinesZeroReg = false;
  if (MI.definesRegister(AArch64::WZR) || MI.definesRegister(AArch64::XZR))
    MIDefinesZeroReg = true;

  switch (MI.getOpcode()) {
  default:
    return MI.getOpcode();
  case AArch64::ADDSWrr:
    return AArch64::ADDWrr;
  case AArch64::ADDSWri:
    return MIDefinesZeroReg ? AArch64::ADDSWri : AArch64::ADDWri;
  case AArch64::ADDSWrs:
    return MIDefinesZeroReg ? AArch64::ADDSWrs : AArch64::ADDWrs;
  case AArch64::ADDSWrx:
    return AArch64::ADDWrx;
  case AArch64::ADDSXrr:
    return AArch64::ADDXrr;
  case AArch64::ADDSXri:
    return MIDefinesZeroReg ? AArch64::ADDSXri : AArch64::ADDXri;
  case AArch64::ADDSXrs:
    return MIDefinesZeroReg ? AArch64::ADDSXrs : AArch64::ADDXrs;
  case AArch64::ADDSXrx:
    return AArch64::ADDXrx;
  case AArch64::SUBSWrr:
    return AArch64::SUBWrr;
  case AArch64::SUBSWri:
    return MIDefinesZeroReg ? AArch64::SUBSWri : AArch64::SUBWri;
  case AArch64::SUBSWrs:
    return MIDefinesZeroReg ? AArch64::SUBSWrs : AArch64::SUBWrs;
  case AArch64::SUBSWrx:
    return AArch64::SUBWrx;
  case AArch64::SUBSXrr:
    return AArch64::SUBXrr;
  case AArch64::SUBSXri:
    return MIDefinesZeroReg ? AArch64::SUBSXri : AArch64::SUBXri;
  case AArch64::SUBSXrs:
    return MIDefinesZeroReg ? AArch64::SUBSXrs : AArch64::SUBXrs;
  case AArch64::SUBSXrx:
    return AArch64::SUBXrx;
  }
}

enum AccessKind { AK_Write = 0x01, AK_Read = 0x10, AK_All = 0x11 };

/// True when condition flags are accessed (either by writing or reading)
/// on the instruction trace starting at From and ending at To.
///
/// Note: If From and To are from different blocks it's assumed CC are accessed
///       on the path.
static bool areCFlagsAccessedBetweenInstrs(
    MachineBasicBlock::iterator From, MachineBasicBlock::iterator To,
    const TargetRegisterInfo *TRI, const AccessKind AccessToCheck = AK_All) {
  // Early exit if To is at the beginning of the BB.
  if (To == To->getParent()->begin())
    return true;

  // Check whether the instructions are in the same basic block
  // If not, assume the condition flags might get modified somewhere.
  if (To->getParent() != From->getParent())
    return true;

  // From must be above To.
  assert(std::find_if(++To.getReverse(), To->getParent()->rend(),
                      [From](MachineInstr &MI) {
                        return MI.getIterator() == From;
                      }) != To->getParent()->rend());

  // We iterate backward starting \p To until we hit \p From.
  for (--To; To != From; --To) {
    const MachineInstr &Instr = *To;

    if (((AccessToCheck & AK_Write) &&
         Instr.modifiesRegister(AArch64::NZCV, TRI)) ||
        ((AccessToCheck & AK_Read) && Instr.readsRegister(AArch64::NZCV, TRI)))
      return true;
  }
  return false;
}

/// Try to optimize a compare instruction. A compare instruction is an
/// instruction which produces AArch64::NZCV. It can be truly compare
/// instruction
/// when there are no uses of its destination register.
///
/// The following steps are tried in order:
/// 1. Convert CmpInstr into an unconditional version.
/// 2. Remove CmpInstr if above there is an instruction producing a needed
///    condition code or an instruction which can be converted into such an
///    instruction.
///    Only comparison with zero is supported.
bool AArch64InstrInfo::optimizeCompareInstr(
    MachineInstr &CmpInstr, unsigned SrcReg, unsigned SrcReg2, int CmpMask,
    int CmpValue, const MachineRegisterInfo *MRI) const {
  assert(CmpInstr.getParent());
  assert(MRI);

  // Replace SUBSWrr with SUBWrr if NZCV is not used.
  int DeadNZCVIdx = CmpInstr.findRegisterDefOperandIdx(AArch64::NZCV, true);
  if (DeadNZCVIdx != -1) {
    if (CmpInstr.definesRegister(AArch64::WZR) ||
        CmpInstr.definesRegister(AArch64::XZR)) {
      CmpInstr.eraseFromParent();
      return true;
    }
    unsigned Opc = CmpInstr.getOpcode();
    unsigned NewOpc = convertToNonFlagSettingOpc(CmpInstr);
    if (NewOpc == Opc)
      return false;
    const MCInstrDesc &MCID = get(NewOpc);
    CmpInstr.setDesc(MCID);
    CmpInstr.RemoveOperand(DeadNZCVIdx);
    bool succeeded = UpdateOperandRegClass(CmpInstr);
    (void)succeeded;
    assert(succeeded && "Some operands reg class are incompatible!");
    return true;
  }

  // Continue only if we have a "ri" where immediate is zero.
  // FIXME:CmpValue has already been converted to 0 or 1 in analyzeCompare
  // function.
  assert((CmpValue == 0 || CmpValue == 1) && "CmpValue must be 0 or 1!");
  if (CmpValue != 0 || SrcReg2 != 0)
    return false;

  // CmpInstr is a Compare instruction if destination register is not used.
  if (!MRI->use_nodbg_empty(CmpInstr.getOperand(0).getReg()))
    return false;

  return substituteCmpToZero(CmpInstr, SrcReg, MRI);
}

/// Get opcode of S version of Instr.
/// If Instr is S version its opcode is returned.
/// AArch64::INSTRUCTION_LIST_END is returned if Instr does not have S version
/// or we are not interested in it.
static unsigned sForm(MachineInstr &Instr) {
  switch (Instr.getOpcode()) {
  default:
    return AArch64::INSTRUCTION_LIST_END;

  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSWri:
  case AArch64::SUBSXrr:
  case AArch64::SUBSXri:
    return Instr.getOpcode();

  case AArch64::ADDWrr:
    return AArch64::ADDSWrr;
  case AArch64::ADDWri:
    return AArch64::ADDSWri;
  case AArch64::ADDXrr:
    return AArch64::ADDSXrr;
  case AArch64::ADDXri:
    return AArch64::ADDSXri;
  case AArch64::ADCWr:
    return AArch64::ADCSWr;
  case AArch64::ADCXr:
    return AArch64::ADCSXr;
  case AArch64::SUBWrr:
    return AArch64::SUBSWrr;
  case AArch64::SUBWri:
    return AArch64::SUBSWri;
  case AArch64::SUBXrr:
    return AArch64::SUBSXrr;
  case AArch64::SUBXri:
    return AArch64::SUBSXri;
  case AArch64::SBCWr:
    return AArch64::SBCSWr;
  case AArch64::SBCXr:
    return AArch64::SBCSXr;
  case AArch64::ANDWri:
    return AArch64::ANDSWri;
  case AArch64::ANDXri:
    return AArch64::ANDSXri;
  }
}

/// Check if AArch64::NZCV should be alive in successors of MBB.
static bool areCFlagsAliveInSuccessors(MachineBasicBlock *MBB) {
  for (auto *BB : MBB->successors())
    if (BB->isLiveIn(AArch64::NZCV))
      return true;
  return false;
}

namespace {

struct UsedNZCV {
  bool N = false;
  bool Z = false;
  bool C = false;
  bool V = false;

  UsedNZCV() = default;

  UsedNZCV &operator|=(const UsedNZCV &UsedFlags) {
    this->N |= UsedFlags.N;
    this->Z |= UsedFlags.Z;
    this->C |= UsedFlags.C;
    this->V |= UsedFlags.V;
    return *this;
  }
};

} // end anonymous namespace

/// Find a condition code used by the instruction.
/// Returns AArch64CC::Invalid if either the instruction does not use condition
/// codes or we don't optimize CmpInstr in the presence of such instructions.
static AArch64CC::CondCode findCondCodeUsedByInstr(const MachineInstr &Instr) {
  switch (Instr.getOpcode()) {
  default:
    return AArch64CC::Invalid;

  case AArch64::Bcc: {
    int Idx = Instr.findRegisterUseOperandIdx(AArch64::NZCV);
    assert(Idx >= 2);
    return static_cast<AArch64CC::CondCode>(Instr.getOperand(Idx - 2).getImm());
  }

  case AArch64::CSINVWr:
  case AArch64::CSINVXr:
  case AArch64::CSINCWr:
  case AArch64::CSINCXr:
  case AArch64::CSELWr:
  case AArch64::CSELXr:
  case AArch64::CSNEGWr:
  case AArch64::CSNEGXr:
  case AArch64::FCSELSrrr:
  case AArch64::FCSELDrrr: {
    int Idx = Instr.findRegisterUseOperandIdx(AArch64::NZCV);
    assert(Idx >= 1);
    return static_cast<AArch64CC::CondCode>(Instr.getOperand(Idx - 1).getImm());
  }
  }
}

static UsedNZCV getUsedNZCV(AArch64CC::CondCode CC) {
  assert(CC != AArch64CC::Invalid);
  UsedNZCV UsedFlags;
  switch (CC) {
  default:
    break;

  case AArch64CC::EQ: // Z set
  case AArch64CC::NE: // Z clear
    UsedFlags.Z = true;
    break;

  case AArch64CC::HI: // Z clear and C set
  case AArch64CC::LS: // Z set   or  C clear
    UsedFlags.Z = true;
    LLVM_FALLTHROUGH;
  case AArch64CC::HS: // C set
  case AArch64CC::LO: // C clear
    UsedFlags.C = true;
    break;

  case AArch64CC::MI: // N set
  case AArch64CC::PL: // N clear
    UsedFlags.N = true;
    break;

  case AArch64CC::VS: // V set
  case AArch64CC::VC: // V clear
    UsedFlags.V = true;
    break;

  case AArch64CC::GT: // Z clear, N and V the same
  case AArch64CC::LE: // Z set,   N and V differ
    UsedFlags.Z = true;
    LLVM_FALLTHROUGH;
  case AArch64CC::GE: // N and V the same
  case AArch64CC::LT: // N and V differ
    UsedFlags.N = true;
    UsedFlags.V = true;
    break;
  }
  return UsedFlags;
}

static bool isADDSRegImm(unsigned Opcode) {
  return Opcode == AArch64::ADDSWri || Opcode == AArch64::ADDSXri;
}

static bool isSUBSRegImm(unsigned Opcode) {
  return Opcode == AArch64::SUBSWri || Opcode == AArch64::SUBSXri;
}

/// Check if CmpInstr can be substituted by MI.
///
/// CmpInstr can be substituted:
/// - CmpInstr is either 'ADDS %vreg, 0' or 'SUBS %vreg, 0'
/// - and, MI and CmpInstr are from the same MachineBB
/// - and, condition flags are not alive in successors of the CmpInstr parent
/// - and, if MI opcode is the S form there must be no defs of flags between
///        MI and CmpInstr
///        or if MI opcode is not the S form there must be neither defs of flags
///        nor uses of flags between MI and CmpInstr.
/// - and  C/V flags are not used after CmpInstr
static bool canInstrSubstituteCmpInstr(MachineInstr *MI, MachineInstr *CmpInstr,
                                       const TargetRegisterInfo *TRI) {
  assert(MI);
  assert(sForm(*MI) != AArch64::INSTRUCTION_LIST_END);
  assert(CmpInstr);

  const unsigned CmpOpcode = CmpInstr->getOpcode();
  if (!isADDSRegImm(CmpOpcode) && !isSUBSRegImm(CmpOpcode))
    return false;

  if (MI->getParent() != CmpInstr->getParent())
    return false;

  if (areCFlagsAliveInSuccessors(CmpInstr->getParent()))
    return false;

  AccessKind AccessToCheck = AK_Write;
  if (sForm(*MI) != MI->getOpcode())
    AccessToCheck = AK_All;
  if (areCFlagsAccessedBetweenInstrs(MI, CmpInstr, TRI, AccessToCheck))
    return false;

  UsedNZCV NZCVUsedAfterCmp;
  for (auto I = std::next(CmpInstr->getIterator()),
            E = CmpInstr->getParent()->instr_end();
       I != E; ++I) {
    const MachineInstr &Instr = *I;
    if (Instr.readsRegister(AArch64::NZCV, TRI)) {
      AArch64CC::CondCode CC = findCondCodeUsedByInstr(Instr);
      if (CC == AArch64CC::Invalid) // Unsupported conditional instruction
        return false;
      NZCVUsedAfterCmp |= getUsedNZCV(CC);
    }

    if (Instr.modifiesRegister(AArch64::NZCV, TRI))
      break;
  }

  return !NZCVUsedAfterCmp.C && !NZCVUsedAfterCmp.V;
}

/// Substitute an instruction comparing to zero with another instruction
/// which produces needed condition flags.
///
/// Return true on success.
bool AArch64InstrInfo::substituteCmpToZero(
    MachineInstr &CmpInstr, unsigned SrcReg,
    const MachineRegisterInfo *MRI) const {
  assert(MRI);
  // Get the unique definition of SrcReg.
  MachineInstr *MI = MRI->getUniqueVRegDef(SrcReg);
  if (!MI)
    return false;

  const TargetRegisterInfo *TRI = &getRegisterInfo();

  unsigned NewOpc = sForm(*MI);
  if (NewOpc == AArch64::INSTRUCTION_LIST_END)
    return false;

  if (!canInstrSubstituteCmpInstr(MI, &CmpInstr, TRI))
    return false;

  // Update the instruction to set NZCV.
  MI->setDesc(get(NewOpc));
  CmpInstr.eraseFromParent();
  bool succeeded = UpdateOperandRegClass(*MI);
  (void)succeeded;
  assert(succeeded && "Some operands reg class are incompatible!");
  MI->addRegisterDefined(AArch64::NZCV, TRI);
  return true;
}

bool AArch64InstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  if (MI.getOpcode() != TargetOpcode::LOAD_STACK_GUARD)
    return false;

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();
  const GlobalValue *GV =
      cast<GlobalValue>((*MI.memoperands_begin())->getValue());
  const TargetMachine &TM = MBB.getParent()->getTarget();
  unsigned char OpFlags = Subtarget.ClassifyGlobalReference(GV, TM);
  const unsigned char MO_NC = AArch64II::MO_NC;

  if ((OpFlags & AArch64II::MO_GOT) != 0) {
    BuildMI(MBB, MI, DL, get(AArch64::LOADgot), Reg)
        .addGlobalAddress(GV, 0, AArch64II::MO_GOT);
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill)
        .addImm(0)
        .addMemOperand(*MI.memoperands_begin());
  } else if (TM.getCodeModel() == CodeModel::Large) {
    BuildMI(MBB, MI, DL, get(AArch64::MOVZXi), Reg)
        .addGlobalAddress(GV, 0, AArch64II::MO_G0 | MO_NC)
        .addImm(0);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G1 | MO_NC)
        .addImm(16);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G2 | MO_NC)
        .addImm(32);
    BuildMI(MBB, MI, DL, get(AArch64::MOVKXi), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, AArch64II::MO_G3)
        .addImm(48);
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill)
        .addImm(0)
        .addMemOperand(*MI.memoperands_begin());
  } else {
    BuildMI(MBB, MI, DL, get(AArch64::ADRP), Reg)
        .addGlobalAddress(GV, 0, OpFlags | AArch64II::MO_PAGE);
    unsigned char LoFlags = OpFlags | AArch64II::MO_PAGEOFF | MO_NC;
    BuildMI(MBB, MI, DL, get(AArch64::LDRXui), Reg)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, LoFlags)
        .addMemOperand(*MI.memoperands_begin());
  }

  MBB.erase(MI);

  return true;
}

/// Return true if this is this instruction has a non-zero immediate
bool AArch64InstrInfo::hasShiftedReg(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::ANDWrs:
  case AArch64::ANDXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
  case AArch64::BICWrs:
  case AArch64::BICXrs:
  case AArch64::EONWrs:
  case AArch64::EONXrs:
  case AArch64::EORWrs:
  case AArch64::EORXrs:
  case AArch64::ORNWrs:
  case AArch64::ORNXrs:
  case AArch64::ORRWrs:
  case AArch64::ORRXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::SUBWrs:
  case AArch64::SUBXrs:
    if (MI.getOperand(3).isImm()) {
      unsigned val = MI.getOperand(3).getImm();
      return (val != 0);
    }
    break;
  }
  return false;
}

/// Return true if this is this instruction has a non-zero immediate
bool AArch64InstrInfo::hasExtendedReg(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::ADDSWrx:
  case AArch64::ADDSXrx:
  case AArch64::ADDSXrx64:
  case AArch64::ADDWrx:
  case AArch64::ADDXrx:
  case AArch64::ADDXrx64:
  case AArch64::SUBSWrx:
  case AArch64::SUBSXrx:
  case AArch64::SUBSXrx64:
  case AArch64::SUBWrx:
  case AArch64::SUBXrx:
  case AArch64::SUBXrx64:
    if (MI.getOperand(3).isImm()) {
      unsigned val = MI.getOperand(3).getImm();
      return (val != 0);
    }
    break;
  }

  return false;
}

// Return true if this instruction simply sets its single destination register
// to zero. This is equivalent to a register rename of the zero-register.
bool AArch64InstrInfo::isGPRZero(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::MOVZWi:
  case AArch64::MOVZXi: // movz Rd, #0 (LSL #0)
    if (MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0) {
      assert(MI.getDesc().getNumOperands() == 3 &&
             MI.getOperand(2).getImm() == 0 && "invalid MOVZi operands");
      return true;
    }
    break;
  case AArch64::ANDWri: // and Rd, Rzr, #imm
    return MI.getOperand(1).getReg() == AArch64::WZR;
  case AArch64::ANDXri:
    return MI.getOperand(1).getReg() == AArch64::XZR;
  case TargetOpcode::COPY:
    return MI.getOperand(1).getReg() == AArch64::WZR;
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool AArch64InstrInfo::isGPRCopy(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // GPR32 copies will by lowered to ORRXrs
    unsigned DstReg = MI.getOperand(0).getReg();
    return (AArch64::GPR32RegClass.contains(DstReg) ||
            AArch64::GPR64RegClass.contains(DstReg));
  }
  case AArch64::ORRXrs: // orr Xd, Xzr, Xm (LSL #0)
    if (MI.getOperand(1).getReg() == AArch64::XZR) {
      assert(MI.getDesc().getNumOperands() == 4 &&
             MI.getOperand(3).getImm() == 0 && "invalid ORRrs operands");
      return true;
    }
    break;
  case AArch64::ADDXri: // add Xd, Xn, #0 (LSL #0)
    if (MI.getOperand(2).getImm() == 0) {
      assert(MI.getDesc().getNumOperands() == 4 &&
             MI.getOperand(3).getImm() == 0 && "invalid ADDXri operands");
      return true;
    }
    break;
  }
  return false;
}

// Return true if this instruction simply renames a general register without
// modifying bits.
bool AArch64InstrInfo::isFPRCopy(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case TargetOpcode::COPY: {
    // FPR64 copies will by lowered to ORR.16b
    unsigned DstReg = MI.getOperand(0).getReg();
    return (AArch64::FPR64RegClass.contains(DstReg) ||
            AArch64::FPR128RegClass.contains(DstReg));
  }
  case AArch64::ORRv16i8:
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      assert(MI.getDesc().getNumOperands() == 3 && MI.getOperand(0).isReg() &&
             "invalid ORRv16i8 operands");
      return true;
    }
    break;
  }
  return false;
}

unsigned AArch64InstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                               int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::LDRWui:
  case AArch64::LDRXui:
  case AArch64::LDRBui:
  case AArch64::LDRHui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:
    if (MI.getOperand(0).getSubReg() == 0 && MI.getOperand(1).isFI() &&
        MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

unsigned AArch64InstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                              int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::STRWui:
  case AArch64::STRXui:
  case AArch64::STRBui:
  case AArch64::STRHui:
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:
    if (MI.getOperand(0).getSubReg() == 0 && MI.getOperand(1).isFI() &&
        MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

/// Return true if this is load/store scales or extends its register offset.
/// This refers to scaling a dynamic index as opposed to scaled immediates.
/// MI should be a memory op that allows scaled addressing.
bool AArch64InstrInfo::isScaledAddr(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    break;
  case AArch64::LDRBBroW:
  case AArch64::LDRBroW:
  case AArch64::LDRDroW:
  case AArch64::LDRHHroW:
  case AArch64::LDRHroW:
  case AArch64::LDRQroW:
  case AArch64::LDRSBWroW:
  case AArch64::LDRSBXroW:
  case AArch64::LDRSHWroW:
  case AArch64::LDRSHXroW:
  case AArch64::LDRSWroW:
  case AArch64::LDRSroW:
  case AArch64::LDRWroW:
  case AArch64::LDRXroW:
  case AArch64::STRBBroW:
  case AArch64::STRBroW:
  case AArch64::STRDroW:
  case AArch64::STRHHroW:
  case AArch64::STRHroW:
  case AArch64::STRQroW:
  case AArch64::STRSroW:
  case AArch64::STRWroW:
  case AArch64::STRXroW:
  case AArch64::LDRBBroX:
  case AArch64::LDRBroX:
  case AArch64::LDRDroX:
  case AArch64::LDRHHroX:
  case AArch64::LDRHroX:
  case AArch64::LDRQroX:
  case AArch64::LDRSBWroX:
  case AArch64::LDRSBXroX:
  case AArch64::LDRSHWroX:
  case AArch64::LDRSHXroX:
  case AArch64::LDRSWroX:
  case AArch64::LDRSroX:
  case AArch64::LDRWroX:
  case AArch64::LDRXroX:
  case AArch64::STRBBroX:
  case AArch64::STRBroX:
  case AArch64::STRDroX:
  case AArch64::STRHHroX:
  case AArch64::STRHroX:
  case AArch64::STRQroX:
  case AArch64::STRSroX:
  case AArch64::STRWroX:
  case AArch64::STRXroX:

    unsigned Val = MI.getOperand(3).getImm();
    AArch64_AM::ShiftExtendType ExtType = AArch64_AM::getMemExtendType(Val);
    return (ExtType != AArch64_AM::UXTX) || AArch64_AM::getMemDoShift(Val);
  }
  return false;
}

/// Check all MachineMemOperands for a hint to suppress pairing.
bool AArch64InstrInfo::isLdStPairSuppressed(const MachineInstr &MI) const {
  return llvm::any_of(MI.memoperands(), [](MachineMemOperand *MMO) {
    return MMO->getFlags() & MOSuppressPair;
  });
}

/// Set a flag on the first MachineMemOperand to suppress pairing.
void AArch64InstrInfo::suppressLdStPair(MachineInstr &MI) const {
  if (MI.memoperands_empty())
    return;
  (*MI.memoperands_begin())->setFlags(MOSuppressPair);
}

/// Check all MachineMemOperands for a hint that the load/store is strided.
bool AArch64InstrInfo::isStridedAccess(const MachineInstr &MI) const {
  return llvm::any_of(MI.memoperands(), [](MachineMemOperand *MMO) {
    return MMO->getFlags() & MOStridedAccess;
  });
}

bool AArch64InstrInfo::isUnscaledLdSt(unsigned Opc) const {
  switch (Opc) {
  default:
    return false;
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURBBi:
  case AArch64::STURHHi:
  case AArch64::STURWi:
  case AArch64::STURXi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURWi:
  case AArch64::LDURXi:
  case AArch64::LDURSWi:
  case AArch64::LDURHHi:
  case AArch64::LDURBBi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSHWi:
    return true;
  }
}

bool AArch64InstrInfo::isUnscaledLdSt(MachineInstr &MI) const {
  return isUnscaledLdSt(MI.getOpcode());
}

// Is this a candidate for ld/st merging or pairing?  For example, we don't
// touch volatiles or load/stores that have a hint to avoid pair formation.
bool AArch64InstrInfo::isCandidateToMergeOrPair(MachineInstr &MI) const {
  // If this is a volatile load/store, don't mess with it.
  if (MI.hasOrderedMemoryRef())
    return false;

  // Make sure this is a reg+imm (as opposed to an address reloc).
  assert(MI.getOperand(1).isReg() && "Expected a reg operand.");
  if (!MI.getOperand(2).isImm())
    return false;

  // Can't merge/pair if the instruction modifies the base register.
  // e.g., ldr x0, [x0]
  unsigned BaseReg = MI.getOperand(1).getReg();
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  if (MI.modifiesRegister(BaseReg, TRI))
    return false;

  // Check if this load/store has a hint to avoid pair formation.
  // MachineMemOperands hints are set by the AArch64StorePairSuppress pass.
  if (isLdStPairSuppressed(MI))
    return false;

  // On some CPUs quad load/store pairs are slower than two single load/stores.
  if (Subtarget.isPaired128Slow()) {
    switch (MI.getOpcode()) {
    default:
      break;
    case AArch64::LDURQi:
    case AArch64::STURQi:
    case AArch64::LDRQui:
    case AArch64::STRQui:
      return false;
    }
  }

  return true;
}

bool AArch64InstrInfo::getMemOpBaseRegImmOfs(
    MachineInstr &LdSt, unsigned &BaseReg, int64_t &Offset,
    const TargetRegisterInfo *TRI) const {
  unsigned Width;
  return getMemOpBaseRegImmOfsWidth(LdSt, BaseReg, Offset, Width, TRI);
}

bool AArch64InstrInfo::getMemOpBaseRegImmOfsWidth(
    MachineInstr &LdSt, unsigned &BaseReg, int64_t &Offset, unsigned &Width,
    const TargetRegisterInfo *TRI) const {
  assert(LdSt.mayLoadOrStore() && "Expected a memory operation.");
  // Handle only loads/stores with base register followed by immediate offset.
  if (LdSt.getNumExplicitOperands() == 3) {
    // Non-paired instruction (e.g., ldr x1, [x0, #8]).
    if (!LdSt.getOperand(1).isReg() || !LdSt.getOperand(2).isImm())
      return false;
  } else if (LdSt.getNumExplicitOperands() == 4) {
    // Paired instruction (e.g., ldp x1, x2, [x0, #8]).
    if (!LdSt.getOperand(1).isReg() || !LdSt.getOperand(2).isReg() ||
        !LdSt.getOperand(3).isImm())
      return false;
  } else
    return false;

  // Get the scaling factor for the instruction and set the width for the
  // instruction.
  unsigned Scale = 0;
  int64_t Dummy1, Dummy2;

  // If this returns false, then it's an instruction we don't want to handle.
  if (!getMemOpInfo(LdSt.getOpcode(), Scale, Width, Dummy1, Dummy2))
    return false;

  // Compute the offset. Offset is calculated as the immediate operand
  // multiplied by the scaling factor. Unscaled instructions have scaling factor
  // set to 1.
  if (LdSt.getNumExplicitOperands() == 3) {
    BaseReg = LdSt.getOperand(1).getReg();
    Offset = LdSt.getOperand(2).getImm() * Scale;
  } else {
    assert(LdSt.getNumExplicitOperands() == 4 && "invalid number of operands");
    BaseReg = LdSt.getOperand(2).getReg();
    Offset = LdSt.getOperand(3).getImm() * Scale;
  }
  return true;
}

MachineOperand &
AArch64InstrInfo::getMemOpBaseRegImmOfsOffsetOperand(MachineInstr &LdSt) const {
  assert(LdSt.mayLoadOrStore() && "Expected a memory operation.");
  MachineOperand &OfsOp = LdSt.getOperand(LdSt.getNumExplicitOperands() - 1);
  assert(OfsOp.isImm() && "Offset operand wasn't immediate.");
  return OfsOp;
}

bool AArch64InstrInfo::getMemOpInfo(unsigned Opcode, unsigned &Scale,
                                    unsigned &Width, int64_t &MinOffset,
                                    int64_t &MaxOffset) const {
  switch (Opcode) {
  // Not a memory operation or something we want to handle.
  default:
    Scale = Width = 0;
    MinOffset = MaxOffset = 0;
    return false;
  case AArch64::STRWpost:
  case AArch64::LDRWpost:
    Width = 32;
    Scale = 4;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDURQi:
  case AArch64::STURQi:
    Width = 16;
    Scale = 1;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDURXi:
  case AArch64::LDURDi:
  case AArch64::STURXi:
  case AArch64::STURDi:
    Width = 8;
    Scale = 1;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDURWi:
  case AArch64::LDURSi:
  case AArch64::LDURSWi:
  case AArch64::STURWi:
  case AArch64::STURSi:
    Width = 4;
    Scale = 1;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDURHi:
  case AArch64::LDURHHi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSHWi:
  case AArch64::STURHi:
  case AArch64::STURHHi:
    Width = 2;
    Scale = 1;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDURBi:
  case AArch64::LDURBBi:
  case AArch64::LDURSBXi:
  case AArch64::LDURSBWi:
  case AArch64::STURBi:
  case AArch64::STURBBi:
    Width = 1;
    Scale = 1;
    MinOffset = -256;
    MaxOffset = 255;
    break;
  case AArch64::LDPQi:
  case AArch64::LDNPQi:
  case AArch64::STPQi:
  case AArch64::STNPQi:
    Scale = 16;
    Width = 32;
    MinOffset = -64;
    MaxOffset = 63;
    break;
  case AArch64::LDRQui:
  case AArch64::STRQui:
    Scale = Width = 16;
    MinOffset = 0;
    MaxOffset = 4095;
    break;
  case AArch64::LDPXi:
  case AArch64::LDPDi:
  case AArch64::LDNPXi:
  case AArch64::LDNPDi:
  case AArch64::STPXi:
  case AArch64::STPDi:
  case AArch64::STNPXi:
  case AArch64::STNPDi:
    Scale = 8;
    Width = 16;
    MinOffset = -64;
    MaxOffset = 63;
    break;
  case AArch64::LDRXui:
  case AArch64::LDRDui:
  case AArch64::STRXui:
  case AArch64::STRDui:
    Scale = Width = 8;
    MinOffset = 0;
    MaxOffset = 4095;
    break;
  case AArch64::LDPWi:
  case AArch64::LDPSi:
  case AArch64::LDNPWi:
  case AArch64::LDNPSi:
  case AArch64::STPWi:
  case AArch64::STPSi:
  case AArch64::STNPWi:
  case AArch64::STNPSi:
    Scale = 4;
    Width = 8;
    MinOffset = -64;
    MaxOffset = 63;
    break;
  case AArch64::LDRWui:
  case AArch64::LDRSui:
  case AArch64::LDRSWui:
  case AArch64::STRWui:
  case AArch64::STRSui:
    Scale = Width = 4;
    MinOffset = 0;
    MaxOffset = 4095;
    break;
  case AArch64::LDRHui:
  case AArch64::LDRHHui:
  case AArch64::STRHui:
  case AArch64::STRHHui:
    Scale = Width = 2;
    MinOffset = 0;
    MaxOffset = 4095;
    break;
  case AArch64::LDRBui:
  case AArch64::LDRBBui:
  case AArch64::STRBui:
  case AArch64::STRBBui:
    Scale = Width = 1;
    MinOffset = 0;
    MaxOffset = 4095;
    break;
  }

  return true;
}

// Scale the unscaled offsets.  Returns false if the unscaled offset can't be
// scaled.
static bool scaleOffset(unsigned Opc, int64_t &Offset) {
  unsigned OffsetStride = 1;
  switch (Opc) {
  default:
    return false;
  case AArch64::LDURQi:
  case AArch64::STURQi:
    OffsetStride = 16;
    break;
  case AArch64::LDURXi:
  case AArch64::LDURDi:
  case AArch64::STURXi:
  case AArch64::STURDi:
    OffsetStride = 8;
    break;
  case AArch64::LDURWi:
  case AArch64::LDURSi:
  case AArch64::LDURSWi:
  case AArch64::STURWi:
  case AArch64::STURSi:
    OffsetStride = 4;
    break;
  }
  // If the byte-offset isn't a multiple of the stride, we can't scale this
  // offset.
  if (Offset % OffsetStride != 0)
    return false;

  // Convert the byte-offset used by unscaled into an "element" offset used
  // by the scaled pair load/store instructions.
  Offset /= OffsetStride;
  return true;
}

static bool canPairLdStOpc(unsigned FirstOpc, unsigned SecondOpc) {
  if (FirstOpc == SecondOpc)
    return true;
  // We can also pair sign-ext and zero-ext instructions.
  switch (FirstOpc) {
  default:
    return false;
  case AArch64::LDRWui:
  case AArch64::LDURWi:
    return SecondOpc == AArch64::LDRSWui || SecondOpc == AArch64::LDURSWi;
  case AArch64::LDRSWui:
  case AArch64::LDURSWi:
    return SecondOpc == AArch64::LDRWui || SecondOpc == AArch64::LDURWi;
  }
  // These instructions can't be paired based on their opcodes.
  return false;
}

/// Detect opportunities for ldp/stp formation.
///
/// Only called for LdSt for which getMemOpBaseRegImmOfs returns true.
bool AArch64InstrInfo::shouldClusterMemOps(MachineInstr &FirstLdSt,
                                           unsigned BaseReg1,
                                           MachineInstr &SecondLdSt,
                                           unsigned BaseReg2,
                                           unsigned NumLoads) const {
  if (BaseReg1 != BaseReg2)
    return false;

  // Only cluster up to a single pair.
  if (NumLoads > 1)
    return false;

  if (!isPairableLdStInst(FirstLdSt) || !isPairableLdStInst(SecondLdSt))
    return false;

  // Can we pair these instructions based on their opcodes?
  unsigned FirstOpc = FirstLdSt.getOpcode();
  unsigned SecondOpc = SecondLdSt.getOpcode();
  if (!canPairLdStOpc(FirstOpc, SecondOpc))
    return false;

  // Can't merge volatiles or load/stores that have a hint to avoid pair
  // formation, for example.
  if (!isCandidateToMergeOrPair(FirstLdSt) ||
      !isCandidateToMergeOrPair(SecondLdSt))
    return false;

  // isCandidateToMergeOrPair guarantees that operand 2 is an immediate.
  int64_t Offset1 = FirstLdSt.getOperand(2).getImm();
  if (isUnscaledLdSt(FirstOpc) && !scaleOffset(FirstOpc, Offset1))
    return false;

  int64_t Offset2 = SecondLdSt.getOperand(2).getImm();
  if (isUnscaledLdSt(SecondOpc) && !scaleOffset(SecondOpc, Offset2))
    return false;

  // Pairwise instructions have a 7-bit signed offset field.
  if (Offset1 > 63 || Offset1 < -64)
    return false;

  // The caller should already have ordered First/SecondLdSt by offset.
  assert(Offset1 <= Offset2 && "Caller should have ordered offsets.");
  return Offset1 + 1 == Offset2;
}

static const MachineInstrBuilder &AddSubReg(const MachineInstrBuilder &MIB,
                                            unsigned Reg, unsigned SubIdx,
                                            unsigned State,
                                            const TargetRegisterInfo *TRI) {
  if (!SubIdx)
    return MIB.addReg(Reg, State);

  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return MIB.addReg(TRI->getSubReg(Reg, SubIdx), State);
  return MIB.addReg(Reg, State, SubIdx);
}

static bool forwardCopyWillClobberTuple(unsigned DestReg, unsigned SrcReg,
                                        unsigned NumRegs) {
  // We really want the positive remainder mod 32 here, that happens to be
  // easily obtainable with a mask.
  return ((DestReg - SrcReg) & 0x1f) < NumRegs;
}

void AArch64InstrInfo::copyPhysRegTuple(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        const DebugLoc &DL, unsigned DestReg,
                                        unsigned SrcReg, bool KillSrc,
                                        unsigned Opcode,
                                        ArrayRef<unsigned> Indices) const {
  assert(Subtarget.hasNEON() && "Unexpected register copy without NEON");
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  uint16_t DestEncoding = TRI->getEncodingValue(DestReg);
  uint16_t SrcEncoding = TRI->getEncodingValue(SrcReg);
  unsigned NumRegs = Indices.size();

  int SubReg = 0, End = NumRegs, Incr = 1;
  if (forwardCopyWillClobberTuple(DestEncoding, SrcEncoding, NumRegs)) {
    SubReg = NumRegs - 1;
    End = -1;
    Incr = -1;
  }

  for (; SubReg != End; SubReg += Incr) {
    const MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(Opcode));
    AddSubReg(MIB, DestReg, Indices[SubReg], RegState::Define, TRI);
    AddSubReg(MIB, SrcReg, Indices[SubReg], 0, TRI);
    AddSubReg(MIB, SrcReg, Indices[SubReg], getKillRegState(KillSrc), TRI);
  }
}

void AArch64InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   const DebugLoc &DL, unsigned DestReg,
                                   unsigned SrcReg, bool KillSrc) const {
  if (AArch64::GPR32spRegClass.contains(DestReg) &&
      (AArch64::GPR32spRegClass.contains(SrcReg) || SrcReg == AArch64::WZR)) {
    const TargetRegisterInfo *TRI = &getRegisterInfo();

    if (DestReg == AArch64::WSP || SrcReg == AArch64::WSP) {
      // If either operand is WSP, expand to ADD #0.
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ADD Xd, Xn, #0" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, AArch64::sub_32,
                                                     &AArch64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, AArch64::sub_32,
                                                    &AArch64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(AArch64::ADDXri), DestRegX)
            .addReg(SrcRegX, RegState::Undef)
            .addImm(0)
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        BuildMI(MBB, I, DL, get(AArch64::ADDWri), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc))
            .addImm(0)
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
      }
    } else if (SrcReg == AArch64::WZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(AArch64::MOVZWi), DestReg)
          .addImm(0)
          .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else {
      if (Subtarget.hasZeroCycleRegMove()) {
        // Cyclone recognizes "ORR Xd, XZR, Xm" as a zero-cycle register move.
        unsigned DestRegX = TRI->getMatchingSuperReg(DestReg, AArch64::sub_32,
                                                     &AArch64::GPR64spRegClass);
        unsigned SrcRegX = TRI->getMatchingSuperReg(SrcReg, AArch64::sub_32,
                                                    &AArch64::GPR64spRegClass);
        // This instruction is reading and writing X registers.  This may upset
        // the register scavenger and machine verifier, so we need to indicate
        // that we are reading an undefined value from SrcRegX, but a proper
        // value from SrcReg.
        BuildMI(MBB, I, DL, get(AArch64::ORRXrr), DestRegX)
            .addReg(AArch64::XZR)
            .addReg(SrcRegX, RegState::Undef)
            .addReg(SrcReg, RegState::Implicit | getKillRegState(KillSrc));
      } else {
        // Otherwise, expand to ORR WZR.
        BuildMI(MBB, I, DL, get(AArch64::ORRWrr), DestReg)
            .addReg(AArch64::WZR)
            .addReg(SrcReg, getKillRegState(KillSrc));
      }
    }
    return;
  }

  if (AArch64::GPR64spRegClass.contains(DestReg) &&
      (AArch64::GPR64spRegClass.contains(SrcReg) || SrcReg == AArch64::XZR)) {
    if (DestReg == AArch64::SP || SrcReg == AArch64::SP) {
      // If either operand is SP, expand to ADD #0.
      BuildMI(MBB, I, DL, get(AArch64::ADDXri), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addImm(0)
          .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else if (SrcReg == AArch64::XZR && Subtarget.hasZeroCycleZeroing()) {
      BuildMI(MBB, I, DL, get(AArch64::MOVZXi), DestReg)
          .addImm(0)
          .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    } else {
      // Otherwise, expand to ORR XZR.
      BuildMI(MBB, I, DL, get(AArch64::ORRXrr), DestReg)
          .addReg(AArch64::XZR)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  // Copy a DDDD register quad by copying the individual sub-registers.
  if (AArch64::DDDDRegClass.contains(DestReg) &&
      AArch64::DDDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::dsub0, AArch64::dsub1,
                                       AArch64::dsub2, AArch64::dsub3};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DDD register triple by copying the individual sub-registers.
  if (AArch64::DDDRegClass.contains(DestReg) &&
      AArch64::DDDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::dsub0, AArch64::dsub1,
                                       AArch64::dsub2};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a DD register pair by copying the individual sub-registers.
  if (AArch64::DDRegClass.contains(DestReg) &&
      AArch64::DDRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::dsub0, AArch64::dsub1};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv8i8,
                     Indices);
    return;
  }

  // Copy a QQQQ register quad by copying the individual sub-registers.
  if (AArch64::QQQQRegClass.contains(DestReg) &&
      AArch64::QQQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::qsub0, AArch64::qsub1,
                                       AArch64::qsub2, AArch64::qsub3};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQQ register triple by copying the individual sub-registers.
  if (AArch64::QQQRegClass.contains(DestReg) &&
      AArch64::QQQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::qsub0, AArch64::qsub1,
                                       AArch64::qsub2};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  // Copy a QQ register pair by copying the individual sub-registers.
  if (AArch64::QQRegClass.contains(DestReg) &&
      AArch64::QQRegClass.contains(SrcReg)) {
    static const unsigned Indices[] = {AArch64::qsub0, AArch64::qsub1};
    copyPhysRegTuple(MBB, I, DL, DestReg, SrcReg, KillSrc, AArch64::ORRv16i8,
                     Indices);
    return;
  }

  if (AArch64::FPR128RegClass.contains(DestReg) &&
      AArch64::FPR128RegClass.contains(SrcReg)) {
    if (Subtarget.hasNEON()) {
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::STRQpre))
          .addReg(AArch64::SP, RegState::Define)
          .addReg(SrcReg, getKillRegState(KillSrc))
          .addReg(AArch64::SP)
          .addImm(-16);
      BuildMI(MBB, I, DL, get(AArch64::LDRQpre))
          .addReg(AArch64::SP, RegState::Define)
          .addReg(DestReg, RegState::Define)
          .addReg(AArch64::SP)
          .addImm(16);
    }
    return;
  }

  if (AArch64::FPR64RegClass.contains(DestReg) &&
      AArch64::FPR64RegClass.contains(SrcReg)) {
    if (Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::dsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::dsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::FMOVDr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR32RegClass.contains(DestReg) &&
      AArch64::FPR32RegClass.contains(SrcReg)) {
    if (Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::ssub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::ssub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR16RegClass.contains(DestReg) &&
      AArch64::FPR16RegClass.contains(SrcReg)) {
    if (Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::hsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::hsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::hsub,
                                       &AArch64::FPR32RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::hsub,
                                      &AArch64::FPR32RegClass);
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  if (AArch64::FPR8RegClass.contains(DestReg) &&
      AArch64::FPR8RegClass.contains(SrcReg)) {
    if (Subtarget.hasNEON()) {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::bsub,
                                       &AArch64::FPR128RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::bsub,
                                      &AArch64::FPR128RegClass);
      BuildMI(MBB, I, DL, get(AArch64::ORRv16i8), DestReg)
          .addReg(SrcReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    } else {
      DestReg = RI.getMatchingSuperReg(DestReg, AArch64::bsub,
                                       &AArch64::FPR32RegClass);
      SrcReg = RI.getMatchingSuperReg(SrcReg, AArch64::bsub,
                                      &AArch64::FPR32RegClass);
      BuildMI(MBB, I, DL, get(AArch64::FMOVSr), DestReg)
          .addReg(SrcReg, getKillRegState(KillSrc));
    }
    return;
  }

  // Copies between GPR64 and FPR64.
  if (AArch64::FPR64RegClass.contains(DestReg) &&
      AArch64::GPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVXDr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (AArch64::GPR64RegClass.contains(DestReg) &&
      AArch64::FPR64RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVDXr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  // Copies between GPR32 and FPR32.
  if (AArch64::FPR32RegClass.contains(DestReg) &&
      AArch64::GPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVWSr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (AArch64::GPR32RegClass.contains(DestReg) &&
      AArch64::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(AArch64::FMOVSWr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (DestReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(SrcReg) && "Invalid NZCV copy");
    BuildMI(MBB, I, DL, get(AArch64::MSR))
        .addImm(AArch64SysReg::NZCV)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(AArch64::NZCV, RegState::Implicit | RegState::Define);
    return;
  }

  if (SrcReg == AArch64::NZCV) {
    assert(AArch64::GPR64RegClass.contains(DestReg) && "Invalid NZCV copy");
    BuildMI(MBB, I, DL, get(AArch64::MRS), DestReg)
        .addImm(AArch64SysReg::NZCV)
        .addReg(AArch64::NZCV, RegState::Implicit | getKillRegState(KillSrc));
    return;
  }

  llvm_unreachable("unimplemented reg-to-reg copy");
}

void AArch64InstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned SrcReg,
    bool isKill, int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOStore, MFI.getObjectSize(FI), Align);
  unsigned Opc = 0;
  bool Offset = true;
  switch (TRI->getSpillSize(*RC)) {
  case 1:
    if (AArch64::FPR8RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRBui;
    break;
  case 2:
    if (AArch64::FPR16RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRHui;
    break;
  case 4:
    if (AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::STRWui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR32RegClass);
      else
        assert(SrcReg != AArch64::WSP);
    } else if (AArch64::FPR32RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRSui;
    break;
  case 8:
    if (AArch64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::STRXui;
      if (TargetRegisterInfo::isVirtualRegister(SrcReg))
        MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR64RegClass);
      else
        assert(SrcReg != AArch64::SP);
    } else if (AArch64::FPR64RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRDui;
    break;
  case 16:
    if (AArch64::FPR128RegClass.hasSubClassEq(RC))
      Opc = AArch64::STRQui;
    else if (AArch64::DDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Twov1d;
      Offset = false;
    }
    break;
  case 24:
    if (AArch64::DDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Threev1d;
      Offset = false;
    }
    break;
  case 32:
    if (AArch64::DDDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Fourv1d;
      Offset = false;
    } else if (AArch64::QQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Twov2d;
      Offset = false;
    }
    break;
  case 48:
    if (AArch64::QQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Threev2d;
      Offset = false;
    }
    break;
  case 64:
    if (AArch64::QQQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register store without NEON");
      Opc = AArch64::ST1Fourv2d;
      Offset = false;
    }
    break;
  }
  assert(Opc && "Unknown register class");

  const MachineInstrBuilder MI = BuildMI(MBB, MBBI, DL, get(Opc))
                                     .addReg(SrcReg, getKillRegState(isKill))
                                     .addFrameIndex(FI);

  if (Offset)
    MI.addImm(0);
  MI.addMemOperand(MMO);
}

void AArch64InstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned DestReg,
    int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, MFI.getObjectSize(FI), Align);

  unsigned Opc = 0;
  bool Offset = true;
  switch (TRI->getSpillSize(*RC)) {
  case 1:
    if (AArch64::FPR8RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRBui;
    break;
  case 2:
    if (AArch64::FPR16RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRHui;
    break;
  case 4:
    if (AArch64::GPR32allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::LDRWui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &AArch64::GPR32RegClass);
      else
        assert(DestReg != AArch64::WSP);
    } else if (AArch64::FPR32RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRSui;
    break;
  case 8:
    if (AArch64::GPR64allRegClass.hasSubClassEq(RC)) {
      Opc = AArch64::LDRXui;
      if (TargetRegisterInfo::isVirtualRegister(DestReg))
        MF.getRegInfo().constrainRegClass(DestReg, &AArch64::GPR64RegClass);
      else
        assert(DestReg != AArch64::SP);
    } else if (AArch64::FPR64RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRDui;
    break;
  case 16:
    if (AArch64::FPR128RegClass.hasSubClassEq(RC))
      Opc = AArch64::LDRQui;
    else if (AArch64::DDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Twov1d;
      Offset = false;
    }
    break;
  case 24:
    if (AArch64::DDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Threev1d;
      Offset = false;
    }
    break;
  case 32:
    if (AArch64::DDDDRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Fourv1d;
      Offset = false;
    } else if (AArch64::QQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Twov2d;
      Offset = false;
    }
    break;
  case 48:
    if (AArch64::QQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Threev2d;
      Offset = false;
    }
    break;
  case 64:
    if (AArch64::QQQQRegClass.hasSubClassEq(RC)) {
      assert(Subtarget.hasNEON() && "Unexpected register load without NEON");
      Opc = AArch64::LD1Fourv2d;
      Offset = false;
    }
    break;
  }
  assert(Opc && "Unknown register class");

  const MachineInstrBuilder MI = BuildMI(MBB, MBBI, DL, get(Opc))
                                     .addReg(DestReg, getDefRegState(true))
                                     .addFrameIndex(FI);
  if (Offset)
    MI.addImm(0);
  MI.addMemOperand(MMO);
}

void llvm::emitFrameOffset(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, const DebugLoc &DL,
                           unsigned DestReg, unsigned SrcReg, int Offset,
                           const TargetInstrInfo *TII,
                           MachineInstr::MIFlag Flag, bool SetNZCV) {
  if (DestReg == SrcReg && Offset == 0)
    return;

  assert((DestReg != AArch64::SP || Offset % 16 == 0) &&
         "SP increment/decrement not 16-byte aligned");

  bool isSub = Offset < 0;
  if (isSub)
    Offset = -Offset;

  // FIXME: If the offset won't fit in 24-bits, compute the offset into a
  // scratch register.  If DestReg is a virtual register, use it as the
  // scratch register; otherwise, create a new virtual register (to be
  // replaced by the scavenger at the end of PEI).  That case can be optimized
  // slightly if DestReg is SP which is always 16-byte aligned, so the scratch
  // register can be loaded with offset%8 and the add/sub can use an extending
  // instruction with LSL#3.
  // Currently the function handles any offsets but generates a poor sequence
  // of code.
  //  assert(Offset < (1 << 24) && "unimplemented reg plus immediate");

  unsigned Opc;
  if (SetNZCV)
    Opc = isSub ? AArch64::SUBSXri : AArch64::ADDSXri;
  else
    Opc = isSub ? AArch64::SUBXri : AArch64::ADDXri;
  const unsigned MaxEncoding = 0xfff;
  const unsigned ShiftSize = 12;
  const unsigned MaxEncodableValue = MaxEncoding << ShiftSize;
  while (((unsigned)Offset) >= (1 << ShiftSize)) {
    unsigned ThisVal;
    if (((unsigned)Offset) > MaxEncodableValue) {
      ThisVal = MaxEncodableValue;
    } else {
      ThisVal = Offset & MaxEncodableValue;
    }
    assert((ThisVal >> ShiftSize) <= MaxEncoding &&
           "Encoding cannot handle value that big");
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
        .addReg(SrcReg)
        .addImm(ThisVal >> ShiftSize)
        .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, ShiftSize))
        .setMIFlag(Flag);

    SrcReg = DestReg;
    Offset -= ThisVal;
    if (Offset == 0)
      return;
  }
  BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
      .addReg(SrcReg)
      .addImm(Offset)
      .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
      .setMIFlag(Flag);
}

MachineInstr *AArch64InstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FrameIndex,
    LiveIntervals *LIS) const {
  // This is a bit of a hack. Consider this instruction:
  //
  //   %0 = COPY %sp; GPR64all:%0
  //
  // We explicitly chose GPR64all for the virtual register so such a copy might
  // be eliminated by RegisterCoalescer. However, that may not be possible, and
  // %0 may even spill. We can't spill %sp, and since it is in the GPR64all
  // register class, TargetInstrInfo::foldMemoryOperand() is going to try.
  //
  // To prevent that, we are going to constrain the %0 register class here.
  //
  // <rdar://problem/11522048>
  //
  if (MI.isFullCopy()) {
    unsigned DstReg = MI.getOperand(0).getReg();
    unsigned SrcReg = MI.getOperand(1).getReg();
    if (SrcReg == AArch64::SP &&
        TargetRegisterInfo::isVirtualRegister(DstReg)) {
      MF.getRegInfo().constrainRegClass(DstReg, &AArch64::GPR64RegClass);
      return nullptr;
    }
    if (DstReg == AArch64::SP &&
        TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      MF.getRegInfo().constrainRegClass(SrcReg, &AArch64::GPR64RegClass);
      return nullptr;
    }
  }

  // Handle the case where a copy is being spilled or filled but the source
  // and destination register class don't match.  For example:
  //
  //   %0 = COPY %xzr; GPR64common:%0
  //
  // In this case we can still safely fold away the COPY and generate the
  // following spill code:
  //
  //   STRXui %xzr, %stack.0
  //
  // This also eliminates spilled cross register class COPYs (e.g. between x and
  // d regs) of the same size.  For example:
  //
  //   %0 = COPY %1; GPR64:%0, FPR64:%1
  //
  // will be filled as
  //
  //   LDRDui %0, fi<#0>
  //
  // instead of
  //
  //   LDRXui %Temp, fi<#0>
  //   %0 = FMOV %Temp
  //
  if (MI.isCopy() && Ops.size() == 1 &&
      // Make sure we're only folding the explicit COPY defs/uses.
      (Ops[0] == 0 || Ops[0] == 1)) {
    bool IsSpill = Ops[0] == 0;
    bool IsFill = !IsSpill;
    const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
    const MachineRegisterInfo &MRI = MF.getRegInfo();
    MachineBasicBlock &MBB = *MI.getParent();
    const MachineOperand &DstMO = MI.getOperand(0);
    const MachineOperand &SrcMO = MI.getOperand(1);
    unsigned DstReg = DstMO.getReg();
    unsigned SrcReg = SrcMO.getReg();
    // This is slightly expensive to compute for physical regs since
    // getMinimalPhysRegClass is slow.
    auto getRegClass = [&](unsigned Reg) {
      return TargetRegisterInfo::isVirtualRegister(Reg)
                 ? MRI.getRegClass(Reg)
                 : TRI.getMinimalPhysRegClass(Reg);
    };

    if (DstMO.getSubReg() == 0 && SrcMO.getSubReg() == 0) {
      assert(TRI.getRegSizeInBits(*getRegClass(DstReg)) ==
                 TRI.getRegSizeInBits(*getRegClass(SrcReg)) &&
             "Mismatched register size in non subreg COPY");
      if (IsSpill)
        storeRegToStackSlot(MBB, InsertPt, SrcReg, SrcMO.isKill(), FrameIndex,
                            getRegClass(SrcReg), &TRI);
      else
        loadRegFromStackSlot(MBB, InsertPt, DstReg, FrameIndex,
                             getRegClass(DstReg), &TRI);
      return &*--InsertPt;
    }

    // Handle cases like spilling def of:
    //
    //   %0:sub_32<def,read-undef> = COPY %wzr; GPR64common:%0
    //
    // where the physical register source can be widened and stored to the full
    // virtual reg destination stack slot, in this case producing:
    //
    //   STRXui %xzr, %stack.0
    //
    if (IsSpill && DstMO.isUndef() &&
        TargetRegisterInfo::isPhysicalRegister(SrcReg)) {
      assert(SrcMO.getSubReg() == 0 &&
             "Unexpected subreg on physical register");
      const TargetRegisterClass *SpillRC;
      unsigned SpillSubreg;
      switch (DstMO.getSubReg()) {
      default:
        SpillRC = nullptr;
        break;
      case AArch64::sub_32:
      case AArch64::ssub:
        if (AArch64::GPR32RegClass.contains(SrcReg)) {
          SpillRC = &AArch64::GPR64RegClass;
          SpillSubreg = AArch64::sub_32;
        } else if (AArch64::FPR32RegClass.contains(SrcReg)) {
          SpillRC = &AArch64::FPR64RegClass;
          SpillSubreg = AArch64::ssub;
        } else
          SpillRC = nullptr;
        break;
      case AArch64::dsub:
        if (AArch64::FPR64RegClass.contains(SrcReg)) {
          SpillRC = &AArch64::FPR128RegClass;
          SpillSubreg = AArch64::dsub;
        } else
          SpillRC = nullptr;
        break;
      }

      if (SpillRC)
        if (unsigned WidenedSrcReg =
                TRI.getMatchingSuperReg(SrcReg, SpillSubreg, SpillRC)) {
          storeRegToStackSlot(MBB, InsertPt, WidenedSrcReg, SrcMO.isKill(),
                              FrameIndex, SpillRC, &TRI);
          return &*--InsertPt;
        }
    }

    // Handle cases like filling use of:
    //
    //   %0:sub_32<def,read-undef> = COPY %1; GPR64:%0, GPR32:%1
    //
    // where we can load the full virtual reg source stack slot, into the subreg
    // destination, in this case producing:
    //
    //   LDRWui %0:sub_32<def,read-undef>, %stack.0
    //
    if (IsFill && SrcMO.getSubReg() == 0 && DstMO.isUndef()) {
      const TargetRegisterClass *FillRC;
      switch (DstMO.getSubReg()) {
      default:
        FillRC = nullptr;
        break;
      case AArch64::sub_32:
        FillRC = &AArch64::GPR32RegClass;
        break;
      case AArch64::ssub:
        FillRC = &AArch64::FPR32RegClass;
        break;
      case AArch64::dsub:
        FillRC = &AArch64::FPR64RegClass;
        break;
      }

      if (FillRC) {
        assert(TRI.getRegSizeInBits(*getRegClass(SrcReg)) ==
                   TRI.getRegSizeInBits(*FillRC) &&
               "Mismatched regclass size on folded subreg COPY");
        loadRegFromStackSlot(MBB, InsertPt, DstReg, FrameIndex, FillRC, &TRI);
        MachineInstr &LoadMI = *--InsertPt;
        MachineOperand &LoadDst = LoadMI.getOperand(0);
        assert(LoadDst.getSubReg() == 0 && "unexpected subreg on fill load");
        LoadDst.setSubReg(DstMO.getSubReg());
        LoadDst.setIsUndef();
        return &LoadMI;
      }
    }
  }

  // Cannot fold.
  return nullptr;
}

int llvm::isAArch64FrameOffsetLegal(const MachineInstr &MI, int &Offset,
                                    bool *OutUseUnscaledOp,
                                    unsigned *OutUnscaledOp,
                                    int *EmittableOffset) {
  int Scale = 1;
  bool IsSigned = false;
  // The ImmIdx should be changed case by case if it is not 2.
  unsigned ImmIdx = 2;
  unsigned UnscaledOp = 0;
  // Set output values in case of early exit.
  if (EmittableOffset)
    *EmittableOffset = 0;
  if (OutUseUnscaledOp)
    *OutUseUnscaledOp = false;
  if (OutUnscaledOp)
    *OutUnscaledOp = 0;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("unhandled opcode in rewriteAArch64FrameIndex");
  // Vector spills/fills can't take an immediate offset.
  case AArch64::LD1Twov2d:
  case AArch64::LD1Threev2d:
  case AArch64::LD1Fourv2d:
  case AArch64::LD1Twov1d:
  case AArch64::LD1Threev1d:
  case AArch64::LD1Fourv1d:
  case AArch64::ST1Twov2d:
  case AArch64::ST1Threev2d:
  case AArch64::ST1Fourv2d:
  case AArch64::ST1Twov1d:
  case AArch64::ST1Threev1d:
  case AArch64::ST1Fourv1d:
    return AArch64FrameOffsetCannotUpdate;
  case AArch64::PRFMui:
    Scale = 8;
    UnscaledOp = AArch64::PRFUMi;
    break;
  case AArch64::LDRXui:
    Scale = 8;
    UnscaledOp = AArch64::LDURXi;
    break;
  case AArch64::LDRWui:
    Scale = 4;
    UnscaledOp = AArch64::LDURWi;
    break;
  case AArch64::LDRBui:
    Scale = 1;
    UnscaledOp = AArch64::LDURBi;
    break;
  case AArch64::LDRHui:
    Scale = 2;
    UnscaledOp = AArch64::LDURHi;
    break;
  case AArch64::LDRSui:
    Scale = 4;
    UnscaledOp = AArch64::LDURSi;
    break;
  case AArch64::LDRDui:
    Scale = 8;
    UnscaledOp = AArch64::LDURDi;
    break;
  case AArch64::LDRQui:
    Scale = 16;
    UnscaledOp = AArch64::LDURQi;
    break;
  case AArch64::LDRBBui:
    Scale = 1;
    UnscaledOp = AArch64::LDURBBi;
    break;
  case AArch64::LDRHHui:
    Scale = 2;
    UnscaledOp = AArch64::LDURHHi;
    break;
  case AArch64::LDRSBXui:
    Scale = 1;
    UnscaledOp = AArch64::LDURSBXi;
    break;
  case AArch64::LDRSBWui:
    Scale = 1;
    UnscaledOp = AArch64::LDURSBWi;
    break;
  case AArch64::LDRSHXui:
    Scale = 2;
    UnscaledOp = AArch64::LDURSHXi;
    break;
  case AArch64::LDRSHWui:
    Scale = 2;
    UnscaledOp = AArch64::LDURSHWi;
    break;
  case AArch64::LDRSWui:
    Scale = 4;
    UnscaledOp = AArch64::LDURSWi;
    break;

  case AArch64::STRXui:
    Scale = 8;
    UnscaledOp = AArch64::STURXi;
    break;
  case AArch64::STRWui:
    Scale = 4;
    UnscaledOp = AArch64::STURWi;
    break;
  case AArch64::STRBui:
    Scale = 1;
    UnscaledOp = AArch64::STURBi;
    break;
  case AArch64::STRHui:
    Scale = 2;
    UnscaledOp = AArch64::STURHi;
    break;
  case AArch64::STRSui:
    Scale = 4;
    UnscaledOp = AArch64::STURSi;
    break;
  case AArch64::STRDui:
    Scale = 8;
    UnscaledOp = AArch64::STURDi;
    break;
  case AArch64::STRQui:
    Scale = 16;
    UnscaledOp = AArch64::STURQi;
    break;
  case AArch64::STRBBui:
    Scale = 1;
    UnscaledOp = AArch64::STURBBi;
    break;
  case AArch64::STRHHui:
    Scale = 2;
    UnscaledOp = AArch64::STURHHi;
    break;

  case AArch64::LDPXi:
  case AArch64::LDPDi:
  case AArch64::STPXi:
  case AArch64::STPDi:
  case AArch64::LDNPXi:
  case AArch64::LDNPDi:
  case AArch64::STNPXi:
  case AArch64::STNPDi:
    ImmIdx = 3;
    IsSigned = true;
    Scale = 8;
    break;
  case AArch64::LDPQi:
  case AArch64::STPQi:
  case AArch64::LDNPQi:
  case AArch64::STNPQi:
    ImmIdx = 3;
    IsSigned = true;
    Scale = 16;
    break;
  case AArch64::LDPWi:
  case AArch64::LDPSi:
  case AArch64::STPWi:
  case AArch64::STPSi:
  case AArch64::LDNPWi:
  case AArch64::LDNPSi:
  case AArch64::STNPWi:
  case AArch64::STNPSi:
    ImmIdx = 3;
    IsSigned = true;
    Scale = 4;
    break;

  case AArch64::LDURXi:
  case AArch64::LDURWi:
  case AArch64::LDURBi:
  case AArch64::LDURHi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURHHi:
  case AArch64::LDURBBi:
  case AArch64::LDURSBXi:
  case AArch64::LDURSBWi:
  case AArch64::LDURSHXi:
  case AArch64::LDURSHWi:
  case AArch64::LDURSWi:
  case AArch64::STURXi:
  case AArch64::STURWi:
  case AArch64::STURBi:
  case AArch64::STURHi:
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURBBi:
  case AArch64::STURHHi:
    Scale = 1;
    break;
  }

  Offset += MI.getOperand(ImmIdx).getImm() * Scale;

  bool useUnscaledOp = false;
  // If the offset doesn't match the scale, we rewrite the instruction to
  // use the unscaled instruction instead. Likewise, if we have a negative
  // offset (and have an unscaled op to use).
  if ((Offset & (Scale - 1)) != 0 || (Offset < 0 && UnscaledOp != 0))
    useUnscaledOp = true;

  // Use an unscaled addressing mode if the instruction has a negative offset
  // (or if the instruction is already using an unscaled addressing mode).
  unsigned MaskBits;
  if (IsSigned) {
    // ldp/stp instructions.
    MaskBits = 7;
    Offset /= Scale;
  } else if (UnscaledOp == 0 || useUnscaledOp) {
    MaskBits = 9;
    IsSigned = true;
    Scale = 1;
  } else {
    MaskBits = 12;
    IsSigned = false;
    Offset /= Scale;
  }

  // Attempt to fold address computation.
  int MaxOff = (1 << (MaskBits - IsSigned)) - 1;
  int MinOff = (IsSigned ? (-MaxOff - 1) : 0);
  if (Offset >= MinOff && Offset <= MaxOff) {
    if (EmittableOffset)
      *EmittableOffset = Offset;
    Offset = 0;
  } else {
    int NewOff = Offset < 0 ? MinOff : MaxOff;
    if (EmittableOffset)
      *EmittableOffset = NewOff;
    Offset = (Offset - NewOff) * Scale;
  }
  if (OutUseUnscaledOp)
    *OutUseUnscaledOp = useUnscaledOp;
  if (OutUnscaledOp)
    *OutUnscaledOp = UnscaledOp;
  return AArch64FrameOffsetCanUpdate |
         (Offset == 0 ? AArch64FrameOffsetIsLegal : 0);
}

bool llvm::rewriteAArch64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                                    unsigned FrameReg, int &Offset,
                                    const AArch64InstrInfo *TII) {
  unsigned Opcode = MI.getOpcode();
  unsigned ImmIdx = FrameRegIdx + 1;

  if (Opcode == AArch64::ADDSXri || Opcode == AArch64::ADDXri) {
    Offset += MI.getOperand(ImmIdx).getImm();
    emitFrameOffset(*MI.getParent(), MI, MI.getDebugLoc(),
                    MI.getOperand(0).getReg(), FrameReg, Offset, TII,
                    MachineInstr::NoFlags, (Opcode == AArch64::ADDSXri));
    MI.eraseFromParent();
    Offset = 0;
    return true;
  }

  int NewOffset;
  unsigned UnscaledOp;
  bool UseUnscaledOp;
  int Status = isAArch64FrameOffsetLegal(MI, Offset, &UseUnscaledOp,
                                         &UnscaledOp, &NewOffset);
  if (Status & AArch64FrameOffsetCanUpdate) {
    if (Status & AArch64FrameOffsetIsLegal)
      // Replace the FrameIndex with FrameReg.
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
    if (UseUnscaledOp)
      MI.setDesc(TII->get(UnscaledOp));

    MI.getOperand(ImmIdx).ChangeToImmediate(NewOffset);
    return Offset == 0;
  }

  return false;
}

void AArch64InstrInfo::getNoop(MCInst &NopInst) const {
  NopInst.setOpcode(AArch64::HINT);
  NopInst.addOperand(MCOperand::createImm(0));
}

// AArch64 supports MachineCombiner.
bool AArch64InstrInfo::useMachineCombiner() const { return true; }

// True when Opc sets flag
static bool isCombineInstrSettingFlag(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSXrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBSWri:
  case AArch64::SUBSXri:
    return true;
  default:
    break;
  }
  return false;
}

// 32b Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate32(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDWrr:
  case AArch64::ADDWri:
  case AArch64::SUBWrr:
  case AArch64::ADDSWrr:
  case AArch64::ADDSWri:
  case AArch64::SUBSWrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBWri:
  case AArch64::SUBSWri:
    return true;
  default:
    break;
  }
  return false;
}

// 64b Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate64(unsigned Opc) {
  switch (Opc) {
  case AArch64::ADDXrr:
  case AArch64::ADDXri:
  case AArch64::SUBXrr:
  case AArch64::ADDSXrr:
  case AArch64::ADDSXri:
  case AArch64::SUBSXrr:
  // Note: MSUB Wd,Wn,Wm,Wi -> Wd = Wi - WnxWm, not Wd=WnxWm - Wi.
  case AArch64::SUBXri:
  case AArch64::SUBSXri:
    return true;
  default:
    break;
  }
  return false;
}

// FP Opcodes that can be combined with a FMUL
static bool isCombineInstrCandidateFP(const MachineInstr &Inst) {
  switch (Inst.getOpcode()) {
  default:
    break;
  case AArch64::FADDSrr:
  case AArch64::FADDDrr:
  case AArch64::FADDv2f32:
  case AArch64::FADDv2f64:
  case AArch64::FADDv4f32:
  case AArch64::FSUBSrr:
  case AArch64::FSUBDrr:
  case AArch64::FSUBv2f32:
  case AArch64::FSUBv2f64:
  case AArch64::FSUBv4f32:
    TargetOptions Options = Inst.getParent()->getParent()->getTarget().Options;
    return (Options.UnsafeFPMath ||
            Options.AllowFPOpFusion == FPOpFusion::Fast);
  }
  return false;
}

// Opcodes that can be combined with a MUL
static bool isCombineInstrCandidate(unsigned Opc) {
  return (isCombineInstrCandidate32(Opc) || isCombineInstrCandidate64(Opc));
}

//
// Utility routine that checks if \param MO is defined by an
// \param CombineOpc instruction in the basic block \param MBB
static bool canCombine(MachineBasicBlock &MBB, MachineOperand &MO,
                       unsigned CombineOpc, unsigned ZeroReg = 0,
                       bool CheckZeroReg = false) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineInstr *MI = nullptr;

  if (MO.isReg() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    MI = MRI.getUniqueVRegDef(MO.getReg());
  // And it needs to be in the trace (otherwise, it won't have a depth).
  if (!MI || MI->getParent() != &MBB || (unsigned)MI->getOpcode() != CombineOpc)
    return false;
  // Must only used by the user we combine with.
  if (!MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
    return false;

  if (CheckZeroReg) {
    assert(MI->getNumOperands() >= 4 && MI->getOperand(0).isReg() &&
           MI->getOperand(1).isReg() && MI->getOperand(2).isReg() &&
           MI->getOperand(3).isReg() && "MAdd/MSub must have a least 4 regs");
    // The third input reg must be zero.
    if (MI->getOperand(3).getReg() != ZeroReg)
      return false;
  }

  return true;
}

//
// Is \param MO defined by an integer multiply and can be combined?
static bool canCombineWithMUL(MachineBasicBlock &MBB, MachineOperand &MO,
                              unsigned MulOpc, unsigned ZeroReg) {
  return canCombine(MBB, MO, MulOpc, ZeroReg, true);
}

//
// Is \param MO defined by a floating-point multiply and can be combined?
static bool canCombineWithFMUL(MachineBasicBlock &MBB, MachineOperand &MO,
                               unsigned MulOpc) {
  return canCombine(MBB, MO, MulOpc);
}

// TODO: There are many more machine instruction opcodes to match:
//       1. Other data types (integer, vectors)
//       2. Other math / logic operations (xor, or)
//       3. Other forms of the same operation (intrinsics and other variants)
bool AArch64InstrInfo::isAssociativeAndCommutative(
    const MachineInstr &Inst) const {
  switch (Inst.getOpcode()) {
  case AArch64::FADDDrr:
  case AArch64::FADDSrr:
  case AArch64::FADDv2f32:
  case AArch64::FADDv2f64:
  case AArch64::FADDv4f32:
  case AArch64::FMULDrr:
  case AArch64::FMULSrr:
  case AArch64::FMULX32:
  case AArch64::FMULX64:
  case AArch64::FMULXv2f32:
  case AArch64::FMULXv2f64:
  case AArch64::FMULXv4f32:
  case AArch64::FMULv2f32:
  case AArch64::FMULv2f64:
  case AArch64::FMULv4f32:
    return Inst.getParent()->getParent()->getTarget().Options.UnsafeFPMath;
  default:
    return false;
  }
}

/// Find instructions that can be turned into madd.
static bool getMaddPatterns(MachineInstr &Root,
                            SmallVectorImpl<MachineCombinerPattern> &Patterns) {
  unsigned Opc = Root.getOpcode();
  MachineBasicBlock &MBB = *Root.getParent();
  bool Found = false;

  if (!isCombineInstrCandidate(Opc))
    return false;
  if (isCombineInstrSettingFlag(Opc)) {
    int Cmp_NZCV = Root.findRegisterDefOperandIdx(AArch64::NZCV, true);
    // When NZCV is live bail out.
    if (Cmp_NZCV == -1)
      return false;
    unsigned NewOpc = convertToNonFlagSettingOpc(Root);
    // When opcode can't change bail out.
    // CHECKME: do we miss any cases for opcode conversion?
    if (NewOpc == Opc)
      return false;
    Opc = NewOpc;
  }

  switch (Opc) {
  default:
    break;
  case AArch64::ADDWrr:
    assert(Root.getOperand(1).isReg() && Root.getOperand(2).isReg() &&
           "ADDWrr does not have register operands");
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDW_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDW_OP2);
      Found = true;
    }
    break;
  case AArch64::ADDXrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDX_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDX_OP2);
      Found = true;
    }
    break;
  case AArch64::SUBWrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBW_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBW_OP2);
      Found = true;
    }
    break;
  case AArch64::SUBXrr:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBX_OP1);
      Found = true;
    }
    if (canCombineWithMUL(MBB, Root.getOperand(2), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBX_OP2);
      Found = true;
    }
    break;
  case AArch64::ADDWri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDWI_OP1);
      Found = true;
    }
    break;
  case AArch64::ADDXri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULADDXI_OP1);
      Found = true;
    }
    break;
  case AArch64::SUBWri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDWrrr,
                          AArch64::WZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBWI_OP1);
      Found = true;
    }
    break;
  case AArch64::SUBXri:
    if (canCombineWithMUL(MBB, Root.getOperand(1), AArch64::MADDXrrr,
                          AArch64::XZR)) {
      Patterns.push_back(MachineCombinerPattern::MULSUBXI_OP1);
      Found = true;
    }
    break;
  }
  return Found;
}
/// Floating-Point Support

/// Find instructions that can be turned into madd.
static bool getFMAPatterns(MachineInstr &Root,
                           SmallVectorImpl<MachineCombinerPattern> &Patterns) {

  if (!isCombineInstrCandidateFP(Root))
    return false;

  MachineBasicBlock &MBB = *Root.getParent();
  bool Found = false;

  switch (Root.getOpcode()) {
  default:
    assert(false && "Unsupported FP instruction in combiner\n");
    break;
  case AArch64::FADDSrr:
    assert(Root.getOperand(1).isReg() && Root.getOperand(2).isReg() &&
           "FADDWrr does not have register operands");
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FMULSrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULADDS_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv1i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv1i32_indexed_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2), AArch64::FMULSrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULADDS_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv1i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv1i32_indexed_OP2);
      Found = true;
    }
    break;
  case AArch64::FADDDrr:
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FMULDrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULADDD_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv1i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv1i64_indexed_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2), AArch64::FMULDrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULADDD_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv1i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv1i64_indexed_OP2);
      Found = true;
    }
    break;
  case AArch64::FADDv2f32:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv2i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2i32_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv2f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2f32_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv2i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2i32_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv2f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2f32_OP2);
      Found = true;
    }
    break;
  case AArch64::FADDv2f64:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv2i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2i64_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv2f64)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2f64_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv2i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2i64_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv2f64)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv2f64_OP2);
      Found = true;
    }
    break;
  case AArch64::FADDv4f32:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv4i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv4i32_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv4f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv4f32_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv4i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv4i32_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv4f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLAv4f32_OP2);
      Found = true;
    }
    break;

  case AArch64::FSUBSrr:
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FMULSrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULSUBS_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2), AArch64::FMULSrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULSUBS_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv1i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv1i32_indexed_OP2);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FNMULSrr)) {
      Patterns.push_back(MachineCombinerPattern::FNMULSUBS_OP1);
      Found = true;
    }
    break;
  case AArch64::FSUBDrr:
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FMULDrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULSUBD_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2), AArch64::FMULDrr)) {
      Patterns.push_back(MachineCombinerPattern::FMULSUBD_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv1i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv1i64_indexed_OP2);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(1), AArch64::FNMULDrr)) {
      Patterns.push_back(MachineCombinerPattern::FNMULSUBD_OP1);
      Found = true;
    }
    break;
  case AArch64::FSUBv2f32:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv2i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2i32_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv2f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2f32_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv2i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2i32_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv2f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2f32_OP2);
      Found = true;
    }
    break;
  case AArch64::FSUBv2f64:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv2i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2i64_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv2f64)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2f64_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv2i64_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2i64_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv2f64)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv2f64_OP2);
      Found = true;
    }
    break;
  case AArch64::FSUBv4f32:
    if (canCombineWithFMUL(MBB, Root.getOperand(1),
                           AArch64::FMULv4i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv4i32_indexed_OP1);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(1),
                                  AArch64::FMULv4f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv4f32_OP1);
      Found = true;
    }
    if (canCombineWithFMUL(MBB, Root.getOperand(2),
                           AArch64::FMULv4i32_indexed)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv4i32_indexed_OP2);
      Found = true;
    } else if (canCombineWithFMUL(MBB, Root.getOperand(2),
                                  AArch64::FMULv4f32)) {
      Patterns.push_back(MachineCombinerPattern::FMLSv4f32_OP2);
      Found = true;
    }
    break;
  }
  return Found;
}

/// Return true when a code sequence can improve throughput. It
/// should be called only for instructions in loops.
/// \param Pattern - combiner pattern
bool AArch64InstrInfo::isThroughputPattern(
    MachineCombinerPattern Pattern) const {
  switch (Pattern) {
  default:
    break;
  case MachineCombinerPattern::FMULADDS_OP1:
  case MachineCombinerPattern::FMULADDS_OP2:
  case MachineCombinerPattern::FMULSUBS_OP1:
  case MachineCombinerPattern::FMULSUBS_OP2:
  case MachineCombinerPattern::FMULADDD_OP1:
  case MachineCombinerPattern::FMULADDD_OP2:
  case MachineCombinerPattern::FMULSUBD_OP1:
  case MachineCombinerPattern::FMULSUBD_OP2:
  case MachineCombinerPattern::FNMULSUBS_OP1:
  case MachineCombinerPattern::FNMULSUBD_OP1:
  case MachineCombinerPattern::FMLAv1i32_indexed_OP1:
  case MachineCombinerPattern::FMLAv1i32_indexed_OP2:
  case MachineCombinerPattern::FMLAv1i64_indexed_OP1:
  case MachineCombinerPattern::FMLAv1i64_indexed_OP2:
  case MachineCombinerPattern::FMLAv2f32_OP2:
  case MachineCombinerPattern::FMLAv2f32_OP1:
  case MachineCombinerPattern::FMLAv2f64_OP1:
  case MachineCombinerPattern::FMLAv2f64_OP2:
  case MachineCombinerPattern::FMLAv2i32_indexed_OP1:
  case MachineCombinerPattern::FMLAv2i32_indexed_OP2:
  case MachineCombinerPattern::FMLAv2i64_indexed_OP1:
  case MachineCombinerPattern::FMLAv2i64_indexed_OP2:
  case MachineCombinerPattern::FMLAv4f32_OP1:
  case MachineCombinerPattern::FMLAv4f32_OP2:
  case MachineCombinerPattern::FMLAv4i32_indexed_OP1:
  case MachineCombinerPattern::FMLAv4i32_indexed_OP2:
  case MachineCombinerPattern::FMLSv1i32_indexed_OP2:
  case MachineCombinerPattern::FMLSv1i64_indexed_OP2:
  case MachineCombinerPattern::FMLSv2i32_indexed_OP2:
  case MachineCombinerPattern::FMLSv2i64_indexed_OP2:
  case MachineCombinerPattern::FMLSv2f32_OP2:
  case MachineCombinerPattern::FMLSv2f64_OP2:
  case MachineCombinerPattern::FMLSv4i32_indexed_OP2:
  case MachineCombinerPattern::FMLSv4f32_OP2:
    return true;
  } // end switch (Pattern)
  return false;
}
/// Return true when there is potentially a faster code sequence for an
/// instruction chain ending in \p Root. All potential patterns are listed in
/// the \p Pattern vector. Pattern should be sorted in priority order since the
/// pattern evaluator stops checking as soon as it finds a faster sequence.

bool AArch64InstrInfo::getMachineCombinerPatterns(
    MachineInstr &Root,
    SmallVectorImpl<MachineCombinerPattern> &Patterns) const {
  // Integer patterns
  if (getMaddPatterns(Root, Patterns))
    return true;
  // Floating point patterns
  if (getFMAPatterns(Root, Patterns))
    return true;

  return TargetInstrInfo::getMachineCombinerPatterns(Root, Patterns);
}

enum class FMAInstKind { Default, Indexed, Accumulator };
/// genFusedMultiply - Generate fused multiply instructions.
/// This function supports both integer and floating point instructions.
/// A typical example:
///  F|MUL I=A,B,0
///  F|ADD R,I,C
///  ==> F|MADD R,A,B,C
/// \param MF Containing MachineFunction
/// \param MRI Register information
/// \param TII Target information
/// \param Root is the F|ADD instruction
/// \param [out] InsInstrs is a vector of machine instructions and will
/// contain the generated madd instruction
/// \param IdxMulOpd is index of operand in Root that is the result of
/// the F|MUL. In the example above IdxMulOpd is 1.
/// \param MaddOpc the opcode fo the f|madd instruction
/// \param RC Register class of operands
/// \param kind of fma instruction (addressing mode) to be generated
/// \param ReplacedAddend is the result register from the instruction
/// replacing the non-combined operand, if any.
static MachineInstr *
genFusedMultiply(MachineFunction &MF, MachineRegisterInfo &MRI,
                 const TargetInstrInfo *TII, MachineInstr &Root,
                 SmallVectorImpl<MachineInstr *> &InsInstrs, unsigned IdxMulOpd,
                 unsigned MaddOpc, const TargetRegisterClass *RC,
                 FMAInstKind kind = FMAInstKind::Default,
                 const unsigned *ReplacedAddend = nullptr) {
  assert(IdxMulOpd == 1 || IdxMulOpd == 2);

  unsigned IdxOtherOpd = IdxMulOpd == 1 ? 2 : 1;
  MachineInstr *MUL = MRI.getUniqueVRegDef(Root.getOperand(IdxMulOpd).getReg());
  unsigned ResultReg = Root.getOperand(0).getReg();
  unsigned SrcReg0 = MUL->getOperand(1).getReg();
  bool Src0IsKill = MUL->getOperand(1).isKill();
  unsigned SrcReg1 = MUL->getOperand(2).getReg();
  bool Src1IsKill = MUL->getOperand(2).isKill();

  unsigned SrcReg2;
  bool Src2IsKill;
  if (ReplacedAddend) {
    // If we just generated a new addend, we must be it's only use.
    SrcReg2 = *ReplacedAddend;
    Src2IsKill = true;
  } else {
    SrcReg2 = Root.getOperand(IdxOtherOpd).getReg();
    Src2IsKill = Root.getOperand(IdxOtherOpd).isKill();
  }

  if (TargetRegisterInfo::isVirtualRegister(ResultReg))
    MRI.constrainRegClass(ResultReg, RC);
  if (TargetRegisterInfo::isVirtualRegister(SrcReg0))
    MRI.constrainRegClass(SrcReg0, RC);
  if (TargetRegisterInfo::isVirtualRegister(SrcReg1))
    MRI.constrainRegClass(SrcReg1, RC);
  if (TargetRegisterInfo::isVirtualRegister(SrcReg2))
    MRI.constrainRegClass(SrcReg2, RC);

  MachineInstrBuilder MIB;
  if (kind == FMAInstKind::Default)
    MIB = BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc), ResultReg)
              .addReg(SrcReg0, getKillRegState(Src0IsKill))
              .addReg(SrcReg1, getKillRegState(Src1IsKill))
              .addReg(SrcReg2, getKillRegState(Src2IsKill));
  else if (kind == FMAInstKind::Indexed)
    MIB = BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc), ResultReg)
              .addReg(SrcReg2, getKillRegState(Src2IsKill))
              .addReg(SrcReg0, getKillRegState(Src0IsKill))
              .addReg(SrcReg1, getKillRegState(Src1IsKill))
              .addImm(MUL->getOperand(3).getImm());
  else if (kind == FMAInstKind::Accumulator)
    MIB = BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc), ResultReg)
              .addReg(SrcReg2, getKillRegState(Src2IsKill))
              .addReg(SrcReg0, getKillRegState(Src0IsKill))
              .addReg(SrcReg1, getKillRegState(Src1IsKill));
  else
    assert(false && "Invalid FMA instruction kind \n");
  // Insert the MADD (MADD, FMA, FMS, FMLA, FMSL)
  InsInstrs.push_back(MIB);
  return MUL;
}

/// genMaddR - Generate madd instruction and combine mul and add using
/// an extra virtual register
/// Example - an ADD intermediate needs to be stored in a register:
///   MUL I=A,B,0
///   ADD R,I,Imm
///   ==> ORR  V, ZR, Imm
///   ==> MADD R,A,B,V
/// \param MF Containing MachineFunction
/// \param MRI Register information
/// \param TII Target information
/// \param Root is the ADD instruction
/// \param [out] InsInstrs is a vector of machine instructions and will
/// contain the generated madd instruction
/// \param IdxMulOpd is index of operand in Root that is the result of
/// the MUL. In the example above IdxMulOpd is 1.
/// \param MaddOpc the opcode fo the madd instruction
/// \param VR is a virtual register that holds the value of an ADD operand
/// (V in the example above).
/// \param RC Register class of operands
static MachineInstr *genMaddR(MachineFunction &MF, MachineRegisterInfo &MRI,
                              const TargetInstrInfo *TII, MachineInstr &Root,
                              SmallVectorImpl<MachineInstr *> &InsInstrs,
                              unsigned IdxMulOpd, unsigned MaddOpc, unsigned VR,
                              const TargetRegisterClass *RC) {
  assert(IdxMulOpd == 1 || IdxMulOpd == 2);

  MachineInstr *MUL = MRI.getUniqueVRegDef(Root.getOperand(IdxMulOpd).getReg());
  unsigned ResultReg = Root.getOperand(0).getReg();
  unsigned SrcReg0 = MUL->getOperand(1).getReg();
  bool Src0IsKill = MUL->getOperand(1).isKill();
  unsigned SrcReg1 = MUL->getOperand(2).getReg();
  bool Src1IsKill = MUL->getOperand(2).isKill();

  if (TargetRegisterInfo::isVirtualRegister(ResultReg))
    MRI.constrainRegClass(ResultReg, RC);
  if (TargetRegisterInfo::isVirtualRegister(SrcReg0))
    MRI.constrainRegClass(SrcReg0, RC);
  if (TargetRegisterInfo::isVirtualRegister(SrcReg1))
    MRI.constrainRegClass(SrcReg1, RC);
  if (TargetRegisterInfo::isVirtualRegister(VR))
    MRI.constrainRegClass(VR, RC);

  MachineInstrBuilder MIB =
      BuildMI(MF, Root.getDebugLoc(), TII->get(MaddOpc), ResultReg)
          .addReg(SrcReg0, getKillRegState(Src0IsKill))
          .addReg(SrcReg1, getKillRegState(Src1IsKill))
          .addReg(VR);
  // Insert the MADD
  InsInstrs.push_back(MIB);
  return MUL;
}

/// When getMachineCombinerPatterns() finds potential patterns,
/// this function generates the instructions that could replace the
/// original code sequence
void AArch64InstrInfo::genAlternativeCodeSequence(
    MachineInstr &Root, MachineCombinerPattern Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const {
  MachineBasicBlock &MBB = *Root.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  MachineInstr *MUL;
  const TargetRegisterClass *RC;
  unsigned Opc;
  switch (Pattern) {
  default:
    // Reassociate instructions.
    TargetInstrInfo::genAlternativeCodeSequence(Root, Pattern, InsInstrs,
                                                DelInstrs, InstrIdxForVirtReg);
    return;
  case MachineCombinerPattern::MULADDW_OP1:
  case MachineCombinerPattern::MULADDX_OP1:
    // MUL I=A,B,0
    // ADD R,I,C
    // ==> MADD R,A,B,C
    // --- Create(MADD);
    if (Pattern == MachineCombinerPattern::MULADDW_OP1) {
      Opc = AArch64::MADDWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      Opc = AArch64::MADDXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC);
    break;
  case MachineCombinerPattern::MULADDW_OP2:
  case MachineCombinerPattern::MULADDX_OP2:
    // MUL I=A,B,0
    // ADD R,C,I
    // ==> MADD R,A,B,C
    // --- Create(MADD);
    if (Pattern == MachineCombinerPattern::MULADDW_OP2) {
      Opc = AArch64::MADDWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      Opc = AArch64::MADDXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC);
    break;
  case MachineCombinerPattern::MULADDWI_OP1:
  case MachineCombinerPattern::MULADDXI_OP1: {
    // MUL I=A,B,0
    // ADD R,I,Imm
    // ==> ORR  V, ZR, Imm
    // ==> MADD R,A,B,V
    // --- Create(MADD);
    const TargetRegisterClass *OrrRC;
    unsigned BitSize, OrrOpc, ZeroReg;
    if (Pattern == MachineCombinerPattern::MULADDWI_OP1) {
      OrrOpc = AArch64::ORRWri;
      OrrRC = &AArch64::GPR32spRegClass;
      BitSize = 32;
      ZeroReg = AArch64::WZR;
      Opc = AArch64::MADDWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      OrrOpc = AArch64::ORRXri;
      OrrRC = &AArch64::GPR64spRegClass;
      BitSize = 64;
      ZeroReg = AArch64::XZR;
      Opc = AArch64::MADDXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    unsigned NewVR = MRI.createVirtualRegister(OrrRC);
    uint64_t Imm = Root.getOperand(2).getImm();

    if (Root.getOperand(3).isImm()) {
      unsigned Val = Root.getOperand(3).getImm();
      Imm = Imm << Val;
    }
    uint64_t UImm = SignExtend64(Imm, BitSize);
    uint64_t Encoding;
    if (AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
      MachineInstrBuilder MIB1 =
          BuildMI(MF, Root.getDebugLoc(), TII->get(OrrOpc), NewVR)
              .addReg(ZeroReg)
              .addImm(Encoding);
      InsInstrs.push_back(MIB1);
      InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
      MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR, RC);
    }
    break;
  }
  case MachineCombinerPattern::MULSUBW_OP1:
  case MachineCombinerPattern::MULSUBX_OP1: {
    // MUL I=A,B,0
    // SUB R,I, C
    // ==> SUB  V, 0, C
    // ==> MADD R,A,B,V // = -C + A*B
    // --- Create(MADD);
    const TargetRegisterClass *SubRC;
    unsigned SubOpc, ZeroReg;
    if (Pattern == MachineCombinerPattern::MULSUBW_OP1) {
      SubOpc = AArch64::SUBWrr;
      SubRC = &AArch64::GPR32spRegClass;
      ZeroReg = AArch64::WZR;
      Opc = AArch64::MADDWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      SubOpc = AArch64::SUBXrr;
      SubRC = &AArch64::GPR64spRegClass;
      ZeroReg = AArch64::XZR;
      Opc = AArch64::MADDXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    unsigned NewVR = MRI.createVirtualRegister(SubRC);
    // SUB NewVR, 0, C
    MachineInstrBuilder MIB1 =
        BuildMI(MF, Root.getDebugLoc(), TII->get(SubOpc), NewVR)
            .addReg(ZeroReg)
            .add(Root.getOperand(2));
    InsInstrs.push_back(MIB1);
    InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
    MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR, RC);
    break;
  }
  case MachineCombinerPattern::MULSUBW_OP2:
  case MachineCombinerPattern::MULSUBX_OP2:
    // MUL I=A,B,0
    // SUB R,C,I
    // ==> MSUB R,A,B,C (computes C - A*B)
    // --- Create(MSUB);
    if (Pattern == MachineCombinerPattern::MULSUBW_OP2) {
      Opc = AArch64::MSUBWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      Opc = AArch64::MSUBXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC);
    break;
  case MachineCombinerPattern::MULSUBWI_OP1:
  case MachineCombinerPattern::MULSUBXI_OP1: {
    // MUL I=A,B,0
    // SUB R,I, Imm
    // ==> ORR  V, ZR, -Imm
    // ==> MADD R,A,B,V // = -Imm + A*B
    // --- Create(MADD);
    const TargetRegisterClass *OrrRC;
    unsigned BitSize, OrrOpc, ZeroReg;
    if (Pattern == MachineCombinerPattern::MULSUBWI_OP1) {
      OrrOpc = AArch64::ORRWri;
      OrrRC = &AArch64::GPR32spRegClass;
      BitSize = 32;
      ZeroReg = AArch64::WZR;
      Opc = AArch64::MADDWrrr;
      RC = &AArch64::GPR32RegClass;
    } else {
      OrrOpc = AArch64::ORRXri;
      OrrRC = &AArch64::GPR64spRegClass;
      BitSize = 64;
      ZeroReg = AArch64::XZR;
      Opc = AArch64::MADDXrrr;
      RC = &AArch64::GPR64RegClass;
    }
    unsigned NewVR = MRI.createVirtualRegister(OrrRC);
    uint64_t Imm = Root.getOperand(2).getImm();
    if (Root.getOperand(3).isImm()) {
      unsigned Val = Root.getOperand(3).getImm();
      Imm = Imm << Val;
    }
    uint64_t UImm = SignExtend64(-Imm, BitSize);
    uint64_t Encoding;
    if (AArch64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
      MachineInstrBuilder MIB1 =
          BuildMI(MF, Root.getDebugLoc(), TII->get(OrrOpc), NewVR)
              .addReg(ZeroReg)
              .addImm(Encoding);
      InsInstrs.push_back(MIB1);
      InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
      MUL = genMaddR(MF, MRI, TII, Root, InsInstrs, 1, Opc, NewVR, RC);
    }
    break;
  }
  // Floating Point Support
  case MachineCombinerPattern::FMULADDS_OP1:
  case MachineCombinerPattern::FMULADDD_OP1:
    // MUL I=A,B,0
    // ADD R,I,C
    // ==> MADD R,A,B,C
    // --- Create(MADD);
    if (Pattern == MachineCombinerPattern::FMULADDS_OP1) {
      Opc = AArch64::FMADDSrrr;
      RC = &AArch64::FPR32RegClass;
    } else {
      Opc = AArch64::FMADDDrrr;
      RC = &AArch64::FPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC);
    break;
  case MachineCombinerPattern::FMULADDS_OP2:
  case MachineCombinerPattern::FMULADDD_OP2:
    // FMUL I=A,B,0
    // FADD R,C,I
    // ==> FMADD R,A,B,C
    // --- Create(FMADD);
    if (Pattern == MachineCombinerPattern::FMULADDS_OP2) {
      Opc = AArch64::FMADDSrrr;
      RC = &AArch64::FPR32RegClass;
    } else {
      Opc = AArch64::FMADDDrrr;
      RC = &AArch64::FPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC);
    break;

  case MachineCombinerPattern::FMLAv1i32_indexed_OP1:
    Opc = AArch64::FMLAv1i32_indexed;
    RC = &AArch64::FPR32RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                           FMAInstKind::Indexed);
    break;
  case MachineCombinerPattern::FMLAv1i32_indexed_OP2:
    Opc = AArch64::FMLAv1i32_indexed;
    RC = &AArch64::FPR32RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                           FMAInstKind::Indexed);
    break;

  case MachineCombinerPattern::FMLAv1i64_indexed_OP1:
    Opc = AArch64::FMLAv1i64_indexed;
    RC = &AArch64::FPR64RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                           FMAInstKind::Indexed);
    break;
  case MachineCombinerPattern::FMLAv1i64_indexed_OP2:
    Opc = AArch64::FMLAv1i64_indexed;
    RC = &AArch64::FPR64RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                           FMAInstKind::Indexed);
    break;

  case MachineCombinerPattern::FMLAv2i32_indexed_OP1:
  case MachineCombinerPattern::FMLAv2f32_OP1:
    RC = &AArch64::FPR64RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv2i32_indexed_OP1) {
      Opc = AArch64::FMLAv2i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv2f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;
  case MachineCombinerPattern::FMLAv2i32_indexed_OP2:
  case MachineCombinerPattern::FMLAv2f32_OP2:
    RC = &AArch64::FPR64RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv2i32_indexed_OP2) {
      Opc = AArch64::FMLAv2i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv2f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMLAv2i64_indexed_OP1:
  case MachineCombinerPattern::FMLAv2f64_OP1:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv2i64_indexed_OP1) {
      Opc = AArch64::FMLAv2i64_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv2f64;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;
  case MachineCombinerPattern::FMLAv2i64_indexed_OP2:
  case MachineCombinerPattern::FMLAv2f64_OP2:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv2i64_indexed_OP2) {
      Opc = AArch64::FMLAv2i64_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv2f64;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMLAv4i32_indexed_OP1:
  case MachineCombinerPattern::FMLAv4f32_OP1:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv4i32_indexed_OP1) {
      Opc = AArch64::FMLAv4i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv4f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMLAv4i32_indexed_OP2:
  case MachineCombinerPattern::FMLAv4f32_OP2:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLAv4i32_indexed_OP2) {
      Opc = AArch64::FMLAv4i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLAv4f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMULSUBS_OP1:
  case MachineCombinerPattern::FMULSUBD_OP1: {
    // FMUL I=A,B,0
    // FSUB R,I,C
    // ==> FNMSUB R,A,B,C // = -C + A*B
    // --- Create(FNMSUB);
    if (Pattern == MachineCombinerPattern::FMULSUBS_OP1) {
      Opc = AArch64::FNMSUBSrrr;
      RC = &AArch64::FPR32RegClass;
    } else {
      Opc = AArch64::FNMSUBDrrr;
      RC = &AArch64::FPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC);
    break;
  }

  case MachineCombinerPattern::FNMULSUBS_OP1:
  case MachineCombinerPattern::FNMULSUBD_OP1: {
    // FNMUL I=A,B,0
    // FSUB R,I,C
    // ==> FNMADD R,A,B,C // = -A*B - C
    // --- Create(FNMADD);
    if (Pattern == MachineCombinerPattern::FNMULSUBS_OP1) {
      Opc = AArch64::FNMADDSrrr;
      RC = &AArch64::FPR32RegClass;
    } else {
      Opc = AArch64::FNMADDDrrr;
      RC = &AArch64::FPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC);
    break;
  }

  case MachineCombinerPattern::FMULSUBS_OP2:
  case MachineCombinerPattern::FMULSUBD_OP2: {
    // FMUL I=A,B,0
    // FSUB R,C,I
    // ==> FMSUB R,A,B,C (computes C - A*B)
    // --- Create(FMSUB);
    if (Pattern == MachineCombinerPattern::FMULSUBS_OP2) {
      Opc = AArch64::FMSUBSrrr;
      RC = &AArch64::FPR32RegClass;
    } else {
      Opc = AArch64::FMSUBDrrr;
      RC = &AArch64::FPR64RegClass;
    }
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC);
    break;
  }

  case MachineCombinerPattern::FMLSv1i32_indexed_OP2:
    Opc = AArch64::FMLSv1i32_indexed;
    RC = &AArch64::FPR32RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                           FMAInstKind::Indexed);
    break;

  case MachineCombinerPattern::FMLSv1i64_indexed_OP2:
    Opc = AArch64::FMLSv1i64_indexed;
    RC = &AArch64::FPR64RegClass;
    MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                           FMAInstKind::Indexed);
    break;

  case MachineCombinerPattern::FMLSv2f32_OP2:
  case MachineCombinerPattern::FMLSv2i32_indexed_OP2:
    RC = &AArch64::FPR64RegClass;
    if (Pattern == MachineCombinerPattern::FMLSv2i32_indexed_OP2) {
      Opc = AArch64::FMLSv2i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLSv2f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMLSv2f64_OP2:
  case MachineCombinerPattern::FMLSv2i64_indexed_OP2:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLSv2i64_indexed_OP2) {
      Opc = AArch64::FMLSv2i64_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLSv2f64;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;

  case MachineCombinerPattern::FMLSv4f32_OP2:
  case MachineCombinerPattern::FMLSv4i32_indexed_OP2:
    RC = &AArch64::FPR128RegClass;
    if (Pattern == MachineCombinerPattern::FMLSv4i32_indexed_OP2) {
      Opc = AArch64::FMLSv4i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Indexed);
    } else {
      Opc = AArch64::FMLSv4f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 2, Opc, RC,
                             FMAInstKind::Accumulator);
    }
    break;
  case MachineCombinerPattern::FMLSv2f32_OP1:
  case MachineCombinerPattern::FMLSv2i32_indexed_OP1: {
    RC = &AArch64::FPR64RegClass;
    unsigned NewVR = MRI.createVirtualRegister(RC);
    MachineInstrBuilder MIB1 =
        BuildMI(MF, Root.getDebugLoc(), TII->get(AArch64::FNEGv2f32), NewVR)
            .add(Root.getOperand(2));
    InsInstrs.push_back(MIB1);
    InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
    if (Pattern == MachineCombinerPattern::FMLSv2i32_indexed_OP1) {
      Opc = AArch64::FMLAv2i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed, &NewVR);
    } else {
      Opc = AArch64::FMLAv2f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator, &NewVR);
    }
    break;
  }
  case MachineCombinerPattern::FMLSv4f32_OP1:
  case MachineCombinerPattern::FMLSv4i32_indexed_OP1: {
    RC = &AArch64::FPR128RegClass;
    unsigned NewVR = MRI.createVirtualRegister(RC);
    MachineInstrBuilder MIB1 =
        BuildMI(MF, Root.getDebugLoc(), TII->get(AArch64::FNEGv4f32), NewVR)
            .add(Root.getOperand(2));
    InsInstrs.push_back(MIB1);
    InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
    if (Pattern == MachineCombinerPattern::FMLSv4i32_indexed_OP1) {
      Opc = AArch64::FMLAv4i32_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed, &NewVR);
    } else {
      Opc = AArch64::FMLAv4f32;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator, &NewVR);
    }
    break;
  }
  case MachineCombinerPattern::FMLSv2f64_OP1:
  case MachineCombinerPattern::FMLSv2i64_indexed_OP1: {
    RC = &AArch64::FPR128RegClass;
    unsigned NewVR = MRI.createVirtualRegister(RC);
    MachineInstrBuilder MIB1 =
        BuildMI(MF, Root.getDebugLoc(), TII->get(AArch64::FNEGv2f64), NewVR)
            .add(Root.getOperand(2));
    InsInstrs.push_back(MIB1);
    InstrIdxForVirtReg.insert(std::make_pair(NewVR, 0));
    if (Pattern == MachineCombinerPattern::FMLSv2i64_indexed_OP1) {
      Opc = AArch64::FMLAv2i64_indexed;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Indexed, &NewVR);
    } else {
      Opc = AArch64::FMLAv2f64;
      MUL = genFusedMultiply(MF, MRI, TII, Root, InsInstrs, 1, Opc, RC,
                             FMAInstKind::Accumulator, &NewVR);
    }
    break;
  }
  } // end switch (Pattern)
  // Record MUL and ADD/SUB for deletion
  DelInstrs.push_back(MUL);
  DelInstrs.push_back(&Root);
}

/// \brief Replace csincr-branch sequence by simple conditional branch
///
/// Examples:
/// 1. \code
///   csinc  w9, wzr, wzr, <condition code>
///   tbnz   w9, #0, 0x44
///    \endcode
/// to
///    \code
///   b.<inverted condition code>
///    \endcode
///
/// 2. \code
///   csinc w9, wzr, wzr, <condition code>
///   tbz   w9, #0, 0x44
///    \endcode
/// to
///    \code
///   b.<condition code>
///    \endcode
///
/// Replace compare and branch sequence by TBZ/TBNZ instruction when the
/// compare's constant operand is power of 2.
///
/// Examples:
///    \code
///   and  w8, w8, #0x400
///   cbnz w8, L1
///    \endcode
/// to
///    \code
///   tbnz w8, #10, L1
///    \endcode
///
/// \param  MI Conditional Branch
/// \return True when the simple conditional branch is generated
///
bool AArch64InstrInfo::optimizeCondBranch(MachineInstr &MI) const {
  bool IsNegativeBranch = false;
  bool IsTestAndBranch = false;
  unsigned TargetBBInMI = 0;
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unknown branch instruction?");
  case AArch64::Bcc:
    return false;
  case AArch64::CBZW:
  case AArch64::CBZX:
    TargetBBInMI = 1;
    break;
  case AArch64::CBNZW:
  case AArch64::CBNZX:
    TargetBBInMI = 1;
    IsNegativeBranch = true;
    break;
  case AArch64::TBZW:
  case AArch64::TBZX:
    TargetBBInMI = 2;
    IsTestAndBranch = true;
    break;
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    TargetBBInMI = 2;
    IsNegativeBranch = true;
    IsTestAndBranch = true;
    break;
  }
  // So we increment a zero register and test for bits other
  // than bit 0? Conservatively bail out in case the verifier
  // missed this case.
  if (IsTestAndBranch && MI.getOperand(1).getImm())
    return false;

  // Find Definition.
  assert(MI.getParent() && "Incomplete machine instruciton\n");
  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  MachineRegisterInfo *MRI = &MF->getRegInfo();
  unsigned VReg = MI.getOperand(0).getReg();
  if (!TargetRegisterInfo::isVirtualRegister(VReg))
    return false;

  MachineInstr *DefMI = MRI->getVRegDef(VReg);

  // Look through COPY instructions to find definition.
  while (DefMI->isCopy()) {
    unsigned CopyVReg = DefMI->getOperand(1).getReg();
    if (!MRI->hasOneNonDBGUse(CopyVReg))
      return false;
    if (!MRI->hasOneDef(CopyVReg))
      return false;
    DefMI = MRI->getVRegDef(CopyVReg);
  }

  switch (DefMI->getOpcode()) {
  default:
    return false;
  // Fold AND into a TBZ/TBNZ if constant operand is power of 2.
  case AArch64::ANDWri:
  case AArch64::ANDXri: {
    if (IsTestAndBranch)
      return false;
    if (DefMI->getParent() != MBB)
      return false;
    if (!MRI->hasOneNonDBGUse(VReg))
      return false;

    bool Is32Bit = (DefMI->getOpcode() == AArch64::ANDWri);
    uint64_t Mask = AArch64_AM::decodeLogicalImmediate(
        DefMI->getOperand(2).getImm(), Is32Bit ? 32 : 64);
    if (!isPowerOf2_64(Mask))
      return false;

    MachineOperand &MO = DefMI->getOperand(1);
    unsigned NewReg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(NewReg))
      return false;

    assert(!MRI->def_empty(NewReg) && "Register must be defined.");

    MachineBasicBlock &RefToMBB = *MBB;
    MachineBasicBlock *TBB = MI.getOperand(1).getMBB();
    DebugLoc DL = MI.getDebugLoc();
    unsigned Imm = Log2_64(Mask);
    unsigned Opc = (Imm < 32)
                       ? (IsNegativeBranch ? AArch64::TBNZW : AArch64::TBZW)
                       : (IsNegativeBranch ? AArch64::TBNZX : AArch64::TBZX);
    MachineInstr *NewMI = BuildMI(RefToMBB, MI, DL, get(Opc))
                              .addReg(NewReg)
                              .addImm(Imm)
                              .addMBB(TBB);
    // Register lives on to the CBZ now.
    MO.setIsKill(false);

    // For immediate smaller than 32, we need to use the 32-bit
    // variant (W) in all cases. Indeed the 64-bit variant does not
    // allow to encode them.
    // Therefore, if the input register is 64-bit, we need to take the
    // 32-bit sub-part.
    if (!Is32Bit && Imm < 32)
      NewMI->getOperand(0).setSubReg(AArch64::sub_32);
    MI.eraseFromParent();
    return true;
  }
  // Look for CSINC
  case AArch64::CSINCWr:
  case AArch64::CSINCXr: {
    if (!(DefMI->getOperand(1).getReg() == AArch64::WZR &&
          DefMI->getOperand(2).getReg() == AArch64::WZR) &&
        !(DefMI->getOperand(1).getReg() == AArch64::XZR &&
          DefMI->getOperand(2).getReg() == AArch64::XZR))
      return false;

    if (DefMI->findRegisterDefOperandIdx(AArch64::NZCV, true) != -1)
      return false;

    AArch64CC::CondCode CC = (AArch64CC::CondCode)DefMI->getOperand(3).getImm();
    // Convert only when the condition code is not modified between
    // the CSINC and the branch. The CC may be used by other
    // instructions in between.
    if (areCFlagsAccessedBetweenInstrs(DefMI, MI, &getRegisterInfo(), AK_Write))
      return false;
    MachineBasicBlock &RefToMBB = *MBB;
    MachineBasicBlock *TBB = MI.getOperand(TargetBBInMI).getMBB();
    DebugLoc DL = MI.getDebugLoc();
    if (IsNegativeBranch)
      CC = AArch64CC::getInvertedCondCode(CC);
    BuildMI(RefToMBB, MI, DL, get(AArch64::Bcc)).addImm(CC).addMBB(TBB);
    MI.eraseFromParent();
    return true;
  }
  }
}

std::pair<unsigned, unsigned>
AArch64InstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  const unsigned Mask = AArch64II::MO_FRAGMENT;
  return std::make_pair(TF & Mask, TF & ~Mask);
}

ArrayRef<std::pair<unsigned, const char *>>
AArch64InstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace AArch64II;

  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_PAGE, "aarch64-page"}, {MO_PAGEOFF, "aarch64-pageoff"},
      {MO_G3, "aarch64-g3"},     {MO_G2, "aarch64-g2"},
      {MO_G1, "aarch64-g1"},     {MO_G0, "aarch64-g0"},
      {MO_HI12, "aarch64-hi12"}};
  return makeArrayRef(TargetFlags);
}

ArrayRef<std::pair<unsigned, const char *>>
AArch64InstrInfo::getSerializableBitmaskMachineOperandTargetFlags() const {
  using namespace AArch64II;

  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_GOT, "aarch64-got"}, {MO_NC, "aarch64-nc"}, {MO_TLS, "aarch64-tls"}};
  return makeArrayRef(TargetFlags);
}

ArrayRef<std::pair<MachineMemOperand::Flags, const char *>>
AArch64InstrInfo::getSerializableMachineMemOperandTargetFlags() const {
  static const std::pair<MachineMemOperand::Flags, const char *> TargetFlags[] =
      {{MOSuppressPair, "aarch64-suppress-pair"},
       {MOStridedAccess, "aarch64-strided-access"}};
  return makeArrayRef(TargetFlags);
}

  /// Constants defining how certain sequences should be outlined.
  /// This encompasses how an outlined function should be called, and what kind of
  /// frame should be emitted for that outlined function.
  ///
  /// \p MachineOutlinerDefault implies that the function should be called with
  /// a save and restore of LR to the stack.
  ///
  /// That is,
  ///
  /// I1     Save LR                    OUTLINED_FUNCTION:
  /// I2 --> BL OUTLINED_FUNCTION       I1
  /// I3     Restore LR                 I2
  ///                                   I3
  ///                                   RET
  ///
  /// * Call construction overhead: 3 (save + BL + restore)
  /// * Frame construction overhead: 1 (ret)
  /// * Requires stack fixups? Yes
  ///
  /// \p MachineOutlinerTailCall implies that the function is being created from
  /// a sequence of instructions ending in a return.
  ///
  /// That is,
  ///
  /// I1                             OUTLINED_FUNCTION:
  /// I2 --> B OUTLINED_FUNCTION     I1
  /// RET                            I2
  ///                                RET
  ///
  /// * Call construction overhead: 1 (B)
  /// * Frame construction overhead: 0 (Return included in sequence)
  /// * Requires stack fixups? No
  ///
  /// \p MachineOutlinerNoLRSave implies that the function should be called using
  /// a BL instruction, but doesn't require LR to be saved and restored. This
  /// happens when LR is known to be dead.
  ///
  /// That is,
  ///
  /// I1                                OUTLINED_FUNCTION:
  /// I2 --> BL OUTLINED_FUNCTION       I1
  /// I3                                I2
  ///                                   I3
  ///                                   RET
  ///
  /// * Call construction overhead: 1 (BL)
  /// * Frame construction overhead: 1 (RET)
  /// * Requires stack fixups? No
  ///
enum MachineOutlinerClass {
  MachineOutlinerDefault,  /// Emit a save, restore, call, and return.
  MachineOutlinerTailCall, /// Only emit a branch.
  MachineOutlinerNoLRSave  /// Emit a call and return.
};

bool AArch64InstrInfo::canOutlineWithoutLRSave(
    MachineBasicBlock::iterator &CallInsertionPt) const {
  // Was LR saved in the function containing this basic block?
  MachineBasicBlock &MBB = *(CallInsertionPt->getParent());
  LiveRegUnits LRU(getRegisterInfo());
  LRU.addLiveOuts(MBB);

  // Get liveness information from the end of the block to the end of the
  // prospective outlined region.
  std::for_each(MBB.rbegin(),
                (MachineBasicBlock::reverse_iterator)CallInsertionPt,
                [&LRU](MachineInstr &MI) { LRU.stepBackward(MI); });

  // If the link register is available at this point, then we can safely outline
  // the region without saving/restoring LR. Otherwise, we must emit a save and
  // restore.
  return LRU.available(AArch64::LR);
}

AArch64GenInstrInfo::MachineOutlinerInfo
AArch64InstrInfo::getOutlininingCandidateInfo(
    std::vector<
        std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>>
        &RepeatedSequenceLocs) const {

  unsigned CallID = MachineOutlinerDefault;
  unsigned FrameID = MachineOutlinerDefault;
  unsigned NumInstrsForCall = 3;
  unsigned NumInstrsToCreateFrame = 1;

  auto DoesntNeedLRSave =
      [this](std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>
                 &I) { return canOutlineWithoutLRSave(I.second); };

  // If the last instruction in any candidate is a terminator, then we should
  // tail call all of the candidates.
  if (RepeatedSequenceLocs[0].second->isTerminator()) {
    CallID = MachineOutlinerTailCall;
    FrameID = MachineOutlinerTailCall;
    NumInstrsForCall = 1;
    NumInstrsToCreateFrame = 0;
  }

  else if (std::all_of(RepeatedSequenceLocs.begin(), RepeatedSequenceLocs.end(),
                       DoesntNeedLRSave)) {
    CallID = MachineOutlinerNoLRSave;
    FrameID = MachineOutlinerNoLRSave;
    NumInstrsForCall = 1;
    NumInstrsToCreateFrame = 1;
  }

  // Check if the range contains a call. These require a save + restore of the
  // link register.
  if (std::any_of(RepeatedSequenceLocs[0].first, RepeatedSequenceLocs[0].second,
                  [](const MachineInstr &MI) { return MI.isCall(); }))
    NumInstrsToCreateFrame += 2; // Save + restore the link register.

  // Handle the last instruction separately. If this is a tail call, then the
  // last instruction is a call. We don't want to save + restore in this case.
  // However, it could be possible that the last instruction is a call without
  // it being valid to tail call this sequence. We should consider this as well.
  else if (RepeatedSequenceLocs[0].second->isCall() &&
           FrameID != MachineOutlinerTailCall)
    NumInstrsToCreateFrame += 2;

  return MachineOutlinerInfo(NumInstrsForCall, NumInstrsToCreateFrame, CallID,
                             FrameID);
}

bool AArch64InstrInfo::isFunctionSafeToOutlineFrom(
    MachineFunction &MF, bool OutlineFromLinkOnceODRs) const {
  const Function &F = MF.getFunction();

  // If F uses a redzone, then don't outline from it because it might mess up
  // the stack.
  if (!F.hasFnAttribute(Attribute::NoRedZone))
    return false;

  // If anyone is using the address of this function, don't outline from it.
  if (F.hasAddressTaken())
    return false;

  // Can F be deduplicated by the linker? If it can, don't outline from it.
  if (!OutlineFromLinkOnceODRs && F.hasLinkOnceODRLinkage())
    return false;

  return true;
}

AArch64GenInstrInfo::MachineOutlinerInstrType
AArch64InstrInfo::getOutliningType(MachineInstr &MI) const {

  MachineFunction *MF = MI.getParent()->getParent();
  AArch64FunctionInfo *FuncInfo = MF->getInfo<AArch64FunctionInfo>();

  // Don't outline LOHs.
  if (FuncInfo->getLOHRelated().count(&MI))
    return MachineOutlinerInstrType::Illegal;

  // Don't allow debug values to impact outlining type.
  if (MI.isDebugValue() || MI.isIndirectDebugValue())
    return MachineOutlinerInstrType::Invisible;

  // Is this a terminator for a basic block?
  if (MI.isTerminator()) {

    // Is this the end of a function?
    if (MI.getParent()->succ_empty())
      return MachineOutlinerInstrType::Legal;

    // It's not, so don't outline it.
    return MachineOutlinerInstrType::Illegal;
  }

  // Outline calls without stack parameters or aggregate parameters.
  if (MI.isCall()) {
    const Module *M = MF->getFunction().getParent();
    assert(M && "No module?");

    // Get the function associated with the call. Look at each operand and find
    // the one that represents the callee and get its name.
    Function *Callee = nullptr;
    for (const MachineOperand &MOP : MI.operands()) {
      if (MOP.isSymbol()) {
        Callee = M->getFunction(MOP.getSymbolName());
        break;
      }

      else if (MOP.isGlobal()) {
        Callee = M->getFunction(MOP.getGlobal()->getGlobalIdentifier());
        break;
      }
    }

    // Only handle functions that we have information about.
    if (!Callee)
      return MachineOutlinerInstrType::Illegal;

    // We have a function we have information about. Check it if it's something
    // can safely outline.

    // If the callee is vararg, it passes parameters on the stack. Don't touch
    // it.
    // FIXME: Functions like printf are very common and we should be able to
    // outline them.
    if (Callee->isVarArg())
      return MachineOutlinerInstrType::Illegal;

    // Check if any of the arguments are a pointer to a struct. We don't want
    // to outline these since they might be loaded in two instructions.
    for (Argument &Arg : Callee->args()) {
      if (Arg.getType()->isPointerTy() &&
          Arg.getType()->getPointerElementType()->isAggregateType())
        return MachineOutlinerInstrType::Illegal;
    }

    // If the thing we're calling doesn't access memory at all, then we're good
    // to go.
    if (Callee->doesNotAccessMemory())
      return MachineOutlinerInstrType::Legal;

    // It accesses memory. Get the machine function for the callee to see if
    // it's safe to outline.
    MachineFunction *CalleeMF = MF->getMMI().getMachineFunction(*Callee);

    // We don't know what's going on with the callee at all. Don't touch it.
    if (!CalleeMF)
      return MachineOutlinerInstrType::Illegal;

    // Does it pass anything on the stack? If it does, don't outline it.
    if (CalleeMF->getInfo<AArch64FunctionInfo>()->getBytesInStackArgArea() != 0)
      return MachineOutlinerInstrType::Illegal;

    // It doesn't, so it's safe to outline and we're done.
    return MachineOutlinerInstrType::Legal;
  }

  // Don't outline positions.
  if (MI.isPosition())
    return MachineOutlinerInstrType::Illegal;

  // Don't touch the link register or W30.
  if (MI.readsRegister(AArch64::W30, &getRegisterInfo()) ||
      MI.modifiesRegister(AArch64::W30, &getRegisterInfo()))
    return MachineOutlinerInstrType::Illegal;

  // Make sure none of the operands are un-outlinable.
  for (const MachineOperand &MOP : MI.operands()) {
    if (MOP.isCPI() || MOP.isJTI() || MOP.isCFIIndex() || MOP.isFI() ||
        MOP.isTargetIndex())
      return MachineOutlinerInstrType::Illegal;

    // Don't outline anything that uses the link register.
    if (MOP.isReg() && getRegisterInfo().regsOverlap(MOP.getReg(), AArch64::LR))
      return MachineOutlinerInstrType::Illegal;
  }

  // Does this use the stack?
  if (MI.modifiesRegister(AArch64::SP, &RI) ||
      MI.readsRegister(AArch64::SP, &RI)) {

    if (MI.mayLoadOrStore()) {
      unsigned Base;  // Filled with the base regiser of MI.
      int64_t Offset; // Filled with the offset of MI.
      unsigned DummyWidth;

      // Does it allow us to offset the base register and is the base SP?
      if (!getMemOpBaseRegImmOfsWidth(MI, Base, Offset, DummyWidth, &RI) ||
          Base != AArch64::SP)
        return MachineOutlinerInstrType::Illegal;

      // Find the minimum/maximum offset for this instruction and check if
      // fixing it up would be in range.
      int64_t MinOffset, MaxOffset; // Unscaled offsets for the instruction.
      unsigned Scale;               // The scale to multiply the offsets by.
      getMemOpInfo(MI.getOpcode(), Scale, DummyWidth, MinOffset, MaxOffset);

      // TODO: We should really test what happens if an instruction overflows.
      // This is tricky to test with IR tests, but when the outliner is moved
      // to a MIR test, it really ought to be checked.
      Offset += 16; // Update the offset to what it would be if we outlined.
      if (Offset < MinOffset * Scale || Offset > MaxOffset * Scale)
        return MachineOutlinerInstrType::Illegal;

      // It's in range, so we can outline it.
      return MachineOutlinerInstrType::Legal;
    }

    // We can't fix it up, so don't outline it.
    return MachineOutlinerInstrType::Illegal;
  }

  return MachineOutlinerInstrType::Legal;
}

void AArch64InstrInfo::fixupPostOutline(MachineBasicBlock &MBB) const {
  for (MachineInstr &MI : MBB) {
    unsigned Base, Width;
    int64_t Offset;

    // Is this a load or store with an immediate offset with SP as the base?
    if (!MI.mayLoadOrStore() ||
        !getMemOpBaseRegImmOfsWidth(MI, Base, Offset, Width, &RI) ||
        Base != AArch64::SP)
      continue;

    // It is, so we have to fix it up.
    unsigned Scale;
    int64_t Dummy1, Dummy2;

    MachineOperand &StackOffsetOperand = getMemOpBaseRegImmOfsOffsetOperand(MI);
    assert(StackOffsetOperand.isImm() && "Stack offset wasn't immediate!");
    getMemOpInfo(MI.getOpcode(), Scale, Width, Dummy1, Dummy2);
    assert(Scale != 0 && "Unexpected opcode!");

    // We've pushed the return address to the stack, so add 16 to the offset.
    // This is safe, since we already checked if it would overflow when we
    // checked if this instruction was legal to outline.
    int64_t NewImm = (Offset + 16) / Scale;
    StackOffsetOperand.setImm(NewImm);
  }
}

void AArch64InstrInfo::insertOutlinerEpilogue(
    MachineBasicBlock &MBB, MachineFunction &MF,
    const MachineOutlinerInfo &MInfo) const {

  // Is there a call in the outlined range?
  if (std::any_of(MBB.instr_begin(), MBB.instr_end(),
                  [](MachineInstr &MI) { return MI.isCall(); })) {
    // Fix up the instructions in the range, since we're going to modify the
    // stack.
    fixupPostOutline(MBB);

    // LR has to be a live in so that we can save it.
    MBB.addLiveIn(AArch64::LR);

    MachineBasicBlock::iterator It = MBB.begin();
    MachineBasicBlock::iterator Et = MBB.end();

    if (MInfo.FrameConstructionID == MachineOutlinerTailCall)
      Et = std::prev(MBB.end());

    // Insert a save before the outlined region
    MachineInstr *STRXpre = BuildMI(MF, DebugLoc(), get(AArch64::STRXpre))
                                .addReg(AArch64::SP, RegState::Define)
                                .addReg(AArch64::LR)
                                .addReg(AArch64::SP)
                                .addImm(-16);
    It = MBB.insert(It, STRXpre);

    // Insert a restore before the terminator for the function.
    MachineInstr *LDRXpost = BuildMI(MF, DebugLoc(), get(AArch64::LDRXpost))
                                 .addReg(AArch64::SP, RegState::Define)
                                 .addReg(AArch64::LR, RegState::Define)
                                 .addReg(AArch64::SP)
                                 .addImm(16);
    Et = MBB.insert(Et, LDRXpost);
  }

  // If this is a tail call outlined function, then there's already a return.
  if (MInfo.FrameConstructionID == MachineOutlinerTailCall)
    return;

  // It's not a tail call, so we have to insert the return ourselves.
  MachineInstr *ret = BuildMI(MF, DebugLoc(), get(AArch64::RET))
                          .addReg(AArch64::LR, RegState::Undef);
  MBB.insert(MBB.end(), ret);

  // Did we have to modify the stack by saving the link register?
  if (MInfo.FrameConstructionID == MachineOutlinerNoLRSave)
    return;

  // We modified the stack.
  // Walk over the basic block and fix up all the stack accesses.
  fixupPostOutline(MBB);
}

void AArch64InstrInfo::insertOutlinerPrologue(
    MachineBasicBlock &MBB, MachineFunction &MF,
    const MachineOutlinerInfo &MInfo) const {}

MachineBasicBlock::iterator AArch64InstrInfo::insertOutlinedCall(
    Module &M, MachineBasicBlock &MBB, MachineBasicBlock::iterator &It,
    MachineFunction &MF, const MachineOutlinerInfo &MInfo) const {

  // Are we tail calling?
  if (MInfo.CallConstructionID == MachineOutlinerTailCall) {
    // If yes, then we can just branch to the label.
    It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(AArch64::B))
                            .addGlobalAddress(M.getNamedValue(MF.getName())));
    return It;
  }

  // Are we saving the link register?
  if (MInfo.CallConstructionID == MachineOutlinerNoLRSave) {
    // No, so just insert the call.
    It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(AArch64::BL))
                            .addGlobalAddress(M.getNamedValue(MF.getName())));
    return It;
  }

  // We have a default call. Save the link register.
  MachineInstr *STRXpre = BuildMI(MF, DebugLoc(), get(AArch64::STRXpre))
                              .addReg(AArch64::SP, RegState::Define)
                              .addReg(AArch64::LR)
                              .addReg(AArch64::SP)
                              .addImm(-16);
  It = MBB.insert(It, STRXpre);
  It++;

  // Insert the call.
  It = MBB.insert(It, BuildMI(MF, DebugLoc(), get(AArch64::BL))
                          .addGlobalAddress(M.getNamedValue(MF.getName())));

  It++;

  // Restore the link register.
  MachineInstr *LDRXpost = BuildMI(MF, DebugLoc(), get(AArch64::LDRXpost))
                               .addReg(AArch64::SP, RegState::Define)
                               .addReg(AArch64::LR, RegState::Define)
                               .addReg(AArch64::SP)
                               .addImm(16);
  It = MBB.insert(It, LDRXpost);

  return It;
}