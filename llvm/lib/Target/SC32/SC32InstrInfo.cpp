#include "SC32InstrInfo.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "SC32GenInstrInfo.inc"

void SC32InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI,
                                const DebugLoc &DL, Register DestReg,
                                Register SrcReg, bool KillSrc,
                                bool RenamableDest, bool RenamableSrc) const {
  BuildMI(MBB, MI, DL, get(SC32::MOV), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

static bool isUnconditionalBranchOpcode(unsigned Opcode) {
  return Opcode == SC32::JMP;
}

static bool isConditionalBranchOpcode(unsigned Opcode) {
  switch (Opcode) {
  case SC32::JEQ:
  case SC32::JNE:
  case SC32::JLE:
  case SC32::JLT:
  case SC32::JGT:
  case SC32::JGE:
  case SC32::JLEU:
  case SC32::JLTU:
  case SC32::JGTU:
  case SC32::JGEU:
    return true;
  default:
    return false;
  }
}

bool SC32InstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();

  if (I == MBB.end()) {
    return false;
  }

  if (isUnconditionalBranchOpcode(I->getOpcode())) {
    TBB = I->getOperand(0).getMBB();

    if (I == MBB.begin()) {
      return false;
    }

    --I;
  }

  if (isConditionalBranchOpcode(I->getOpcode())) {
    FBB = TBB;
    TBB = I->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(I->getOpcode()));
  }

  return false;
}

unsigned SC32InstrInfo::removeBranch(MachineBasicBlock &MBB,
                                     int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;
  int Removed = 0;

  while (I != MBB.begin()) {
    --I;

    unsigned Opcode = I->getOpcode();

    if (isUnconditionalBranchOpcode(Opcode) ||
        isConditionalBranchOpcode(Opcode)) {
      Removed += getInstSizeInBytes(*I);
      I->eraseFromParent();
      I = MBB.end();
      ++Count;
    } else if (!I->isDebugInstr()) {
      break;
    }
  }

  if (BytesRemoved) {
    *BytesRemoved = Removed;
  }

  return Count;
}

unsigned SC32InstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  unsigned Count = 0;
  int Added = 0;

  if (!Cond.empty()) {
    unsigned Opcode = Cond[0].getImm();
    MachineInstr *I = BuildMI(&MBB, DL, get(Opcode)).addMBB(TBB);
    Added += getInstSizeInBytes(*I);
    ++Count;
    TBB = FBB;
  }

  if (TBB) {
    MachineInstr *I = BuildMI(&MBB, DL, get(SC32::JMP)).addMBB(TBB);
    Added += getInstSizeInBytes(*I);
    ++Count;
  }

  if (BytesAdded) {
    *BytesAdded = Added;
  }

  return Count;
}

void SC32InstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool IsKill, int FrameIndex, const TargetRegisterClass *RC, Register VReg,
    MachineInstr::MIFlag Flags) const {
  DebugLoc DL;

  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }

  BuildMI(MBB, MI, DL, get(SC32::ST))
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addImm(0);
}
void SC32InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         Register DestReg, int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         Register VReg, unsigned SubReg,
                                         MachineInstr::MIFlag Flags) const {
  DebugLoc DL;

  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }

  BuildMI(MBB, MI, DL, get(SC32::LD), DestReg)
      .addFrameIndex(FrameIndex)
      .addImm(0);
}
