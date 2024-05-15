#include "InArchInstrInfo.h"
#include "InArch.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "InArchGenInstrInfo.inc"

#define DEBUG_TYPE "InArch-inst-info"

InArchInstrInfo::InArchInstrInfo()
    : InArchGenInstrInfo(InArch::ADJCALLSTACKDOWN, InArch::ADJCALLSTACKUP) {
}

unsigned InArchInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
    // TODO: load opcodes
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }
  return 0;
}

unsigned InArchInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  llvm_unreachable("");
  return 0;
}

static InArchCC::CondCode getCondFromBranchOpc(unsigned Opc) {
  switch (Opc) {
  default:
    return InArchCC::INVALID;
  case InArch::BEQ:
    return InArchCC::EQ;
  case InArch::BNE:
    return InArchCC::NE;
  case InArch::BLE:
    return InArchCC::LE;
  case InArch::BGT:
    return InArchCC::GT;
  case InArch::BLEU:
    return InArchCC::LEU;
  case InArch::BGTU:
    return InArchCC::GTU;
  }
}

static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  unsigned CC = getCondFromBranchOpc(LastInst.getOpcode());
  Cond.push_back(MachineOperand::CreateImm(CC));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

const MCInstrDesc &InArchInstrInfo::getBrCond(InArchCC::CondCode CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case InArchCC::EQ:
    return get(InArch::BEQ);
  case InArchCC::NE:
    return get(InArch::BNE);
  case InArchCC::LE:
    return get(InArch::BLE);
  case InArchCC::GT:
    return get(InArch::BGT);
  case InArchCC::LEU:
    return get(InArch::BLEU);
  case InArchCC::GTU:
    return get(InArch::BGTU);
  }
}

InArchCC::CondCode InArchCC::getOppositeBranchCondition(InArchCC::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case InArchCC::EQ:
    return InArchCC::NE;
  case InArchCC::NE:
    return InArchCC::EQ;
  case InArchCC::LE:
    return InArchCC::GT;
  case InArchCC::GT:
    return InArchCC::LE;
  case InArchCC::LEU:
    return InArchCC::GTU;
  case InArchCC::GTU:
    return InArchCC::LEU;
  }
}

// TODO: inherited from riscv
bool InArchInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // We can't handle blocks that end in an indirect branch.
  if (I->getDesc().isIndirectBranch())
    return true;

  // We can't handle blocks with more than 2 terminators.
  if (NumTerminators > 2)
    return true;

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = getBranchDestBlock(*I);
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = getBranchDestBlock(*I);
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

// TODO: explore
unsigned InArchInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                     int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isUnconditionalBranch() &&
      !I->getDesc().isConditionalBranch())
    return 0;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 2;
}

MachineBasicBlock *
InArchInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  return MI.getOperand(MI.getNumExplicitOperands() - 1).getMBB();
}

void InArchInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                const DebugLoc &DL, MCRegister DstReg,
                                MCRegister SrcReg, bool KillSrc) const {
  if (InArch::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(InArch::ADDI), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
    return;
  }
  llvm_unreachable("can't copyPhysReg");
}

void InArchInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        Register SrcReg, bool IsKill, int FI,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI, Register VReg) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(InArch::STI_))
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

void InArchInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         Register DstReg, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI, Register VReg) const {
  DebugLoc DL;
  if (I != MBB.end())
    DL = I->getDebugLoc();

  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  BuildMI(MBB, I, DL, get(InArch::LDI), DstReg)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

bool InArchInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() == 3) && "Invalid branch condition!");
  auto CC = static_cast<InArchCC::CondCode>(Cond[0].getImm());
  Cond[0].setImm(getOppositeBranchCondition(CC));
  return false;
}

// TODO: explore
unsigned InArchInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "Wrong number of components");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(InArch::B)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  auto CC = static_cast<InArchCC::CondCode>(Cond[0].getImm());
  MachineInstr &CondMI =
      *BuildMI(&MBB, DL, getBrCond(CC)).add(Cond[1]).add(Cond[2]).addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(InArch::B)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

unsigned InArchInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  llvm_unreachable("");
}

bool InArchInstrInfo::getBaseAndOffsetPosition(const MachineInstr &MI,
                                             unsigned &BasePos,
                                             unsigned &OffsetPos) const {
  llvm_unreachable("");
}