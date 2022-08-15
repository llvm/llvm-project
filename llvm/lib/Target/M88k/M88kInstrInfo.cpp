//===-- M88kInstrInfo.cpp - M88k instruction information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the M88k implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "M88kInstrInfo.h"
#include "M88k.h"
#include "MCTargetDesc/M88kBaseInfo.h"
#include "MCTargetDesc/M88kMCTargetDesc.h"
#include "M88kSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdint>
#include <iterator>

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRMAP_INFO
#include "M88kGenInstrInfo.inc"

#define DEBUG_TYPE "m88k-ii"

// Pin the vtable to this file.
void M88kInstrInfo::anchor() {}

M88kInstrInfo::M88kInstrInfo(const M88kSubtarget &STI)
    : M88kGenInstrInfo(M88k::ADJCALLSTACKDOWN, M88k::ADJCALLSTACKUP), STI(STI), RI() {}

std::pair<unsigned, unsigned>
M88kInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  return std::make_pair(TF, 0u);
}

ArrayRef<std::pair<unsigned, const char *>>
M88kInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace M88kII;

  static const std::pair<unsigned, const char *> Flags[] = {
      {MO_ABS_HI, "m88k-abs-hi"},
      {MO_ABS_LO, "m88k-abs-lo"},
  };
  return makeArrayRef(Flags);
}

static MachineMemOperand *getMachineMemOperand(MachineBasicBlock &MBB, int FI,
                                               MachineMemOperand::Flags Flags) {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 Flags, MFI.getObjectSize(FI),
                                 MFI.getObjectAlign(FI));
}

static bool isAnalyzableBranchOpc(unsigned BranchOpc) {
  return BranchOpc == M88k::BR || BranchOpc == M88k::BSR ||
         BranchOpc == M88k::BCND || BranchOpc == M88k::BB0 ||
         BranchOpc == M88k::BB1;
}

bool M88kInstrInfo::isBranchOffsetInRange(unsigned BranchOpc,
                                          int64_t BrOffset) const {
  assert(isAnalyzableBranchOpc(BranchOpc) && "Unexpected branch opcode");
  int Bits = (BranchOpc == M88k::BR || BranchOpc == M88k::BSR) ? 26 : 16;
  return isIntN(Bits, BrOffset / 4);
}

MachineBasicBlock *
M88kInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  int NumOp = MI.getNumExplicitOperands();
  return MI.getOperand(NumOp - 1).getMBB();
}

void M88kInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                         MachineBasicBlock &NewDestBB,
                                         MachineBasicBlock &RestoreBB,
                                         const DebugLoc &DL, int64_t BrOffset,
                                         RegScavenger *RS) const {
  assert(RS && "RegScavenger required for long branching");
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  // FIXME: A virtual register must be used initially, as the register
  // scavenger won't work with empty blocks (SIInstrInfo::insertIndirectBranch
  // uses the same workaround).
  Register ScratchReg = MRI.createVirtualRegister(&M88k::GPRRCRegClass);
  auto I = MBB.end();

  // Load address of destination BB.
  BuildMI(MBB, I, DL, get(M88k::ORriu))
      .addReg(ScratchReg, RegState::Define | RegState::Dead)
      .addReg(M88k::R0)
      .addMBB(&NewDestBB, M88kII::MO_ABS_HI);
  BuildMI(MBB, I, DL, get(M88k::ORri))
      .addReg(ScratchReg)
      .addReg(ScratchReg)
      .addMBB(&NewDestBB, M88kII::MO_ABS_LO);

  MachineInstr *MI = BuildMI(MBB, I, DL, get(M88k::JMP)).addReg(ScratchReg);

  RS->enterBasicBlockEnd(MBB);
  unsigned Scav = RS->scavengeRegisterBackwards(M88k::GPRRCRegClass,
                                                MI->getIterator(), false, 0);

  // TODO: The case when there is no scavenged register needs special handling.
  assert(Scav != M88k::NoRegister && "No register is scavenged!");
  MRI.replaceRegWith(ScratchReg, Scav);
  MRI.clearVirtRegs();
  RS->setRegUsed(Scav);
}

// The contents of values added to Cond are not examined outside of
// M88kInstrInfo, giving us flexibility in what to push to it. For M88k, we push
// BranchOpcode, CC, Reg.
static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  Cond.push_back(MachineOperand::CreateImm(LastInst.getOpcode()));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

bool M88kInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
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

  // We can't handle all branch opcodes.
  if (!isAnalyzableBranchOpc(I->getOpcode()) ||
      (NumTerminators == 2 &&
       !isAnalyzableBranchOpc(std::prev(I)->getOpcode())))
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

unsigned M88kInstrInfo::removeBranch(MachineBasicBlock &MBB,
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

unsigned M88kInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "wrong number of m88k branch conditions");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(M88k::BR)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  unsigned BrOpc = Cond[0].getImm();
  unsigned CC = Cond[1].getImm();
  MachineInstr &CondMI =
      *BuildMI(&MBB, DL, get(BrOpc)).addImm(CC).add(Cond[2]).addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(M88k::BR)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

bool M88kInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 3 && "wrong number of m88k branch conditions");

  switch (Cond[0].getImm()) {
  case M88k::BCND:
    // Invert bits to get reverse condition.
    Cond[1].setImm(~Cond[1].getImm() & 0x0f);
    break;
#if 0
  case M88k::BB1:
  case M88k::BB0: {
    // TODO This only works if the value was produced by cmp/fcmp.
    // Inverse condition is:
    // - even bit number: next bit numner
    // - odd bit number: previous bit numner
    unsigned CC = Cond[1].getImm();
    CC = (CC & ~1) | (1 - (CC & 0x01));
    Cond[1].setImm(CC);
    break;
  }
#else
  case M88k::BB1:
  case M88k::BB0:
    // The save way is to delare that the condition cannot be reversed.
    return true;
#endif
  default:
    return true;
  }
  return false;
}

unsigned M88kInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  switch (MI.getOpcode()) {
  // TODO Check which LD instructions are really selected.
  case M88k::LDrib:
  case M88k::LDrih:
  case M88k::LDriw:
  case M88k::LDrid:
  case M88k::LDrrsb:
  case M88k::LDrrsbu:
  case M88k::LDrrsd:
  case M88k::LDrrsdu:
  case M88k::LDrrsh:
  case M88k::LDrrshu:
  case M88k::LDrrsw:
  case M88k::LDrrswu:
  case M88k::LDrrub:
  case M88k::LDrrubu:
  case M88k::LDrrud:
  case M88k::LDrrudu:
  case M88k::LDrruh:
  case M88k::LDrruhu:
  case M88k::LDrruw:
  case M88k::LDrruwu:
    if (MI.getOperand(1).isFI()) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned M88kInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  switch (MI.getOpcode()) {
  // TODO Check which ST instructions are really selected.
  case M88k::STrib:
  case M88k::STrih:
  case M88k::STriw:
  case M88k::STrid:
  case M88k::STrrsb:
  case M88k::STrrsbt:
  case M88k::STrrsbu:
  case M88k::STrrsbut:
  case M88k::STrrsd:
  case M88k::STrrsdt:
  case M88k::STrrsdu:
  case M88k::STrrsdut:
  case M88k::STrrsh:
  case M88k::STrrsht:
  case M88k::STrrshu:
  case M88k::STrrshut:
  case M88k::STrrsw:
  case M88k::STrrswt:
  case M88k::STrrswu:
  case M88k::STrrswut:
  case M88k::STrrub:
  case M88k::STrrubt:
  case M88k::STrrubu:
  case M88k::STrrubut:
  case M88k::STrrud:
  case M88k::STrrudt:
  case M88k::STrrudu:
  case M88k::STrrudut:
  case M88k::STrruh:
  case M88k::STrruht:
  case M88k::STrruhu:
  case M88k::STrruhut:
  case M88k::STrruw:
  case M88k::STrruwt:
  case M88k::STrruwu:
  case M88k::STrruwut:
  case M88k::STxid:
  case M88k::STxis:
  case M88k::STxix:
  case M88k::STxrd:
  case M88k::STxrdt:
  case M88k::STxrdu:
  case M88k::STxrdut:
  case M88k::STxrss:
  case M88k::STxrsst:
  case M88k::STxrssu:
  case M88k::STxrssut:
  case M88k::STxrsx:
  case M88k::STxrsxt:
  case M88k::STxrsxu:
  case M88k::STxrsxut:
  case M88k::STxrud:
  case M88k::STxrudt:
  case M88k::STxrudu:
  case M88k::STxrudut:
  case M88k::STxrus:
  case M88k::STxrust:
  case M88k::STxrusu:
  case M88k::STxrusut:
  case M88k::STxrux:
  case M88k::STxruxt:
  case M88k::STxruxu:
  case M88k::STxruxut:
    if (MI.getOperand(1).isFI()) {
      FrameIndex = MI.getOperand(1).getIndex();
      return MI.getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

void M88kInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        Register SrcReg, bool isKill,
                                        int FrameIndex,
                                        const TargetRegisterClass *RC,
                                        const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  MachineMemOperand *MMO =
      getMachineMemOperand(MBB, FrameIndex, MachineMemOperand::MOStore);

  // Build an STriw instruction.
  BuildMI(MBB, MBBI, DL, get(M88k::STriw))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO);
}

void M88kInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         Register DestReg, int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  MachineMemOperand *MMO =
      getMachineMemOperand(MBB, FrameIndex, MachineMemOperand::MOLoad);

  // Build an LDriw instruction.
  BuildMI(MBB, MBBI, DL, get(M88k::LDriw))
      .addReg(DestReg, RegState::Define)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO);
}

unsigned M88kInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isInlineAsm()) {
    const MachineFunction *MF = MI.getParent()->getParent();
    const char *AsmStr = MI.getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  return MI.getDesc().getSize();
}

void M88kInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                const DebugLoc &DL, MCRegister DestReg,
                                MCRegister SrcReg, bool KillSrc) const {
  // Split 64-bit GPR moves into two 64-bit moves. Add implicit uses of the
  // super register in case one of the subregs is undefined.
  if (M88k::GPR64RCRegClass.contains(DestReg, SrcReg)) {
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, M88k::sub_hi),
                RI.getSubReg(SrcReg, M88k::sub_hi), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
        .addReg(SrcReg, RegState::Implicit);
    copyPhysReg(MBB, MBBI, DL, RI.getSubReg(DestReg, M88k::sub_lo),
                RI.getSubReg(SrcReg, M88k::sub_lo), KillSrc);
    MachineInstrBuilder(*MBB.getParent(), std::prev(MBBI))
        .addReg(SrcReg, (getKillRegState(KillSrc) | RegState::Implicit));
    return;
  }

  if (M88k::GPRRCRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(M88k::ORrr), DestReg)
        .addReg(M88k::R0)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  unsigned Opc;
  if (M88k::XRRCRegClass.contains(DestReg, SrcReg))
    Opc = M88k::MOVxx;
  else if (M88k::GPRRCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxs;
  else if (M88k::GPR64RCRegClass.contains(DestReg) &&
           M88k::XRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVrxd;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::GPRRCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrs;
  else if (M88k::XRRCRegClass.contains(DestReg) &&
           M88k::GPR64RCRegClass.contains(SrcReg))
    Opc = M88k::MOVxrd;
  else
    llvm_unreachable("m88: Impossible reg-to-reg copy");

  BuildMI(MBB, MBBI, DL, get(Opc), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}

Optional<DestSourcePair>
M88kInstrInfo::isCopyInstrImpl(const MachineInstr &MI) const {
  if (MI.isMoveReg() ||
      (MI.getOpcode() == M88k::ORri && MI.getOperand(2).getImm() == 0))
    return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};

  if (MI.getOpcode() == M88k::ORrr) {
    if (MI.getOperand(2).getReg() == M88k::R0)
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    if (MI.getOperand(1).getReg() == M88k::R0)
      return DestSourcePair{MI.getOperand(0), MI.getOperand(2)};
  }
  return None;
}

void M88kInstrInfo::insertNoop(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(M88k::ORrr), M88k::R0)
      .addReg(M88k::R0)
      .addReg(M88k::R0);
}

MCInst M88kInstrInfo::getNop() const {
  return MCInstBuilder(M88k::ORrr)
      .addReg(M88k::R0)
      .addReg(M88k::R0)
      .addReg(M88k::R0);
}

bool M88kInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  // This isn't needed since LOAD_STACK_GUARD is not used.
  case TargetOpcode::LOAD_STACK_GUARD: {
    MachineBasicBlock &MBB = *MI.getParent();
    const Register Reg = MI.getOperand(0).getReg();
    auto MMO = *MI.memoperands_begin();
    const GlobalValue *GV = cast<GlobalValue>(MMO->getValue());

    // Load stack guard value.
    BuildMI(MBB, &MI, MI.getDebugLoc(), get(M88k::ORriu))
        .addReg(Reg, RegState::Define | RegState::Dead)
        .addReg(M88k::R0)
        .addGlobalAddress(GV, 0, M88kII::MO_ABS_HI);
    BuildMI(MBB, &MI, MI.getDebugLoc(), get(M88k::LDriw))
        .addReg(Reg, RegState::Define)
        .addReg(Reg, RegState::Kill)
        .addGlobalAddress(GV, 0, M88kII::MO_ABS_LO)
        .addMemOperand(MMO);

    // Erase the LOAD_STACK_GUARD instruction.
    MBB.erase(MI);
    return true;
  }
  default:
    return false;
  }
}

bool M88kInstrInfo::isReallyTriviallyReMaterializable(
    const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    // This function should only be called for opcodes with the ReMaterializable
    // flag set.
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    MI.dump();
#endif
    llvm_unreachable("Unknown rematerializable operation!");
    break;
  case M88k::ADDri:
  case M88k::ADDUri:
  case M88k::SUBri:
  case M88k::SUBUri:
  case M88k::ANDri:
  case M88k::ANDriu:
  case M88k::ORri:
  case M88k::ORriu:
  case M88k::XORri:
  case M88k::XORriu:
  case M88k::MASKri:
  case M88k::MASKriu:
    return MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == M88k::R0;
  }
  return false;
}

void M88kInstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  Register DestReg, unsigned SubIdx,
                                  const MachineInstr &Orig,
                                  const TargetRegisterInfo &TRI) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(&Orig);
  MBB.insert(I, MI);
  MI->substituteRegister(Orig.getOperand(0).getReg(), DestReg, SubIdx, TRI);
}
