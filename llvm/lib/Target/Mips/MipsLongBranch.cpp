//===-- MipsLongBranch.cpp - Emit long branches ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass expands a branch or jump instruction into a long branch if its
// offset is too large to fit into its immediate field.
//
// FIXME:
// 1. Fix pc-region jump instructions which cross 256MB segment boundaries.
// 2. If program has inline assembly statements whose size cannot be
//    determined accurately, load branch target addresses from the GOT.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-long-branch"

#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

STATISTIC(LongBranches, "Number of long branches.");

static cl::opt<bool> SkipLongBranch(
  "skip-mips-long-branch",
  cl::init(false),
  cl::desc("MIPS: Skip long branch pass."),
  cl::Hidden);

static cl::opt<bool> ForceLongBranch(
  "force-mips-long-branch",
  cl::init(false),
  cl::desc("MIPS: Expand all branches to long format."),
  cl::Hidden);

namespace {
  typedef MachineBasicBlock::iterator Iter;
  typedef MachineBasicBlock::reverse_iterator ReverseIter;

  struct MBBInfo {
    uint64_t Size, Address;
    bool HasLongBranch;
    MachineInstr *Br;

    MBBInfo() : Size(0), HasLongBranch(false), Br(0) {}
  };

  class MipsLongBranch : public MachineFunctionPass {

  public:
    static char ID;
    MipsLongBranch(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm),
        IsPIC(TM.getRelocationModel() == Reloc::PIC_),
        ABI(TM.getSubtarget<MipsSubtarget>().getTargetABI()),
        LongBranchSeqSize(!IsPIC ? 2 : (ABI == MipsSubtarget::N64 ? 13 : 9)) {}

    virtual const char *getPassName() const {
      return "Mips Long Branch";
    }

    bool runOnMachineFunction(MachineFunction &F);

  private:
    void splitMBB(MachineBasicBlock *MBB);
    void initMBBInfo();
    int64_t computeOffset(const MachineInstr *Br);
    void replaceBranch(MachineBasicBlock &MBB, Iter Br, DebugLoc DL,
                       MachineBasicBlock *MBBOpnd);
    void expandToLongBranch(MBBInfo &Info);

    const TargetMachine &TM;
    MachineFunction *MF;
    SmallVector<MBBInfo, 16> MBBInfos;
    bool IsPIC;
    unsigned ABI;
    unsigned LongBranchSeqSize;
  };

  char MipsLongBranch::ID = 0;
} // end of anonymous namespace

/// createMipsLongBranchPass - Returns a pass that converts branches to long
/// branches.
FunctionPass *llvm::createMipsLongBranchPass(MipsTargetMachine &tm) {
  return new MipsLongBranch(tm);
}

/// Iterate over list of Br's operands and search for a MachineBasicBlock
/// operand.
static MachineBasicBlock *getTargetMBB(const MachineInstr &Br) {
  for (unsigned I = 0, E = Br.getDesc().getNumOperands(); I < E; ++I) {
    const MachineOperand &MO = Br.getOperand(I);

    if (MO.isMBB())
      return MO.getMBB();
  }

  assert(false && "This instruction does not have an MBB operand.");
  return 0;
}

// Traverse the list of instructions backwards until a non-debug instruction is
// found or it reaches E.
static ReverseIter getNonDebugInstr(ReverseIter B, ReverseIter E) {
  for (; B != E; ++B)
    if (!B->isDebugValue())
      return B;

  return E;
}

// Split MBB if it has two direct jumps/branches.
void MipsLongBranch::splitMBB(MachineBasicBlock *MBB) {
  ReverseIter End = MBB->rend();
  ReverseIter LastBr = getNonDebugInstr(MBB->rbegin(), End);

  // Return if MBB has no branch instructions.
  if ((LastBr == End) ||
      (!LastBr->isConditionalBranch() && !LastBr->isUnconditionalBranch()))
    return;

  ReverseIter FirstBr = getNonDebugInstr(llvm::next(LastBr), End);

  // MBB has only one branch instruction if FirstBr is not a branch
  // instruction.
  if ((FirstBr == End) ||
      (!FirstBr->isConditionalBranch() && !FirstBr->isUnconditionalBranch()))
    return;

  assert(!FirstBr->isIndirectBranch() && "Unexpected indirect branch found.");

  // Create a new MBB. Move instructions in MBB to the newly created MBB.
  MachineBasicBlock *NewMBB =
    MF->CreateMachineBasicBlock(MBB->getBasicBlock());

  // Insert NewMBB and fix control flow.
  MachineBasicBlock *Tgt = getTargetMBB(*FirstBr);
  NewMBB->transferSuccessors(MBB);
  NewMBB->removeSuccessor(Tgt);
  MBB->addSuccessor(NewMBB);
  MBB->addSuccessor(Tgt);
  MF->insert(llvm::next(MachineFunction::iterator(MBB)), NewMBB);

  NewMBB->splice(NewMBB->end(), MBB, (++LastBr).base(), MBB->end());
}

// Fill MBBInfos.
void MipsLongBranch::initMBBInfo() {
  // Split the MBBs if they have two branches. Each basic block should have at
  // most one branch after this loop is executed.
  for (MachineFunction::iterator I = MF->begin(), E = MF->end(); I != E;)
    splitMBB(I++);

  MF->RenumberBlocks();
  MBBInfos.clear();
  MBBInfos.resize(MF->size());

  const MipsInstrInfo *TII =
    static_cast<const MipsInstrInfo*>(TM.getInstrInfo());
  for (unsigned I = 0, E = MBBInfos.size(); I < E; ++I) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(I);

    // Compute size of MBB.
    for (MachineBasicBlock::instr_iterator MI = MBB->instr_begin();
         MI != MBB->instr_end(); ++MI)
      MBBInfos[I].Size += TII->GetInstSizeInBytes(&*MI);

    // Search for MBB's branch instruction.
    ReverseIter End = MBB->rend();
    ReverseIter Br = getNonDebugInstr(MBB->rbegin(), End);

    if ((Br != End) && !Br->isIndirectBranch() &&
        (Br->isConditionalBranch() ||
         (Br->isUnconditionalBranch() &&
          TM.getRelocationModel() == Reloc::PIC_)))
      MBBInfos[I].Br = (++Br).base();
  }
}

// Compute offset of branch in number of bytes.
int64_t MipsLongBranch::computeOffset(const MachineInstr *Br) {
  int64_t Offset = 0;
  int ThisMBB = Br->getParent()->getNumber();
  int TargetMBB = getTargetMBB(*Br)->getNumber();

  // Compute offset of a forward branch.
  if (ThisMBB < TargetMBB) {
    for (int N = ThisMBB + 1; N < TargetMBB; ++N)
      Offset += MBBInfos[N].Size;

    return Offset + 4;
  }

  // Compute offset of a backward branch.
  for (int N = ThisMBB; N >= TargetMBB; --N)
    Offset += MBBInfos[N].Size;

  return -Offset + 4;
}

// Replace Br with a branch which has the opposite condition code and a
// MachineBasicBlock operand MBBOpnd.
void MipsLongBranch::replaceBranch(MachineBasicBlock &MBB, Iter Br,
                                   DebugLoc DL, MachineBasicBlock *MBBOpnd) {
  const MipsInstrInfo *TII =
    static_cast<const MipsInstrInfo*>(TM.getInstrInfo());
  unsigned NewOpc = TII->getOppositeBranchOpc(Br->getOpcode());
  const MCInstrDesc &NewDesc = TII->get(NewOpc);

  MachineInstrBuilder MIB = BuildMI(MBB, Br, DL, NewDesc);

  for (unsigned I = 0, E = Br->getDesc().getNumOperands(); I < E; ++I) {
    MachineOperand &MO = Br->getOperand(I);

    if (!MO.isReg()) {
      assert(MO.isMBB() && "MBB operand expected.");
      break;
    }

    MIB.addReg(MO.getReg());
  }

  MIB.addMBB(MBBOpnd);

  // Bundle the instruction in the delay slot to the newly created branch
  // and erase the original branch.
  assert(Br->isBundledWithSucc());
  MachineBasicBlock::instr_iterator II(Br);
  MIBundleBuilder(&*MIB).append((++II)->removeFromBundle());
  Br->eraseFromParent();
}

// Expand branch instructions to long branches.
void MipsLongBranch::expandToLongBranch(MBBInfo &I) {
  MachineBasicBlock::iterator Pos;
  MachineBasicBlock *MBB = I.Br->getParent(), *TgtMBB = getTargetMBB(*I.Br);
  DebugLoc DL = I.Br->getDebugLoc();
  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator FallThroughMBB = ++MachineFunction::iterator(MBB);
  MachineBasicBlock *LongBrMBB = MF->CreateMachineBasicBlock(BB);

  const MipsInstrInfo *TII =
    static_cast<const MipsInstrInfo*>(TM.getInstrInfo());

  MF->insert(FallThroughMBB, LongBrMBB);
  MBB->removeSuccessor(TgtMBB);
  MBB->addSuccessor(LongBrMBB);

  if (IsPIC) {
    MachineBasicBlock *BalTgtMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(FallThroughMBB, BalTgtMBB);
    LongBrMBB->addSuccessor(BalTgtMBB);
    BalTgtMBB->addSuccessor(TgtMBB);

    int64_t TgtAddress = MBBInfos[TgtMBB->getNumber()].Address;
    unsigned BalTgtMBBSize = 5;
    int64_t Offset = TgtAddress - (I.Address + I.Size - BalTgtMBBSize * 4);
    int64_t Lo = SignExtend64<16>(Offset & 0xffff);
    int64_t Hi = SignExtend64<16>(((Offset + 0x8000) >> 16) & 0xffff);

    if (ABI != MipsSubtarget::N64) {
      // $longbr:
      //  addiu $sp, $sp, -8
      //  sw $ra, 0($sp)
      //  bal $baltgt
      //  lui $at, %hi($tgt - $baltgt)
      // $baltgt:
      //  addiu $at, $at, %lo($tgt - $baltgt)
      //  addu $at, $ra, $at
      //  lw $ra, 0($sp)
      //  jr $at
      //  addiu $sp, $sp, 8
      // $fallthrough:
      //

      Pos = LongBrMBB->begin();

      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::ADDiu), Mips::SP)
        .addReg(Mips::SP).addImm(-8);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::SW)).addReg(Mips::RA)
        .addReg(Mips::SP).addImm(0);

      MIBundleBuilder(*LongBrMBB, Pos)
        .append(BuildMI(*MF, DL, TII->get(Mips::BAL_BR)).addMBB(BalTgtMBB))
        .append(BuildMI(*MF, DL, TII->get(Mips::LUi), Mips::AT).addImm(Hi));

      Pos = BalTgtMBB->begin();

      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::ADDiu), Mips::AT)
        .addReg(Mips::AT).addImm(Lo);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::ADDu), Mips::AT)
        .addReg(Mips::RA).addReg(Mips::AT);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::LW), Mips::RA)
        .addReg(Mips::SP).addImm(0);

      MIBundleBuilder(*BalTgtMBB, Pos)
        .append(BuildMI(*MF, DL, TII->get(Mips::JR)).addReg(Mips::AT))
        .append(BuildMI(*MF, DL, TII->get(Mips::ADDiu), Mips::SP)
                .addReg(Mips::SP).addImm(8));
    } else {
      // $longbr:
      //  daddiu $sp, $sp, -16
      //  sd $ra, 0($sp)
      //  lui64 $at, %highest($tgt - $baltgt)
      //  daddiu $at, $at, %higher($tgt - $baltgt)
      //  dsll $at, $at, 16
      //  daddiu $at, $at, %hi($tgt - $baltgt)
      //  bal $baltgt
      //  dsll $at, $at, 16
      // $baltgt:
      //  daddiu $at, $at, %lo($tgt - $baltgt)
      //  daddu $at, $ra, $at
      //  ld $ra, 0($sp)
      //  jr64 $at
      //  daddiu $sp, $sp, 16
      // $fallthrough:
      //

      int64_t Higher = SignExtend64<16>(((Offset + 0x80008000) >> 32) & 0xffff);
      int64_t Highest =
        SignExtend64<16>(((Offset + 0x800080008000LL) >> 48) & 0xffff);

      Pos = LongBrMBB->begin();

      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::SP_64)
        .addReg(Mips::SP_64).addImm(-16);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::SD)).addReg(Mips::RA_64)
        .addReg(Mips::SP_64).addImm(0);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::LUi64), Mips::AT_64)
        .addImm(Highest);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::AT_64)
        .addReg(Mips::AT_64).addImm(Higher);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DSLL), Mips::AT_64)
        .addReg(Mips::AT_64).addImm(16);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::AT_64)
        .addReg(Mips::AT_64).addImm(Hi);

      MIBundleBuilder(*LongBrMBB, Pos)
        .append(BuildMI(*MF, DL, TII->get(Mips::BAL_BR)).addMBB(BalTgtMBB))
        .append(BuildMI(*MF, DL, TII->get(Mips::DSLL), Mips::AT_64)
                .addReg(Mips::AT_64).addImm(16));

      Pos = BalTgtMBB->begin();

      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::AT_64)
        .addReg(Mips::AT_64).addImm(Lo);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DADDu), Mips::AT_64)
        .addReg(Mips::RA_64).addReg(Mips::AT_64);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::LD), Mips::RA_64)
        .addReg(Mips::SP_64).addImm(0);

      MIBundleBuilder(*BalTgtMBB, Pos)
        .append(BuildMI(*MF, DL, TII->get(Mips::JR64)).addReg(Mips::AT_64))
        .append(BuildMI(*MF, DL, TII->get(Mips::DADDiu), Mips::SP_64)
                .addReg(Mips::SP_64).addImm(16));
    }

    assert(BalTgtMBBSize == BalTgtMBB->size());
    assert(LongBrMBB->size() + BalTgtMBBSize == LongBranchSeqSize);
  } else {
    // $longbr:
    //  j $tgt
    //  nop
    // $fallthrough:
    //
    Pos = LongBrMBB->begin();
    LongBrMBB->addSuccessor(TgtMBB);
    MIBundleBuilder(*LongBrMBB, Pos)
      .append(BuildMI(*MF, DL, TII->get(Mips::J)).addMBB(TgtMBB))
      .append(BuildMI(*MF, DL, TII->get(Mips::NOP)));

    assert(LongBrMBB->size() == LongBranchSeqSize);
  }

  if (I.Br->isUnconditionalBranch()) {
    // Change branch destination.
    assert(I.Br->getDesc().getNumOperands() == 1);
    I.Br->RemoveOperand(0);
    I.Br->addOperand(MachineOperand::CreateMBB(LongBrMBB));
  } else
    // Change branch destination and reverse condition.
    replaceBranch(*MBB, I.Br, DL, FallThroughMBB);
}

static void emitGPDisp(MachineFunction &F, const MipsInstrInfo *TII) {
  MachineBasicBlock &MBB = F.front();
  MachineBasicBlock::iterator I = MBB.begin();
  DebugLoc DL = MBB.findDebugLoc(MBB.begin());
  BuildMI(MBB, I, DL, TII->get(Mips::LUi), Mips::V0)
    .addExternalSymbol("_gp_disp", MipsII::MO_ABS_HI);
  BuildMI(MBB, I, DL, TII->get(Mips::ADDiu), Mips::V0)
    .addReg(Mips::V0).addExternalSymbol("_gp_disp", MipsII::MO_ABS_LO);
  MBB.removeLiveIn(Mips::V0);
}

bool MipsLongBranch::runOnMachineFunction(MachineFunction &F) {
  const MipsInstrInfo *TII =
    static_cast<const MipsInstrInfo*>(TM.getInstrInfo());

  if (TM.getSubtarget<MipsSubtarget>().inMips16Mode())
    return false;
  if ((TM.getRelocationModel() == Reloc::PIC_) &&
      TM.getSubtarget<MipsSubtarget>().isABI_O32() &&
      F.getInfo<MipsFunctionInfo>()->globalBaseRegSet())
    emitGPDisp(F, TII);

  if (SkipLongBranch)
    return true;

  MF = &F;
  initMBBInfo();

  SmallVectorImpl<MBBInfo>::iterator I, E = MBBInfos.end();
  bool EverMadeChange = false, MadeChange = true;

  while (MadeChange) {
    MadeChange = false;

    for (I = MBBInfos.begin(); I != E; ++I) {
      // Skip if this MBB doesn't have a branch or the branch has already been
      // converted to a long branch.
      if (!I->Br || I->HasLongBranch)
        continue;

      int ShVal = TM.getSubtarget<MipsSubtarget>().inMicroMipsMode() ? 2 : 4;

      // Check if offset fits into 16-bit immediate field of branches.
      if (!ForceLongBranch && isInt<16>(computeOffset(I->Br) / ShVal))
        continue;

      I->HasLongBranch = true;
      I->Size += LongBranchSeqSize * 4;
      ++LongBranches;
      EverMadeChange = MadeChange = true;
    }
  }

  if (!EverMadeChange)
    return true;

  // Compute basic block addresses.
  if (TM.getRelocationModel() == Reloc::PIC_) {
    uint64_t Address = 0;

    for (I = MBBInfos.begin(); I != E; Address += I->Size, ++I)
      I->Address = Address;
  }

  // Do the expansion.
  for (I = MBBInfos.begin(); I != E; ++I)
    if (I->HasLongBranch)
      expandToLongBranch(*I);

  MF->RenumberBlocks();

  return true;
}
