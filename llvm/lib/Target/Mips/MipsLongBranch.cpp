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
// FIXME: Fix pc-region jump instructions which cross 256MB segment boundaries.
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsMCNaCl.h"
#include "MipsMachineFunction.h"
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

#define DEBUG_TYPE "mips-long-branch"

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

    MBBInfo() : Size(0), HasLongBranch(false), Br(nullptr) {}
  };

  class MipsLongBranch : public MachineFunctionPass {

  public:
    static char ID;
    MipsLongBranch(TargetMachine &tm)
        : MachineFunctionPass(ID), TM(tm), IsPIC(TM.isPositionIndependent()),
          ABI(static_cast<const MipsTargetMachine &>(TM).getABI()) {}

    const char *getPassName() const override {
      return "Mips Long Branch";
    }

    bool runOnMachineFunction(MachineFunction &F) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::AllVRegsAllocated);
    }

  private:
    void splitMBB(MachineBasicBlock *MBB);
    void initMBBInfo();
    int64_t computeOffset(const MachineInstr *Br);
    void replaceBranch(MachineBasicBlock &MBB, Iter Br, const DebugLoc &DL,
                       MachineBasicBlock *MBBOpnd);
    void expandToLongBranch(MBBInfo &Info);

    const TargetMachine &TM;
    MachineFunction *MF;
    SmallVector<MBBInfo, 16> MBBInfos;
    bool IsPIC;
    MipsABIInfo ABI;
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

  llvm_unreachable("This instruction does not have an MBB operand.");
}

// Traverse the list of instructions backwards until a non-debug instruction is
// found or it reaches E.
static ReverseIter getNonDebugInstr(ReverseIter B, const ReverseIter &E) {
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

  ReverseIter FirstBr = getNonDebugInstr(std::next(LastBr), End);

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
  NewMBB->removeSuccessor(Tgt, true);
  MBB->addSuccessor(NewMBB);
  MBB->addSuccessor(Tgt);
  MF->insert(std::next(MachineFunction::iterator(MBB)), NewMBB);

  NewMBB->splice(NewMBB->end(), MBB, (++LastBr).base(), MBB->end());
}

// Fill MBBInfos.
void MipsLongBranch::initMBBInfo() {
  // Split the MBBs if they have two branches. Each basic block should have at
  // most one branch after this loop is executed.
  for (auto &MBB : *MF)
    splitMBB(&MBB);

  MF->RenumberBlocks();
  MBBInfos.clear();
  MBBInfos.resize(MF->size());

  const MipsInstrInfo *TII =
      static_cast<const MipsInstrInfo *>(MF->getSubtarget().getInstrInfo());
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
        (Br->isConditionalBranch() || (Br->isUnconditionalBranch() && IsPIC)))
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
                                   const DebugLoc &DL,
                                   MachineBasicBlock *MBBOpnd) {
  const MipsInstrInfo *TII = static_cast<const MipsInstrInfo *>(
      MBB.getParent()->getSubtarget().getInstrInfo());
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

  if (Br->hasDelaySlot()) {
    // Bundle the instruction in the delay slot to the newly created branch
    // and erase the original branch.
    assert(Br->isBundledWithSucc());
    MachineBasicBlock::instr_iterator II(Br);
    MIBundleBuilder(&*MIB).append((++II)->removeFromBundle());
  }
  Br->eraseFromParent();
}

// Expand branch instructions to long branches.
// TODO: This function has to be fixed for beqz16 and bnez16, because it
// currently assumes that all branches have 16-bit offsets, and will produce
// wrong code if branches whose allowed offsets are [-128, -126, ..., 126]
// are present.
void MipsLongBranch::expandToLongBranch(MBBInfo &I) {
  MachineBasicBlock::iterator Pos;
  MachineBasicBlock *MBB = I.Br->getParent(), *TgtMBB = getTargetMBB(*I.Br);
  DebugLoc DL = I.Br->getDebugLoc();
  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator FallThroughMBB = ++MachineFunction::iterator(MBB);
  MachineBasicBlock *LongBrMBB = MF->CreateMachineBasicBlock(BB);
  const MipsSubtarget &Subtarget =
      static_cast<const MipsSubtarget &>(MF->getSubtarget());
  const MipsInstrInfo *TII =
      static_cast<const MipsInstrInfo *>(Subtarget.getInstrInfo());

  MF->insert(FallThroughMBB, LongBrMBB);
  MBB->replaceSuccessor(TgtMBB, LongBrMBB);

  if (IsPIC) {
    MachineBasicBlock *BalTgtMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(FallThroughMBB, BalTgtMBB);
    LongBrMBB->addSuccessor(BalTgtMBB);
    BalTgtMBB->addSuccessor(TgtMBB);

    // We must select between the MIPS32r6/MIPS64r6 BAL (which is a normal
    // instruction) and the pre-MIPS32r6/MIPS64r6 definition (which is an
    // pseudo-instruction wrapping BGEZAL).
    unsigned BalOp = Subtarget.hasMips32r6() ? Mips::BAL : Mips::BAL_BR;

    if (!ABI.IsN64()) {
      // $longbr:
      //  addiu $sp, $sp, -8
      //  sw $ra, 0($sp)
      //  lui $at, %hi($tgt - $baltgt)
      //  bal $baltgt
      //  addiu $at, $at, %lo($tgt - $baltgt)
      // $baltgt:
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

      // LUi and ADDiu instructions create 32-bit offset of the target basic
      // block from the target of BAL instruction.  We cannot use immediate
      // value for this offset because it cannot be determined accurately when
      // the program has inline assembly statements.  We therefore use the
      // relocation expressions %hi($tgt-$baltgt) and %lo($tgt-$baltgt) which
      // are resolved during the fixup, so the values will always be correct.
      //
      // Since we cannot create %hi($tgt-$baltgt) and %lo($tgt-$baltgt)
      // expressions at this point (it is possible only at the MC layer),
      // we replace LUi and ADDiu with pseudo instructions
      // LONG_BRANCH_LUi and LONG_BRANCH_ADDiu, and add both basic
      // blocks as operands to these instructions.  When lowering these pseudo
      // instructions to LUi and ADDiu in the MC layer, we will create
      // %hi($tgt-$baltgt) and %lo($tgt-$baltgt) expressions and add them as
      // operands to lowered instructions.

      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::LONG_BRANCH_LUi), Mips::AT)
        .addMBB(TgtMBB).addMBB(BalTgtMBB);
      MIBundleBuilder(*LongBrMBB, Pos)
          .append(BuildMI(*MF, DL, TII->get(BalOp)).addMBB(BalTgtMBB))
          .append(BuildMI(*MF, DL, TII->get(Mips::LONG_BRANCH_ADDiu), Mips::AT)
                      .addReg(Mips::AT)
                      .addMBB(TgtMBB)
                      .addMBB(BalTgtMBB));

      Pos = BalTgtMBB->begin();

      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::ADDu), Mips::AT)
        .addReg(Mips::RA).addReg(Mips::AT);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::LW), Mips::RA)
        .addReg(Mips::SP).addImm(0);

      // In NaCl, modifying the sp is not allowed in branch delay slot.
      if (Subtarget.isTargetNaCl())
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::ADDiu), Mips::SP)
          .addReg(Mips::SP).addImm(8);

      if (Subtarget.hasMips32r6())
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::JALR))
          .addReg(Mips::ZERO).addReg(Mips::AT);
      else
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::JR)).addReg(Mips::AT);

      if (Subtarget.isTargetNaCl()) {
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::NOP));
        // Bundle-align the target of indirect branch JR.
        TgtMBB->setAlignment(MIPS_NACL_BUNDLE_ALIGN);
      } else
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::ADDiu), Mips::SP)
          .addReg(Mips::SP).addImm(8);

      BalTgtMBB->rbegin()->bundleWithPred();
    } else {
      // $longbr:
      //  daddiu $sp, $sp, -16
      //  sd $ra, 0($sp)
      //  daddiu $at, $zero, %hi($tgt - $baltgt)
      //  dsll $at, $at, 16
      //  bal $baltgt
      //  daddiu $at, $at, %lo($tgt - $baltgt)
      // $baltgt:
      //  daddu $at, $ra, $at
      //  ld $ra, 0($sp)
      //  jr64 $at
      //  daddiu $sp, $sp, 16
      // $fallthrough:
      //

      // We assume the branch is within-function, and that offset is within
      // +/- 2GB.  High 32 bits will therefore always be zero.

      // Note that this will work even if the offset is negative, because
      // of the +1 modification that's added in that case.  For example, if the
      // offset is -1MB (0xFFFFFFFFFFF00000), the computation for %higher is
      //
      // 0xFFFFFFFFFFF00000 + 0x80008000 = 0x000000007FF08000
      //
      // and the bits [47:32] are zero.  For %highest
      //
      // 0xFFFFFFFFFFF00000 + 0x800080008000 = 0x000080007FF08000
      //
      // and the bits [63:48] are zero.

      Pos = LongBrMBB->begin();

      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::SP_64)
        .addReg(Mips::SP_64).addImm(-16);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::SD)).addReg(Mips::RA_64)
        .addReg(Mips::SP_64).addImm(0);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::LONG_BRANCH_DADDiu),
              Mips::AT_64).addReg(Mips::ZERO_64)
                          .addMBB(TgtMBB, MipsII::MO_ABS_HI).addMBB(BalTgtMBB);
      BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::DSLL), Mips::AT_64)
        .addReg(Mips::AT_64).addImm(16);

      MIBundleBuilder(*LongBrMBB, Pos)
          .append(BuildMI(*MF, DL, TII->get(BalOp)).addMBB(BalTgtMBB))
          .append(
              BuildMI(*MF, DL, TII->get(Mips::LONG_BRANCH_DADDiu), Mips::AT_64)
                  .addReg(Mips::AT_64)
                  .addMBB(TgtMBB, MipsII::MO_ABS_LO)
                  .addMBB(BalTgtMBB));

      Pos = BalTgtMBB->begin();

      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DADDu), Mips::AT_64)
        .addReg(Mips::RA_64).addReg(Mips::AT_64);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::LD), Mips::RA_64)
        .addReg(Mips::SP_64).addImm(0);

      if (Subtarget.hasMips64r6())
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::JALR64))
          .addReg(Mips::ZERO_64).addReg(Mips::AT_64);
      else
        BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::JR64)).addReg(Mips::AT_64);

      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DADDiu), Mips::SP_64)
        .addReg(Mips::SP_64).addImm(16);
      BalTgtMBB->rbegin()->bundleWithPred();
    }

    assert(LongBrMBB->size() + BalTgtMBB->size() == LongBranchSeqSize);
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
    replaceBranch(*MBB, I.Br, DL, &*FallThroughMBB);
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
  const MipsSubtarget &STI =
      static_cast<const MipsSubtarget &>(F.getSubtarget());
  const MipsInstrInfo *TII =
      static_cast<const MipsInstrInfo *>(STI.getInstrInfo());
  LongBranchSeqSize =
      !IsPIC ? 2 : (ABI.IsN64() ? 10 : (!STI.isTargetNaCl() ? 9 : 10));

  if (STI.inMips16Mode() || !STI.enableLongBranchPass())
    return false;
  if (IsPIC && static_cast<const MipsTargetMachine &>(TM).getABI().IsO32() &&
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

      int ShVal = STI.inMicroMipsMode() ? 2 : 4;
      int64_t Offset = computeOffset(I->Br) / ShVal;

      if (STI.isTargetNaCl()) {
        // The offset calculation does not include sandboxing instructions
        // that will be added later in the MC layer.  Since at this point we
        // don't know the exact amount of code that "sandboxing" will add, we
        // conservatively estimate that code will not grow more than 100%.
        Offset *= 2;
      }

      // Check if offset fits into 16-bit immediate field of branches.
      if (!ForceLongBranch && isInt<16>(Offset))
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
  if (IsPIC) {
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
