//===-- AArch64BranchRelaxation.cpp - AArch64 branch relaxation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64-branch-relax"

STATISTIC(NumSplit, "Number of basic blocks split");
STATISTIC(NumConditionalRelaxed, "Number of conditional branches relaxed");

namespace llvm {
void initializeAArch64BranchRelaxationPass(PassRegistry &);
}

#define AARCH64_BR_RELAX_NAME "AArch64 branch relaxation pass"

namespace {
class AArch64BranchRelaxation : public MachineFunctionPass {
  /// BasicBlockInfo - Information about the offset and size of a single
  /// basic block.
  struct BasicBlockInfo {
    /// Offset - Distance from the beginning of the function to the beginning
    /// of this basic block.
    ///
    /// The offset is always aligned as required by the basic block.
    unsigned Offset;

    /// Size - Size of the basic block in bytes.  If the block contains
    /// inline assembly, this is a worst case estimate.
    ///
    /// The size does not include any alignment padding whether from the
    /// beginning of the block, or from an aligned jump table at the end.
    unsigned Size;

    BasicBlockInfo() : Offset(0), Size(0) {}

    /// Compute the offset immediately following this block.  If LogAlign is
    /// specified, return the offset the successor block will get if it has
    /// this alignment.
    unsigned postOffset(unsigned LogAlign = 0) const {
      unsigned PO = Offset + Size;
      unsigned Align = 1 << LogAlign;
      return (PO + Align - 1) / Align * Align;
    }
  };

  SmallVector<BasicBlockInfo, 16> BlockInfo;

  MachineFunction *MF;
  const AArch64InstrInfo *TII;

  bool relaxBranchInstructions();
  void scanFunction();
  MachineBasicBlock *splitBlockBeforeInstr(MachineInstr &MI);
  void adjustBlockOffsets(MachineBasicBlock &MBB);
  bool isBlockInRange(const MachineInstr &MI, const MachineBasicBlock &BB) const;

  unsigned insertInvertedConditionalBranch(MachineBasicBlock &SrcBB,
                                           MachineBasicBlock::iterator InsPt,
                                           const DebugLoc &DL,
                                           const MachineInstr &OldBr,
                                           MachineBasicBlock &NewDestBB) const;
  unsigned insertUnconditionalBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock &NewDestBB,
                                     const DebugLoc &DL) const;

  bool fixupConditionalBranch(MachineInstr &MI);
  void computeBlockSize(const MachineBasicBlock &MBB);
  unsigned getInstrOffset(const MachineInstr &MI) const;
  void dumpBBs();
  void verify();

public:
  static char ID;
  AArch64BranchRelaxation() : MachineFunctionPass(ID) {
    initializeAArch64BranchRelaxationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return AARCH64_BR_RELAX_NAME;
  }
};
char AArch64BranchRelaxation::ID = 0;
}

INITIALIZE_PASS(AArch64BranchRelaxation, "aarch64-branch-relax",
                AARCH64_BR_RELAX_NAME, false, false)

/// verify - check BBOffsets, BBSizes, alignment of islands
void AArch64BranchRelaxation::verify() {
#ifndef NDEBUG
  unsigned PrevNum = MF->begin()->getNumber();
  for (MachineBasicBlock &MBB : *MF) {
    unsigned Align = MBB.getAlignment();
    unsigned Num = MBB.getNumber();
    assert(BlockInfo[Num].Offset % (1u << Align) == 0);
    assert(!Num || BlockInfo[PrevNum].postOffset() <= BlockInfo[Num].Offset);
    PrevNum = Num;
  }
#endif
}

/// print block size and offset information - debugging
void AArch64BranchRelaxation::dumpBBs() {
  for (auto &MBB : *MF) {
    const BasicBlockInfo &BBI = BlockInfo[MBB.getNumber()];
    dbgs() << format("BB#%u\toffset=%08x\t", MBB.getNumber(), BBI.Offset)
           << format("size=%#x\n", BBI.Size);
  }
}

// FIXME: This is a less precise version of MachineBasicBlock::canFallThrough?

/// \returns true if the specified basic block can fallthrough
/// into the block immediately after it.
static bool hasFallthrough(const MachineBasicBlock &MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::const_iterator MBBI(MBB);

  // Can't fall off end of function.
  auto NextBB = std::next(MBBI);
  if (NextBB == MBB.getParent()->end())
    return false;

  return MBB.isSuccessor(&*NextBB);
}

/// scanFunction - Do the initial scan of the function, building up
/// information about each block.
void AArch64BranchRelaxation::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it. If so, we have to be conservative about
  // alignment assumptions, as we don't know for sure the size of any
  // instructions in the inline assembly.
  for (MachineBasicBlock &MBB : *MF)
    computeBlockSize(MBB);

  // Compute block offsets and known bits.
  adjustBlockOffsets(*MF->begin());
}

/// computeBlockSize - Compute the size for MBB.
/// This function updates BlockInfo directly.
void AArch64BranchRelaxation::computeBlockSize(const MachineBasicBlock &MBB) {
  unsigned Size = 0;
  for (const MachineInstr &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  BlockInfo[MBB.getNumber()].Size = Size;
}

/// getInstrOffset - Return the current offset of the specified machine
/// instruction from the start of the function.  This offset changes as stuff is
/// moved around inside the function.
unsigned AArch64BranchRelaxation::getInstrOffset(const MachineInstr &MI) const {
  const MachineBasicBlock *MBB = MI.getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::const_iterator I = MBB->begin(); &*I != &MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }

  return Offset;
}

void AArch64BranchRelaxation::adjustBlockOffsets(MachineBasicBlock &Start) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB : make_range(MachineFunction::iterator(Start), MF->end())) {
    unsigned Num = MBB.getNumber();
    if (!Num) // block zero is never changed from offset zero.
      continue;
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    unsigned LogAlign = MBB.getAlignment();
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(LogAlign);
    PrevNum = Num;
  }
}

/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
/// NOTE: Successor list of the original BB is out of date after this function,
/// and must be updated by the caller! Other transforms follow using this
/// utility function, so no point updating now rather than waiting.
MachineBasicBlock *
AArch64BranchRelaxation::splitBlockBeforeInstr(MachineInstr &MI) {
  MachineBasicBlock *OrigBB = MI.getParent();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
      MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MF->insert(++OrigBB->getIterator(), NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI.getIterator(), OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  insertUnconditionalBranch(*OrigBB, *NewBB, DebugLoc());

  // Insert an entry into BlockInfo to align it properly with the block numbers.
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Figure out how large the OrigBB is.  As the first half of the original
  // block, it cannot contain a tablejump.  The size includes
  // the new jump we added.  (It should be possible to do this without
  // recounting everything, but it's very confusing, and this is rarely
  // executed.)
  computeBlockSize(*OrigBB);

  // Figure out how large the NewMBB is.  As the second half of the original
  // block, it may contain a tablejump.
  computeBlockSize(*NewBB);

  // All BBOffsets following these blocks must be modified.
  adjustBlockOffsets(*OrigBB);

  ++NumSplit;

  return NewBB;
}

/// isBlockInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool AArch64BranchRelaxation::isBlockInRange(
  const MachineInstr &MI, const MachineBasicBlock &DestBB) const {
  unsigned BrOffset = getInstrOffset(MI);
  unsigned DestOffset = BlockInfo[DestBB.getNumber()].Offset;

  if (TII->isBranchInRange(MI.getOpcode(), BrOffset, DestOffset))
    return true;

  DEBUG(
    dbgs() << "Out of range branch to destination BB#" << DestBB.getNumber()
           << " from BB#" << MI.getParent()->getNumber()
           << " to " << DestOffset
           << " offset " << static_cast<int>(DestOffset - BrOffset)
           << '\t' << MI
  );

  return false;
}

static MachineBasicBlock *getDestBlock(const MachineInstr &MI) {
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

static unsigned getOppositeConditionOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("unexpected opcode!");
  case AArch64::TBNZW:   return AArch64::TBZW;
  case AArch64::TBNZX:   return AArch64::TBZX;
  case AArch64::TBZW:    return AArch64::TBNZW;
  case AArch64::TBZX:    return AArch64::TBNZX;
  case AArch64::CBNZW:   return AArch64::CBZW;
  case AArch64::CBNZX:   return AArch64::CBZX;
  case AArch64::CBZW:    return AArch64::CBNZW;
  case AArch64::CBZX:    return AArch64::CBNZX;
  case AArch64::Bcc:     return AArch64::Bcc; // Condition is an operand for Bcc.
  }
}

static inline void invertBccCondition(MachineInstr &MI) {
  assert(MI.getOpcode() == AArch64::Bcc && "Unexpected opcode!");
  MachineOperand &CCOp = MI.getOperand(0);

  AArch64CC::CondCode CC = static_cast<AArch64CC::CondCode>(CCOp.getImm());
  CCOp.setImm(AArch64CC::getInvertedCondCode(CC));
}

/// Insert a conditional branch at the end of \p MBB to \p NewDestBB, using the
/// inverse condition of branch \p OldBr.
/// \returns The number of bytes added to the block.
unsigned AArch64BranchRelaxation::insertInvertedConditionalBranch(
  MachineBasicBlock &SrcMBB,
  MachineBasicBlock::iterator InsPt,
  const DebugLoc &DL,
  const MachineInstr &OldBr,
  MachineBasicBlock &NewDestBB) const {
  unsigned OppositeCondOpc = getOppositeConditionOpcode(OldBr.getOpcode());

  MachineInstrBuilder MIB =
    BuildMI(SrcMBB, InsPt, DL, TII->get(OppositeCondOpc))
    .addOperand(OldBr.getOperand(0));

  unsigned Opc = OldBr.getOpcode();

  if (Opc == AArch64::TBZW || Opc == AArch64::TBNZW ||
      Opc == AArch64::TBZX || Opc == AArch64::TBNZX)
    MIB.addOperand(OldBr.getOperand(1));

  if (OldBr.getOpcode() == AArch64::Bcc)
    invertBccCondition(*MIB);

  MIB.addMBB(&NewDestBB);

  return TII->getInstSizeInBytes(*MIB);
}

/// Insert an unconditional branch at the end of \p MBB to \p DestBB.
/// \returns the number of bytes emitted.
unsigned AArch64BranchRelaxation::insertUnconditionalBranch(
  MachineBasicBlock &MBB,
  MachineBasicBlock &DestBB,
  const DebugLoc &DL) const {
  MachineInstr *MI = BuildMI(&MBB, DL, TII->get(AArch64::B))
    .addMBB(&DestBB);

  return TII->getInstSizeInBytes(*MI);
}

static void changeBranchDestBlock(MachineInstr &MI,
                                  MachineBasicBlock &NewDestBB) {
  unsigned OpNum = 0;
  unsigned Opc = MI.getOpcode();

  if (Opc != AArch64::B) {
    OpNum = (Opc == AArch64::TBZW ||
             Opc == AArch64::TBNZW ||
             Opc == AArch64::TBZX ||
             Opc == AArch64::TBNZX) ? 2 : 1;
  }

  MI.getOperand(OpNum).setMBB(&NewDestBB);
}

/// fixupConditionalBranch - Fix up a conditional branch whose destination is
/// too far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool AArch64BranchRelaxation::fixupConditionalBranch(MachineInstr &MI) {
  MachineBasicBlock *DestBB = getDestBlock(MI);

  // Add an unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // tbz L1
  // =>
  // tbnz L2
  // b   L1
  // L2:

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through block. Otherwise,
  // split the MBB before the next instruction.
  MachineBasicBlock *MBB = MI.getParent();
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != &MI) || !hasFallthrough(*MBB);

  if (BMI != &MI) {
    if (std::next(MachineBasicBlock::iterator(MI)) ==
            std::prev(MBB->getLastNonDebugInstr()) &&
        BMI->isUnconditionalBranch()) {
      // Last MI in the BB is an unconditional branch. We can simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      MachineBasicBlock *NewDest = getDestBlock(*BMI);
      if (isBlockInRange(MI, *NewDest)) {
        DEBUG(dbgs() << "  Invert condition and swap its destination with "
                     << *BMI);
        changeBranchDestBlock(*BMI, *DestBB);

        int NewSize =
          insertInvertedConditionalBranch(*MBB, MI.getIterator(),
                                          MI.getDebugLoc(), MI, *NewDest);
        int OldSize = TII->getInstSizeInBytes(MI);
        BlockInfo[MBB->getNumber()].Size += (NewSize - OldSize);
        MI.eraseFromParent();
        return true;
      }
    }
  }

  if (NeedSplit) {
    // Analyze the branch so we know how to update the successor lists.
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 2> Cond;
    bool Fail = TII->analyzeBranch(*MBB, TBB, FBB, Cond, false);
    assert(!Fail && "branches to relax should be analyzable");
    (void)Fail;

    MachineBasicBlock *NewBB = splitBlockBeforeInstr(MI);
    // No need for the branch to the next block. We're adding an unconditional
    // branch to the destination.
    int delta = TII->getInstSizeInBytes(MBB->back());
    BlockInfo[MBB->getNumber()].Size -= delta;
    MBB->back().eraseFromParent();
    // BlockInfo[SplitBB].Offset is wrong temporarily, fixed below

    // Update the successor lists according to the transformation to follow.
    // Do it here since if there's no split, no update is needed.
    MBB->replaceSuccessor(FBB, NewBB);
    NewBB->addSuccessor(FBB);
  }

  MachineBasicBlock &NextBB = *std::next(MachineFunction::iterator(MBB));

  DEBUG(dbgs() << "  Insert B to BB#" << DestBB->getNumber()
               << ", invert condition and change dest. to BB#"
               << NextBB.getNumber() << '\n');

  unsigned &MBBSize = BlockInfo[MBB->getNumber()].Size;

  // Insert a new conditional branch and a new unconditional branch.
  MBBSize += insertInvertedConditionalBranch(*MBB, MBB->end(),
                                             MI.getDebugLoc(), MI, NextBB);

  MBBSize += insertUnconditionalBranch(*MBB, *DestBB, MI.getDebugLoc());

  // Remove the old conditional branch.  It may or may not still be in MBB.
  MBBSize -= TII->getInstSizeInBytes(MI);
  MI.eraseFromParent();

  // Finally, keep the block offsets up to date.
  adjustBlockOffsets(*MBB);
  return true;
}

bool AArch64BranchRelaxation::relaxBranchInstructions() {
  bool Changed = false;
  // Relaxing branches involves creating new basic blocks, so re-eval
  // end() for termination.
  for (MachineFunction::iterator I = MF->begin(); I != MF->end(); ++I) {
    MachineBasicBlock &MBB = *I;
    MachineBasicBlock::iterator J = MBB.getFirstTerminator();
    if (J == MBB.end())
      continue;

    MachineBasicBlock::iterator Next;
    for (MachineBasicBlock::iterator J = MBB.getFirstTerminator();
         J != MBB.end(); J = Next) {
      Next = std::next(J);
      MachineInstr &MI = *J;

      if (MI.isConditionalBranch()) {
        MachineBasicBlock *DestBB = getDestBlock(MI);
        if (!isBlockInRange(MI, *DestBB)) {
          if (Next != MBB.end() && Next->isConditionalBranch()) {
            // If there are multiple conditional branches, this isn't an
            // analyzable block. Split later terminators into a new block so
            // each one will be analyzable.

            MachineBasicBlock *NewBB = splitBlockBeforeInstr(*Next);
            NewBB->transferSuccessors(&MBB);
            MBB.addSuccessor(NewBB);
            MBB.addSuccessor(DestBB);

            // Cleanup potential unconditional branch to successor block.
            NewBB->updateTerminator();
            MBB.updateTerminator();
          } else {
            fixupConditionalBranch(MI);
            ++NumConditionalRelaxed;
          }

          Changed = true;

          // This may have modified all of the terminators, so start over.
          Next = MBB.getFirstTerminator();
        }

      }
    }
  }

  return Changed;
}

bool AArch64BranchRelaxation::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;

  DEBUG(dbgs() << "***** AArch64BranchRelaxation *****\n");

  TII = MF->getSubtarget<AArch64Subtarget>().getInstrInfo();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Do the initial scan of the function, building up information about the
  // sizes of each block.
  scanFunction();

  DEBUG(dbgs() << "  Basic blocks before relaxation\n"; dumpBBs(););

  bool MadeChange = false;
  while (relaxBranchInstructions())
    MadeChange = true;

  // After a while, this might be made debug-only, but it is not expensive.
  verify();

  DEBUG(dbgs() << "  Basic blocks after relaxation\n\n"; dumpBBs());

  BlockInfo.clear();

  return MadeChange;
}

/// Returns an instance of the AArch64 Branch Relaxation pass.
FunctionPass *llvm::createAArch64BranchRelaxation() {
  return new AArch64BranchRelaxation();
}
