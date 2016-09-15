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
  TII->insertUnconditionalBranch(*OrigBB, NewBB, DebugLoc());

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

/// fixupConditionalBranch - Fix up a conditional branch whose destination is
/// too far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool AArch64BranchRelaxation::fixupConditionalBranch(MachineInstr &MI) {
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;

  bool Fail = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  assert(!Fail && "branches to be relaxed must be analyzable");
  (void)Fail;

  // Add an unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // tbz L1
  // =>
  // tbnz L2
  // b   L1
  // L2:

  if (FBB && isBlockInRange(MI, *FBB)) {
    // Last MI in the BB is an unconditional branch. We can simply invert the
    // condition and swap destinations:
    // beq L1
    // b   L2
    // =>
    // bne L2
    // b   L1
    DEBUG(dbgs() << "  Invert condition and swap "
                    "its destination with " << MBB->back());

    TII->reverseBranchCondition(Cond);
    int OldSize = 0, NewSize = 0;
    TII->removeBranch(*MBB, &OldSize);
    TII->insertBranch(*MBB, FBB, TBB, Cond, DL, &NewSize);

    BlockInfo[MBB->getNumber()].Size += (NewSize - OldSize);
    return true;
  } else if (FBB) {
    // We need to split the basic block here to obtain two long-range
    // unconditional branches.
    auto &NewBB = *MF->CreateMachineBasicBlock(MBB->getBasicBlock());
    MF->insert(++MBB->getIterator(), &NewBB);

    // Insert an entry into BlockInfo to align it properly with the block
    // numbers.
    BlockInfo.insert(BlockInfo.begin() + NewBB.getNumber(), BasicBlockInfo());

    unsigned &NewBBSize = BlockInfo[NewBB.getNumber()].Size;
    int NewBrSize;
    TII->insertUnconditionalBranch(NewBB, FBB, DL, &NewBrSize);
    NewBBSize += NewBrSize;

    // Update the successor lists according to the transformation to follow.
    // Do it here since if there's no split, no update is needed.
    MBB->replaceSuccessor(FBB, &NewBB);
    NewBB.addSuccessor(FBB);
  }

  // We now have an appropriate fall-through block in place (either naturally or
  // just created), so we can invert the condition.
  MachineBasicBlock &NextBB = *std::next(MachineFunction::iterator(MBB));

  DEBUG(dbgs() << "  Insert B to BB#" << TBB->getNumber()
               << ", invert condition and change dest. to BB#"
               << NextBB.getNumber() << '\n');

  unsigned &MBBSize = BlockInfo[MBB->getNumber()].Size;

  // Insert a new conditional branch and a new unconditional branch.
  int RemovedSize = 0;
  TII->reverseBranchCondition(Cond);
  TII->removeBranch(*MBB, &RemovedSize);
  MBBSize -= RemovedSize;

  int AddedSize = 0;
  TII->insertBranch(*MBB, &NextBB, TBB, Cond, DL, &AddedSize);
  MBBSize += AddedSize;

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
