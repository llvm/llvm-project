//===-- AArch64BranchRelaxation.cpp - AArch64 branch relaxation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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

static cl::opt<bool>
BranchRelaxation("aarch64-branch-relax", cl::Hidden, cl::init(true),
                 cl::desc("Relax out of range conditional branches"));

static cl::opt<unsigned>
TBZDisplacementBits("aarch64-tbz-offset-bits", cl::Hidden, cl::init(14),
                    cl::desc("Restrict range of TB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned>
CBZDisplacementBits("aarch64-cbz-offset-bits", cl::Hidden, cl::init(19),
                    cl::desc("Restrict range of CB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned>
BCCDisplacementBits("aarch64-bcc-offset-bits", cl::Hidden, cl::init(19),
                    cl::desc("Restrict range of Bcc instructions (DEBUG)"));

STATISTIC(NumSplit, "Number of basic blocks split");
STATISTIC(NumRelaxed, "Number of conditional branches relaxed");

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
  MachineBasicBlock *splitBlockBeforeInstr(MachineInstr *MI);
  void adjustBlockOffsets(MachineBasicBlock &MBB);
  bool isBlockInRange(MachineInstr *MI, MachineBasicBlock *BB, unsigned Disp);
  bool fixupConditionalBranch(MachineInstr *MI);
  void computeBlockSize(const MachineBasicBlock &MBB);
  unsigned getInstrOffset(MachineInstr *MI) const;
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

/// BBHasFallthrough - Return true if the specified basic block can fallthrough
/// into the block immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI(MBB);
  // Can't fall off end of function.
  auto NextBB = std::next(MBBI);
  if (NextBB == MBB->getParent()->end())
    return false;

  for (MachineBasicBlock *S : MBB->successors())
    if (S == &*NextBB)
      return true;

  return false;
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
    Size += TII->GetInstSizeInBytes(&MI);
  BlockInfo[MBB.getNumber()].Size = Size;
}

/// getInstrOffset - Return the current offset of the specified machine
/// instruction from the start of the function.  This offset changes as stuff is
/// moved around inside the function.
unsigned AArch64BranchRelaxation::getInstrOffset(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); &*I != MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->GetInstSizeInBytes(I);
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
AArch64BranchRelaxation::splitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
      MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MF->insert(++OrigBB->getIterator(), NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  BuildMI(OrigBB, DebugLoc(), TII->get(AArch64::B)).addMBB(NewBB);

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
bool AArch64BranchRelaxation::isBlockInRange(MachineInstr *MI,
                                             MachineBasicBlock *DestBB,
                                             unsigned Bits) {
  unsigned MaxOffs = ((1 << (Bits - 1)) - 1) << 2;
  unsigned BrOffset = getInstrOffset(MI);
  unsigned DestOffset = BlockInfo[DestBB->getNumber()].Offset;

  DEBUG(dbgs() << "Branch of destination BB#" << DestBB->getNumber()
               << " from BB#" << MI->getParent()->getNumber()
               << " max delta=" << MaxOffs << " from " << getInstrOffset(MI)
               << " to " << DestOffset << " offset "
               << int(DestOffset - BrOffset) << "\t" << *MI);

  // Branch before the Dest.
  if (BrOffset <= DestOffset)
    return (DestOffset - BrOffset <= MaxOffs);
  return (BrOffset - DestOffset <= MaxOffs);
}

static bool isConditionalBranch(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case AArch64::TBZW:
  case AArch64::TBNZW:
  case AArch64::TBZX:
  case AArch64::TBNZX:
  case AArch64::CBZW:
  case AArch64::CBNZW:
  case AArch64::CBZX:
  case AArch64::CBNZX:
  case AArch64::Bcc:
    return true;
  }
}

static MachineBasicBlock *getDestBlock(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default:
    llvm_unreachable("unexpected opcode!");
  case AArch64::TBZW:
  case AArch64::TBNZW:
  case AArch64::TBZX:
  case AArch64::TBNZX:
    return MI->getOperand(2).getMBB();
  case AArch64::CBZW:
  case AArch64::CBNZW:
  case AArch64::CBZX:
  case AArch64::CBNZX:
  case AArch64::Bcc:
    return MI->getOperand(1).getMBB();
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

static unsigned getBranchDisplacementBits(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("unexpected opcode!");
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

static inline void invertBccCondition(MachineInstr *MI) {
  assert(MI->getOpcode() == AArch64::Bcc && "Unexpected opcode!");
  AArch64CC::CondCode CC = (AArch64CC::CondCode)MI->getOperand(0).getImm();
  CC = AArch64CC::getInvertedCondCode(CC);
  MI->getOperand(0).setImm((int64_t)CC);
}

/// fixupConditionalBranch - Fix up a conditional branch whose destination is
/// too far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool AArch64BranchRelaxation::fixupConditionalBranch(MachineInstr *MI) {
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
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != MI) || !BBHasFallthrough(MBB);

  if (BMI != MI) {
    if (std::next(MachineBasicBlock::iterator(MI)) ==
            std::prev(MBB->getLastNonDebugInstr()) &&
        BMI->getOpcode() == AArch64::B) {
      // Last MI in the BB is an unconditional branch. Can we simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      MachineBasicBlock *NewDest = BMI->getOperand(0).getMBB();
      if (isBlockInRange(MI, NewDest,
                         getBranchDisplacementBits(MI->getOpcode()))) {
        DEBUG(dbgs() << "  Invert condition and swap its destination with "
                     << *BMI);
        BMI->getOperand(0).setMBB(DestBB);
        unsigned OpNum = (MI->getOpcode() == AArch64::TBZW ||
                          MI->getOpcode() == AArch64::TBNZW ||
                          MI->getOpcode() == AArch64::TBZX ||
                          MI->getOpcode() == AArch64::TBNZX)
                             ? 2
                             : 1;
        MI->getOperand(OpNum).setMBB(NewDest);
        MI->setDesc(TII->get(getOppositeConditionOpcode(MI->getOpcode())));
        if (MI->getOpcode() == AArch64::Bcc)
          invertBccCondition(MI);
        return true;
      }
    }
  }

  if (NeedSplit) {
    // Analyze the branch so we know how to update the successor lists.
    MachineBasicBlock *TBB, *FBB;
    SmallVector<MachineOperand, 2> Cond;
    TII->AnalyzeBranch(*MBB, TBB, FBB, Cond, false);

    MachineBasicBlock *NewBB = splitBlockBeforeInstr(MI);
    // No need for the branch to the next block. We're adding an unconditional
    // branch to the destination.
    int delta = TII->GetInstSizeInBytes(&MBB->back());
    BlockInfo[MBB->getNumber()].Size -= delta;
    MBB->back().eraseFromParent();
    // BlockInfo[SplitBB].Offset is wrong temporarily, fixed below

    // Update the successor lists according to the transformation to follow.
    // Do it here since if there's no split, no update is needed.
    MBB->replaceSuccessor(FBB, NewBB);
    NewBB->addSuccessor(FBB);
  }
  MachineBasicBlock *NextBB = &*std::next(MachineFunction::iterator(MBB));

  DEBUG(dbgs() << "  Insert B to BB#" << DestBB->getNumber()
               << ", invert condition and change dest. to BB#"
               << NextBB->getNumber() << "\n");

  // Insert a new conditional branch and a new unconditional branch.
  MachineInstrBuilder MIB = BuildMI(
      MBB, DebugLoc(), TII->get(getOppositeConditionOpcode(MI->getOpcode())))
                                .addOperand(MI->getOperand(0));
  if (MI->getOpcode() == AArch64::TBZW || MI->getOpcode() == AArch64::TBNZW ||
      MI->getOpcode() == AArch64::TBZX || MI->getOpcode() == AArch64::TBNZX)
    MIB.addOperand(MI->getOperand(1));
  if (MI->getOpcode() == AArch64::Bcc)
    invertBccCondition(MIB);
  MIB.addMBB(NextBB);
  BlockInfo[MBB->getNumber()].Size += TII->GetInstSizeInBytes(&MBB->back());
  BuildMI(MBB, DebugLoc(), TII->get(AArch64::B)).addMBB(DestBB);
  BlockInfo[MBB->getNumber()].Size += TII->GetInstSizeInBytes(&MBB->back());

  // Remove the old conditional branch.  It may or may not still be in MBB.
  BlockInfo[MI->getParent()->getNumber()].Size -= TII->GetInstSizeInBytes(MI);
  MI->eraseFromParent();

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
    MachineInstr *MI = MBB.getFirstTerminator();
    if (isConditionalBranch(MI->getOpcode()) &&
        !isBlockInRange(MI, getDestBlock(MI),
                        getBranchDisplacementBits(MI->getOpcode()))) {
      fixupConditionalBranch(MI);
      ++NumRelaxed;
      Changed = true;
    }
  }
  return Changed;
}

bool AArch64BranchRelaxation::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;

  // If the pass is disabled, just bail early.
  if (!BranchRelaxation)
    return false;

  DEBUG(dbgs() << "***** AArch64BranchRelaxation *****\n");

  TII = (const AArch64InstrInfo *)MF->getSubtarget().getInstrInfo();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Do the initial scan of the function, building up information about the
  // sizes of each block.
  scanFunction();

  DEBUG(dbgs() << "  Basic blocks before relaxation\n");
  DEBUG(dumpBBs());

  bool MadeChange = false;
  while (relaxBranchInstructions())
    MadeChange = true;

  // After a while, this might be made debug-only, but it is not expensive.
  verify();

  DEBUG(dbgs() << "  Basic blocks after relaxation\n");
  DEBUG(dbgs() << '\n'; dumpBBs());

  BlockInfo.clear();

  return MadeChange;
}

/// Returns an instance of the AArch64 Branch Relaxation pass.
FunctionPass *llvm::createAArch64BranchRelaxation() {
  return new AArch64BranchRelaxation();
}
