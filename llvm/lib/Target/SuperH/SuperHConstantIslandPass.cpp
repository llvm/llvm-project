//===- SuperHConstantIslandPass.cpp - SuperH constant islands -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that splits the constant pool up into 'islands'
// which are scattered through-out the function.  This is required due to the
// limited pc-relative displacements that SuperH has.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHInstPrinter.h"
#include "SuperHBasicBlockInfo.h"
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSubtarget.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include <memory>
using namespace llvm;

#define DEBUG_TYPE "sh-cp-islands"

#define SH_CP_ISLANDS_OPT_NAME \
  "SH constant island placement and branch shortening pass"

static cl::opt<unsigned>
CPMaxIteration("sh-constant-island-max-iteration", cl::Hidden, cl::init(30),
          cl::desc("The max number of iteration for converge"));

namespace {

  /// SuperHConstantIslands - Due to limited PC-relative displacements, SH
  /// requires constant pool entries to be scattered among the instructions
  /// inside a function.  To do this, it completely ignores the normal LLVM
  /// constant pool; instead, it places constants wherever it feels like with
  /// special instructions.
  ///
  /// The terminology used in this pass includes:
  ///   Islands - Clumps of constants placed in the function.
  ///   Water   - Potential places where an island could be formed.
  ///   CPE     - A constant pool entry that has been placed somewhere, which
  ///             tracks a list of users.
  class SuperHConstantIslands : public MachineFunctionPass {
  public:
    static char ID;

  private:
    std::unique_ptr<SuperHBasicBlockUtils> BBUtils = nullptr;

    /// WaterList - A sorted list of basic blocks where islands could be placed
    /// (i.e. blocks that don't fall through to the following block, due
    /// to a return, unreachable, or unconditional branch).
    std::vector<MachineBasicBlock*> WaterList;

    /// NewWaterList - The subset of WaterList that was created since the
    /// previous iteration by inserting unconditional branches.
    SmallPtrSet<MachineBasicBlock *, 4> NewWaterList;

    using water_iterator = std::vector<MachineBasicBlock *>::iterator;

    /// CPUser - One user of a constant pool, keeping the machine instruction
    /// pointer, the constant pool being referenced, and the max displacement
    /// allowed from the instruction to the CP.  The HighWaterMark records the
    /// highest basic block where a new CPEntry can be placed.  To ensure this
    /// pass terminates, the CP entries are initially placed at the end of the
    /// function and then move monotonically to lower addresses.  The
    /// exception to this rule is when the current CP entry for a particular
    /// CPUser is out of range, but there is another CP entry for the same
    /// constant value in range.  We want to use the existing in-range CP
    /// entry, but if it later moves out of range, the search for new water
    /// should resume where it left off.  The HighWaterMark is used to record
    /// that point.
    struct CPUser {
      MachineInstr *MI;
      MachineInstr *CPEMI;
      MachineBasicBlock *HighWaterMark;
      unsigned MaxDisp;
      bool NegOk;
      bool IsSoImm;
      bool KnownAlignment = false;

      CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned maxdisp,
             bool neg, bool soimm)
        : MI(mi), CPEMI(cpemi), MaxDisp(maxdisp), NegOk(neg), IsSoImm(soimm) {
        HighWaterMark = CPEMI->getParent();
      }

      /// getMaxDisp - Returns the maximum displacement supported by MI.
      /// Correct for unknown alignment.
      /// Conservatively subtract 2 bytes to handle weird alignment effects.
      unsigned getMaxDisp() const {
        return (KnownAlignment ? MaxDisp : MaxDisp - 2) - 2;
      }
    };

    /// CPUsers - Keep track of all of the machine instructions that use various
    /// constant pools and their max displacement.
    std::vector<CPUser> CPUsers;

    /// CPEntry - One per constant pool entry, keeping the machine instruction
    /// pointer, the constpool index, and the number of CPUser's which
    /// reference this entry.
    struct CPEntry {
      MachineInstr *CPEMI;
      unsigned CPI;
      unsigned RefCount;

      CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
        : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
    };

    /// CPEntries - Keep track of all of the constant pool entry machine
    /// instructions. For each original constpool index (i.e. those that existed
    /// upon entry to this pass), it keeps a vector of entries.  Original
    /// elements are cloned as we go along; the clones are put in the vector of
    /// the original element, but have distinct CPIs.
    ///
    /// The first half of CPEntries contains generic constants, the second half
    /// contains jump tables. Use getCombinedIndex on a generic CPEMI to look up
    /// which vector it will be in here.
    std::vector<std::vector<CPEntry>> CPEntries;

    /// ImmBranch - One per immediate branch, keeping the machine instruction
    /// pointer, conditional or unconditional, the max displacement,
    /// and (if isCond is true) the corresponding unconditional branch
    /// opcode.
    struct ImmBranch {
      MachineInstr *MI;
      unsigned MaxDisp : 12;

      ImmBranch(MachineInstr *mi, unsigned maxdisp)
        : MI(mi), MaxDisp(maxdisp) {}
    };

    /// ImmBranches - Keep track of all the immediate branch instructions.
    std::vector<ImmBranch> ImmBranches;

    MachineFunction *MF;
    MachineConstantPool *MCP;
    const SuperHInstrInfo *TII;
    const SuperHSubtarget *STI;
    SuperHMachineFunctionInfo *SFI;
    MachineDominatorTree *DT = nullptr;
    bool isPIC;

  public:

    SuperHConstantIslands() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<MachineDominatorTreeWrapperPass>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().setNoVRegs();
    }

    StringRef getPassName() const override {
      return SH_CP_ISLANDS_OPT_NAME;
    }

  private:
    bool BBHasFallthrough(MachineBasicBlock *MBB);
    void doInitialConstPlacement(std::vector<MachineInstr *> &CPEMIs);
    CPEntry *findConstPoolEntry(unsigned CPI, const MachineInstr *CPEMI);
    Align getCPEAlign(const MachineInstr *CPEMI);
    void initializeFunctionInfo(const std::vector<MachineInstr*> &CPEMIs);
    MachineBasicBlock *splitBlockBeforeInstr(MachineInstr *MI);
    void updateForInsertedWaterBlock(MachineBasicBlock *NewBB);
    bool decrementCPEReferenceCount(unsigned CPI, MachineInstr* CPEMI);
    unsigned getCombinedIndex(const MachineInstr *CPEMI);
    int findInRangeCPEntry(CPUser& U, unsigned UserOffset);
    bool findAvailableWater(CPUser&U, unsigned UserOffset,
                            water_iterator &WaterIter, bool CloserWater);
    void createNewWater(unsigned CPUserIndex, unsigned UserOffset,
                        MachineBasicBlock *&NewMBB);
    bool handleConstantPoolUser(unsigned CPUserIndex, bool CloserWater);
    void removeDeadCPEMI(MachineInstr *CPEMI);
    bool removeUnusedCPEntries();
    bool isCPEntryInRange(MachineInstr *MI, unsigned UserOffset,
                          MachineInstr *CPEMI, unsigned Disp, bool NegOk,
                          bool DoDump = false);
    bool isWaterInRange(unsigned UserOffset, MachineBasicBlock *Water,
                        CPUser &U, unsigned &Growth);
    bool fixupImmediateBr(ImmBranch &Br);

    unsigned getUserOffset(CPUser&) const;
    void verify();

    bool isOffsetInRange(unsigned UserOffset, unsigned TrialOffset,
                         unsigned Disp, bool NegativeOK, bool IsSoImm = false);
    bool isOffsetInRange(unsigned UserOffset, unsigned TrialOffset,
                         const CPUser &U) {
      return isOffsetInRange(UserOffset, TrialOffset,
                             U.getMaxDisp(), U.NegOk, U.IsSoImm);
    }

  };

} // end anonymous namespace

char SuperHConstantIslands::ID = 0;




//===----------------------------------------------------------------------===//
//                                  HELPERS
//===----------------------------------------------------------------------===//

// Align blocks where the previous block does not fall through. This may add
// extra NOP's but they will not be executed. It uses the PrefLoopAlignment as a
// measure of how much to align, and only runs at CodeGenOptLevel::Aggressive.
static bool AlignBlocks(MachineFunction *MF, const SuperHSubtarget *STI) {
  if (MF->getTarget().getOptLevel() != CodeGenOptLevel::Aggressive ||
      MF->getFunction().hasOptSize())
    return false;

  auto *TLI = STI->getTargetLowering();
  const Align Alignment = TLI->getPrefLoopAlignment();
  if (Alignment < 4)
    return false;

  bool Changed = false;
  bool PrevCanFallthrough = true;
  for (auto &MBB : *MF) {
    if (!PrevCanFallthrough) {
      Changed = true;
      MBB.setAlignment(Alignment);
    }
  }

  return Changed;
}

/// CompareMBBNumbers - Little predicate function to sort the WaterList by MBB
/// ID.
static bool CompareMBBNumbers(const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
  return LHS->getNumber() < RHS->getNumber();
}

/// getDispRange - Returns the maximum displacement that can fit in
/// the specific unconditional branch instruction.
static inline unsigned getDispRange(int Opc) {
  return ((1<<8)-1)*4;
}

/// isOffsetInRange - Checks whether UserOffset (the location of a constant pool
/// reference) is within MaxDisp of TrialOffset (a proposed location of a
/// constant pool entry).
/// UserOffset is computed by getUserOffset above to include PC adjustments. If
/// the mod 4 alignment of UserOffset is not known, the uncertainty must be
/// subtracted from MaxDisp instead. CPUser::getMaxDisp() does that.
bool SuperHConstantIslands::isOffsetInRange(unsigned UserOffset,
                                         unsigned TrialOffset, unsigned MaxDisp,
                                         bool NegativeOK, bool IsSoImm) {
  if (UserOffset <= TrialOffset) {
    // User before the Trial.
    if (TrialOffset - UserOffset <= MaxDisp)
      return true;
    // FIXME: Make use full range of soimm values.
  } else if (NegativeOK) {
    if (UserOffset - TrialOffset <= MaxDisp)
      return true;
    // FIXME: Make use full range of soimm values.
  }
  return false;
}

/// BBHasFallthrough - Return true if the specified basic block can fallthrough
/// into the block immediately after it.
bool SuperHConstantIslands::BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB->getIterator();
  // Can't fall off end of function.
  if (std::next(MBBI) == MBB->getParent()->end())
    return false;

  MachineBasicBlock *NextBB = &*std::next(MBBI);
  if (!MBB->isSuccessor(NextBB))
    return false;

  // Try to analyze the end of the block. A potential fallthrough may already
  // have an unconditional branch for whatever reason.
  MachineBasicBlock *TBB, *FBB;
  SmallVector<MachineOperand, 4> Cond;
  bool TooDifficult = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  return TooDifficult || FBB == nullptr;
}

/// getCPEAlign - Returns the required alignment of the constant pool entry
/// represented by CPEMI.
Align SuperHConstantIslands::getCPEAlign(const MachineInstr *CPEMI) {
  unsigned CPI = getCombinedIndex(CPEMI);
  assert(CPI < MCP->getConstants().size() && "Invalid constant pool index.");
  return MCP->getConstants()[CPI].getAlign();
}

/// isWaterInRange - Returns true if a CPE placed after the specified
/// Water (a basic block) will be in range for the specific MI.
///
/// Compute how much the function will grow by inserting a CPE after Water.
bool SuperHConstantIslands::isWaterInRange(unsigned UserOffset,
                                        MachineBasicBlock* Water, CPUser &U,
                                        unsigned &Growth) {
  BBInfoVector &BBInfo = BBUtils->getBBInfo();
  const Align CPEAlign = getCPEAlign(U.CPEMI);
  const unsigned CPEOffset = BBInfo[Water->getNumber()].postOffset(CPEAlign);
  unsigned NextBlockOffset;
  Align NextBlockAlignment;
  MachineFunction::const_iterator NextBlock = Water->getIterator();
  if (++NextBlock == MF->end()) {
    NextBlockOffset = BBInfo[Water->getNumber()].postOffset();
  } else {
    NextBlockOffset = BBInfo[NextBlock->getNumber()].Offset;
    NextBlockAlignment = NextBlock->getAlignment();
  }
  unsigned Size = U.CPEMI->getOperand(2).getImm();
  unsigned CPEEnd = CPEOffset + Size;

  // The CPE may be able to hide in the alignment padding before the next
  // block. It may also cause more padding to be required if it is more aligned
  // that the next block.
  if (CPEEnd > NextBlockOffset) {
    Growth = CPEEnd - NextBlockOffset;
    // Compute the padding that would go at the end of the CPE to align the next
    // block.
    Growth += offsetToAlignment(CPEEnd, NextBlockAlignment);

    // If the CPE is to be inserted before the instruction, that will raise
    // the offset of the instruction. Also account for unknown alignment padding
    // in blocks between CPE and the user.
    if (CPEOffset < UserOffset)
      UserOffset += Growth + UnknownPadding(MF->getAlignment(), Log2(CPEAlign));
  } else
    // CPE fits in existing padding.
    Growth = 0;

  return isOffsetInRange(UserOffset, CPEOffset, U);
}

/// isCPEntryInRange - Returns true if the distance between specific MI and
/// specific ConstPool entry instruction can fit in MI's displacement field.
bool SuperHConstantIslands::isCPEntryInRange(MachineInstr *MI, unsigned UserOffset,
                                      MachineInstr *CPEMI, unsigned MaxDisp,
                                      bool NegOk, bool DoDump) {
  unsigned CPEOffset = BBUtils->getOffsetOf(CPEMI);

  if (DoDump) {
    LLVM_DEBUG({
        BBInfoVector &BBInfo = BBUtils->getBBInfo();
      unsigned Block = MI->getParent()->getNumber();
      const BasicBlockInfo &BBI = BBInfo[Block];
      dbgs() << "User of CPE#" << CPEMI->getOperand(0).getImm()
             << " max delta=" << MaxDisp
             << format(" insn address=%#x", UserOffset) << " in "
             << printMBBReference(*MI->getParent()) << ": "
             << format("%#x-%x\t", BBI.Offset, BBI.postOffset()) << *MI
             << format("CPE address=%#x offset=%+d: ", CPEOffset,
                       int(CPEOffset - UserOffset));
    });
  }

  return isOffsetInRange(UserOffset, CPEOffset, MaxDisp, NegOk);
}

/// getUserOffset - Compute the offset of U.MI as seen by the hardware
/// displacement computation.  Update U.KnownAlignment to match its current
/// basic block location.
unsigned SuperHConstantIslands::getUserOffset(CPUser &U) const {
  unsigned UserOffset = BBUtils->getOffsetOf(U.MI);

  SmallVectorImpl<BasicBlockInfo> &BBInfo = BBUtils->getBBInfo();
  const BasicBlockInfo &BBI = BBInfo[U.MI->getParent()->getNumber()];
  unsigned KnownBits = BBI.internalKnownBits();

  // The value read from PC is offset from the actual instruction address.
  UserOffset += 4;

  // Because of inline assembly, we may not know the alignment (mod 4) of U.MI.
  // Make sure U.getMaxDisp() returns a constrained range.
  U.KnownAlignment = (KnownBits >= 2);
  return UserOffset;
}




//===----------------------------------------------------------------------===//
//                          BASE IMPLEMENTATION
//===----------------------------------------------------------------------===//

bool SuperHConstantIslands::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MCP = MF->getConstantPool();
  BBUtils = std::make_unique<SuperHBasicBlockUtils>(mf);

  LLVM_DEBUG(dbgs() << "***** SuperHConstantIslands: "
                    << MCP->getConstants().size() << " CP entries, aligned to "
                    << MCP->getConstantPoolAlign().value() << " bytes *****\n");

  STI = &MF->getSubtarget<SuperHSubtarget>();
  TII = STI->getInstrInfo();
  isPIC = STI->getTargetLowering()->isPositionIndependent();
  SFI = MF->getInfo<SuperHMachineFunctionInfo>();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  bool MadeChange = false;

  // Align any non-fallthrough blocks
  MadeChange |= AlignBlocks(MF, STI);

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr*> CPEMIs;
  if (!MCP->isEmpty())
    doInitialConstPlacement(CPEMIs);


  // Iteratively place constant pool entries and fix up branches until there
  // is no change.
  unsigned NoCPIters = 0, NoBRIters = 0;
  while (true) {
    LLVM_DEBUG(dbgs() << "Beginning CP iteration #" << NoCPIters << " with " <<
                         CPUsers.size() << " entries..." << '\n');
    bool CPChange = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i)
      // For most inputs, it converges in no more than 5 iterations.
      // If it doesn't end in 10, the input may have huge BB or many CPEs.
      // In this case, we will try different heuristics.
      CPChange |= handleConstantPoolUser(i, NoCPIters >= CPMaxIteration / 2);
    if (CPChange && ++NoCPIters > CPMaxIteration)
      report_fatal_error("Constant Island pass failed to converge!");

    // Clear NewWaterList now.  If we split a block for branches, it should
    // appear as "new water" for the next iteration of constant pool placement.
    NewWaterList.clear();

    LLVM_DEBUG(dbgs() << "Beginning BR iteration #" << NoBRIters << " with " <<
                         ImmBranches.size() << " entries..." << '\n');
    bool BRChange = false;
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i) {
      // Note: fixupImmediateBr can append to ImmBranches.
      BRChange |= fixupImmediateBr(ImmBranches[i]);
    }
    if (BRChange && ++NoBRIters > 30)
      report_fatal_error("Branch Fix Up pass failed to converge!");

    if (!CPChange && !BRChange)
      break;
    MadeChange = true;
  }

  BBUtils->clear();
  CPUsers.clear();
  CPEntries.clear();
  WaterList.clear();
  ImmBranches.clear();
  return MadeChange;
}

/// initializeFunctionInfo - Do the initial scan of the function, building up
/// information about the sizes of each block, the location of all the water,
/// and finding all of the constant pool users.
void SuperHConstantIslands::
initializeFunctionInfo(const std::vector<MachineInstr*> &CPEMIs) {

  BBUtils->computeAllBlockSizes();
  BBInfoVector &BBInfo = BBUtils->getBBInfo();

  // The known bits of the entry block offset are determined by the function
  // alignment.
  BBInfo.front().KnownBits = Log2(MF->getAlignment());

  // Compute block offsets and known bits.
  BBUtils->adjustBBOffsetsAfter(&MF->front());

  // Now go back through the instructions and build up our data structures.
  for (MachineBasicBlock &MBB : *MF) {

    // If this block doesn't fall through into the next MBB, then this is
    // 'water' that a constant pool island could be placed.
    if (!BBHasFallthrough(&MBB))
      WaterList.push_back(&MBB);

    for (MachineInstr &I : MBB) {
      if (I.isDebugInstr())
        continue;

      unsigned Opc = I.getOpcode();
      if (I.isBranch()) {
        unsigned Bits = 8;
        unsigned Scale = 4;
        switch (Opc) {
        default:
          continue;

        case SH::BF:
        case SH::BFS:
        case SH::BT:
        case SH::BTS:
          Bits = 8;
          Scale = 4;
          break;
        case SH::BRA:
        case SH::BSR:
          Bits = 12;
          Scale = 4;
          break;
        }


        // Record this immediate branch.
        unsigned MaxOffs = ((1 << (Bits-1))-1) * Scale;
        ImmBranches.push_back(ImmBranch(&I, MaxOffs));
      }

      if (Opc == SH::CONSTPOOL_ENTRY)
        continue;

      // Scan the instructions for constant pool operands.
      for (unsigned op = 0, e = I.getNumOperands(); op != e; ++op) {
        if (I.getOperand(op).isCPI()) {
          bool NegOk = false;
          bool IsSoImm = false;

          unsigned CPI = I.getOperand(op).getIndex();
          MachineInstr *CPEMI = CPEMIs[CPI];
          unsigned MaxOffs = ((1 << 8)-1) * 4;
          CPUsers.push_back(CPUser(&I, CPEMI, MaxOffs, NegOk, IsSoImm));

          // Increment corresponding CPEntry reference count.
          CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
          assert(CPE && "Cannot find a corresponding CPEntry!");
          CPE->RefCount++;

          // Instructions can only use one CP entry, don't bother scanning the
          // rest of the operands.
          break;

        }
      }
    }
  }
}




//===----------------------------------------------------------------------===//
//                            CONSTPOOL MANAGMENT
//===----------------------------------------------------------------------===//

/// removeDeadCPEMI - Remove a dead constant pool entry instruction. Update
/// sizes and offsets of impacted basic blocks.
void SuperHConstantIslands::removeDeadCPEMI(MachineInstr *CPEMI) {
  MachineBasicBlock *CPEBB = CPEMI->getParent();
  unsigned Size = CPEMI->getOperand(2).getImm();
  CPEMI->eraseFromParent();
  BBInfoVector &BBInfo = BBUtils->getBBInfo();
  BBUtils->adjustBBSize(CPEBB, -Size);
  // All succeeding offsets have the current size value added in, fix this.
  if (CPEBB->empty()) {
    BBInfo[CPEBB->getNumber()].Size = 0;

    // This block no longer needs to be aligned.
    CPEBB->setAlignment(Align(1));
  } else {
    // Entries are sorted by descending alignment, so realign from the front.
    CPEBB->setAlignment(getCPEAlign(&*CPEBB->begin()));
  }

  BBUtils->adjustBBOffsetsAfter(CPEBB);
}

/// removeUnusedCPEntries - Remove constant pool entries whose refcounts
/// are zero.
bool SuperHConstantIslands::removeUnusedCPEntries() {
  unsigned MadeChange = false;
  for (std::vector<CPEntry> &CPEs : CPEntries) {
    for (CPEntry &CPE : CPEs) {
      if (CPE.RefCount == 0 && CPE.CPEMI) {
        removeDeadCPEMI(CPE.CPEMI);
        CPE.CPEMI = nullptr;
        MadeChange = true;
      }
    }
  }
  return MadeChange;
}

/// decrementCPEReferenceCount - find the constant pool entry with index CPI
/// and instruction CPEMI, and decrement its refcount.  If the refcount
/// becomes 0 remove the entry and instruction.  Returns true if we removed
/// the entry, false if we didn't.
bool SuperHConstantIslands::decrementCPEReferenceCount(unsigned CPI,
                                                    MachineInstr *CPEMI) {
  // Find the old entry. Eliminate it if it is no longer used.
  CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
  assert(CPE && "Unexpected!");
  if (--CPE->RefCount == 0) {
    removeDeadCPEMI(CPEMI);
    CPE->CPEMI = nullptr;
    return true;
  }
  return false;
}

unsigned SuperHConstantIslands::getCombinedIndex(const MachineInstr *CPEMI) {
  if (CPEMI->getOperand(1).isCPI())
    return CPEMI->getOperand(1).getIndex();

  return 0;
}
/// LookForCPEntryInRange - see if the currently referenced CPE is in range;
/// if not, see if an in-range clone of the CPE is in range, and if so,
/// change the data structures so the user references the clone.  Returns:
/// 0 = no existing entry found
/// 1 = entry found, and there were no code insertions or deletions
/// 2 = entry found, and there were code insertions or deletions
int SuperHConstantIslands::findInRangeCPEntry(CPUser& U, unsigned UserOffset) {
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;

  // Check to see if the CPE is already in-range.
  if (isCPEntryInRange(UserMI, UserOffset, CPEMI, U.getMaxDisp(), U.NegOk,
                       true)) {
    LLVM_DEBUG(dbgs() << "In range\n");
    return 1;
  }

  // No.  Look for previously created clones of the CPE that are in range.
  unsigned CPI = getCombinedIndex(CPEMI);
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  for (CPEntry &CPE : CPEs) {
    // We already tried this one
    if (CPE.CPEMI == CPEMI)
      continue;
    // Removing CPEs can leave empty entries, skip
    if (CPE.CPEMI == nullptr)
      continue;
    if (isCPEntryInRange(UserMI, UserOffset, CPE.CPEMI, U.getMaxDisp(),
                         U.NegOk)) {
      LLVM_DEBUG(dbgs() << "Replacing CPE#" << CPI << " with CPE#" << CPE.CPI
                        << "\n");
      // Point the CPUser node to the replacement
      U.CPEMI = CPE.CPEMI;
      // Change the CPI in the instruction operand to refer to the clone.
      for (MachineOperand &MO : UserMI->operands())
        if (MO.isCPI()) {
          MO.setIndex(CPE.CPI);
          break;
        }
      // Adjust the refcount of the clone...
      CPE.RefCount++;
      // ...and the original.  If we didn't remove the old entry, none of the
      // addresses changed, so we don't need another pass.
      return decrementCPEReferenceCount(CPI, CPEMI) ? 2 : 1;
    }
  }
  return 0;
}

/// handleConstantPoolUser - Analyze the specified user, checking to see if it
/// is out-of-range.  If so, pick up the constant pool value and move it some
/// place in-range.  Return true if we changed any addresses (thus must run
/// another pass of branch lengthening), false otherwise.
bool SuperHConstantIslands::handleConstantPoolUser(unsigned CPUserIndex,
                                                bool CloserWater) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;
  unsigned CPI = getCombinedIndex(CPEMI);
  unsigned Size = CPEMI->getOperand(2).getImm();
  // Compute this only once, it's expensive.
  unsigned UserOffset = getUserOffset(U);

  // See if the current entry is within range, or there is a clone of it
  // in range.
  int result = findInRangeCPEntry(U, UserOffset);
  if (result==1) return false;
  else if (result==2) return true;

  // No existing clone of this CPE is within range.
  // We will be generating a new clone.  Get a UID for it.
  unsigned ID = SFI->createConstIndex();

  // Look for water where we can place this CPE.
  MachineBasicBlock *NewIsland = MF->CreateMachineBasicBlock();
  MachineBasicBlock *NewMBB;
  water_iterator IP;
  if (findAvailableWater(U, UserOffset, IP, CloserWater)) {
    LLVM_DEBUG(dbgs() << "Found water in range\n");
    MachineBasicBlock *WaterBB = *IP;

    // If the original WaterList entry was "new water" on this iteration,
    // propagate that to the new island.  This is just keeping NewWaterList
    // updated to match the WaterList, which will be updated below.
    if (NewWaterList.erase(WaterBB))
      NewWaterList.insert(NewIsland);

    // The new CPE goes before the following block (NewMBB).
    NewMBB = &*++WaterBB->getIterator();
  } else {
    // No water found.
    LLVM_DEBUG(dbgs() << "No water found\n");
    createNewWater(CPUserIndex, UserOffset, NewMBB);

    // splitBlockBeforeInstr adds to WaterList, which is important when it is
    // called while handling branches so that the water will be seen on the
    // next iteration for constant pools, but in this context, we don't want
    // it.  Check for this so it will be removed from the WaterList.
    // Also remove any entry from NewWaterList.
    MachineBasicBlock *WaterBB = &*--NewMBB->getIterator();
    IP = find(WaterList, WaterBB);
    if (IP != WaterList.end())
      NewWaterList.erase(WaterBB);

    // We are adding new water.  Update NewWaterList.
    NewWaterList.insert(NewIsland);
  }
  // Always align the new block because CP entries can be smaller than 4
  // bytes. Be careful not to decrease the existing alignment, e.g. NewMBB may
  // be an already aligned constant pool block.
  const Align Alignment = Align(4);
  if (NewMBB->getAlignment() < Alignment)
    NewMBB->setAlignment(Alignment);

  // Remove the original WaterList entry; we want subsequent insertions in
  // this vicinity to go after the one we're about to insert.  This
  // considerably reduces the number of times we have to move the same CPE
  // more than once and is also important to ensure the algorithm terminates.
  if (IP != WaterList.end())
    WaterList.erase(IP);

  // Okay, we know we can put an island before NewMBB now, do it!
  MF->insert(NewMBB->getIterator(), NewIsland);

  // Update internal data structures to account for the newly inserted MBB.
  updateForInsertedWaterBlock(NewIsland);

  // Now that we have an island to add the CPE to, clone the original CPE and
  // add it to the island.
  U.HighWaterMark = NewIsland;
  U.CPEMI = BuildMI(NewIsland, DebugLoc(), CPEMI->getDesc())
                .addImm(ID)
                .add(CPEMI->getOperand(1))
                .addImm(Size);
  CPEntries[CPI].push_back(CPEntry(U.CPEMI, ID, 1));

  // Decrement the old entry, and remove it if refcount becomes 0.
  decrementCPEReferenceCount(CPI, CPEMI);

  // Mark the basic block as aligned as required by the const-pool entry.
  NewIsland->setAlignment(getCPEAlign(U.CPEMI));

  // Increase the size of the island block to account for the new entry.
  BBUtils->adjustBBSize(NewIsland, Size);
  BBUtils->adjustBBOffsetsAfter(&*--NewIsland->getIterator());

  // Finally, change the CPI in the instruction operand to be ID.
  for (MachineOperand &MO : UserMI->operands())
    if (MO.isCPI()) {
      MO.setIndex(ID);
      break;
    }

  LLVM_DEBUG(
      dbgs() << "  Moved CPE to #" << ID << " CPI=" << CPI
             << format(" offset=%#x\n",
                       BBUtils->getBBInfo()[NewIsland->getNumber()].Offset));

  return true;
}

/// findConstPoolEntry - Given the constpool index and CONSTPOOL_ENTRY MI,
/// look up the corresponding CPEntry.
SuperHConstantIslands::CPEntry *
SuperHConstantIslands::findConstPoolEntry(unsigned CPI,
                                       const MachineInstr *CPEMI) {
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  // Number of entries per constpool index should be small, just do a
  // linear search.
  for (CPEntry &CPE : CPEs)
    if (CPE.CPEMI == CPEMI)
      return &CPE;
  return nullptr;
}

/// findAvailableWater - Look for an existing entry in the WaterList in which
/// we can place the CPE referenced from U so it's within range of U's MI.
/// Returns true if found, false if not.  If it returns true, WaterIter
/// is set to the WaterList entry.  For Thumb, prefer water that will not
/// introduce padding to water that will.  To ensure that this pass
/// terminates, the CPE location for a particular CPUser is only allowed to
/// move to a lower address, so search backward from the end of the list and
/// prefer the first water that is in range.
bool SuperHConstantIslands::findAvailableWater(CPUser &U, unsigned UserOffset,
                                            water_iterator &WaterIter,
                                            bool CloserWater) {
  if (WaterList.empty())
    return false;

  unsigned BestGrowth = ~0u;
  // The nearest water without splitting the UserBB is right after it.
  // If the distance is still large (we have a big BB), then we need to split it
  // if we don't converge after certain iterations. This helps the following
  // situation to converge:
  //   BB0:
  //      Big BB
  //   BB1:
  //      Constant Pool
  // When a CP access is out of range, BB0 may be used as water. However,
  // inserting islands between BB0 and BB1 makes other accesses out of range.
  MachineBasicBlock *UserBB = U.MI->getParent();
  BBInfoVector &BBInfo = BBUtils->getBBInfo();
  const Align CPEAlign = getCPEAlign(U.CPEMI);
  unsigned MinNoSplitDisp = BBInfo[UserBB->getNumber()].postOffset(CPEAlign);
  if (CloserWater && MinNoSplitDisp > U.getMaxDisp() / 2)
    return false;
  for (water_iterator IP = std::prev(WaterList.end()), B = WaterList.begin();;
       --IP) {
    MachineBasicBlock* WaterBB = *IP;
    // Check if water is in range and is either at a lower address than the
    // current "high water mark" or a new water block that was created since
    // the previous iteration by inserting an unconditional branch.  In the
    // latter case, we want to allow resetting the high water mark back to
    // this new water since we haven't seen it before.  Inserting branches
    // should be relatively uncommon and when it does happen, we want to be
    // sure to take advantage of it for all the CPEs near that block, so that
    // we don't insert more branches than necessary.
    // When CloserWater is true, we try to find the lowest address after (or
    // equal to) user MI's BB no matter of padding growth.
    unsigned Growth;
    if (isWaterInRange(UserOffset, WaterBB, U, Growth) &&
        (WaterBB->getNumber() < U.HighWaterMark->getNumber() ||
         NewWaterList.count(WaterBB) || WaterBB == U.MI->getParent()) &&
        Growth < BestGrowth) {
      // This is the least amount of required padding seen so far.
      BestGrowth = Growth;
      WaterIter = IP;
      LLVM_DEBUG(dbgs() << "Found water after " << printMBBReference(*WaterBB)
                        << " Growth=" << Growth << '\n');

      if (CloserWater && WaterBB == U.MI->getParent())
        return true;
      // Keep looking unless it is perfect and we're not looking for the lowest
      // possible address.
      if (!CloserWater && BestGrowth == 0)
        return true;
    }
    if (IP == B)
      break;
  }
  return BestGrowth != ~0u;
}

/// createNewWater - No existing WaterList entry will work for
/// CPUsers[CPUserIndex], so create a place to put the CPE.  The end of the
/// block is used if in range, and the conditional branch munged so control
/// flow is correct.  Otherwise the block is split to create a hole with an
/// unconditional branch around it.  In either case NewMBB is set to a
/// block following which the new island can be inserted (the WaterList
/// is not adjusted).
void SuperHConstantIslands::createNewWater(unsigned CPUserIndex,
                                        unsigned UserOffset,
                                        MachineBasicBlock *&NewMBB) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;
  const Align CPEAlign = getCPEAlign(CPEMI);
  MachineBasicBlock *UserMBB = UserMI->getParent();
  BBInfoVector &BBInfo = BBUtils->getBBInfo();
  const BasicBlockInfo &UserBBI = BBInfo[UserMBB->getNumber()];

  // If the block does not end in an unconditional branch already, and if the
  // end of the block is within range, make new water there.  (The addition
  // below is for the unconditional branch we will be adding: 4 bytes on ARM +
  // Thumb2, 2 on Thumb1.
  if (BBHasFallthrough(UserMBB)) {
    // Size of branch to insert.
    unsigned Delta = 4;
    // Compute the offset where the CPE will begin.
    unsigned CPEOffset = UserBBI.postOffset(CPEAlign) + Delta;

    if (isOffsetInRange(UserOffset, CPEOffset, U)) {
      LLVM_DEBUG(dbgs() << "Split at end of " << printMBBReference(*UserMBB)
                        << format(", expected CPE offset %#x\n", CPEOffset));
      NewMBB = &*++UserMBB->getIterator();

      // Add an unconditional branch from UserMBB to fallthrough block.  Record
      // it for branch lengthening; this new branch will not get out of range,
      // but if the preceding conditional branch is out of range, the targets
      // will be exchanged, and the altered branch may be out of range, so the
      // machinery has to know about it.
      BuildMI(UserMBB, DebugLoc(), TII->get(SH::BRA))
          .addMBB(NewMBB);

      unsigned MaxDisp = getDispRange(SH::BRA);
      ImmBranches.push_back(ImmBranch(&UserMBB->back(), MaxDisp));
      BBUtils->computeBlockSize(UserMBB);
      BBUtils->adjustBBOffsetsAfter(UserMBB);
      return;
    }
  }

  // What a big block.  Find a place within the block to split it.  This is a
  // little tricky on Thumb1 since instructions are 2 bytes and constant pool
  // entries are 4 bytes: if instruction I references island CPE, and
  // instruction I+1 references CPE', it will not work well to put CPE as far
  // forward as possible, since then CPE' cannot immediately follow it (that
  // location is 2 bytes farther away from I+1 than CPE was from I) and we'd
  // need to create a new island.  So, we make a first guess, then walk through
  // the instructions between the one currently being looked at and the
  // possible insertion point, and make sure any other instructions that
  // reference CPEs will be able to use the same island area; if not, we back
  // up the insertion point.

  // Try to split the block so it's fully aligned.  Compute the latest split
  // point where we can add a 4-byte branch instruction, and then align to
  // Align which is the largest possible alignment in the function.
  const Align Align = MF->getAlignment();
  assert(Align >= CPEAlign && "Over-aligned constant pool entry");
  unsigned KnownBits = UserBBI.internalKnownBits();
  unsigned UPad = UnknownPadding(Align, KnownBits);
  unsigned BaseInsertOffset = UserOffset + U.getMaxDisp() - UPad;
  LLVM_DEBUG(dbgs() << format("Split in middle of big block before %#x",
                              BaseInsertOffset));

  // The 4 in the following is for the unconditional branch we'll be inserting
  // (allows for long branch on Thumb1).  Alignment of the island is handled
  // inside isOffsetInRange.
  BaseInsertOffset -= 4;

  LLVM_DEBUG(dbgs() << format(", adjusted to %#x", BaseInsertOffset)
                    << " la=" << Log2(Align) << " kb=" << KnownBits
                    << " up=" << UPad << '\n');

  unsigned EndInsertOffset = BaseInsertOffset + 4 + UPad +
    CPEMI->getOperand(2).getImm();
  MachineBasicBlock::iterator MI = UserMI;
  ++MI;
  unsigned CPUIndex = CPUserIndex+1;
  unsigned NumCPUsers = CPUsers.size();
  for (unsigned Offset = UserOffset + TII->getInstSizeInBytes(*UserMI);
       Offset < BaseInsertOffset;
       Offset += TII->getInstSizeInBytes(*MI), MI = std::next(MI)) {
    assert(MI != UserMBB->end() && "Fell off end of block");
    if (CPUIndex < NumCPUsers && CPUsers[CPUIndex].MI == &*MI) {
      CPUser &U = CPUsers[CPUIndex];
      if (!isOffsetInRange(Offset, EndInsertOffset, U)) {
        // Shift intertion point by one unit of alignment so it is within reach.
        BaseInsertOffset -= Align.value();
        EndInsertOffset -= Align.value();
      }
      // This is overly conservative, as we don't account for CPEMIs being
      // reused within the block, but it doesn't matter much.  Also assume CPEs
      // are added in order with alignment padding.  We may eventually be able
      // to pack the aligned CPEs better.
      EndInsertOffset += U.CPEMI->getOperand(2).getImm();
      CPUIndex++;
    }
  }

  --MI;

  // We really must not split an IT block.
  NewMBB = splitBlockBeforeInstr(&*MI);
}

/// updateForInsertedWaterBlock - When a block is newly inserted into the
/// machine function, it upsets all of the block numbers.  Renumber the blocks
/// and update the arrays that parallel this numbering.
void SuperHConstantIslands::updateForInsertedWaterBlock(MachineBasicBlock *NewBB) {
  // Renumber the MBB's to keep them consecutive.
  NewBB->getParent()->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBUtils->insert(NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add NewMBB as having
  // available water after it.
  water_iterator IP = llvm::lower_bound(WaterList, NewBB, CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}

/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *SuperHConstantIslands::splitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();

  // Collect liveness information at MI.
  LivePhysRegs LRs(*MF->getSubtarget().getRegisterInfo());
  LRs.addLiveOuts(*OrigBB);
  auto LivenessEnd = ++MachineBasicBlock::iterator(MI).getReverse();
  for (MachineInstr &LiveMI : make_range(OrigBB->rbegin(), LivenessEnd))
    LRs.stepBackward(LiveMI);

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
    MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = ++OrigBB->getIterator();
  MF->insert(MBBI, NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  BuildMI(OrigBB, DebugLoc(), TII->get(SH::BRA)).addMBB(NewBB);

  // Update the CFG.  All succs of OrigBB are now succs of NewBB.
  NewBB->transferSuccessors(OrigBB);

  // OrigBB branches to NewBB.
  OrigBB->addSuccessor(NewBB);

  // Update live-in information in the new block.
  MachineRegisterInfo &MRI = MF->getRegInfo();
  for (MCPhysReg L : LRs)
    if (!MRI.isReserved(L))
      NewBB->addLiveIn(L);

  // Update internal data structures to account for the newly inserted MBB.
  // This is almost the same as updateForInsertedWaterBlock, except that
  // the Water goes after OrigBB, not NewBB.
  MF->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBUtils->insert(NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add OrigMBB as having
  // available water after it (but not if it's already there, which happens
  // when splitting before a conditional branch that is followed by an
  // unconditional branch - in that case we want to insert NewBB).
  water_iterator IP = llvm::lower_bound(WaterList, OrigBB, CompareMBBNumbers);
  MachineBasicBlock* WaterBB = *IP;
  if (WaterBB == OrigBB)
    WaterList.insert(std::next(IP), NewBB);
  else
    WaterList.insert(IP, OrigBB);
  NewWaterList.insert(OrigBB);

  // Figure out how large the OrigBB is.  As the first half of the original
  // block, it cannot contain a tablejump.  The size includes
  // the new jump we added.  (It should be possible to do this without
  // recounting everything, but it's very confusing, and this is rarely
  // executed.)
  BBUtils->computeBlockSize(OrigBB);

  // Figure out how large the NewMBB is.  As the second half of the original
  // block, it may contain a tablejump.
  BBUtils->computeBlockSize(NewBB);

  // All BBOffsets following these blocks must be modified.
  BBUtils->adjustBBOffsetsAfter(OrigBB);

  return NewBB;
}




//===----------------------------------------------------------------------===//
//                            CONST PLACEMENT
//===----------------------------------------------------------------------===//

/// Perform the initial placement of the regular constant pool entries.
/// To start with, we put them all at the end of the function.
void
SuperHConstantIslands::doInitialConstPlacement(std::vector<MachineInstr *> &CPEMIs) {
  
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = MF->CreateMachineBasicBlock();
  MF->push_back(BB);

  // MachineConstantPool measures alignment in bytes.
  const Align MaxAlign = MCP->getConstantPoolAlign();
  const unsigned MaxLogAlign = Log2(MaxAlign);

  // Mark the basic block as required by the const-pool.
  BB->setAlignment(MaxAlign);

  // The function needs to be as aligned as the basic blocks. The linker may
  // move functions around based on their alignment.
  // Special case: halfword literals still need word alignment on the function.
  Align FuncAlign = MaxAlign;
  if (MaxAlign == 2)
    FuncAlign = Align(4);
  MF->ensureAlignment(FuncAlign);

  // Order the entries in BB by descending alignment.  That ensures correct
  // alignment of all entries as long as BB is sufficiently aligned.  Keep
  // track of the insertion point for each alignment.  We are going to bucket
  // sort the entries as they are created.
  SmallVector<MachineBasicBlock::iterator, 8> InsPoint(MaxLogAlign + 1,
                                                       BB->end());

  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs = MCP->getConstants();

  const DataLayout &TD = MF->getDataLayout();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = CPs[i].getSizeInBytes(TD);
    Align Alignment = CPs[i].getAlign();
    // Verify that all constant pool entries are a multiple of their alignment.
    // If not, we would have to pad them out so that instructions stay aligned.
    assert(isAligned(Alignment, Size) && "CP Entry not multiple of 4 bytes!");

    // Insert CONSTPOOL_ENTRY before entries with a smaller alignment.
    unsigned LogAlign = Log2(Alignment);
    MachineBasicBlock::iterator InsAt = InsPoint[LogAlign];
    MachineInstr *CPEMI =
      BuildMI(*BB, InsAt, DebugLoc(), TII->get(SH::CONSTPOOL_ENTRY))
        .addImm(i).addConstantPoolIndex(i).addImm(Size);
    CPEMIs.push_back(CPEMI);

    // Ensure that future entries with higher alignment get inserted before
    // CPEMI. This is bucket sort with iterators.
    for (unsigned a = LogAlign + 1; a <= MaxLogAlign; ++a)
      if (InsPoint[a] == InsAt)
        InsPoint[a] = CPEMI;

    // Add a new CPEntry, but no corresponding CPUser yet.
    CPEntries.emplace_back(1, CPEntry(CPEMI, i));
    LLVM_DEBUG(dbgs() << "Moved CPI#" << i << " to end of function, size = "
                      << Size << ", align = " << Alignment.value() << '\n');
  }
  LLVM_DEBUG(BB->dump());
}

/// fixupImmediateBr - Fix up an immediate branch whose destination is too far
/// away to fit in its displacement field.
bool SuperHConstantIslands::fixupImmediateBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock *DestBB = MI->getOperand(0).getMBB();

  // Check to see if the DestBB is already in-range.
  if (BBUtils->isBBInRange(MI, DestBB, Br.MaxDisp))
    return false;

  // Use BRA to implement far jump.
  Br.MaxDisp = (1 << 11);
  MI->setDesc(TII->get(SH::BRA));
  BBInfoVector &BBInfo = BBUtils->getBBInfo();
  BBInfo[MBB->getNumber()].Size += 2;
  BBUtils->adjustBBOffsetsAfter(MBB);
  return true;
}




//===----------------------------------------------------------------------===//
//                             INITIALIZATION
//===----------------------------------------------------------------------===//

/// createARMConstantIslandPass - returns an instance of the constpool
/// island pass.
FunctionPass *llvm::createSuperHConstantIslandPass() {
  return new SuperHConstantIslands();
}


INITIALIZE_PASS(SuperHConstantIslands, "sh-cp-islands", SH_CP_ISLANDS_OPT_NAME,
                false, false)
