//===-- X86FloatingPoint.cpp - Floating point Reg -> Stack converter ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which converts floating point instructions from
// pseudo registers into register stack instructions.  This pass uses live
// variable information to indicate where the FPn registers are used and their
// lifetimes.
//
// The x87 hardware tracks liveness of the stack registers, so it is necessary
// to implement exact liveness tracking between basic blocks. The CFG edges are
// partitioned into bundles where the same FP registers must be live in
// identical stack positions. Instructions are inserted at the end of each basic
// block to rearrange the live registers to match the outgoing bundle.
//
// This approach avoids splitting critical edges at the potential cost of more
// live register shuffling instructions when critical edges are present.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-codegen"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/InlineAsm.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumFXCH, "Number of fxch instructions inserted");
STATISTIC(NumFP  , "Number of floating point instructions");

namespace {
  struct FPS : public MachineFunctionPass {
    static char ID;
    FPS() : MachineFunctionPass(ID) {
      initializeEdgeBundlesPass(*PassRegistry::getPassRegistry());
      // This is really only to keep valgrind quiet.
      // The logic in isLive() is too much for it.
      memset(Stack, 0, sizeof(Stack));
      memset(RegMap, 0, sizeof(RegMap));
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<EdgeBundles>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const { return "X86 FP Stackifier"; }

  private:
    const TargetInstrInfo *TII; // Machine instruction info.

    // Two CFG edges are related if they leave the same block, or enter the same
    // block. The transitive closure of an edge under this relation is a
    // LiveBundle. It represents a set of CFG edges where the live FP stack
    // registers must be allocated identically in the x87 stack.
    //
    // A LiveBundle is usually all the edges leaving a block, or all the edges
    // entering a block, but it can contain more edges if critical edges are
    // present.
    //
    // The set of live FP registers in a LiveBundle is calculated by bundleCFG,
    // but the exact mapping of FP registers to stack slots is fixed later.
    struct LiveBundle {
      // Bit mask of live FP registers. Bit 0 = FP0, bit 1 = FP1, &c.
      unsigned Mask;

      // Number of pre-assigned live registers in FixStack. This is 0 when the
      // stack order has not yet been fixed.
      unsigned FixCount;

      // Assigned stack order for live-in registers.
      // FixStack[i] == getStackEntry(i) for all i < FixCount.
      unsigned char FixStack[8];

      LiveBundle() : Mask(0), FixCount(0) {}

      // Have the live registers been assigned a stack order yet?
      bool isFixed() const { return !Mask || FixCount; }
    };

    // Numbered LiveBundle structs. LiveBundles[0] is used for all CFG edges
    // with no live FP registers.
    SmallVector<LiveBundle, 8> LiveBundles;

    // The edge bundle analysis provides indices into the LiveBundles vector.
    EdgeBundles *Bundles;

    // Return a bitmask of FP registers in block's live-in list.
    unsigned calcLiveInMask(MachineBasicBlock *MBB) {
      unsigned Mask = 0;
      for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
           E = MBB->livein_end(); I != E; ++I) {
        unsigned Reg = *I - X86::FP0;
        if (Reg < 8)
          Mask |= 1 << Reg;
      }
      return Mask;
    }

    // Partition all the CFG edges into LiveBundles.
    void bundleCFG(MachineFunction &MF);

    MachineBasicBlock *MBB;     // Current basic block

    // The hardware keeps track of how many FP registers are live, so we have
    // to model that exactly. Usually, each live register corresponds to an
    // FP<n> register, but when dealing with calls, returns, and inline
    // assembly, it is sometimes necessary to have live scratch registers.
    unsigned Stack[8];          // FP<n> Registers in each stack slot...
    unsigned StackTop;          // The current top of the FP stack.

    enum {
      NumFPRegs = 16            // Including scratch pseudo-registers.
    };

    // For each live FP<n> register, point to its Stack[] entry.
    // The first entries correspond to FP0-FP6, the rest are scratch registers
    // used when we need slightly different live registers than what the
    // register allocator thinks.
    unsigned RegMap[NumFPRegs];

    // Pending fixed registers - Inline assembly needs FP registers to appear
    // in fixed stack slot positions. This is handled by copying FP registers
    // to ST registers before the instruction, and copying back after the
    // instruction.
    //
    // This is modeled with pending ST registers. NumPendingSTs is the number
    // of ST registers (ST0-STn) we are tracking. PendingST[n] points to an FP
    // register that holds the ST value. The ST registers are not moved into
    // place until immediately before the instruction that needs them.
    //
    // It can happen that we need an ST register to be live when no FP register
    // holds the value:
    //
    //   %ST0 = COPY %FP4<kill>
    //
    // When that happens, we allocate a scratch FP register to hold the ST
    // value. That means every register in PendingST must be live.

    unsigned NumPendingSTs;
    unsigned char PendingST[8];

    // Set up our stack model to match the incoming registers to MBB.
    void setupBlockStack();

    // Shuffle live registers to match the expectations of successor blocks.
    void finishBlockStack();

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    void dumpStack() const {
      dbgs() << "Stack contents:";
      for (unsigned i = 0; i != StackTop; ++i) {
        dbgs() << " FP" << Stack[i];
        assert(RegMap[Stack[i]] == i && "Stack[] doesn't match RegMap[]!");
      }
      for (unsigned i = 0; i != NumPendingSTs; ++i)
        dbgs() << ", ST" << i << " in FP" << unsigned(PendingST[i]);
      dbgs() << "\n";
    }
#endif

    /// getSlot - Return the stack slot number a particular register number is
    /// in.
    unsigned getSlot(unsigned RegNo) const {
      assert(RegNo < NumFPRegs && "Regno out of range!");
      return RegMap[RegNo];
    }

    /// isLive - Is RegNo currently live in the stack?
    bool isLive(unsigned RegNo) const {
      unsigned Slot = getSlot(RegNo);
      return Slot < StackTop && Stack[Slot] == RegNo;
    }

    /// getScratchReg - Return an FP register that is not currently in use.
    unsigned getScratchReg() {
      for (int i = NumFPRegs - 1; i >= 8; --i)
        if (!isLive(i))
          return i;
      llvm_unreachable("Ran out of scratch FP registers");
    }

    /// isScratchReg - Returns trus if RegNo is a scratch FP register.
    bool isScratchReg(unsigned RegNo) {
      return RegNo > 8 && RegNo < NumFPRegs;
    }

    /// getStackEntry - Return the X86::FP<n> register in register ST(i).
    unsigned getStackEntry(unsigned STi) const {
      if (STi >= StackTop)
        report_fatal_error("Access past stack top!");
      return Stack[StackTop-1-STi];
    }

    /// getSTReg - Return the X86::ST(i) register which contains the specified
    /// FP<RegNo> register.
    unsigned getSTReg(unsigned RegNo) const {
      return StackTop - 1 - getSlot(RegNo) + X86::ST0;
    }

    // pushReg - Push the specified FP<n> register onto the stack.
    void pushReg(unsigned Reg) {
      assert(Reg < NumFPRegs && "Register number out of range!");
      if (StackTop >= 8)
        report_fatal_error("Stack overflow!");
      Stack[StackTop] = Reg;
      RegMap[Reg] = StackTop++;
    }

    bool isAtTop(unsigned RegNo) const { return getSlot(RegNo) == StackTop-1; }
    void moveToTop(unsigned RegNo, MachineBasicBlock::iterator I) {
      DebugLoc dl = I == MBB->end() ? DebugLoc() : I->getDebugLoc();
      if (isAtTop(RegNo)) return;

      unsigned STReg = getSTReg(RegNo);
      unsigned RegOnTop = getStackEntry(0);

      // Swap the slots the regs are in.
      std::swap(RegMap[RegNo], RegMap[RegOnTop]);

      // Swap stack slot contents.
      if (RegMap[RegOnTop] >= StackTop)
        report_fatal_error("Access past stack top!");
      std::swap(Stack[RegMap[RegOnTop]], Stack[StackTop-1]);

      // Emit an fxch to update the runtime processors version of the state.
      BuildMI(*MBB, I, dl, TII->get(X86::XCH_F)).addReg(STReg);
      ++NumFXCH;
    }

    void duplicateToTop(unsigned RegNo, unsigned AsReg, MachineInstr *I) {
      DebugLoc dl = I == MBB->end() ? DebugLoc() : I->getDebugLoc();
      unsigned STReg = getSTReg(RegNo);
      pushReg(AsReg);   // New register on top of stack

      BuildMI(*MBB, I, dl, TII->get(X86::LD_Frr)).addReg(STReg);
    }

    /// duplicatePendingSTBeforeKill - The instruction at I is about to kill
    /// RegNo. If any PendingST registers still need the RegNo value, duplicate
    /// them to new scratch registers.
    void duplicatePendingSTBeforeKill(unsigned RegNo, MachineInstr *I) {
      for (unsigned i = 0; i != NumPendingSTs; ++i) {
        if (PendingST[i] != RegNo)
          continue;
        unsigned SR = getScratchReg();
        DEBUG(dbgs() << "Duplicating pending ST" << i
                     << " in FP" << RegNo << " to FP" << SR << '\n');
        duplicateToTop(RegNo, SR, I);
        PendingST[i] = SR;
      }
    }

    /// popStackAfter - Pop the current value off of the top of the FP stack
    /// after the specified instruction.
    void popStackAfter(MachineBasicBlock::iterator &I);

    /// freeStackSlotAfter - Free the specified register from the register
    /// stack, so that it is no longer in a register.  If the register is
    /// currently at the top of the stack, we just pop the current instruction,
    /// otherwise we store the current top-of-stack into the specified slot,
    /// then pop the top of stack.
    void freeStackSlotAfter(MachineBasicBlock::iterator &I, unsigned Reg);

    /// freeStackSlotBefore - Just the pop, no folding. Return the inserted
    /// instruction.
    MachineBasicBlock::iterator
    freeStackSlotBefore(MachineBasicBlock::iterator I, unsigned FPRegNo);

    /// Adjust the live registers to be the set in Mask.
    void adjustLiveRegs(unsigned Mask, MachineBasicBlock::iterator I);

    /// Shuffle the top FixCount stack entries such that FP reg FixStack[0] is
    /// st(0), FP reg FixStack[1] is st(1) etc.
    void shuffleStackTop(const unsigned char *FixStack, unsigned FixCount,
                         MachineBasicBlock::iterator I);

    bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

    void handleZeroArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFP(MachineBasicBlock::iterator &I);
    void handleOneArgFPRW(MachineBasicBlock::iterator &I);
    void handleTwoArgFP(MachineBasicBlock::iterator &I);
    void handleCompareFP(MachineBasicBlock::iterator &I);
    void handleCondMovFP(MachineBasicBlock::iterator &I);
    void handleSpecialFP(MachineBasicBlock::iterator &I);

    // Check if a COPY instruction is using FP registers.
    bool isFPCopy(MachineInstr *MI) {
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned SrcReg = MI->getOperand(1).getReg();

      return X86::RFP80RegClass.contains(DstReg) ||
        X86::RFP80RegClass.contains(SrcReg);
    }
  };
  char FPS::ID = 0;
}

FunctionPass *llvm::createX86FloatingPointStackifierPass() { return new FPS(); }

/// getFPReg - Return the X86::FPx register number for the specified operand.
/// For example, this returns 3 for X86::FP3.
static unsigned getFPReg(const MachineOperand &MO) {
  assert(MO.isReg() && "Expected an FP register!");
  unsigned Reg = MO.getReg();
  assert(Reg >= X86::FP0 && Reg <= X86::FP6 && "Expected FP register!");
  return Reg - X86::FP0;
}

/// runOnMachineFunction - Loop over all of the basic blocks, transforming FP
/// register references into FP stack references.
///
bool FPS::runOnMachineFunction(MachineFunction &MF) {
  // We only need to run this pass if there are any FP registers used in this
  // function.  If it is all integer, there is nothing for us to do!
  bool FPIsUsed = false;

  assert(X86::FP6 == X86::FP0+6 && "Register enums aren't sorted right!");
  for (unsigned i = 0; i <= 6; ++i)
    if (MF.getRegInfo().isPhysRegUsed(X86::FP0+i)) {
      FPIsUsed = true;
      break;
    }

  // Early exit.
  if (!FPIsUsed) return false;

  Bundles = &getAnalysis<EdgeBundles>();
  TII = MF.getTarget().getInstrInfo();

  // Prepare cross-MBB liveness.
  bundleCFG(MF);

  StackTop = 0;

  // Process the function in depth first order so that we process at least one
  // of the predecessors for every reachable block in the function.
  SmallPtrSet<MachineBasicBlock*, 8> Processed;
  MachineBasicBlock *Entry = MF.begin();

  bool Changed = false;
  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*, 8> >
         I = df_ext_begin(Entry, Processed), E = df_ext_end(Entry, Processed);
       I != E; ++I)
    Changed |= processBasicBlock(MF, **I);

  // Process any unreachable blocks in arbitrary order now.
  if (MF.size() != Processed.size())
    for (MachineFunction::iterator BB = MF.begin(), E = MF.end(); BB != E; ++BB)
      if (Processed.insert(BB))
        Changed |= processBasicBlock(MF, *BB);

  LiveBundles.clear();

  return Changed;
}

/// bundleCFG - Scan all the basic blocks to determine consistent live-in and
/// live-out sets for the FP registers. Consistent means that the set of
/// registers live-out from a block is identical to the live-in set of all
/// successors. This is not enforced by the normal live-in lists since
/// registers may be implicitly defined, or not used by all successors.
void FPS::bundleCFG(MachineFunction &MF) {
  assert(LiveBundles.empty() && "Stale data in LiveBundles");
  LiveBundles.resize(Bundles->getNumBundles());

  // Gather the actual live-in masks for all MBBs.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    const unsigned Mask = calcLiveInMask(MBB);
    if (!Mask)
      continue;
    // Update MBB ingoing bundle mask.
    LiveBundles[Bundles->getBundle(MBB->getNumber(), false)].Mask |= Mask;
  }
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// transforming FP instructions into their stack form.
///
bool FPS::processBasicBlock(MachineFunction &MF, MachineBasicBlock &BB) {
  bool Changed = false;
  MBB = &BB;
  NumPendingSTs = 0;

  setupBlockStack();

  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    MachineInstr *MI = I;
    uint64_t Flags = MI->getDesc().TSFlags;

    unsigned FPInstClass = Flags & X86II::FPTypeMask;
    if (MI->isInlineAsm())
      FPInstClass = X86II::SpecialFP;

    if (MI->isCopy() && isFPCopy(MI))
      FPInstClass = X86II::SpecialFP;

    if (MI->isImplicitDef() &&
        X86::RFP80RegClass.contains(MI->getOperand(0).getReg()))
      FPInstClass = X86II::SpecialFP;

    if (FPInstClass == X86II::NotFP)
      continue;  // Efficiently ignore non-fp insts!

    MachineInstr *PrevMI = 0;
    if (I != BB.begin())
      PrevMI = prior(I);

    ++NumFP;  // Keep track of # of pseudo instrs
    DEBUG(dbgs() << "\nFPInst:\t" << *MI);

    // Get dead variables list now because the MI pointer may be deleted as part
    // of processing!
    SmallVector<unsigned, 8> DeadRegs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDead())
        DeadRegs.push_back(MO.getReg());
    }

    switch (FPInstClass) {
    case X86II::ZeroArgFP:  handleZeroArgFP(I); break;
    case X86II::OneArgFP:   handleOneArgFP(I);  break;  // fstp ST(0)
    case X86II::OneArgFPRW: handleOneArgFPRW(I); break; // ST(0) = fsqrt(ST(0))
    case X86II::TwoArgFP:   handleTwoArgFP(I);  break;
    case X86II::CompareFP:  handleCompareFP(I); break;
    case X86II::CondMovFP:  handleCondMovFP(I); break;
    case X86II::SpecialFP:  handleSpecialFP(I); break;
    default: llvm_unreachable("Unknown FP Type!");
    }

    // Check to see if any of the values defined by this instruction are dead
    // after definition.  If so, pop them.
    for (unsigned i = 0, e = DeadRegs.size(); i != e; ++i) {
      unsigned Reg = DeadRegs[i];
      if (Reg >= X86::FP0 && Reg <= X86::FP6) {
        DEBUG(dbgs() << "Register FP#" << Reg-X86::FP0 << " is dead!\n");
        freeStackSlotAfter(I, Reg-X86::FP0);
      }
    }

    // Print out all of the instructions expanded to if -debug
    DEBUG(
      MachineBasicBlock::iterator PrevI(PrevMI);
      if (I == PrevI) {
        dbgs() << "Just deleted pseudo instruction\n";
      } else {
        MachineBasicBlock::iterator Start = I;
        // Rewind to first instruction newly inserted.
        while (Start != BB.begin() && prior(Start) != PrevI) --Start;
        dbgs() << "Inserted instructions:\n\t";
        Start->print(dbgs(), &MF.getTarget());
        while (++Start != llvm::next(I)) {}
      }
      dumpStack();
    );
    (void)PrevMI;

    Changed = true;
  }

  finishBlockStack();

  return Changed;
}

/// setupBlockStack - Use the live bundles to set up our model of the stack
/// to match predecessors' live out stack.
void FPS::setupBlockStack() {
  DEBUG(dbgs() << "\nSetting up live-ins for BB#" << MBB->getNumber()
               << " derived from " << MBB->getName() << ".\n");
  StackTop = 0;
  // Get the live-in bundle for MBB.
  const LiveBundle &Bundle =
    LiveBundles[Bundles->getBundle(MBB->getNumber(), false)];

  if (!Bundle.Mask) {
    DEBUG(dbgs() << "Block has no FP live-ins.\n");
    return;
  }

  // Depth-first iteration should ensure that we always have an assigned stack.
  assert(Bundle.isFixed() && "Reached block before any predecessors");

  // Push the fixed live-in registers.
  for (unsigned i = Bundle.FixCount; i > 0; --i) {
    MBB->addLiveIn(X86::ST0+i-1);
    DEBUG(dbgs() << "Live-in st(" << (i-1) << "): %FP"
                 << unsigned(Bundle.FixStack[i-1]) << '\n');
    pushReg(Bundle.FixStack[i-1]);
  }

  // Kill off unwanted live-ins. This can happen with a critical edge.
  // FIXME: We could keep these live registers around as zombies. They may need
  // to be revived at the end of a short block. It might save a few instrs.
  adjustLiveRegs(calcLiveInMask(MBB), MBB->begin());
  DEBUG(MBB->dump());
}

/// finishBlockStack - Revive live-outs that are implicitly defined out of
/// MBB. Shuffle live registers to match the expected fixed stack of any
/// predecessors, and ensure that all predecessors are expecting the same
/// stack.
void FPS::finishBlockStack() {
  // The RET handling below takes care of return blocks for us.
  if (MBB->succ_empty())
    return;

  DEBUG(dbgs() << "Setting up live-outs for BB#" << MBB->getNumber()
               << " derived from " << MBB->getName() << ".\n");

  // Get MBB's live-out bundle.
  unsigned BundleIdx = Bundles->getBundle(MBB->getNumber(), true);
  LiveBundle &Bundle = LiveBundles[BundleIdx];

  // We may need to kill and define some registers to match successors.
  // FIXME: This can probably be combined with the shuffle below.
  MachineBasicBlock::iterator Term = MBB->getFirstTerminator();
  adjustLiveRegs(Bundle.Mask, Term);

  if (!Bundle.Mask) {
    DEBUG(dbgs() << "No live-outs.\n");
    return;
  }

  // Has the stack order been fixed yet?
  DEBUG(dbgs() << "LB#" << BundleIdx << ": ");
  if (Bundle.isFixed()) {
    DEBUG(dbgs() << "Shuffling stack to match.\n");
    shuffleStackTop(Bundle.FixStack, Bundle.FixCount, Term);
  } else {
    // Not fixed yet, we get to choose.
    DEBUG(dbgs() << "Fixing stack order now.\n");
    Bundle.FixCount = StackTop;
    for (unsigned i = 0; i < StackTop; ++i)
      Bundle.FixStack[i] = getStackEntry(i);
  }
}


//===----------------------------------------------------------------------===//
// Efficient Lookup Table Support
//===----------------------------------------------------------------------===//

namespace {
  struct TableEntry {
    uint16_t from;
    uint16_t to;
    bool operator<(const TableEntry &TE) const { return from < TE.from; }
    friend bool operator<(const TableEntry &TE, unsigned V) {
      return TE.from < V;
    }
    friend bool LLVM_ATTRIBUTE_UNUSED operator<(unsigned V,
                                                const TableEntry &TE) {
      return V < TE.from;
    }
  };
}

#ifndef NDEBUG
static bool TableIsSorted(const TableEntry *Table, unsigned NumEntries) {
  for (unsigned i = 0; i != NumEntries-1; ++i)
    if (!(Table[i] < Table[i+1])) return false;
  return true;
}
#endif

static int Lookup(const TableEntry *Table, unsigned N, unsigned Opcode) {
  const TableEntry *I = std::lower_bound(Table, Table+N, Opcode);
  if (I != Table+N && I->from == Opcode)
    return I->to;
  return -1;
}

#ifdef NDEBUG
#define ASSERT_SORTED(TABLE)
#else
#define ASSERT_SORTED(TABLE)                                              \
  { static bool TABLE##Checked = false;                                   \
    if (!TABLE##Checked) {                                                \
       assert(TableIsSorted(TABLE, array_lengthof(TABLE)) &&              \
              "All lookup tables must be sorted for efficient access!");  \
       TABLE##Checked = true;                                             \
    }                                                                     \
  }
#endif

//===----------------------------------------------------------------------===//
// Register File -> Register Stack Mapping Methods
//===----------------------------------------------------------------------===//

// OpcodeTable - Sorted map of register instructions to their stack version.
// The first element is an register file pseudo instruction, the second is the
// concrete X86 instruction which uses the register stack.
//
static const TableEntry OpcodeTable[] = {
  { X86::ABS_Fp32     , X86::ABS_F     },
  { X86::ABS_Fp64     , X86::ABS_F     },
  { X86::ABS_Fp80     , X86::ABS_F     },
  { X86::ADD_Fp32m    , X86::ADD_F32m  },
  { X86::ADD_Fp64m    , X86::ADD_F64m  },
  { X86::ADD_Fp64m32  , X86::ADD_F32m  },
  { X86::ADD_Fp80m32  , X86::ADD_F32m  },
  { X86::ADD_Fp80m64  , X86::ADD_F64m  },
  { X86::ADD_FpI16m32 , X86::ADD_FI16m },
  { X86::ADD_FpI16m64 , X86::ADD_FI16m },
  { X86::ADD_FpI16m80 , X86::ADD_FI16m },
  { X86::ADD_FpI32m32 , X86::ADD_FI32m },
  { X86::ADD_FpI32m64 , X86::ADD_FI32m },
  { X86::ADD_FpI32m80 , X86::ADD_FI32m },
  { X86::CHS_Fp32     , X86::CHS_F     },
  { X86::CHS_Fp64     , X86::CHS_F     },
  { X86::CHS_Fp80     , X86::CHS_F     },
  { X86::CMOVBE_Fp32  , X86::CMOVBE_F  },
  { X86::CMOVBE_Fp64  , X86::CMOVBE_F  },
  { X86::CMOVBE_Fp80  , X86::CMOVBE_F  },
  { X86::CMOVB_Fp32   , X86::CMOVB_F   },
  { X86::CMOVB_Fp64   , X86::CMOVB_F  },
  { X86::CMOVB_Fp80   , X86::CMOVB_F  },
  { X86::CMOVE_Fp32   , X86::CMOVE_F  },
  { X86::CMOVE_Fp64   , X86::CMOVE_F   },
  { X86::CMOVE_Fp80   , X86::CMOVE_F   },
  { X86::CMOVNBE_Fp32 , X86::CMOVNBE_F },
  { X86::CMOVNBE_Fp64 , X86::CMOVNBE_F },
  { X86::CMOVNBE_Fp80 , X86::CMOVNBE_F },
  { X86::CMOVNB_Fp32  , X86::CMOVNB_F  },
  { X86::CMOVNB_Fp64  , X86::CMOVNB_F  },
  { X86::CMOVNB_Fp80  , X86::CMOVNB_F  },
  { X86::CMOVNE_Fp32  , X86::CMOVNE_F  },
  { X86::CMOVNE_Fp64  , X86::CMOVNE_F  },
  { X86::CMOVNE_Fp80  , X86::CMOVNE_F  },
  { X86::CMOVNP_Fp32  , X86::CMOVNP_F  },
  { X86::CMOVNP_Fp64  , X86::CMOVNP_F  },
  { X86::CMOVNP_Fp80  , X86::CMOVNP_F  },
  { X86::CMOVP_Fp32   , X86::CMOVP_F   },
  { X86::CMOVP_Fp64   , X86::CMOVP_F   },
  { X86::CMOVP_Fp80   , X86::CMOVP_F   },
  { X86::COS_Fp32     , X86::COS_F     },
  { X86::COS_Fp64     , X86::COS_F     },
  { X86::COS_Fp80     , X86::COS_F     },
  { X86::DIVR_Fp32m   , X86::DIVR_F32m },
  { X86::DIVR_Fp64m   , X86::DIVR_F64m },
  { X86::DIVR_Fp64m32 , X86::DIVR_F32m },
  { X86::DIVR_Fp80m32 , X86::DIVR_F32m },
  { X86::DIVR_Fp80m64 , X86::DIVR_F64m },
  { X86::DIVR_FpI16m32, X86::DIVR_FI16m},
  { X86::DIVR_FpI16m64, X86::DIVR_FI16m},
  { X86::DIVR_FpI16m80, X86::DIVR_FI16m},
  { X86::DIVR_FpI32m32, X86::DIVR_FI32m},
  { X86::DIVR_FpI32m64, X86::DIVR_FI32m},
  { X86::DIVR_FpI32m80, X86::DIVR_FI32m},
  { X86::DIV_Fp32m    , X86::DIV_F32m  },
  { X86::DIV_Fp64m    , X86::DIV_F64m  },
  { X86::DIV_Fp64m32  , X86::DIV_F32m  },
  { X86::DIV_Fp80m32  , X86::DIV_F32m  },
  { X86::DIV_Fp80m64  , X86::DIV_F64m  },
  { X86::DIV_FpI16m32 , X86::DIV_FI16m },
  { X86::DIV_FpI16m64 , X86::DIV_FI16m },
  { X86::DIV_FpI16m80 , X86::DIV_FI16m },
  { X86::DIV_FpI32m32 , X86::DIV_FI32m },
  { X86::DIV_FpI32m64 , X86::DIV_FI32m },
  { X86::DIV_FpI32m80 , X86::DIV_FI32m },
  { X86::ILD_Fp16m32  , X86::ILD_F16m  },
  { X86::ILD_Fp16m64  , X86::ILD_F16m  },
  { X86::ILD_Fp16m80  , X86::ILD_F16m  },
  { X86::ILD_Fp32m32  , X86::ILD_F32m  },
  { X86::ILD_Fp32m64  , X86::ILD_F32m  },
  { X86::ILD_Fp32m80  , X86::ILD_F32m  },
  { X86::ILD_Fp64m32  , X86::ILD_F64m  },
  { X86::ILD_Fp64m64  , X86::ILD_F64m  },
  { X86::ILD_Fp64m80  , X86::ILD_F64m  },
  { X86::ISTT_Fp16m32 , X86::ISTT_FP16m},
  { X86::ISTT_Fp16m64 , X86::ISTT_FP16m},
  { X86::ISTT_Fp16m80 , X86::ISTT_FP16m},
  { X86::ISTT_Fp32m32 , X86::ISTT_FP32m},
  { X86::ISTT_Fp32m64 , X86::ISTT_FP32m},
  { X86::ISTT_Fp32m80 , X86::ISTT_FP32m},
  { X86::ISTT_Fp64m32 , X86::ISTT_FP64m},
  { X86::ISTT_Fp64m64 , X86::ISTT_FP64m},
  { X86::ISTT_Fp64m80 , X86::ISTT_FP64m},
  { X86::IST_Fp16m32  , X86::IST_F16m  },
  { X86::IST_Fp16m64  , X86::IST_F16m  },
  { X86::IST_Fp16m80  , X86::IST_F16m  },
  { X86::IST_Fp32m32  , X86::IST_F32m  },
  { X86::IST_Fp32m64  , X86::IST_F32m  },
  { X86::IST_Fp32m80  , X86::IST_F32m  },
  { X86::IST_Fp64m32  , X86::IST_FP64m },
  { X86::IST_Fp64m64  , X86::IST_FP64m },
  { X86::IST_Fp64m80  , X86::IST_FP64m },
  { X86::LD_Fp032     , X86::LD_F0     },
  { X86::LD_Fp064     , X86::LD_F0     },
  { X86::LD_Fp080     , X86::LD_F0     },
  { X86::LD_Fp132     , X86::LD_F1     },
  { X86::LD_Fp164     , X86::LD_F1     },
  { X86::LD_Fp180     , X86::LD_F1     },
  { X86::LD_Fp32m     , X86::LD_F32m   },
  { X86::LD_Fp32m64   , X86::LD_F32m   },
  { X86::LD_Fp32m80   , X86::LD_F32m   },
  { X86::LD_Fp64m     , X86::LD_F64m   },
  { X86::LD_Fp64m80   , X86::LD_F64m   },
  { X86::LD_Fp80m     , X86::LD_F80m   },
  { X86::MUL_Fp32m    , X86::MUL_F32m  },
  { X86::MUL_Fp64m    , X86::MUL_F64m  },
  { X86::MUL_Fp64m32  , X86::MUL_F32m  },
  { X86::MUL_Fp80m32  , X86::MUL_F32m  },
  { X86::MUL_Fp80m64  , X86::MUL_F64m  },
  { X86::MUL_FpI16m32 , X86::MUL_FI16m },
  { X86::MUL_FpI16m64 , X86::MUL_FI16m },
  { X86::MUL_FpI16m80 , X86::MUL_FI16m },
  { X86::MUL_FpI32m32 , X86::MUL_FI32m },
  { X86::MUL_FpI32m64 , X86::MUL_FI32m },
  { X86::MUL_FpI32m80 , X86::MUL_FI32m },
  { X86::SIN_Fp32     , X86::SIN_F     },
  { X86::SIN_Fp64     , X86::SIN_F     },
  { X86::SIN_Fp80     , X86::SIN_F     },
  { X86::SQRT_Fp32    , X86::SQRT_F    },
  { X86::SQRT_Fp64    , X86::SQRT_F    },
  { X86::SQRT_Fp80    , X86::SQRT_F    },
  { X86::ST_Fp32m     , X86::ST_F32m   },
  { X86::ST_Fp64m     , X86::ST_F64m   },
  { X86::ST_Fp64m32   , X86::ST_F32m   },
  { X86::ST_Fp80m32   , X86::ST_F32m   },
  { X86::ST_Fp80m64   , X86::ST_F64m   },
  { X86::ST_FpP80m    , X86::ST_FP80m  },
  { X86::SUBR_Fp32m   , X86::SUBR_F32m },
  { X86::SUBR_Fp64m   , X86::SUBR_F64m },
  { X86::SUBR_Fp64m32 , X86::SUBR_F32m },
  { X86::SUBR_Fp80m32 , X86::SUBR_F32m },
  { X86::SUBR_Fp80m64 , X86::SUBR_F64m },
  { X86::SUBR_FpI16m32, X86::SUBR_FI16m},
  { X86::SUBR_FpI16m64, X86::SUBR_FI16m},
  { X86::SUBR_FpI16m80, X86::SUBR_FI16m},
  { X86::SUBR_FpI32m32, X86::SUBR_FI32m},
  { X86::SUBR_FpI32m64, X86::SUBR_FI32m},
  { X86::SUBR_FpI32m80, X86::SUBR_FI32m},
  { X86::SUB_Fp32m    , X86::SUB_F32m  },
  { X86::SUB_Fp64m    , X86::SUB_F64m  },
  { X86::SUB_Fp64m32  , X86::SUB_F32m  },
  { X86::SUB_Fp80m32  , X86::SUB_F32m  },
  { X86::SUB_Fp80m64  , X86::SUB_F64m  },
  { X86::SUB_FpI16m32 , X86::SUB_FI16m },
  { X86::SUB_FpI16m64 , X86::SUB_FI16m },
  { X86::SUB_FpI16m80 , X86::SUB_FI16m },
  { X86::SUB_FpI32m32 , X86::SUB_FI32m },
  { X86::SUB_FpI32m64 , X86::SUB_FI32m },
  { X86::SUB_FpI32m80 , X86::SUB_FI32m },
  { X86::TST_Fp32     , X86::TST_F     },
  { X86::TST_Fp64     , X86::TST_F     },
  { X86::TST_Fp80     , X86::TST_F     },
  { X86::UCOM_FpIr32  , X86::UCOM_FIr  },
  { X86::UCOM_FpIr64  , X86::UCOM_FIr  },
  { X86::UCOM_FpIr80  , X86::UCOM_FIr  },
  { X86::UCOM_Fpr32   , X86::UCOM_Fr   },
  { X86::UCOM_Fpr64   , X86::UCOM_Fr   },
  { X86::UCOM_Fpr80   , X86::UCOM_Fr   },
};

static unsigned getConcreteOpcode(unsigned Opcode) {
  ASSERT_SORTED(OpcodeTable);
  int Opc = Lookup(OpcodeTable, array_lengthof(OpcodeTable), Opcode);
  assert(Opc != -1 && "FP Stack instruction not in OpcodeTable!");
  return Opc;
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

// PopTable - Sorted map of instructions to their popping version.  The first
// element is an instruction, the second is the version which pops.
//
static const TableEntry PopTable[] = {
  { X86::ADD_FrST0 , X86::ADD_FPrST0  },

  { X86::DIVR_FrST0, X86::DIVR_FPrST0 },
  { X86::DIV_FrST0 , X86::DIV_FPrST0  },

  { X86::IST_F16m  , X86::IST_FP16m   },
  { X86::IST_F32m  , X86::IST_FP32m   },

  { X86::MUL_FrST0 , X86::MUL_FPrST0  },

  { X86::ST_F32m   , X86::ST_FP32m    },
  { X86::ST_F64m   , X86::ST_FP64m    },
  { X86::ST_Frr    , X86::ST_FPrr     },

  { X86::SUBR_FrST0, X86::SUBR_FPrST0 },
  { X86::SUB_FrST0 , X86::SUB_FPrST0  },

  { X86::UCOM_FIr  , X86::UCOM_FIPr   },

  { X86::UCOM_FPr  , X86::UCOM_FPPr   },
  { X86::UCOM_Fr   , X86::UCOM_FPr    },
};

/// popStackAfter - Pop the current value off of the top of the FP stack after
/// the specified instruction.  This attempts to be sneaky and combine the pop
/// into the instruction itself if possible.  The iterator is left pointing to
/// the last instruction, be it a new pop instruction inserted, or the old
/// instruction if it was modified in place.
///
void FPS::popStackAfter(MachineBasicBlock::iterator &I) {
  MachineInstr* MI = I;
  DebugLoc dl = MI->getDebugLoc();
  ASSERT_SORTED(PopTable);
  if (StackTop == 0)
    report_fatal_error("Cannot pop empty stack!");
  RegMap[Stack[--StackTop]] = ~0;     // Update state

  // Check to see if there is a popping version of this instruction...
  int Opcode = Lookup(PopTable, array_lengthof(PopTable), I->getOpcode());
  if (Opcode != -1) {
    I->setDesc(TII->get(Opcode));
    if (Opcode == X86::UCOM_FPPr)
      I->RemoveOperand(0);
  } else {    // Insert an explicit pop
    I = BuildMI(*MBB, ++I, dl, TII->get(X86::ST_FPrr)).addReg(X86::ST0);
  }
}

/// freeStackSlotAfter - Free the specified register from the register stack, so
/// that it is no longer in a register.  If the register is currently at the top
/// of the stack, we just pop the current instruction, otherwise we store the
/// current top-of-stack into the specified slot, then pop the top of stack.
void FPS::freeStackSlotAfter(MachineBasicBlock::iterator &I, unsigned FPRegNo) {
  if (getStackEntry(0) == FPRegNo) {  // already at the top of stack? easy.
    popStackAfter(I);
    return;
  }

  // Otherwise, store the top of stack into the dead slot, killing the operand
  // without having to add in an explicit xchg then pop.
  //
  I = freeStackSlotBefore(++I, FPRegNo);
}

/// freeStackSlotBefore - Free the specified register without trying any
/// folding.
MachineBasicBlock::iterator
FPS::freeStackSlotBefore(MachineBasicBlock::iterator I, unsigned FPRegNo) {
  unsigned STReg    = getSTReg(FPRegNo);
  unsigned OldSlot  = getSlot(FPRegNo);
  unsigned TopReg   = Stack[StackTop-1];
  Stack[OldSlot]    = TopReg;
  RegMap[TopReg]    = OldSlot;
  RegMap[FPRegNo]   = ~0;
  Stack[--StackTop] = ~0;
  return BuildMI(*MBB, I, DebugLoc(), TII->get(X86::ST_FPrr)).addReg(STReg);
}

/// adjustLiveRegs - Kill and revive registers such that exactly the FP
/// registers with a bit in Mask are live.
void FPS::adjustLiveRegs(unsigned Mask, MachineBasicBlock::iterator I) {
  unsigned Defs = Mask;
  unsigned Kills = 0;
  for (unsigned i = 0; i < StackTop; ++i) {
    unsigned RegNo = Stack[i];
    if (!(Defs & (1 << RegNo)))
      // This register is live, but we don't want it.
      Kills |= (1 << RegNo);
    else
      // We don't need to imp-def this live register.
      Defs &= ~(1 << RegNo);
  }
  assert((Kills & Defs) == 0 && "Register needs killing and def'ing?");

  // Produce implicit-defs for free by using killed registers.
  while (Kills && Defs) {
    unsigned KReg = CountTrailingZeros_32(Kills);
    unsigned DReg = CountTrailingZeros_32(Defs);
    DEBUG(dbgs() << "Renaming %FP" << KReg << " as imp %FP" << DReg << "\n");
    std::swap(Stack[getSlot(KReg)], Stack[getSlot(DReg)]);
    std::swap(RegMap[KReg], RegMap[DReg]);
    Kills &= ~(1 << KReg);
    Defs &= ~(1 << DReg);
  }

  // Kill registers by popping.
  if (Kills && I != MBB->begin()) {
    MachineBasicBlock::iterator I2 = llvm::prior(I);
    while (StackTop) {
      unsigned KReg = getStackEntry(0);
      if (!(Kills & (1 << KReg)))
        break;
      DEBUG(dbgs() << "Popping %FP" << KReg << "\n");
      popStackAfter(I2);
      Kills &= ~(1 << KReg);
    }
  }

  // Manually kill the rest.
  while (Kills) {
    unsigned KReg = CountTrailingZeros_32(Kills);
    DEBUG(dbgs() << "Killing %FP" << KReg << "\n");
    freeStackSlotBefore(I, KReg);
    Kills &= ~(1 << KReg);
  }

  // Load zeros for all the imp-defs.
  while(Defs) {
    unsigned DReg = CountTrailingZeros_32(Defs);
    DEBUG(dbgs() << "Defining %FP" << DReg << " as 0\n");
    BuildMI(*MBB, I, DebugLoc(), TII->get(X86::LD_F0));
    pushReg(DReg);
    Defs &= ~(1 << DReg);
  }

  // Now we should have the correct registers live.
  DEBUG(dumpStack());
  assert(StackTop == CountPopulation_32(Mask) && "Live count mismatch");
}

/// shuffleStackTop - emit fxch instructions before I to shuffle the top
/// FixCount entries into the order given by FixStack.
/// FIXME: Is there a better algorithm than insertion sort?
void FPS::shuffleStackTop(const unsigned char *FixStack,
                          unsigned FixCount,
                          MachineBasicBlock::iterator I) {
  // Move items into place, starting from the desired stack bottom.
  while (FixCount--) {
    // Old register at position FixCount.
    unsigned OldReg = getStackEntry(FixCount);
    // Desired register at position FixCount.
    unsigned Reg = FixStack[FixCount];
    if (Reg == OldReg)
      continue;
    // (Reg st0) (OldReg st0) = (Reg OldReg st0)
    moveToTop(Reg, I);
    if (FixCount > 0)
      moveToTop(OldReg, I);
  }
  DEBUG(dumpStack());
}


//===----------------------------------------------------------------------===//
// Instruction transformation implementation
//===----------------------------------------------------------------------===//

/// handleZeroArgFP - ST(0) = fld0    ST(0) = flds <mem>
///
void FPS::handleZeroArgFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  unsigned DestReg = getFPReg(MI->getOperand(0));

  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(0);   // Remove the explicit ST(0) operand
  MI->setDesc(TII->get(getConcreteOpcode(MI->getOpcode())));

  // Result gets pushed on the stack.
  pushReg(DestReg);
}

/// handleOneArgFP - fst <mem>, ST(0)
///
void FPS::handleOneArgFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  unsigned NumOps = MI->getDesc().getNumOperands();
  assert((NumOps == X86::AddrNumOperands + 1 || NumOps == 1) &&
         "Can only handle fst* & ftst instructions!");

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(NumOps-1));
  bool KillsSrc = MI->killsRegister(X86::FP0+Reg);

  if (KillsSrc)
    duplicatePendingSTBeforeKill(Reg, I);

  // FISTP64m is strange because there isn't a non-popping versions.
  // If we have one _and_ we don't want to pop the operand, duplicate the value
  // on the stack instead of moving it.  This ensure that popping the value is
  // always ok.
  // Ditto FISTTP16m, FISTTP32m, FISTTP64m, ST_FpP80m.
  //
  if (!KillsSrc &&
      (MI->getOpcode() == X86::IST_Fp64m32 ||
       MI->getOpcode() == X86::ISTT_Fp16m32 ||
       MI->getOpcode() == X86::ISTT_Fp32m32 ||
       MI->getOpcode() == X86::ISTT_Fp64m32 ||
       MI->getOpcode() == X86::IST_Fp64m64 ||
       MI->getOpcode() == X86::ISTT_Fp16m64 ||
       MI->getOpcode() == X86::ISTT_Fp32m64 ||
       MI->getOpcode() == X86::ISTT_Fp64m64 ||
       MI->getOpcode() == X86::IST_Fp64m80 ||
       MI->getOpcode() == X86::ISTT_Fp16m80 ||
       MI->getOpcode() == X86::ISTT_Fp32m80 ||
       MI->getOpcode() == X86::ISTT_Fp64m80 ||
       MI->getOpcode() == X86::ST_FpP80m)) {
    duplicateToTop(Reg, getScratchReg(), I);
  } else {
    moveToTop(Reg, I);            // Move to the top of the stack...
  }

  // Convert from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(NumOps-1);    // Remove explicit ST(0) operand
  MI->setDesc(TII->get(getConcreteOpcode(MI->getOpcode())));

  if (MI->getOpcode() == X86::IST_FP64m ||
      MI->getOpcode() == X86::ISTT_FP16m ||
      MI->getOpcode() == X86::ISTT_FP32m ||
      MI->getOpcode() == X86::ISTT_FP64m ||
      MI->getOpcode() == X86::ST_FP80m) {
    if (StackTop == 0)
      report_fatal_error("Stack empty??");
    --StackTop;
  } else if (KillsSrc) { // Last use of operand?
    popStackAfter(I);
  }
}


/// handleOneArgFPRW: Handle instructions that read from the top of stack and
/// replace the value with a newly computed value.  These instructions may have
/// non-fp operands after their FP operands.
///
///  Examples:
///     R1 = fchs R2
///     R1 = fadd R2, [mem]
///
void FPS::handleOneArgFPRW(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
#ifndef NDEBUG
  unsigned NumOps = MI->getDesc().getNumOperands();
  assert(NumOps >= 2 && "FPRW instructions must have 2 ops!!");
#endif

  // Is this the last use of the source register?
  unsigned Reg = getFPReg(MI->getOperand(1));
  bool KillsSrc = MI->killsRegister(X86::FP0+Reg);

  if (KillsSrc) {
    duplicatePendingSTBeforeKill(Reg, I);
    // If this is the last use of the source register, just make sure it's on
    // the top of the stack.
    moveToTop(Reg, I);
    if (StackTop == 0)
      report_fatal_error("Stack cannot be empty!");
    --StackTop;
    pushReg(getFPReg(MI->getOperand(0)));
  } else {
    // If this is not the last use of the source register, _copy_ it to the top
    // of the stack.
    duplicateToTop(Reg, getFPReg(MI->getOperand(0)), I);
  }

  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(1);   // Drop the source operand.
  MI->RemoveOperand(0);   // Drop the destination operand.
  MI->setDesc(TII->get(getConcreteOpcode(MI->getOpcode())));
}


//===----------------------------------------------------------------------===//
// Define tables of various ways to map pseudo instructions
//

// ForwardST0Table - Map: A = B op C  into: ST(0) = ST(0) op ST(i)
static const TableEntry ForwardST0Table[] = {
  { X86::ADD_Fp32  , X86::ADD_FST0r },
  { X86::ADD_Fp64  , X86::ADD_FST0r },
  { X86::ADD_Fp80  , X86::ADD_FST0r },
  { X86::DIV_Fp32  , X86::DIV_FST0r },
  { X86::DIV_Fp64  , X86::DIV_FST0r },
  { X86::DIV_Fp80  , X86::DIV_FST0r },
  { X86::MUL_Fp32  , X86::MUL_FST0r },
  { X86::MUL_Fp64  , X86::MUL_FST0r },
  { X86::MUL_Fp80  , X86::MUL_FST0r },
  { X86::SUB_Fp32  , X86::SUB_FST0r },
  { X86::SUB_Fp64  , X86::SUB_FST0r },
  { X86::SUB_Fp80  , X86::SUB_FST0r },
};

// ReverseST0Table - Map: A = B op C  into: ST(0) = ST(i) op ST(0)
static const TableEntry ReverseST0Table[] = {
  { X86::ADD_Fp32  , X86::ADD_FST0r  },   // commutative
  { X86::ADD_Fp64  , X86::ADD_FST0r  },   // commutative
  { X86::ADD_Fp80  , X86::ADD_FST0r  },   // commutative
  { X86::DIV_Fp32  , X86::DIVR_FST0r },
  { X86::DIV_Fp64  , X86::DIVR_FST0r },
  { X86::DIV_Fp80  , X86::DIVR_FST0r },
  { X86::MUL_Fp32  , X86::MUL_FST0r  },   // commutative
  { X86::MUL_Fp64  , X86::MUL_FST0r  },   // commutative
  { X86::MUL_Fp80  , X86::MUL_FST0r  },   // commutative
  { X86::SUB_Fp32  , X86::SUBR_FST0r },
  { X86::SUB_Fp64  , X86::SUBR_FST0r },
  { X86::SUB_Fp80  , X86::SUBR_FST0r },
};

// ForwardSTiTable - Map: A = B op C  into: ST(i) = ST(0) op ST(i)
static const TableEntry ForwardSTiTable[] = {
  { X86::ADD_Fp32  , X86::ADD_FrST0  },   // commutative
  { X86::ADD_Fp64  , X86::ADD_FrST0  },   // commutative
  { X86::ADD_Fp80  , X86::ADD_FrST0  },   // commutative
  { X86::DIV_Fp32  , X86::DIVR_FrST0 },
  { X86::DIV_Fp64  , X86::DIVR_FrST0 },
  { X86::DIV_Fp80  , X86::DIVR_FrST0 },
  { X86::MUL_Fp32  , X86::MUL_FrST0  },   // commutative
  { X86::MUL_Fp64  , X86::MUL_FrST0  },   // commutative
  { X86::MUL_Fp80  , X86::MUL_FrST0  },   // commutative
  { X86::SUB_Fp32  , X86::SUBR_FrST0 },
  { X86::SUB_Fp64  , X86::SUBR_FrST0 },
  { X86::SUB_Fp80  , X86::SUBR_FrST0 },
};

// ReverseSTiTable - Map: A = B op C  into: ST(i) = ST(i) op ST(0)
static const TableEntry ReverseSTiTable[] = {
  { X86::ADD_Fp32  , X86::ADD_FrST0 },
  { X86::ADD_Fp64  , X86::ADD_FrST0 },
  { X86::ADD_Fp80  , X86::ADD_FrST0 },
  { X86::DIV_Fp32  , X86::DIV_FrST0 },
  { X86::DIV_Fp64  , X86::DIV_FrST0 },
  { X86::DIV_Fp80  , X86::DIV_FrST0 },
  { X86::MUL_Fp32  , X86::MUL_FrST0 },
  { X86::MUL_Fp64  , X86::MUL_FrST0 },
  { X86::MUL_Fp80  , X86::MUL_FrST0 },
  { X86::SUB_Fp32  , X86::SUB_FrST0 },
  { X86::SUB_Fp64  , X86::SUB_FrST0 },
  { X86::SUB_Fp80  , X86::SUB_FrST0 },
};


/// handleTwoArgFP - Handle instructions like FADD and friends which are virtual
/// instructions which need to be simplified and possibly transformed.
///
/// Result: ST(0) = fsub  ST(0), ST(i)
///         ST(i) = fsub  ST(0), ST(i)
///         ST(0) = fsubr ST(0), ST(i)
///         ST(i) = fsubr ST(0), ST(i)
///
void FPS::handleTwoArgFP(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(ForwardST0Table); ASSERT_SORTED(ReverseST0Table);
  ASSERT_SORTED(ForwardSTiTable); ASSERT_SORTED(ReverseSTiTable);
  MachineInstr *MI = I;

  unsigned NumOperands = MI->getDesc().getNumOperands();
  assert(NumOperands == 3 && "Illegal TwoArgFP instruction!");
  unsigned Dest = getFPReg(MI->getOperand(0));
  unsigned Op0 = getFPReg(MI->getOperand(NumOperands-2));
  unsigned Op1 = getFPReg(MI->getOperand(NumOperands-1));
  bool KillsOp0 = MI->killsRegister(X86::FP0+Op0);
  bool KillsOp1 = MI->killsRegister(X86::FP0+Op1);
  DebugLoc dl = MI->getDebugLoc();

  unsigned TOS = getStackEntry(0);

  // One of our operands must be on the top of the stack.  If neither is yet, we
  // need to move one.
  if (Op0 != TOS && Op1 != TOS) {   // No operand at TOS?
    // We can choose to move either operand to the top of the stack.  If one of
    // the operands is killed by this instruction, we want that one so that we
    // can update right on top of the old version.
    if (KillsOp0) {
      moveToTop(Op0, I);         // Move dead operand to TOS.
      TOS = Op0;
    } else if (KillsOp1) {
      moveToTop(Op1, I);
      TOS = Op1;
    } else {
      // All of the operands are live after this instruction executes, so we
      // cannot update on top of any operand.  Because of this, we must
      // duplicate one of the stack elements to the top.  It doesn't matter
      // which one we pick.
      //
      duplicateToTop(Op0, Dest, I);
      Op0 = TOS = Dest;
      KillsOp0 = true;
    }
  } else if (!KillsOp0 && !KillsOp1) {
    // If we DO have one of our operands at the top of the stack, but we don't
    // have a dead operand, we must duplicate one of the operands to a new slot
    // on the stack.
    duplicateToTop(Op0, Dest, I);
    Op0 = TOS = Dest;
    KillsOp0 = true;
  }

  // Now we know that one of our operands is on the top of the stack, and at
  // least one of our operands is killed by this instruction.
  assert((TOS == Op0 || TOS == Op1) && (KillsOp0 || KillsOp1) &&
         "Stack conditions not set up right!");

  // We decide which form to use based on what is on the top of the stack, and
  // which operand is killed by this instruction.
  const TableEntry *InstTable;
  bool isForward = TOS == Op0;
  bool updateST0 = (TOS == Op0 && !KillsOp1) || (TOS == Op1 && !KillsOp0);
  if (updateST0) {
    if (isForward)
      InstTable = ForwardST0Table;
    else
      InstTable = ReverseST0Table;
  } else {
    if (isForward)
      InstTable = ForwardSTiTable;
    else
      InstTable = ReverseSTiTable;
  }

  int Opcode = Lookup(InstTable, array_lengthof(ForwardST0Table),
                      MI->getOpcode());
  assert(Opcode != -1 && "Unknown TwoArgFP pseudo instruction!");

  // NotTOS - The register which is not on the top of stack...
  unsigned NotTOS = (TOS == Op0) ? Op1 : Op0;

  // Replace the old instruction with a new instruction
  MBB->remove(I++);
  I = BuildMI(*MBB, I, dl, TII->get(Opcode)).addReg(getSTReg(NotTOS));

  // If both operands are killed, pop one off of the stack in addition to
  // overwriting the other one.
  if (KillsOp0 && KillsOp1 && Op0 != Op1) {
    assert(!updateST0 && "Should have updated other operand!");
    popStackAfter(I);   // Pop the top of stack
  }

  // Update stack information so that we know the destination register is now on
  // the stack.
  unsigned UpdatedSlot = getSlot(updateST0 ? TOS : NotTOS);
  assert(UpdatedSlot < StackTop && Dest < 7);
  Stack[UpdatedSlot]   = Dest;
  RegMap[Dest]         = UpdatedSlot;
  MBB->getParent()->DeleteMachineInstr(MI); // Remove the old instruction
}

/// handleCompareFP - Handle FUCOM and FUCOMI instructions, which have two FP
/// register arguments and no explicit destinations.
///
void FPS::handleCompareFP(MachineBasicBlock::iterator &I) {
  ASSERT_SORTED(ForwardST0Table); ASSERT_SORTED(ReverseST0Table);
  ASSERT_SORTED(ForwardSTiTable); ASSERT_SORTED(ReverseSTiTable);
  MachineInstr *MI = I;

  unsigned NumOperands = MI->getDesc().getNumOperands();
  assert(NumOperands == 2 && "Illegal FUCOM* instruction!");
  unsigned Op0 = getFPReg(MI->getOperand(NumOperands-2));
  unsigned Op1 = getFPReg(MI->getOperand(NumOperands-1));
  bool KillsOp0 = MI->killsRegister(X86::FP0+Op0);
  bool KillsOp1 = MI->killsRegister(X86::FP0+Op1);

  // Make sure the first operand is on the top of stack, the other one can be
  // anywhere.
  moveToTop(Op0, I);

  // Change from the pseudo instruction to the concrete instruction.
  MI->getOperand(0).setReg(getSTReg(Op1));
  MI->RemoveOperand(1);
  MI->setDesc(TII->get(getConcreteOpcode(MI->getOpcode())));

  // If any of the operands are killed by this instruction, free them.
  if (KillsOp0) freeStackSlotAfter(I, Op0);
  if (KillsOp1 && Op0 != Op1) freeStackSlotAfter(I, Op1);
}

/// handleCondMovFP - Handle two address conditional move instructions.  These
/// instructions move a st(i) register to st(0) iff a condition is true.  These
/// instructions require that the first operand is at the top of the stack, but
/// otherwise don't modify the stack at all.
void FPS::handleCondMovFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;

  unsigned Op0 = getFPReg(MI->getOperand(0));
  unsigned Op1 = getFPReg(MI->getOperand(2));
  bool KillsOp1 = MI->killsRegister(X86::FP0+Op1);

  // The first operand *must* be on the top of the stack.
  moveToTop(Op0, I);

  // Change the second operand to the stack register that the operand is in.
  // Change from the pseudo instruction to the concrete instruction.
  MI->RemoveOperand(0);
  MI->RemoveOperand(1);
  MI->getOperand(0).setReg(getSTReg(Op1));
  MI->setDesc(TII->get(getConcreteOpcode(MI->getOpcode())));

  // If we kill the second operand, make sure to pop it from the stack.
  if (Op0 != Op1 && KillsOp1) {
    // Get this value off of the register stack.
    freeStackSlotAfter(I, Op1);
  }
}


/// handleSpecialFP - Handle special instructions which behave unlike other
/// floating point instructions.  This is primarily intended for use by pseudo
/// instructions.
///
void FPS::handleSpecialFP(MachineBasicBlock::iterator &I) {
  MachineInstr *MI = I;
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown SpecialFP instruction!");
  case TargetOpcode::COPY: {
    // We handle three kinds of copies: FP <- FP, FP <- ST, and ST <- FP.
    const MachineOperand &MO1 = MI->getOperand(1);
    const MachineOperand &MO0 = MI->getOperand(0);
    unsigned DstST = MO0.getReg() - X86::ST0;
    unsigned SrcST = MO1.getReg() - X86::ST0;
    bool KillsSrc = MI->killsRegister(MO1.getReg());

    // ST = COPY FP. Set up a pending ST register.
    if (DstST < 8) {
      unsigned SrcFP = getFPReg(MO1);
      assert(isLive(SrcFP) && "Cannot copy dead register");
      assert(!MO0.isDead() && "Cannot copy to dead ST register");

      // Unallocated STs are marked as the nonexistent FP255.
      while (NumPendingSTs <= DstST)
        PendingST[NumPendingSTs++] = NumFPRegs;

      // STi could still be live from a previous inline asm.
      if (isScratchReg(PendingST[DstST])) {
        DEBUG(dbgs() << "Clobbering old ST in FP" << unsigned(PendingST[DstST])
                     << '\n');
        freeStackSlotBefore(MI, PendingST[DstST]);
      }

      // When the source is killed, allocate a scratch FP register.
      if (KillsSrc) {
        duplicatePendingSTBeforeKill(SrcFP, I);
        unsigned Slot = getSlot(SrcFP);
        unsigned SR = getScratchReg();
        PendingST[DstST] = SR;
        Stack[Slot] = SR;
        RegMap[SR] = Slot;
      } else
        PendingST[DstST] = SrcFP;
      break;
    }

    // FP = COPY ST. Extract fixed stack value.
    // Any instruction defining ST registers must have assigned them to a
    // scratch register.
    if (SrcST < 8) {
      unsigned DstFP = getFPReg(MO0);
      assert(!isLive(DstFP) && "Cannot copy ST to live FP register");
      assert(NumPendingSTs > SrcST && "Cannot copy from dead ST register");
      unsigned SrcFP = PendingST[SrcST];
      assert(isScratchReg(SrcFP) && "Expected ST in a scratch register");
      assert(isLive(SrcFP) && "Scratch holding ST is dead");

      // DstFP steals the stack slot from SrcFP.
      unsigned Slot = getSlot(SrcFP);
      Stack[Slot] = DstFP;
      RegMap[DstFP] = Slot;

      // Always treat the ST as killed.
      PendingST[SrcST] = NumFPRegs;
      while (NumPendingSTs && PendingST[NumPendingSTs - 1] == NumFPRegs)
        --NumPendingSTs;
      break;
    }

    // FP <- FP copy.
    unsigned DstFP = getFPReg(MO0);
    unsigned SrcFP = getFPReg(MO1);
    assert(isLive(SrcFP) && "Cannot copy dead register");
    if (KillsSrc) {
      // If the input operand is killed, we can just change the owner of the
      // incoming stack slot into the result.
      unsigned Slot = getSlot(SrcFP);
      Stack[Slot] = DstFP;
      RegMap[DstFP] = Slot;
    } else {
      // For COPY we just duplicate the specified value to a new stack slot.
      // This could be made better, but would require substantial changes.
      duplicateToTop(SrcFP, DstFP, I);
    }
    break;
  }

  case TargetOpcode::IMPLICIT_DEF: {
    // All FP registers must be explicitly defined, so load a 0 instead.
    unsigned Reg = MI->getOperand(0).getReg() - X86::FP0;
    DEBUG(dbgs() << "Emitting LD_F0 for implicit FP" << Reg << '\n');
    BuildMI(*MBB, I, MI->getDebugLoc(), TII->get(X86::LD_F0));
    pushReg(Reg);
    break;
  }

  case X86::FpPOP_RETVAL: {
    // The FpPOP_RETVAL instruction is used after calls that return a value on
    // the floating point stack. We cannot model this with ST defs since CALL
    // instructions have fixed clobber lists. This instruction is interpreted
    // to mean that there is one more live register on the stack than we
    // thought.
    //
    // This means that StackTop does not match the hardware stack between a
    // call and the FpPOP_RETVAL instructions.  We do tolerate FP instructions
    // between CALL and FpPOP_RETVAL as long as they don't overflow the
    // hardware stack.
    unsigned DstFP = getFPReg(MI->getOperand(0));

    // Move existing stack elements up to reflect reality.
    assert(StackTop < 8 && "Stack overflowed before FpPOP_RETVAL");
    if (StackTop) {
      std::copy_backward(Stack, Stack + StackTop, Stack + StackTop + 1);
      for (unsigned i = 0; i != NumFPRegs; ++i)
        ++RegMap[i];
    }
    ++StackTop;

    // DstFP is the new bottom of the stack.
    Stack[0] = DstFP;
    RegMap[DstFP] = 0;

    // DstFP will be killed by processBasicBlock if this was a dead def.
    break;
  }

  case TargetOpcode::INLINEASM: {
    // The inline asm MachineInstr currently only *uses* FP registers for the
    // 'f' constraint.  These should be turned into the current ST(x) register
    // in the machine instr.
    //
    // There are special rules for x87 inline assembly. The compiler must know
    // exactly how many registers are popped and pushed implicitly by the asm.
    // Otherwise it is not possible to restore the stack state after the inline
    // asm.
    //
    // There are 3 kinds of input operands:
    //
    // 1. Popped inputs. These must appear at the stack top in ST0-STn. A
    //    popped input operand must be in a fixed stack slot, and it is either
    //    tied to an output operand, or in the clobber list. The MI has ST use
    //    and def operands for these inputs.
    //
    // 2. Fixed inputs. These inputs appear in fixed stack slots, but are
    //    preserved by the inline asm. The fixed stack slots must be STn-STm
    //    following the popped inputs. A fixed input operand cannot be tied to
    //    an output or appear in the clobber list. The MI has ST use operands
    //    and no defs for these inputs.
    //
    // 3. Preserved inputs. These inputs use the "f" constraint which is
    //    represented as an FP register. The inline asm won't change these
    //    stack slots.
    //
    // Outputs must be in ST registers, FP outputs are not allowed. Clobbered
    // registers do not count as output operands. The inline asm changes the
    // stack as if it popped all the popped inputs and then pushed all the
    // output operands.

    // Scan the assembly for ST registers used, defined and clobbered. We can
    // only tell clobbers from defs by looking at the asm descriptor.
    unsigned STUses = 0, STDefs = 0, STClobbers = 0, STDeadDefs = 0;
    unsigned NumOps = 0;
    for (unsigned i = InlineAsm::MIOp_FirstOperand, e = MI->getNumOperands();
         i != e && MI->getOperand(i).isImm(); i += 1 + NumOps) {
      unsigned Flags = MI->getOperand(i).getImm();
      NumOps = InlineAsm::getNumOperandRegisters(Flags);
      if (NumOps != 1)
        continue;
      const MachineOperand &MO = MI->getOperand(i + 1);
      if (!MO.isReg())
        continue;
      unsigned STReg = MO.getReg() - X86::ST0;
      if (STReg >= 8)
        continue;

      switch (InlineAsm::getKind(Flags)) {
      case InlineAsm::Kind_RegUse:
        STUses |= (1u << STReg);
        break;
      case InlineAsm::Kind_RegDef:
      case InlineAsm::Kind_RegDefEarlyClobber:
        STDefs |= (1u << STReg);
        if (MO.isDead())
          STDeadDefs |= (1u << STReg);
        break;
      case InlineAsm::Kind_Clobber:
        STClobbers |= (1u << STReg);
        break;
      default:
        break;
      }
    }

    if (STUses && !isMask_32(STUses))
      MI->emitError("fixed input regs must be last on the x87 stack");
    unsigned NumSTUses = CountTrailingOnes_32(STUses);

    // Defs must be contiguous from the stack top. ST0-STn.
    if (STDefs && !isMask_32(STDefs)) {
      MI->emitError("output regs must be last on the x87 stack");
      STDefs = NextPowerOf2(STDefs) - 1;
    }
    unsigned NumSTDefs = CountTrailingOnes_32(STDefs);

    // So must the clobbered stack slots. ST0-STm, m >= n.
    if (STClobbers && !isMask_32(STDefs | STClobbers))
      MI->emitError("clobbers must be last on the x87 stack");

    // Popped inputs are the ones that are also clobbered or defined.
    unsigned STPopped = STUses & (STDefs | STClobbers);
    if (STPopped && !isMask_32(STPopped))
      MI->emitError("implicitly popped regs must be last on the x87 stack");
    unsigned NumSTPopped = CountTrailingOnes_32(STPopped);

    DEBUG(dbgs() << "Asm uses " << NumSTUses << " fixed regs, pops "
                 << NumSTPopped << ", and defines " << NumSTDefs << " regs.\n");

    // Scan the instruction for FP uses corresponding to "f" constraints.
    // Collect FP registers to kill afer the instruction.
    // Always kill all the scratch regs.
    unsigned FPKills = ((1u << NumFPRegs) - 1) & ~0xff;
    unsigned FPUsed = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || Op.getReg() < X86::FP0 || Op.getReg() > X86::FP6)
        continue;
      if (!Op.isUse())
        MI->emitError("illegal \"f\" output constraint");
      unsigned FPReg = getFPReg(Op);
      FPUsed |= 1U << FPReg;

      // If we kill this operand, make sure to pop it from the stack after the
      // asm.  We just remember it for now, and pop them all off at the end in
      // a batch.
      if (Op.isKill())
        FPKills |= 1U << FPReg;
    }

    // The popped inputs will be killed by the instruction, so duplicate them
    // if the FP register needs to be live after the instruction, or if it is
    // used in the instruction itself. We effectively treat the popped inputs
    // as early clobbers.
    for (unsigned i = 0; i < NumSTPopped; ++i) {
      if ((FPKills & ~FPUsed) & (1u << PendingST[i]))
        continue;
      unsigned SR = getScratchReg();
      duplicateToTop(PendingST[i], SR, I);
      DEBUG(dbgs() << "Duplicating ST" << i << " in FP"
                   << unsigned(PendingST[i]) << " to avoid clobbering it.\n");
      PendingST[i] = SR;
    }

    // Make sure we have a unique live register for every fixed use. Some of
    // them could be undef uses, and we need to emit LD_F0 instructions.
    for (unsigned i = 0; i < NumSTUses; ++i) {
      if (i < NumPendingSTs && PendingST[i] < NumFPRegs) {
        // Check for shared assignments.
        for (unsigned j = 0; j < i; ++j) {
          if (PendingST[j] != PendingST[i])
            continue;
          // STi and STj are inn the same register, create a copy.
          unsigned SR = getScratchReg();
          duplicateToTop(PendingST[i], SR, I);
          DEBUG(dbgs() << "Duplicating ST" << i << " in FP"
                       << unsigned(PendingST[i])
                       << " to avoid collision with ST" << j << '\n');
          PendingST[i] = SR;
        }
        continue;
      }
      unsigned SR = getScratchReg();
      DEBUG(dbgs() << "Emitting LD_F0 for ST" << i << " in FP" << SR << '\n');
      BuildMI(*MBB, I, MI->getDebugLoc(), TII->get(X86::LD_F0));
      pushReg(SR);
      PendingST[i] = SR;
      if (NumPendingSTs == i)
        ++NumPendingSTs;
    }
    assert(NumPendingSTs >= NumSTUses && "Fixed registers should be assigned");

    // Now we can rearrange the live registers to match what was requested.
    shuffleStackTop(PendingST, NumPendingSTs, I);
    DEBUG({dbgs() << "Before asm: "; dumpStack();});

    // With the stack layout fixed, rewrite the FP registers.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || Op.getReg() < X86::FP0 || Op.getReg() > X86::FP6)
        continue;
      unsigned FPReg = getFPReg(Op);
      Op.setReg(getSTReg(FPReg));
    }

    // Simulate the inline asm popping its inputs and pushing its outputs.
    StackTop -= NumSTPopped;

    // Hold the fixed output registers in scratch FP registers. They will be
    // transferred to real FP registers by copies.
    NumPendingSTs = 0;
    for (unsigned i = 0; i < NumSTDefs; ++i) {
      unsigned SR = getScratchReg();
      pushReg(SR);
      FPKills &= ~(1u << SR);
    }
    for (unsigned i = 0; i < NumSTDefs; ++i)
      PendingST[NumPendingSTs++] = getStackEntry(i);
    DEBUG({dbgs() << "After asm: "; dumpStack();});

    // If any of the ST defs were dead, pop them immediately. Our caller only
    // handles dead FP defs.
    MachineBasicBlock::iterator InsertPt = MI;
    for (unsigned i = 0; STDefs & (1u << i); ++i) {
      if (!(STDeadDefs & (1u << i)))
        continue;
      freeStackSlotAfter(InsertPt, PendingST[i]);
      PendingST[i] = NumFPRegs;
    }
    while (NumPendingSTs && PendingST[NumPendingSTs - 1] == NumFPRegs)
      --NumPendingSTs;

    // If this asm kills any FP registers (is the last use of them) we must
    // explicitly emit pop instructions for them.  Do this now after the asm has
    // executed so that the ST(x) numbers are not off (which would happen if we
    // did this inline with operand rewriting).
    //
    // Note: this might be a non-optimal pop sequence.  We might be able to do
    // better by trying to pop in stack order or something.
    while (FPKills) {
      unsigned FPReg = CountTrailingZeros_32(FPKills);
      if (isLive(FPReg))
        freeStackSlotAfter(InsertPt, FPReg);
      FPKills &= ~(1U << FPReg);
    }
    // Don't delete the inline asm!
    return;
  }

  case X86::WIN_FTOL_32:
  case X86::WIN_FTOL_64: {
    // Push the operand into ST0.
    MachineOperand &Op = MI->getOperand(0);
    assert(Op.isUse() && Op.isReg() &&
      Op.getReg() >= X86::FP0 && Op.getReg() <= X86::FP6);
    unsigned FPReg = getFPReg(Op);
    if (Op.isKill())
      moveToTop(FPReg, I);
    else
      duplicateToTop(FPReg, FPReg, I);

    // Emit the call. This will pop the operand.
    BuildMI(*MBB, I, MI->getDebugLoc(), TII->get(X86::CALLpcrel32))
      .addExternalSymbol("_ftol2")
      .addReg(X86::ST0, RegState::ImplicitKill)
      .addReg(X86::EAX, RegState::Define | RegState::Implicit)
      .addReg(X86::EDX, RegState::Define | RegState::Implicit)
      .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit);
    --StackTop;

    break;
  }

  case X86::RET:
  case X86::RETI:
    // If RET has an FP register use operand, pass the first one in ST(0) and
    // the second one in ST(1).

    // Find the register operands.
    unsigned FirstFPRegOp = ~0U, SecondFPRegOp = ~0U;
    unsigned LiveMask = 0;

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || Op.getReg() < X86::FP0 || Op.getReg() > X86::FP6)
        continue;
      // FP Register uses must be kills unless there are two uses of the same
      // register, in which case only one will be a kill.
      assert(Op.isUse() &&
             (Op.isKill() ||                        // Marked kill.
              getFPReg(Op) == FirstFPRegOp ||       // Second instance.
              MI->killsRegister(Op.getReg())) &&    // Later use is marked kill.
             "Ret only defs operands, and values aren't live beyond it");

      if (FirstFPRegOp == ~0U)
        FirstFPRegOp = getFPReg(Op);
      else {
        assert(SecondFPRegOp == ~0U && "More than two fp operands!");
        SecondFPRegOp = getFPReg(Op);
      }
      LiveMask |= (1 << getFPReg(Op));

      // Remove the operand so that later passes don't see it.
      MI->RemoveOperand(i);
      --i, --e;
    }

    // We may have been carrying spurious live-ins, so make sure only the returned
    // registers are left live.
    adjustLiveRegs(LiveMask, MI);
    if (!LiveMask) return;  // Quick check to see if any are possible.

    // There are only four possibilities here:
    // 1) we are returning a single FP value.  In this case, it has to be in
    //    ST(0) already, so just declare success by removing the value from the
    //    FP Stack.
    if (SecondFPRegOp == ~0U) {
      // Assert that the top of stack contains the right FP register.
      assert(StackTop == 1 && FirstFPRegOp == getStackEntry(0) &&
             "Top of stack not the right register for RET!");

      // Ok, everything is good, mark the value as not being on the stack
      // anymore so that our assertion about the stack being empty at end of
      // block doesn't fire.
      StackTop = 0;
      return;
    }

    // Otherwise, we are returning two values:
    // 2) If returning the same value for both, we only have one thing in the FP
    //    stack.  Consider:  RET FP1, FP1
    if (StackTop == 1) {
      assert(FirstFPRegOp == SecondFPRegOp && FirstFPRegOp == getStackEntry(0)&&
             "Stack misconfiguration for RET!");

      // Duplicate the TOS so that we return it twice.  Just pick some other FPx
      // register to hold it.
      unsigned NewReg = getScratchReg();
      duplicateToTop(FirstFPRegOp, NewReg, MI);
      FirstFPRegOp = NewReg;
    }

    /// Okay we know we have two different FPx operands now:
    assert(StackTop == 2 && "Must have two values live!");

    /// 3) If SecondFPRegOp is currently in ST(0) and FirstFPRegOp is currently
    ///    in ST(1).  In this case, emit an fxch.
    if (getStackEntry(0) == SecondFPRegOp) {
      assert(getStackEntry(1) == FirstFPRegOp && "Unknown regs live");
      moveToTop(FirstFPRegOp, MI);
    }

    /// 4) Finally, FirstFPRegOp must be in ST(0) and SecondFPRegOp must be in
    /// ST(1).  Just remove both from our understanding of the stack and return.
    assert(getStackEntry(0) == FirstFPRegOp && "Unknown regs live");
    assert(getStackEntry(1) == SecondFPRegOp && "Unknown regs live");
    StackTop = 0;
    return;
  }

  I = MBB->erase(I);  // Remove the pseudo instruction

  // We want to leave I pointing to the previous instruction, but what if we
  // just erased the first instruction?
  if (I == MBB->begin()) {
    DEBUG(dbgs() << "Inserting dummy KILL\n");
    I = BuildMI(*MBB, I, DebugLoc(), TII->get(TargetOpcode::KILL));
  } else
    --I;
}
