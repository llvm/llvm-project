//===-- PrologEpilogInserter.h - Prolog/Epilog code insertion -*- C++ -* --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
// This pass also implements a shrink wrapping variant of prolog/epilog
// insertion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PEI_H
#define LLVM_CODEGEN_PEI_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class RegScavenger;
  class MachineBasicBlock;

  class PEI : public MachineFunctionPass {
  public:
    static char ID;
    PEI() : MachineFunctionPass(ID) {
      initializePEIPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn);

  private:
    RegScavenger *RS;

    // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee saved
    // stack frame indexes.
    unsigned MinCSFrameIndex, MaxCSFrameIndex;

    // Analysis info for spill/restore placement.
    // "CSR": "callee saved register".

    // CSRegSet contains indices into the Callee Saved Register Info
    // vector built by calculateCalleeSavedRegisters() and accessed
    // via MF.getFrameInfo()->getCalleeSavedInfo().
    typedef SparseBitVector<> CSRegSet;

    // CSRegBlockMap maps MachineBasicBlocks to sets of callee
    // saved register indices.
    typedef DenseMap<MachineBasicBlock*, CSRegSet> CSRegBlockMap;

    // Set and maps for computing CSR spill/restore placement:
    //  used in function (UsedCSRegs)
    //  used in a basic block (CSRUsed)
    //  anticipatable in a basic block (Antic{In,Out})
    //  available in a basic block (Avail{In,Out})
    //  to be spilled at the entry to a basic block (CSRSave)
    //  to be restored at the end of a basic block (CSRRestore)
    CSRegSet UsedCSRegs;
    CSRegBlockMap CSRUsed;
    CSRegBlockMap AnticIn, AnticOut;
    CSRegBlockMap AvailIn, AvailOut;
    CSRegBlockMap CSRSave;
    CSRegBlockMap CSRRestore;

    // Entry and return blocks of the current function.
    MachineBasicBlock* EntryBlock;
    SmallVector<MachineBasicBlock*, 4> ReturnBlocks;

    // Map of MBBs to top level MachineLoops.
    DenseMap<MachineBasicBlock*, MachineLoop*> TLLoops;

    // Flag to control shrink wrapping per-function:
    // may choose to skip shrink wrapping for certain
    // functions.
    bool ShrinkWrapThisFunction;

    // Flag to control whether to use the register scavenger to resolve
    // frame index materialization registers. Set according to
    // TRI->requiresFrameIndexScavenging() for the curren function.
    bool FrameIndexVirtualScavenging;

#ifndef NDEBUG
    // Machine function handle.
    MachineFunction* MF;

    // Flag indicating that the current function
    // has at least one "short" path in the machine
    // CFG from the entry block to an exit block.
    bool HasFastExitPath;
#endif

    bool calculateSets(MachineFunction &Fn);
    bool calcAnticInOut(MachineBasicBlock* MBB);
    bool calcAvailInOut(MachineBasicBlock* MBB);
    void calculateAnticAvail(MachineFunction &Fn);
    bool addUsesForMEMERegion(MachineBasicBlock* MBB,
                              SmallVector<MachineBasicBlock*, 4>& blks);
    bool addUsesForTopLevelLoops(SmallVector<MachineBasicBlock*, 4>& blks);
    bool calcSpillPlacements(MachineBasicBlock* MBB,
                             SmallVector<MachineBasicBlock*, 4> &blks,
                             CSRegBlockMap &prevSpills);
    bool calcRestorePlacements(MachineBasicBlock* MBB,
                               SmallVector<MachineBasicBlock*, 4> &blks,
                               CSRegBlockMap &prevRestores);
    void placeSpillsAndRestores(MachineFunction &Fn);
    void placeCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateCallsInformation(MachineFunction &Fn);
    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void insertCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void scavengeFrameVirtualRegs(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);

    // Initialize DFA sets, called before iterations.
    void clearAnticAvailSets();
    // Clear all sets constructed by shrink wrapping.
    void clearAllSets();

    // Initialize all shrink wrapping data.
    void initShrinkWrappingInfo();

    // Convienences for dealing with machine loops.
    MachineBasicBlock* getTopLevelLoopPreheader(MachineLoop* LP);
    MachineLoop* getTopLevelLoopParent(MachineLoop *LP);

    // Propgate CSRs used in MBB to all MBBs of loop LP.
    void propagateUsesAroundLoop(MachineBasicBlock* MBB, MachineLoop* LP);

    // Convenience for recognizing return blocks.
    bool isReturnBlock(MachineBasicBlock* MBB);

#ifndef NDEBUG
    // Debugging methods.

    // Mark this function as having fast exit paths.
    void findFastExitPath();

    // Verify placement of spills/restores.
    void verifySpillRestorePlacement();

    std::string getBasicBlockName(const MachineBasicBlock* MBB);
    std::string stringifyCSRegSet(const CSRegSet& s);
    void dumpSet(const CSRegSet& s);
    void dumpUsed(MachineBasicBlock* MBB);
    void dumpAllUsed();
    void dumpSets(MachineBasicBlock* MBB);
    void dumpSets1(MachineBasicBlock* MBB);
    void dumpAllSets();
    void dumpSRSets();
#endif

  };
} // End llvm namespace
#endif
