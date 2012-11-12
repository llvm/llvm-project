//===-- ShrinkWrapping.cpp - Reduce spills/restores of callee-saved regs --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a shrink wrapping variant of prolog/epilog insertion:
// - Spills and restores of callee-saved registers (CSRs) are placed in the
//   machine CFG to tightly surround their uses so that execution paths that
//   do not use CSRs do not pay the spill/restore penalty.
//
// - Avoiding placment of spills/restores in loops: if a CSR is used inside a
//   loop the spills are placed in the loop preheader, and restores are
//   placed in the loop exit nodes (the successors of loop _exiting_ nodes).
//
// - Covering paths without CSR uses:
//   If a region in a CFG uses CSRs and has multiple entry and/or exit points,
//   the use info for the CSRs inside the region is propagated outward in the
//   CFG to ensure validity of the spill/restore placements. This decreases
//   the effectiveness of shrink wrapping but does not require edge splitting
//   in the machine CFG.
//
// This shrink wrapping implementation uses an iterative analysis to determine
// which basic blocks require spills and restores for CSRs.
//
// This pass uses MachineDominators and MachineLoopInfo. Loop information
// is used to prevent placement of callee-saved register spills/restores
// in the bodies of loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "shrink-wrap"

#include "PrologEpilogInserter.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <sstream>

using namespace llvm;

STATISTIC(numSRReduced, "Number of CSR spills+restores reduced.");

// Shrink Wrapping:
static cl::opt<bool>
ShrinkWrapping("shrink-wrap",
               cl::desc("Shrink wrap callee-saved register spills/restores"));

// Shrink wrap only the specified function, a debugging aid.
static cl::opt<std::string>
ShrinkWrapFunc("shrink-wrap-func", cl::Hidden,
               cl::desc("Shrink wrap the specified function"),
               cl::value_desc("funcname"),
               cl::init(""));

// Debugging level for shrink wrapping.
enum ShrinkWrapDebugLevel {
  None, BasicInfo, Iterations, Details
};

static cl::opt<enum ShrinkWrapDebugLevel>
ShrinkWrapDebugging("shrink-wrap-dbg", cl::Hidden,
  cl::desc("Print shrink wrapping debugging information"),
  cl::values(
    clEnumVal(None      , "disable debug output"),
    clEnumVal(BasicInfo , "print basic DF sets"),
    clEnumVal(Iterations, "print SR sets for each iteration"),
    clEnumVal(Details   , "print all DF sets"),
    clEnumValEnd));


void PEI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  if (ShrinkWrapping || ShrinkWrapFunc != "") {
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<MachineDominatorTree>();
  }
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

//===----------------------------------------------------------------------===//
//  ShrinkWrapping implementation
//===----------------------------------------------------------------------===//

// Convienences for dealing with machine loops.
MachineBasicBlock* PEI::getTopLevelLoopPreheader(MachineLoop* LP) {
  assert(LP && "Machine loop is NULL.");
  MachineBasicBlock* PHDR = LP->getLoopPreheader();
  MachineLoop* PLP = LP->getParentLoop();
  while (PLP) {
    PHDR = PLP->getLoopPreheader();
    PLP = PLP->getParentLoop();
  }
  return PHDR;
}

MachineLoop* PEI::getTopLevelLoopParent(MachineLoop *LP) {
  if (LP == 0)
    return 0;
  MachineLoop* PLP = LP->getParentLoop();
  while (PLP) {
    LP = PLP;
    PLP = PLP->getParentLoop();
  }
  return LP;
}

bool PEI::isReturnBlock(MachineBasicBlock* MBB) {
  return (MBB && !MBB->empty() && MBB->back().isReturn());
}

// Initialize shrink wrapping DFA sets, called before iterations.
void PEI::clearAnticAvailSets() {
  AnticIn.clear();
  AnticOut.clear();
  AvailIn.clear();
  AvailOut.clear();
}

// Clear all sets constructed by shrink wrapping.
void PEI::clearAllSets() {
  ReturnBlocks.clear();
  clearAnticAvailSets();
  UsedCSRegs.clear();
  CSRUsed.clear();
  TLLoops.clear();
  CSRSave.clear();
  CSRRestore.clear();
}

// Initialize all shrink wrapping data.
void PEI::initShrinkWrappingInfo() {
  clearAllSets();
  EntryBlock = 0;
#ifndef NDEBUG
  HasFastExitPath = false;
#endif
  ShrinkWrapThisFunction = ShrinkWrapping;
  // DEBUG: enable or disable shrink wrapping for the current function
  // via --shrink-wrap-func=<funcname>.
#ifndef NDEBUG
  if (ShrinkWrapFunc != "") {
    std::string MFName = MF->getName().str();
    ShrinkWrapThisFunction = (MFName == ShrinkWrapFunc);
  }
#endif
}


/// placeCSRSpillsAndRestores - determine which MBBs of the function
/// need save, restore code for callee-saved registers by doing a DF analysis
/// similar to the one used in code motion (GVNPRE). This produces maps of MBBs
/// to sets of registers (CSRs) for saves and restores. MachineLoopInfo
/// is used to ensure that CSR save/restore code is not placed inside loops.
/// This function computes the maps of MBBs -> CSRs to spill and restore
/// in CSRSave, CSRRestore.
///
/// If shrink wrapping is not being performed, place all spills in
/// the entry block, all restores in return blocks. In this case,
/// CSRSave has a single mapping, CSRRestore has mappings for each
/// return block.
///
void PEI::placeCSRSpillsAndRestores(MachineFunction &Fn) {

  DEBUG(MF = &Fn);

  initShrinkWrappingInfo();

  DEBUG(if (ShrinkWrapThisFunction) {
      dbgs() << "Place CSR spills/restores for "
             << MF->getName() << "\n";
    });

  if (calculateSets(Fn))
    placeSpillsAndRestores(Fn);
}

/// calcAnticInOut - calculate the anticipated in/out reg sets
/// for the given MBB by looking forward in the MCFG at MBB's
/// successors.
///
bool PEI::calcAnticInOut(MachineBasicBlock* MBB) {
  bool changed = false;

  // AnticOut[MBB] = INTERSECT(AnticIn[S] for S in SUCCESSORS(MBB))
  SmallVector<MachineBasicBlock*, 4> successors;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;
    if (SUCC != MBB)
      successors.push_back(SUCC);
  }

  unsigned i = 0, e = successors.size();
  if (i != e) {
    CSRegSet prevAnticOut = AnticOut[MBB];
    MachineBasicBlock* SUCC = successors[i];

    AnticOut[MBB] = AnticIn[SUCC];
    for (++i; i != e; ++i) {
      SUCC = successors[i];
      AnticOut[MBB] &= AnticIn[SUCC];
    }
    if (prevAnticOut != AnticOut[MBB])
      changed = true;
  }

  // AnticIn[MBB] = UNION(CSRUsed[MBB], AnticOut[MBB]);
  CSRegSet prevAnticIn = AnticIn[MBB];
  AnticIn[MBB] = CSRUsed[MBB] | AnticOut[MBB];
  if (prevAnticIn != AnticIn[MBB])
    changed = true;
  return changed;
}

/// calcAvailInOut - calculate the available in/out reg sets
/// for the given MBB by looking backward in the MCFG at MBB's
/// predecessors.
///
bool PEI::calcAvailInOut(MachineBasicBlock* MBB) {
  bool changed = false;

  // AvailIn[MBB] = INTERSECT(AvailOut[P] for P in PREDECESSORS(MBB))
  SmallVector<MachineBasicBlock*, 4> predecessors;
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock* PRED = *PI;
    if (PRED != MBB)
      predecessors.push_back(PRED);
  }

  unsigned i = 0, e = predecessors.size();
  if (i != e) {
    CSRegSet prevAvailIn = AvailIn[MBB];
    MachineBasicBlock* PRED = predecessors[i];

    AvailIn[MBB] = AvailOut[PRED];
    for (++i; i != e; ++i) {
      PRED = predecessors[i];
      AvailIn[MBB] &= AvailOut[PRED];
    }
    if (prevAvailIn != AvailIn[MBB])
      changed = true;
  }

  // AvailOut[MBB] = UNION(CSRUsed[MBB], AvailIn[MBB]);
  CSRegSet prevAvailOut = AvailOut[MBB];
  AvailOut[MBB] = CSRUsed[MBB] | AvailIn[MBB];
  if (prevAvailOut != AvailOut[MBB])
    changed = true;
  return changed;
}

/// calculateAnticAvail - build the sets anticipated and available
/// registers in the MCFG of the current function iteratively,
/// doing a combined forward and backward analysis.
///
void PEI::calculateAnticAvail(MachineFunction &Fn) {
  // Initialize data flow sets.
  clearAnticAvailSets();

  // Calculate Antic{In,Out} and Avail{In,Out} iteratively on the MCFG.
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    ++iterations;
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;

      // Calculate anticipated in, out regs at MBB from
      // anticipated at successors of MBB.
      changed |= calcAnticInOut(MBB);

      // Calculate available in, out regs at MBB from
      // available at predecessors of MBB.
      changed |= calcAvailInOut(MBB);
    }
  }

  DEBUG({
      if (ShrinkWrapDebugging >= Details) {
        dbgs()
          << "-----------------------------------------------------------\n"
          << " Antic/Avail Sets:\n"
          << "-----------------------------------------------------------\n"
          << "iterations = " << iterations << "\n"
          << "-----------------------------------------------------------\n"
          << "MBB | USED | ANTIC_IN | ANTIC_OUT | AVAIL_IN | AVAIL_OUT\n"
          << "-----------------------------------------------------------\n";

        for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
             MBBI != MBBE; ++MBBI) {
          MachineBasicBlock* MBB = MBBI;
          dumpSets(MBB);
        }

        dbgs()
          << "-----------------------------------------------------------\n";
      }
    });
}

/// propagateUsesAroundLoop - copy used register info from MBB to all blocks
/// of the loop given by LP and its parent loops. This prevents spills/restores
/// from being placed in the bodies of loops.
///
void PEI::propagateUsesAroundLoop(MachineBasicBlock* MBB, MachineLoop* LP) {
  if (! MBB || !LP)
    return;

  std::vector<MachineBasicBlock*> loopBlocks = LP->getBlocks();
  for (unsigned i = 0, e = loopBlocks.size(); i != e; ++i) {
    MachineBasicBlock* LBB = loopBlocks[i];
    if (LBB == MBB)
      continue;
    if (CSRUsed[LBB].contains(CSRUsed[MBB]))
      continue;
    CSRUsed[LBB] |= CSRUsed[MBB];
  }
}

/// calculateSets - collect the CSRs used in this function, compute
/// the DF sets that describe the initial minimal regions in the
/// Machine CFG around which CSR spills and restores must be placed.
///
/// Additionally, this function decides if shrink wrapping should
/// be disabled for the current function, checking the following:
///  1. the current function has more than 500 MBBs: heuristic limit
///     on function size to reduce compile time impact of the current
///     iterative algorithm.
///  2. all CSRs are used in the entry block.
///  3. all CSRs are used in all immediate successors of the entry block.
///  4. all CSRs are used in a subset of blocks, each of which dominates
///     all return blocks. These blocks, taken as a subgraph of the MCFG,
///     are equivalent to the entry block since all execution paths pass
///     through them.
///
bool PEI::calculateSets(MachineFunction &Fn) {
  // Sets used to compute spill, restore placement sets.
  const std::vector<CalleeSavedInfo> CSI =
    Fn.getFrameInfo()->getCalleeSavedInfo();

  // If no CSRs used, we are done.
  if (CSI.empty()) {
    DEBUG(if (ShrinkWrapThisFunction)
            dbgs() << "DISABLED: " << Fn.getName()
                   << ": uses no callee-saved registers\n");
    return false;
  }

  // Save refs to entry and return blocks.
  EntryBlock = Fn.begin();
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB)
    if (isReturnBlock(MBB))
      ReturnBlocks.push_back(MBB);

  // Determine if this function has fast exit paths.
  DEBUG(if (ShrinkWrapThisFunction)
          findFastExitPath());

  // Limit shrink wrapping via the current iterative bit vector
  // implementation to functions with <= 500 MBBs.
  if (Fn.size() > 500) {
    DEBUG(if (ShrinkWrapThisFunction)
            dbgs() << "DISABLED: " << Fn.getName()
                   << ": too large (" << Fn.size() << " MBBs)\n");
    ShrinkWrapThisFunction = false;
  }

  // Return now if not shrink wrapping.
  if (! ShrinkWrapThisFunction)
    return false;

  // Collect set of used CSRs.
  for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
    UsedCSRegs.set(inx);
  }

  // Walk instructions in all MBBs, create CSRUsed[] sets, choose
  // whether or not to shrink wrap this function.
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();
  MachineDominatorTree &DT = getAnalysis<MachineDominatorTree>();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  bool allCSRUsesInEntryBlock = true;
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I) {
      for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
        unsigned Reg = CSI[inx].getReg();
        // If instruction I reads or modifies Reg, add it to UsedCSRegs,
        // CSRUsed map for the current block.
        for (unsigned opInx = 0, opEnd = I->getNumOperands();
             opInx != opEnd; ++opInx) {
          const MachineOperand &MO = I->getOperand(opInx);
          if (! (MO.isReg() && (MO.isUse() || MO.isDef())))
            continue;
          unsigned MOReg = MO.getReg();
          if (!MOReg)
            continue;
          if (MOReg == Reg ||
              (TargetRegisterInfo::isPhysicalRegister(MOReg) &&
               TargetRegisterInfo::isPhysicalRegister(Reg) &&
               TRI->isSubRegister(Reg, MOReg))) {
            // CSR Reg is defined/used in block MBB.
            CSRUsed[MBB].set(inx);
            // Check for uses in EntryBlock.
            if (MBB != EntryBlock)
              allCSRUsesInEntryBlock = false;
          }
        }
      }
    }

    if (CSRUsed[MBB].empty())
      continue;

    // Propagate CSRUsed[MBB] in loops
    if (MachineLoop* LP = LI.getLoopFor(MBB)) {
      // Add top level loop to work list.
      MachineBasicBlock* HDR = getTopLevelLoopPreheader(LP);
      MachineLoop* PLP = getTopLevelLoopParent(LP);

      if (! HDR) {
        HDR = PLP->getHeader();
        assert(HDR->pred_size() > 0 && "Loop header has no predecessors?");
        MachineBasicBlock::pred_iterator PI = HDR->pred_begin();
        HDR = *PI;
      }
      TLLoops[HDR] = PLP;

      // Push uses from inside loop to its parent loops,
      // or to all other MBBs in its loop.
      if (LP->getLoopDepth() > 1) {
        for (MachineLoop* PLP = LP->getParentLoop(); PLP;
             PLP = PLP->getParentLoop()) {
          propagateUsesAroundLoop(MBB, PLP);
        }
      } else {
        propagateUsesAroundLoop(MBB, LP);
      }
    }
  }

  if (allCSRUsesInEntryBlock) {
    DEBUG(dbgs() << "DISABLED: " << Fn.getName()
                 << ": all CSRs used in EntryBlock\n");
    ShrinkWrapThisFunction = false;
  } else {
    bool allCSRsUsedInEntryFanout = true;
    for (MachineBasicBlock::succ_iterator SI = EntryBlock->succ_begin(),
           SE = EntryBlock->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock* SUCC = *SI;
      if (CSRUsed[SUCC] != UsedCSRegs)
        allCSRsUsedInEntryFanout = false;
    }
    if (allCSRsUsedInEntryFanout) {
      DEBUG(dbgs() << "DISABLED: " << Fn.getName()
                   << ": all CSRs used in imm successors of EntryBlock\n");
      ShrinkWrapThisFunction = false;
    }
  }

  if (ShrinkWrapThisFunction) {
    // Check if MBB uses CSRs and dominates all exit nodes.
    // Such nodes are equiv. to the entry node w.r.t.
    // CSR uses: every path through the function must
    // pass through this node. If each CSR is used at least
    // once by these nodes, shrink wrapping is disabled.
    CSRegSet CSRUsedInChokePoints;
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;
      if (MBB == EntryBlock || CSRUsed[MBB].empty() || MBB->succ_size() < 1)
        continue;
      bool dominatesExitNodes = true;
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        if (! DT.dominates(MBB, ReturnBlocks[ri])) {
          dominatesExitNodes = false;
          break;
        }
      if (dominatesExitNodes) {
        CSRUsedInChokePoints |= CSRUsed[MBB];
        if (CSRUsedInChokePoints == UsedCSRegs) {
          DEBUG(dbgs() << "DISABLED: " << Fn.getName()
                       << ": all CSRs used in choke point(s) at "
                       << getBasicBlockName(MBB) << "\n");
          ShrinkWrapThisFunction = false;
          break;
        }
      }
    }
  }

  // Return now if we have decided not to apply shrink wrapping
  // to the current function.
  if (! ShrinkWrapThisFunction)
    return false;

  DEBUG({
      dbgs() << "ENABLED: " << Fn.getName();
      if (HasFastExitPath)
        dbgs() << " (fast exit path)";
      dbgs() << "\n";
      if (ShrinkWrapDebugging >= BasicInfo) {
        dbgs() << "------------------------------"
             << "-----------------------------\n";
        dbgs() << "UsedCSRegs = " << stringifyCSRegSet(UsedCSRegs) << "\n";
        if (ShrinkWrapDebugging >= Details) {
          dbgs() << "------------------------------"
               << "-----------------------------\n";
          dumpAllUsed();
        }
      }
    });

  // Build initial DF sets to determine minimal regions in the
  // Machine CFG around which CSRs must be spilled and restored.
  calculateAnticAvail(Fn);

  return true;
}

/// addUsesForMEMERegion - add uses of CSRs spilled or restored in
/// multi-entry, multi-exit (MEME) regions so spill and restore
/// placement will not break code that enters or leaves a
/// shrink-wrapped region by inducing spills with no matching
/// restores or restores with no matching spills. A MEME region
/// is a subgraph of the MCFG with multiple entry edges, multiple
/// exit edges, or both. This code propagates use information
/// through the MCFG until all paths requiring spills and restores
/// _outside_ the computed minimal placement regions have been covered.
///
bool PEI::addUsesForMEMERegion(MachineBasicBlock* MBB,
                               SmallVector<MachineBasicBlock*, 4>& blks) {
  if (MBB->succ_size() < 2 && MBB->pred_size() < 2) {
    bool processThisBlock = false;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock* SUCC = *SI;
      if (SUCC->pred_size() > 1) {
        processThisBlock = true;
        break;
      }
    }
    if (!CSRRestore[MBB].empty() && MBB->succ_size() > 0) {
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
        MachineBasicBlock* PRED = *PI;
        if (PRED->succ_size() > 1) {
          processThisBlock = true;
          break;
        }
      }
    }
    if (! processThisBlock)
      return false;
  }

  CSRegSet prop;
  if (!CSRSave[MBB].empty())
    prop = CSRSave[MBB];
  else if (!CSRRestore[MBB].empty())
    prop = CSRRestore[MBB];
  else
    prop = CSRUsed[MBB];
  if (prop.empty())
    return false;

  // Propagate selected bits to successors, predecessors of MBB.
  bool addedUses = false;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;
    // Self-loop
    if (SUCC == MBB)
      continue;
    if (! CSRUsed[SUCC].contains(prop)) {
      CSRUsed[SUCC] |= prop;
      addedUses = true;
      blks.push_back(SUCC);
      DEBUG(if (ShrinkWrapDebugging >= Iterations)
              dbgs() << getBasicBlockName(MBB)
                   << "(" << stringifyCSRegSet(prop) << ")->"
                   << "successor " << getBasicBlockName(SUCC) << "\n");
    }
  }
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock* PRED = *PI;
    // Self-loop
    if (PRED == MBB)
      continue;
    if (! CSRUsed[PRED].contains(prop)) {
      CSRUsed[PRED] |= prop;
      addedUses = true;
      blks.push_back(PRED);
      DEBUG(if (ShrinkWrapDebugging >= Iterations)
              dbgs() << getBasicBlockName(MBB)
                   << "(" << stringifyCSRegSet(prop) << ")->"
                   << "predecessor " << getBasicBlockName(PRED) << "\n");
    }
  }
  return addedUses;
}

/// addUsesForTopLevelLoops - add uses for CSRs used inside top
/// level loops to the exit blocks of those loops.
///
bool PEI::addUsesForTopLevelLoops(SmallVector<MachineBasicBlock*, 4>& blks) {
  bool addedUses = false;

  // Place restores for top level loops where needed.
  for (DenseMap<MachineBasicBlock*, MachineLoop*>::iterator
         I = TLLoops.begin(), E = TLLoops.end(); I != E; ++I) {
    MachineBasicBlock* MBB = I->first;
    MachineLoop* LP = I->second;
    MachineBasicBlock* HDR = LP->getHeader();
    SmallVector<MachineBasicBlock*, 4> exitBlocks;
    CSRegSet loopSpills;

    loopSpills = CSRSave[MBB];
    if (CSRSave[MBB].empty()) {
      loopSpills = CSRUsed[HDR];
      assert(!loopSpills.empty() && "No CSRs used in loop?");
    } else if (CSRRestore[MBB].contains(CSRSave[MBB]))
      continue;

    LP->getExitBlocks(exitBlocks);
    assert(exitBlocks.size() > 0 && "Loop has no top level exit blocks?");
    for (unsigned i = 0, e = exitBlocks.size(); i != e; ++i) {
      MachineBasicBlock* EXB = exitBlocks[i];
      if (! CSRUsed[EXB].contains(loopSpills)) {
        CSRUsed[EXB] |= loopSpills;
        addedUses = true;
        DEBUG(if (ShrinkWrapDebugging >= Iterations)
                dbgs() << "LOOP " << getBasicBlockName(MBB)
                     << "(" << stringifyCSRegSet(loopSpills) << ")->"
                     << getBasicBlockName(EXB) << "\n");
        if (EXB->succ_size() > 1 || EXB->pred_size() > 1)
          blks.push_back(EXB);
      }
    }
  }
  return addedUses;
}

/// calcSpillPlacements - determine which CSRs should be spilled
/// in MBB using AnticIn sets of MBB's predecessors, keeping track
/// of changes to spilled reg sets. Add MBB to the set of blocks
/// that need to be processed for propagating use info to cover
/// multi-entry/exit regions.
///
bool PEI::calcSpillPlacements(MachineBasicBlock* MBB,
                              SmallVector<MachineBasicBlock*, 4> &blks,
                              CSRegBlockMap &prevSpills) {
  bool placedSpills = false;
  // Intersect (CSRegs - AnticIn[P]) for P in Predecessors(MBB)
  CSRegSet anticInPreds;
  SmallVector<MachineBasicBlock*, 4> predecessors;
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock* PRED = *PI;
    if (PRED != MBB)
      predecessors.push_back(PRED);
  }
  unsigned i = 0, e = predecessors.size();
  if (i != e) {
    MachineBasicBlock* PRED = predecessors[i];
    anticInPreds = UsedCSRegs - AnticIn[PRED];
    for (++i; i != e; ++i) {
      PRED = predecessors[i];
      anticInPreds &= (UsedCSRegs - AnticIn[PRED]);
    }
  } else {
    // Handle uses in entry blocks (which have no predecessors).
    // This is necessary because the DFA formulation assumes the
    // entry and (multiple) exit nodes cannot have CSR uses, which
    // is not the case in the real world.
    anticInPreds = UsedCSRegs;
  }
  // Compute spills required at MBB:
  CSRSave[MBB] |= (AnticIn[MBB] - AvailIn[MBB]) & anticInPreds;

  if (! CSRSave[MBB].empty()) {
    if (MBB == EntryBlock) {
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        CSRRestore[ReturnBlocks[ri]] |= CSRSave[MBB];
    } else {
      // Reset all regs spilled in MBB that are also spilled in EntryBlock.
      if (CSRSave[EntryBlock].intersects(CSRSave[MBB])) {
        CSRSave[MBB] = CSRSave[MBB] - CSRSave[EntryBlock];
      }
    }
  }
  placedSpills = (CSRSave[MBB] != prevSpills[MBB]);
  prevSpills[MBB] = CSRSave[MBB];
  // Remember this block for adding restores to successor
  // blocks for multi-entry region.
  if (placedSpills)
    blks.push_back(MBB);

  DEBUG(if (! CSRSave[MBB].empty() && ShrinkWrapDebugging >= Iterations)
          dbgs() << "SAVE[" << getBasicBlockName(MBB) << "] = "
               << stringifyCSRegSet(CSRSave[MBB]) << "\n");

  return placedSpills;
}

/// calcRestorePlacements - determine which CSRs should be restored
/// in MBB using AvailOut sets of MBB's succcessors, keeping track
/// of changes to restored reg sets. Add MBB to the set of blocks
/// that need to be processed for propagating use info to cover
/// multi-entry/exit regions.
///
bool PEI::calcRestorePlacements(MachineBasicBlock* MBB,
                                SmallVector<MachineBasicBlock*, 4> &blks,
                                CSRegBlockMap &prevRestores) {
  bool placedRestores = false;
  // Intersect (CSRegs - AvailOut[S]) for S in Successors(MBB)
  CSRegSet availOutSucc;
  SmallVector<MachineBasicBlock*, 4> successors;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;
    if (SUCC != MBB)
      successors.push_back(SUCC);
  }
  unsigned i = 0, e = successors.size();
  if (i != e) {
    MachineBasicBlock* SUCC = successors[i];
    availOutSucc = UsedCSRegs - AvailOut[SUCC];
    for (++i; i != e; ++i) {
      SUCC = successors[i];
      availOutSucc &= (UsedCSRegs - AvailOut[SUCC]);
    }
  } else {
    if (! CSRUsed[MBB].empty() || ! AvailOut[MBB].empty()) {
      // Handle uses in return blocks (which have no successors).
      // This is necessary because the DFA formulation assumes the
      // entry and (multiple) exit nodes cannot have CSR uses, which
      // is not the case in the real world.
      availOutSucc = UsedCSRegs;
    }
  }
  // Compute restores required at MBB:
  CSRRestore[MBB] |= (AvailOut[MBB] - AnticOut[MBB]) & availOutSucc;

  // Postprocess restore placements at MBB.
  // Remove the CSRs that are restored in the return blocks.
  // Lest this be confusing, note that:
  // CSRSave[EntryBlock] == CSRRestore[B] for all B in ReturnBlocks.
  if (MBB->succ_size() && ! CSRRestore[MBB].empty()) {
    if (! CSRSave[EntryBlock].empty())
      CSRRestore[MBB] = CSRRestore[MBB] - CSRSave[EntryBlock];
  }
  placedRestores = (CSRRestore[MBB] != prevRestores[MBB]);
  prevRestores[MBB] = CSRRestore[MBB];
  // Remember this block for adding saves to predecessor
  // blocks for multi-entry region.
  if (placedRestores)
    blks.push_back(MBB);

  DEBUG(if (! CSRRestore[MBB].empty() && ShrinkWrapDebugging >= Iterations)
          dbgs() << "RESTORE[" << getBasicBlockName(MBB) << "] = "
               << stringifyCSRegSet(CSRRestore[MBB]) << "\n");

  return placedRestores;
}

/// placeSpillsAndRestores - place spills and restores of CSRs
/// used in MBBs in minimal regions that contain the uses.
///
void PEI::placeSpillsAndRestores(MachineFunction &Fn) {
  CSRegBlockMap prevCSRSave;
  CSRegBlockMap prevCSRRestore;
  SmallVector<MachineBasicBlock*, 4> cvBlocks, ncvBlocks;
  bool changed = true;
  unsigned iterations = 0;

  // Iterate computation of spill and restore placements in the MCFG until:
  //   1. CSR use info has been fully propagated around the MCFG, and
  //   2. computation of CSRSave[], CSRRestore[] reach fixed points.
  while (changed) {
    changed = false;
    ++iterations;

    DEBUG(if (ShrinkWrapDebugging >= Iterations)
            dbgs() << "iter " << iterations
                 << " --------------------------------------------------\n");

    // Calculate CSR{Save,Restore} sets using Antic, Avail on the MCFG,
    // which determines the placements of spills and restores.
    // Keep track of changes to spills, restores in each iteration to
    // minimize the total iterations.
    bool SRChanged = false;
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;

      // Place spills for CSRs in MBB.
      SRChanged |= calcSpillPlacements(MBB, cvBlocks, prevCSRSave);

      // Place restores for CSRs in MBB.
      SRChanged |= calcRestorePlacements(MBB, cvBlocks, prevCSRRestore);
    }

    // Add uses of CSRs used inside loops where needed.
    changed |= addUsesForTopLevelLoops(cvBlocks);

    // Add uses for CSRs spilled or restored at branch, join points.
    if (changed || SRChanged) {
      while (! cvBlocks.empty()) {
        MachineBasicBlock* MBB = cvBlocks.pop_back_val();
        changed |= addUsesForMEMERegion(MBB, ncvBlocks);
      }
      if (! ncvBlocks.empty()) {
        cvBlocks = ncvBlocks;
        ncvBlocks.clear();
      }
    }

    if (changed) {
      calculateAnticAvail(Fn);
      CSRSave.clear();
      CSRRestore.clear();
    }
  }

  // Check for effectiveness:
  //  SR0 = {r | r in CSRSave[EntryBlock], CSRRestore[RB], RB in ReturnBlocks}
  //  numSRReduced = |(UsedCSRegs - SR0)|, approx. SR0 by CSRSave[EntryBlock]
  // Gives a measure of how many CSR spills have been moved from EntryBlock
  // to minimal regions enclosing their uses.
  CSRegSet notSpilledInEntryBlock = (UsedCSRegs - CSRSave[EntryBlock]);
  unsigned numSRReducedThisFunc = notSpilledInEntryBlock.count();
  numSRReduced += numSRReducedThisFunc;
  DEBUG(if (ShrinkWrapDebugging >= BasicInfo) {
      dbgs() << "-----------------------------------------------------------\n";
      dbgs() << "total iterations = " << iterations << " ( "
           << Fn.getName()
           << " " << numSRReducedThisFunc
           << " " << Fn.size()
           << " )\n";
      dbgs() << "-----------------------------------------------------------\n";
      dumpSRSets();
      dbgs() << "-----------------------------------------------------------\n";
      if (numSRReducedThisFunc)
        verifySpillRestorePlacement();
    });
}

// Debugging methods.
#ifndef NDEBUG
/// findFastExitPath - debugging method used to detect functions
/// with at least one path from the entry block to a return block
/// directly or which has a very small number of edges.
///
void PEI::findFastExitPath() {
  if (! EntryBlock)
    return;
  // Fina a path from EntryBlock to any return block that does not branch:
  //        Entry
  //          |     ...
  //          v      |
  //         B1<-----+
  //          |
  //          v
  //       Return
  for (MachineBasicBlock::succ_iterator SI = EntryBlock->succ_begin(),
         SE = EntryBlock->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;

    // Assume positive, disprove existence of fast path.
    HasFastExitPath = true;

    // Check the immediate successors.
    if (isReturnBlock(SUCC)) {
      if (ShrinkWrapDebugging >= BasicInfo)
        dbgs() << "Fast exit path: " << getBasicBlockName(EntryBlock)
             << "->" << getBasicBlockName(SUCC) << "\n";
      break;
    }
    // Traverse df from SUCC, look for a branch block.
    std::string exitPath = getBasicBlockName(SUCC);
    for (df_iterator<MachineBasicBlock*> BI = df_begin(SUCC),
           BE = df_end(SUCC); BI != BE; ++BI) {
      MachineBasicBlock* SBB = *BI;
      // Reject paths with branch nodes.
      if (SBB->succ_size() > 1) {
        HasFastExitPath = false;
        break;
      }
      exitPath += "->" + getBasicBlockName(SBB);
    }
    if (HasFastExitPath) {
      if (ShrinkWrapDebugging >= BasicInfo)
        dbgs() << "Fast exit path: " << getBasicBlockName(EntryBlock)
             << "->" << exitPath << "\n";
      break;
    }
  }
}

/// verifySpillRestorePlacement - check the current spill/restore
/// sets for safety. Attempt to find spills without restores or
/// restores without spills.
/// Spills: walk df from each MBB in spill set ensuring that
///         all CSRs spilled at MMBB are restored on all paths
///         from MBB to all exit blocks.
/// Restores: walk idf from each MBB in restore set ensuring that
///           all CSRs restored at MBB are spilled on all paths
///           reaching MBB.
///
void PEI::verifySpillRestorePlacement() {
  unsigned numReturnBlocks = 0;
  for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    if (isReturnBlock(MBB) || MBB->succ_size() == 0)
      ++numReturnBlocks;
  }
  for (CSRegBlockMap::iterator BI = CSRSave.begin(),
         BE = CSRSave.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet spilled = BI->second;
    CSRegSet restored;

    if (spilled.empty())
      continue;

    DEBUG(dbgs() << "SAVE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(spilled)
                 << "  RESTORE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(CSRRestore[MBB]) << "\n");

    if (CSRRestore[MBB].intersects(spilled)) {
      restored |= (CSRRestore[MBB] & spilled);
    }

    // Walk depth first from MBB to find restores of all CSRs spilled at MBB:
    // we must find restores for all spills w/no intervening spills on all
    // paths from MBB to all return blocks.
    for (df_iterator<MachineBasicBlock*> BI = df_begin(MBB),
           BE = df_end(MBB); BI != BE; ++BI) {
      MachineBasicBlock* SBB = *BI;
      if (SBB == MBB)
        continue;
      // Stop when we encounter spills of any CSRs spilled at MBB that
      // have not yet been seen to be restored.
      if (CSRSave[SBB].intersects(spilled) &&
          !restored.contains(CSRSave[SBB] & spilled))
        break;
      // Collect the CSRs spilled at MBB that are restored
      // at this DF successor of MBB.
      if (CSRRestore[SBB].intersects(spilled))
        restored |= (CSRRestore[SBB] & spilled);
      // If we are at a retun block, check that the restores
      // we have seen so far exhaust the spills at MBB, then
      // reset the restores.
      if (isReturnBlock(SBB) || SBB->succ_size() == 0) {
        if (restored != spilled) {
          CSRegSet notRestored = (spilled - restored);
          DEBUG(dbgs() << MF->getName() << ": "
                       << stringifyCSRegSet(notRestored)
                       << " spilled at " << getBasicBlockName(MBB)
                       << " are never restored on path to return "
                       << getBasicBlockName(SBB) << "\n");
        }
        restored.clear();
      }
    }
  }

  // Check restore placements.
  for (CSRegBlockMap::iterator BI = CSRRestore.begin(),
         BE = CSRRestore.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet restored = BI->second;
    CSRegSet spilled;

    if (restored.empty())
      continue;

    DEBUG(dbgs() << "SAVE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(CSRSave[MBB])
                 << "  RESTORE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(restored) << "\n");

    if (CSRSave[MBB].intersects(restored)) {
      spilled |= (CSRSave[MBB] & restored);
    }
    // Walk inverse depth first from MBB to find spills of all
    // CSRs restored at MBB:
    for (idf_iterator<MachineBasicBlock*> BI = idf_begin(MBB),
           BE = idf_end(MBB); BI != BE; ++BI) {
      MachineBasicBlock* PBB = *BI;
      if (PBB == MBB)
        continue;
      // Stop when we encounter restores of any CSRs restored at MBB that
      // have not yet been seen to be spilled.
      if (CSRRestore[PBB].intersects(restored) &&
          !spilled.contains(CSRRestore[PBB] & restored))
        break;
      // Collect the CSRs restored at MBB that are spilled
      // at this DF predecessor of MBB.
      if (CSRSave[PBB].intersects(restored))
        spilled |= (CSRSave[PBB] & restored);
    }
    if (spilled != restored) {
      CSRegSet notSpilled = (restored - spilled);
      DEBUG(dbgs() << MF->getName() << ": "
                   << stringifyCSRegSet(notSpilled)
                   << " restored at " << getBasicBlockName(MBB)
                   << " are never spilled\n");
    }
  }
}

// Debugging print methods.
std::string PEI::getBasicBlockName(const MachineBasicBlock* MBB) {
  if (!MBB)
    return "";

  if (MBB->getBasicBlock())
    return MBB->getBasicBlock()->getName().str();

  std::ostringstream name;
  name << "_MBB_" << MBB->getNumber();
  return name.str();
}

std::string PEI::stringifyCSRegSet(const CSRegSet& s) {
  const TargetRegisterInfo* TRI = MF->getTarget().getRegisterInfo();
  const std::vector<CalleeSavedInfo> CSI =
    MF->getFrameInfo()->getCalleeSavedInfo();

  std::ostringstream srep;
  if (CSI.size() == 0) {
    srep << "[]";
    return srep.str();
  }
  srep << "[";
  CSRegSet::iterator I = s.begin(), E = s.end();
  if (I != E) {
    unsigned reg = CSI[*I].getReg();
    srep << TRI->getName(reg);
    for (++I; I != E; ++I) {
      reg = CSI[*I].getReg();
      srep << ",";
      srep << TRI->getName(reg);
    }
  }
  srep << "]";
  return srep.str();
}

void PEI::dumpSet(const CSRegSet& s) {
  DEBUG(dbgs() << stringifyCSRegSet(s) << "\n");
}

void PEI::dumpUsed(MachineBasicBlock* MBB) {
  DEBUG({
      if (MBB)
        dbgs() << "CSRUsed[" << getBasicBlockName(MBB) << "] = "
               << stringifyCSRegSet(CSRUsed[MBB])  << "\n";
    });
}

void PEI::dumpAllUsed() {
    for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;
      dumpUsed(MBB);
    }
}

void PEI::dumpSets(MachineBasicBlock* MBB) {
  DEBUG({
      if (MBB)
        dbgs() << getBasicBlockName(MBB)           << " | "
               << stringifyCSRegSet(CSRUsed[MBB])  << " | "
               << stringifyCSRegSet(AnticIn[MBB])  << " | "
               << stringifyCSRegSet(AnticOut[MBB]) << " | "
               << stringifyCSRegSet(AvailIn[MBB])  << " | "
               << stringifyCSRegSet(AvailOut[MBB]) << "\n";
    });
}

void PEI::dumpSets1(MachineBasicBlock* MBB) {
  DEBUG({
      if (MBB)
        dbgs() << getBasicBlockName(MBB)             << " | "
               << stringifyCSRegSet(CSRUsed[MBB])    << " | "
               << stringifyCSRegSet(AnticIn[MBB])    << " | "
               << stringifyCSRegSet(AnticOut[MBB])   << " | "
               << stringifyCSRegSet(AvailIn[MBB])    << " | "
               << stringifyCSRegSet(AvailOut[MBB])   << " | "
               << stringifyCSRegSet(CSRSave[MBB])    << " | "
               << stringifyCSRegSet(CSRRestore[MBB]) << "\n";
    });
}

void PEI::dumpAllSets() {
    for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;
      dumpSets1(MBB);
    }
}

void PEI::dumpSRSets() {
  DEBUG({
      for (MachineFunction::iterator MBB = MF->begin(), E = MF->end();
           MBB != E; ++MBB) {
        if (!CSRSave[MBB].empty()) {
          dbgs() << "SAVE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(CSRSave[MBB]);
          if (CSRRestore[MBB].empty())
            dbgs() << '\n';
        }

        if (!CSRRestore[MBB].empty() && !CSRSave[MBB].empty())
          dbgs() << "    "
                 << "RESTORE[" << getBasicBlockName(MBB) << "] = "
                 << stringifyCSRegSet(CSRRestore[MBB]) << "\n";
      }
    });
}
#endif
