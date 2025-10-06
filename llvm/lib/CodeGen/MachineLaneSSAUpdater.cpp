//===- MachineLaneSSAUpdater.cpp - SSA repair for Machine IR (lane-aware) ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineLaneSSAUpdater - a universal SSA repair utility
// for Machine IR that handles both regular new definitions and reload-after-
// spill scenarios with full subregister lane awareness.
//
// Key features:
//  - Two explicit entry points: addDefAndRepairNewDef and addDefAndRepairAfterSpill
//  - Lane-aware PHI insertion with per-edge masks
//  - Pruned IDF computation (NewDefBlocks ∩ LiveIn(OldVR))
//  - Precise LiveInterval extension using captured EndPoints
//  - REG_SEQUENCE insertion only when necessary
//  - Preservation of undef/dead flags on partial definitions
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "machine-lane-ssa-updater"

using namespace llvm;

//===----------------------------------------------------------------------===//
// SpillCutCollector Implementation
//===----------------------------------------------------------------------===//

CutEndPoints SpillCutCollector::cut(Register OrigVReg, SlotIndex CutIdx, 
                                    LaneBitmask LanesToCut) {
  LLVM_DEBUG(dbgs() << "SpillCutCollector::cut VReg=" << OrigVReg 
                    << " at " << CutIdx << " lanes=" << PrintLaneMask(LanesToCut) << "\n");
  
  assert(OrigVReg.isVirtual() && "Only virtual registers can be cut for spilling");
  
  LiveInterval &LI = LIS.getInterval(OrigVReg);
  SmallVector<LaneBitmask, 4> TouchedLanes;
  SmallVector<LiveRange::Segment, 4> DebugSegsBefore;
  SmallVector<SlotIndex, 8> MainEndPoints;
  DenseMap<LaneBitmask, SmallVector<SlotIndex, 8>> SubrangeEndPoints;
  
  // Store debug information before pruning
  for (const LiveRange::Segment &S : LI.segments) {
    DebugSegsBefore.push_back(S);
  }
  
  // Use MRI to get the accurate full mask for this register class
  LaneBitmask RegClassFullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
  bool HasSubranges = !LI.subranges().empty();
  bool IsFullRegSpill = (LanesToCut == RegClassFullMask) || (!HasSubranges && MRI.shouldTrackSubRegLiveness(OrigVReg));
  
  LLVM_DEBUG(dbgs() << "  HasSubranges=" << HasSubranges 
                    << " RegClassFullMask=" << PrintLaneMask(RegClassFullMask)
                    << " shouldTrackSubRegLiveness=" << MRI.shouldTrackSubRegLiveness(OrigVReg)
                    << " IsFullRegSpill=" << IsFullRegSpill << "\n");
  
  if (IsFullRegSpill) {
    // Whole-register spill: prune main range only
  if (LI.liveAt(CutIdx)) {
      TouchedLanes.push_back(LanesToCut);
    LIS.pruneValue(LI, CutIdx, &MainEndPoints);
      LLVM_DEBUG(dbgs() << "  Pruned main range (whole-reg) with " << MainEndPoints.size() 
                      << " endpoints\n");
  }
  } else {
    // Partial-lane spill: refine-then-operate on subranges
    LLVM_DEBUG(dbgs() << "  Partial-lane spill: refining subranges for " 
                      << PrintLaneMask(LanesToCut) << "\n");
    
    // Step 1: Collect subranges that need refinement
    SmallVector<LiveInterval::SubRange *, 4> SubrangesToRefine;
    SmallVector<LiveInterval::SubRange *, 4> PreciseMatches;
    
  for (LiveInterval::SubRange &SR : LI.subranges()) {
      LaneBitmask Overlap = SR.LaneMask & LanesToCut;
      if (Overlap.none()) {
        continue; // No intersection, skip
      }
      
      if (Overlap == SR.LaneMask) {
        // SR is completely contained in LanesToCut
        PreciseMatches.push_back(&SR);
        LLVM_DEBUG(dbgs() << "    Found " << (SR.LaneMask == LanesToCut ? "precise" : "subset") 
                          << " match: " << PrintLaneMask(SR.LaneMask) << "\n");
      } else {
        // Partial overlap: need to refine this subrange
        SubrangesToRefine.push_back(&SR);
        LLVM_DEBUG(dbgs() << "    Need to refine: " << PrintLaneMask(SR.LaneMask) 
                          << " (overlap=" << PrintLaneMask(Overlap) << ")\n");
      }
    }
    
    // Step 2: Refine overlapping subranges into disjoint ones
    for (LiveInterval::SubRange *SR : SubrangesToRefine) {
      LaneBitmask OrigMask = SR->LaneMask;
      LaneBitmask SpillMask = OrigMask & LanesToCut;
      LaneBitmask KeepMask = OrigMask & ~LanesToCut;
      
      LLVM_DEBUG(dbgs() << "    Refining " << PrintLaneMask(OrigMask) 
                        << " into Spill=" << PrintLaneMask(SpillMask) 
                        << " Keep=" << PrintLaneMask(KeepMask) << "\n");
      
      // Create new subrange for spilled portion (SpillMask is always non-empty here)
      LiveInterval::SubRange *SpillSR = LI.createSubRange(LIS.getVNInfoAllocator(), SpillMask);
      // Copy liveness from original subrange
      SpillSR->assign(*SR, LIS.getVNInfoAllocator());
      PreciseMatches.push_back(SpillSR);
      LLVM_DEBUG(dbgs() << "      Created spill subrange: " << PrintLaneMask(SpillMask) << "\n");
      
      // Update original subrange to keep-only portion (KeepMask is always non-empty here)
      SR->LaneMask = KeepMask;
      LLVM_DEBUG(dbgs() << "      Updated original to keep: " << PrintLaneMask(KeepMask) << "\n");
    }
    
    // Step 3: Prune only the precise matches for LanesToCut
    for (LiveInterval::SubRange *SR : PreciseMatches) {
      if (SR->liveAt(CutIdx) && (SR->LaneMask & LanesToCut).any()) {
        TouchedLanes.push_back(SR->LaneMask);
        SmallVector<SlotIndex, 8> SubEndPoints;
        LIS.pruneValue(*SR, CutIdx, &SubEndPoints);
        SubrangeEndPoints[SR->LaneMask] = std::move(SubEndPoints);
        LLVM_DEBUG(dbgs() << "    Pruned subrange " << PrintLaneMask(SR->LaneMask)
                          << " with " << SubrangeEndPoints[SR->LaneMask].size() << " endpoints\n");
      }
    }
    
    // Note: Do NOT prune main range for partial spills - subranges are authoritative
  }
  
  LLVM_DEBUG(dbgs() << "  Cut complete: " << TouchedLanes.size() 
                    << " touched lane masks\n");
  
  return CutEndPoints(OrigVReg, CutIdx, std::move(TouchedLanes), 
                      std::move(MainEndPoints), std::move(SubrangeEndPoints),
                      std::move(DebugSegsBefore));
}

//===----------------------------------------------------------------------===//
// MachineLaneSSAUpdater Implementation
//===----------------------------------------------------------------------===//

Register MachineLaneSSAUpdater::addDefAndRepairNewDef(MachineInstr &NewDefMI,
                                                       Register OrigVReg,
                                                       LaneBitmask DefMask) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::addDefAndRepairNewDef VReg=" << OrigVReg
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Step 1: Index the new instruction in SlotIndexes/LIS
  indexNewInstr(NewDefMI);
  
  // Step 2: Extract the new SSA register from the definition instruction
  Register NewSSAVReg = NewDefMI.defs().begin()->getReg();
  assert(NewSSAVReg.isValid() && NewSSAVReg.isVirtual() &&
         "NewDefMI should define a valid virtual register");
  
  // Step 3: Derive necessary data from intact LiveIntervals
  // The LiveInterval should already exist and be properly computed
  if (!LIS.hasInterval(NewSSAVReg)) {
    LIS.createAndComputeVirtRegInterval(NewSSAVReg);
  }
  
  // Step 4: Perform common SSA repair (PHI placement + use rewriting)
  performSSARepair(NewSSAVReg, OrigVReg, DefMask, NewDefMI.getParent());
  
  LLVM_DEBUG(dbgs() << "  New def SSA repair complete, returning " << NewSSAVReg << "\n");
  return NewSSAVReg;
}

Register MachineLaneSSAUpdater::addDefAndRepairAfterSpill(MachineInstr &ReloadMI,
                                                           Register OrigVReg,
                                                           LaneBitmask DefMask,
                                                           const CutEndPoints &EP) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::addDefAndRepairAfterSpill VReg=" << OrigVReg
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Safety checks as specified in the design
  assert(EP.getOrigVReg() == OrigVReg && 
         "CutEndPoints OrigVReg mismatch");
  
  // Validate that DefMask is a subset of the lanes that were actually spilled
  // This allows partial reloads (e.g., reload 32-bit subreg from 64-bit spill)
  LaneBitmask SpilledLanes = LaneBitmask::getNone();
  for (LaneBitmask TouchedMask : EP.getTouchedLaneMasks()) {
    SpilledLanes |= TouchedMask;
  }
  assert((DefMask & SpilledLanes) == DefMask && 
         "DefMask must be a subset of the lanes that were spilled");
  
  LLVM_DEBUG(dbgs() << "  DefMask=" << PrintLaneMask(DefMask) 
                    << " is subset of SpilledLanes=" << PrintLaneMask(SpilledLanes) << "\n");
  
  // Step 1: Index the reload instruction and get its SlotIndex
  SlotIndex ReloadIdx = indexNewInstr(ReloadMI);
  assert(ReloadIdx >= EP.getCutIdx() && 
         "Reload index must be >= cut index");
  
  // Step 2: Extract the new SSA register from the reload instruction
  // The caller should have already created NewVReg and built ReloadMI with it
  Register NewSSAVReg = ReloadMI.defs().begin()->getReg();
  assert(NewSSAVReg.isValid() && NewSSAVReg.isVirtual() &&
         "ReloadMI should define a valid virtual register");
  
  // Step 3: Create and extend NewSSAVReg's LiveInterval using captured EndPoints
  // The endpoints capture where the original register was live after the spill point
  // We need to reconstruct this liveness for the new SSA register
  LiveInterval &NewLI = LIS.createAndComputeVirtRegInterval(NewSSAVReg);
  
  // Extend main live range using the captured endpoints
  if (!EP.getMainEndPoints().empty()) {
    LIS.extendToIndices(NewLI, EP.getMainEndPoints());
    LLVM_DEBUG(dbgs() << "  Extended NewSSA main range with " << EP.getMainEndPoints().size() 
                      << " endpoints\n");
  }
  
  // Extend subranges for lane-aware liveness reconstruction
  // Create subranges on-demand for each LaneMask that was captured during spill
  for (const auto &[LaneMask, EndPoints] : EP.getSubrangeEndPoints()) {
    if (!EndPoints.empty()) {
      // Always create a new subrange since NewLI.subranges() is initially empty
      LiveInterval::SubRange *NewSR = NewLI.createSubRange(LIS.getVNInfoAllocator(), LaneMask);
      
      LIS.extendToIndices(*NewSR, EndPoints);
      LLVM_DEBUG(dbgs() << "  Created and extended NewSSA subrange " << PrintLaneMask(LaneMask)
                        << " with " << EndPoints.size() << " endpoints\n");
    }
  }
  
  // Step 4: Perform common SSA repair (PHI placement + use rewriting)
  performSSARepair(NewSSAVReg, OrigVReg, DefMask, ReloadMI.getParent());
  
  LLVM_DEBUG(dbgs() << "  SSA repair complete, returning " << NewSSAVReg << "\n");
  return NewSSAVReg;
}

//===----------------------------------------------------------------------===//
// Common SSA Repair Logic
//===----------------------------------------------------------------------===//

void MachineLaneSSAUpdater::performSSARepair(Register NewVReg, Register OrigVReg, 
                                              LaneBitmask DefMask, MachineBasicBlock *DefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::performSSARepair NewVReg=" << NewVReg
                    << " OrigVReg=" << OrigVReg << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Step 1: Use worklist-driven PHI placement
  SmallVector<Register> AllPHIVRegs = insertLaneAwarePHI(NewVReg, OrigVReg, DefMask, DefBB);
  
  // Step 2: Rewrite dominated uses once for each new register
  rewriteDominatedUses(OrigVReg, NewVReg, DefMask);
  for (Register PHIVReg : AllPHIVRegs) {
    rewriteDominatedUses(OrigVReg, PHIVReg, DefMask);
  }
  
  // Step 3: Renumber values if needed
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  NewLI.RenumberValues();
  
  // Also renumber PHI intervals
  for (Register PHIVReg : AllPHIVRegs) {
    if (LIS.hasInterval(PHIVReg)) {
      LiveInterval &PHILI = LIS.getInterval(PHIVReg);
      PHILI.RenumberValues();
    }
  }
  
  // Also renumber original interval if it was modified
  LiveInterval &OrigLI = LIS.getInterval(OrigVReg);
  OrigLI.RenumberValues();
  
  // Step 4: Verification if enabled
  if (VerifyOnExit) {
    LLVM_DEBUG(dbgs() << "  Verifying after SSA repair...\n");
    // TODO: Add verification calls
  }
  
  LLVM_DEBUG(dbgs() << "  performSSARepair complete\n");
}

//===----------------------------------------------------------------------===//
// Internal Helper Methods (Stubs)
//===----------------------------------------------------------------------===//

SlotIndex MachineLaneSSAUpdater::indexNewInstr(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::indexNewInstr: " << MI);
  
  // Register the instruction in SlotIndexes and LiveIntervals
  // This is typically done automatically when instructions are inserted,
  // but we need to ensure it's properly indexed
  SlotIndexes *SI = LIS.getSlotIndexes();
  
  // Check if instruction is already indexed
  if (SI->hasIndex(MI)) {
    SlotIndex Idx = SI->getInstructionIndex(MI);
    LLVM_DEBUG(dbgs() << "  Already indexed at " << Idx << "\n");
    return Idx;
  }
  
  // Insert the instruction in maps - this should be done by the caller
  // before calling our SSA repair methods, but we can verify
  LIS.InsertMachineInstrInMaps(MI);
  
  SlotIndex Idx = SI->getInstructionIndex(MI);
  LLVM_DEBUG(dbgs() << "  Indexed at " << Idx << "\n");
  return Idx;
}

void MachineLaneSSAUpdater::extendPreciselyAt(const Register VReg,
                                              const SmallVector<LaneBitmask, 4> &LaneMasks,
                                              const MachineInstr &AtMI) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::extendPreciselyAt VReg=" << VReg 
                    << " at " << LIS.getInstructionIndex(AtMI) << "\n");
  
  if (!VReg.isVirtual()) {
    return; // Only handle virtual registers
  }
  
  SlotIndex DefIdx = LIS.getInstructionIndex(AtMI).getRegSlot();
  
  // Create or get the LiveInterval for this register
  LiveInterval &LI = LIS.getInterval(VReg);
  
  // Extend the main live range to include the definition point
  SmallVector<SlotIndex, 2> DefPoint = { DefIdx };
  LIS.extendToIndices(LI, DefPoint);
  
  // For each lane mask, ensure appropriate subranges exist and are extended
  // For now, assume all lanes are valid - we'll refine this later based on register class
  LaneBitmask RegCoverageMask = MF.getRegInfo().getMaxLaneMaskForVReg(VReg);
  
  for (LaneBitmask LaneMask : LaneMasks) {
    if (LaneMask == MF.getRegInfo().getMaxLaneMaskForVReg(VReg) || LaneMask == LaneBitmask::getNone()) {
      continue; // Main range handles getAll(), skip getNone()
    }
    
    // Only process lanes that are valid for this register class
    LaneBitmask ValidLanes = LaneMask & RegCoverageMask;
    if (ValidLanes.none()) {
      continue;
    }
    
    // Find or create the appropriate subrange
    LiveInterval::SubRange *SR = nullptr;
    for (LiveInterval::SubRange &Sub : LI.subranges()) {
      if (Sub.LaneMask == ValidLanes) {
        SR = &Sub;
        break;
      }
    }
    if (!SR) {
      SR = LI.createSubRange(LIS.getVNInfoAllocator(), ValidLanes);
    }
    
    // Extend this subrange to include the definition point
    LIS.extendToIndices(*SR, DefPoint);
    
    LLVM_DEBUG(dbgs() << "  Extended subrange " << PrintLaneMask(ValidLanes) << "\n");
  }
  
  LLVM_DEBUG(dbgs() << "  LiveInterval extension complete\n");
}

void MachineLaneSSAUpdater::computePrunedIDF(Register OrigVReg,
                                              LaneBitmask DefMask,
                                              ArrayRef<MachineBasicBlock *> NewDefBlocks,
                                              SmallVectorImpl<MachineBasicBlock *> &OutIDFBlocks) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::computePrunedIDF VReg=" << OrigVReg 
                    << " DefMask=" << PrintLaneMask(DefMask)
                    << " with " << NewDefBlocks.size() << " new def blocks\n");
  
  // Clear output vector at entry
  OutIDFBlocks.clear();
  
  // Early bail-out checks for robustness
  if (!OrigVReg.isVirtual()) {
    LLVM_DEBUG(dbgs() << "  Skipping non-virtual register\n");
    return;
  }
  
  if (!LIS.hasInterval(OrigVReg)) {
    LLVM_DEBUG(dbgs() << "  OrigVReg not tracked by LiveIntervals, bailing out\n");
    return;
  }
  
  // Get the main LiveInterval for OrigVReg
  LiveInterval &LI = LIS.getInterval(OrigVReg);
  
  // Build prune set: blocks where specified lanes (DefMask) are live-in at entry
  SmallPtrSet<MachineBasicBlock *, 32> LiveIn;
  for (MachineBasicBlock &BB : MF) {
    SlotIndex Start = LIS.getMBBStartIdx(&BB);
    
    // Collect live lanes at block entry
    LaneBitmask LiveLanes = LaneBitmask::getNone();
    
    if (DefMask == MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg)) {
      // For full register (e.g., reload case), check main interval
      if (LI.liveAt(Start)) {
        LiveLanes = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg);
      }
    } else {
      // For specific lanes, check subranges
      for (LiveInterval::SubRange &S : LI.subranges()) {
        if (S.liveAt(Start)) {
          LiveLanes |= S.LaneMask;
        }
      }
      
      // If no subranges found but main interval is live,
      // assume all lanes are covered by the main interval
      if (LiveLanes == LaneBitmask::getNone() && LI.liveAt(Start)) {
        LiveLanes = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg);
      }
    }
    
    // Check if any of the requested lanes (DefMask) are live
    if ((LiveLanes & DefMask).any()) {
      LiveIn.insert(&BB);
    }
  }
  
  // Seed set: the blocks where new defs exist (e.g., reload or prior PHIs)
  SmallPtrSet<MachineBasicBlock *, 8> DefBlocks;
  for (MachineBasicBlock *B : NewDefBlocks) {
    if (B) { // Robust to null entries
      DefBlocks.insert(B);
    }
  }
  
  // Early exit if either set is empty
  if (DefBlocks.empty() || LiveIn.empty()) {
    LLVM_DEBUG(dbgs() << "  DefBlocks=" << DefBlocks.size() << " LiveIn=" << LiveIn.size() 
                      << ", early exit\n");
    return;
  }
  
  LLVM_DEBUG(dbgs() << "  DefBlocks=" << DefBlocks.size() << " LiveIn=" << LiveIn.size() << "\n");
  
  // Use LLVM's IDFCalculatorBase for MachineBasicBlock with forward dominance
  using NodeTy = MachineBasicBlock;
  
  // Access the underlying DomTreeBase from MachineDominatorTree
  // MachineDominatorTree inherits from DomTreeBase<MachineBasicBlock>
  DomTreeBase<NodeTy> &DT = MDT;
  
  // Compute pruned IDF (forward dominance, IsPostDom=false)
  llvm::IDFCalculatorBase<NodeTy, /*IsPostDom=*/false> IDF(DT);
  IDF.setDefiningBlocks(DefBlocks);
  IDF.setLiveInBlocks(LiveIn);
  IDF.calculate(OutIDFBlocks);
  
  LLVM_DEBUG(dbgs() << "  Computed " << OutIDFBlocks.size() << " IDF blocks\n");
  
  // Note: We do not place PHIs here; this function only computes candidate 
  // join blocks. The IDFCalculator handles deduplication automatically.
}

SmallVector<Register> MachineLaneSSAUpdater::insertLaneAwarePHI(Register InitialVReg,
                                                                Register OrigVReg,
                                                                LaneBitmask DefMask,
                                                                MachineBasicBlock *InitialDefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::insertLaneAwarePHI InitialVReg=" << InitialVReg
                    << " OrigVReg=" << OrigVReg << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Worklist item: (VReg, DefBB) pairs that need PHI placement
  struct WorkItem {
    Register VReg;
    MachineBasicBlock *DefBB;
    WorkItem(Register V, MachineBasicBlock *BB) : VReg(V), DefBB(BB) {}
  };
  
  SmallVector<Register> AllCreatedPHIs;
  SmallVector<WorkItem> Worklist;
  DenseSet<MachineBasicBlock *> ProcessedBlocks; // Avoid duplicate PHIs in same block
  
  // Seed worklist with initial definition
  Worklist.emplace_back(InitialVReg, InitialDefBB);
  
  LLVM_DEBUG(dbgs() << "  Starting worklist processing...\n");
  
  while (!Worklist.empty()) {
    WorkItem Item = Worklist.pop_back_val();
    
    LLVM_DEBUG(dbgs() << "  Processing VReg=" << Item.VReg 
                      << " DefBB=#" << Item.DefBB->getNumber() << "\n");
    
    // Step 1: Compute pruned IDF for this definition
    SmallVector<MachineBasicBlock *> DefBlocks = {Item.DefBB};
    SmallVector<MachineBasicBlock *> IDFBlocks;
    computePrunedIDF(OrigVReg, DefMask, DefBlocks, IDFBlocks);
    
    LLVM_DEBUG(dbgs() << "    Found " << IDFBlocks.size() << " IDF blocks\n");
    
    // Step 2: Create PHIs in each IDF block
    for (MachineBasicBlock *JoinMBB : IDFBlocks) {
      // Skip if we already processed this join block (avoid duplicate PHIs)
      if (ProcessedBlocks.contains(JoinMBB)) {
        LLVM_DEBUG(dbgs() << "    Skipping already processed BB#" << JoinMBB->getNumber() << "\n");
        continue;
      }
      ProcessedBlocks.insert(JoinMBB);
      
      LLVM_DEBUG(dbgs() << "    Creating PHI in BB#" << JoinMBB->getNumber() << "\n");
      
      // Create PHI using the original per-edge analysis logic
      Register PHIResult = createPHIInBlock(*JoinMBB, OrigVReg, Item.VReg);
      
      // Add PHI result to worklist for further processing and to result collection
      if (PHIResult.isValid()) {
        Worklist.emplace_back(PHIResult, JoinMBB);
        AllCreatedPHIs.push_back(PHIResult);
        LLVM_DEBUG(dbgs() << "      Created PHI result VReg=" << PHIResult 
                          << ", added to worklist\n");
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "  Worklist processing complete. Created " 
                    << AllCreatedPHIs.size() << " PHI registers total.\n");
  
  return AllCreatedPHIs;
}

// Helper: Create lane-specific PHI in a join block
Register MachineLaneSSAUpdater::createPHIInBlock(MachineBasicBlock &JoinMBB,
                                                 Register OrigVReg,
                                                 Register NewVReg) {
  LLVM_DEBUG(dbgs() << "    createPHIInBlock in BB#" << JoinMBB.getNumber()
                    << " OrigVReg=" << OrigVReg << " NewVReg=" << NewVReg << "\n");
  
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const LaneBitmask FullMask = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg);
  
  // Derive DefMask from NewVReg's register class (matches reload size)
  const LaneBitmask ReloadMask = MF.getRegInfo().getMaxLaneMaskForVReg(NewVReg);
  const bool IsPartialReload = (FullMask & ~ReloadMask).any();
  
  // Collect PHI operands for the specific reload lanes
  SmallVector<MachineOperand> PHIOperands;
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  
  LLVM_DEBUG(dbgs() << "      Creating PHI for " << (IsPartialReload ? "partial reload" : "full reload")
                    << " ReloadMask=" << PrintLaneMask(ReloadMask) << "\n");
  
  for (MachineBasicBlock *PredMBB : JoinMBB.predecessors()) {
    // Check if NewVReg (reloaded register) is live-out from this predecessor
    bool NewVRegLive = LIS.isLiveOutOfMBB(NewLI, PredMBB);
    
    if (NewVRegLive) {
      // This is the reload path - use NewVReg (always full register for its class)
      LLVM_DEBUG(dbgs() << "        Pred BB#" << PredMBB->getNumber() 
                        << " contributes NewVReg (reload path)\n");
      
      PHIOperands.push_back(MachineOperand::CreateReg(NewVReg, /*isDef*/ false));
      PHIOperands.push_back(MachineOperand::CreateMBB(PredMBB));
      
    } else {
      // This is the original path - use OrigVReg with appropriate subregister
      LLVM_DEBUG(dbgs() << "        Pred BB#" << PredMBB->getNumber() 
                        << " contributes OrigVReg (original path)\n");
      
      if (IsPartialReload) {
        // Partial case: z = PHI(y, BB1, x.sub0, BB0)
        unsigned SubIdx = getSubRegIndexForLaneMask(ReloadMask, &TRI);
        PHIOperands.push_back(MachineOperand::CreateReg(OrigVReg, /*isDef*/ false,
                                                       /*isImp*/ false, /*isKill*/ false,
                                                       /*isDead*/ false, /*isUndef*/ false,
                                                       /*isEarlyClobber*/ false, SubIdx));
      } else {
        // Full register case: z = PHI(y, BB1, x, BB0)
        PHIOperands.push_back(MachineOperand::CreateReg(OrigVReg, /*isDef*/ false));
      }
      PHIOperands.push_back(MachineOperand::CreateMBB(PredMBB));
    }
  }
  
  // Create the single lane-specific PHI
  if (!PHIOperands.empty()) {
    const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(NewVReg);
    Register PHIVReg = MF.getRegInfo().createVirtualRegister(RC);
    
    auto PHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                          TII->get(TargetOpcode::PHI), PHIVReg);
    for (const MachineOperand &Op : PHIOperands) {
      PHINode.add(Op);
    }
    
    MachineInstr *PHI = PHINode.getInstr();
    LIS.InsertMachineInstrInMaps(*PHI);
    LIS.createAndComputeVirtRegInterval(PHIVReg);
    
    LLVM_DEBUG(dbgs() << "      Created lane-specific PHI: ");
    LLVM_DEBUG(PHI->print(dbgs()));
    return PHIVReg;
  }
  
  return Register();
}

void MachineLaneSSAUpdater::rewriteDominatedUses(Register OrigVReg,
                                                  Register NewSSA,
                                                  LaneBitmask MaskToRewrite) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::rewriteDominatedUses OrigVReg=" << OrigVReg
                    << " NewSSA=" << NewSSA << " Mask=" << PrintLaneMask(MaskToRewrite) << "\n");
  
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  // Find the definition instruction for NewSSA
  MachineInstr *DefMI = MRI.getVRegDef(NewSSA);
  if (!DefMI) {
    LLVM_DEBUG(dbgs() << "  No definition found for NewSSA, skipping\n");
    return;
  }
  
  // Get the LiveInterval and VNInfo for the definition
  LiveInterval &LI = LIS.getInterval(OrigVReg);
  SlotIndex DefIdx = LIS.getInstructionIndex(*DefMI).getRegSlot();
  VNInfo *VNI = LI.getVNInfoAt(DefIdx);
  if (!VNI) {
    LLVM_DEBUG(dbgs() << "  No VNInfo found for definition, skipping\n");
    return;
  }

  const TargetRegisterClass *NewRC = MRI.getRegClass(NewSSA);

  LLVM_DEBUG(dbgs() << "  Rewriting uses reached by VNI " << VNI->id << " from: ");
  LLVM_DEBUG(DefMI->print(dbgs()));

  // Iterate through all uses of OrigVReg
  for (MachineOperand &MO : llvm::make_early_inc_range(MRI.use_operands(OrigVReg))) {
    MachineInstr *UseMI = MO.getParent();
    
    // Skip the definition instruction itself
    if (UseMI == DefMI)
      continue;

    // Check if this use is reached by our VNI
    if (!reachedByThisVNI(LI, DefMI, UseMI, MO, VNI))
      continue;

    // Get the lane mask for this operand
    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    LLVM_DEBUG(dbgs() << "    Processing use with OpMask=" << PrintLaneMask(OpMask) << ": ");
    LLVM_DEBUG(UseMI->print(dbgs()));

    const TargetRegisterClass *OpRC = MRI.getRegClass(OrigVReg);

    // Case 1: Exact match - direct replacement
    if (OpMask == MaskToRewrite) {
      // Check register class compatibility
      if (TRI.getCommonSubClass(NewRC, OpRC)) {
        LLVM_DEBUG(dbgs() << "      Exact match -> direct replacement\n");
        MO.setReg(NewSSA);
        MO.setSubReg(0); // Clear subregister
        continue;
      }
      
      // Incompatible register classes with same lane mask indicates corrupted MIR
      llvm_unreachable("Incompatible register classes with same lane mask - invalid MIR");
    }

    // Case 2: Super/Mixed - use needs more lanes than we're rewriting
    if ((OpMask & ~MaskToRewrite).any()) {
      LLVM_DEBUG(dbgs() << "      Super/Mixed case -> building REG_SEQUENCE\n");
      
      SmallVector<LaneBitmask, 4> LanesToExtend;
      SlotIndex RSIdx;
      Register RSReg = buildRSForSuperUse(UseMI, MO, OrigVReg, NewSSA, MaskToRewrite,
                                          LI, OpRC, RSIdx, LanesToExtend);
      extendAt(LI, RSIdx, LanesToExtend);
      MO.setReg(RSReg);
      MO.setSubReg(0);
      
    } else {
      // Case 3: Subset - use needs fewer lanes, keep subregister index
      LLVM_DEBUG(dbgs() << "      Subset case -> keeping subregister\n");
      unsigned SubReg = MO.getSubReg();
      assert(SubReg && "Subset case should have subregister");
      
      MO.setReg(NewSSA);
      // Keep the existing subregister index
    }
  }
  
  LLVM_DEBUG(dbgs() << "  Completed rewriting dominated uses\n");
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Return the VNInfo reaching this PHI operand along its predecessor edge.
VNInfo *MachineLaneSSAUpdater::incomingOnEdge(LiveInterval &LI, MachineInstr *Phi,
                                               MachineOperand &PhiOp) {
  unsigned OpIdx = Phi->getOperandNo(&PhiOp);
  MachineBasicBlock *Pred = Phi->getOperand(OpIdx + 1).getMBB();
  SlotIndex EndB = LIS.getMBBEndIdx(Pred);
  return LI.getVNInfoBefore(EndB);
}

/// True if \p UseMI's operand is reached by \p VNI (PHIs, same-block order,
/// cross-block dominance).
bool MachineLaneSSAUpdater::reachedByThisVNI(LiveInterval &LI, MachineInstr *DefMI,
                                              MachineInstr *UseMI, MachineOperand &UseOp,
                                              VNInfo *VNI) {
  if (UseMI->isPHI())
    return incomingOnEdge(LI, UseMI, UseOp) == VNI;

  if (UseMI->getParent() == DefMI->getParent()) {
    SlotIndex DefIdx = LIS.getInstructionIndex(*DefMI);
    SlotIndex UseIdx = LIS.getInstructionIndex(*UseMI);
    return DefIdx < UseIdx; // strict within-block order
  }
  return MDT.dominates(DefMI->getParent(), UseMI->getParent());
}

/// What lanes does this operand read?
LaneBitmask MachineLaneSSAUpdater::operandLaneMask(const MachineOperand &MO) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  if (unsigned Sub = MO.getSubReg())
    return TRI.getSubRegIndexLaneMask(Sub);
  return MRI.getMaxLaneMaskForVReg(MO.getReg());
}

/// Build a REG_SEQUENCE to materialize a super-reg/mixed-lane use.
/// Inserts at the PHI predecessor terminator (for PHI uses) or right before
/// UseMI otherwise. Returns the new full-width vreg, the RS index via OutIdx,
/// and the subrange lane masks that should be extended to that point.
Register MachineLaneSSAUpdater::buildRSForSuperUse(MachineInstr *UseMI, MachineOperand &MO,
                                                   Register OldVR, Register NewVR,
                                                   LaneBitmask MaskToRewrite, LiveInterval &LI,
                                                   const TargetRegisterClass *OpRC,
                                                   SlotIndex &OutIdx,
                                                   SmallVectorImpl<LaneBitmask> &LanesToExtend) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  MachineBasicBlock *InsertBB = UseMI->getParent();
  MachineBasicBlock::iterator IP(UseMI);
  SlotIndex QueryIdx;

  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(&MO);
    MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
    InsertBB = Pred;
    IP = Pred->getFirstTerminator(); // ok if == end()
    QueryIdx = LIS.getMBBEndIdx(Pred).getPrevSlot();
  } else {
    QueryIdx = LIS.getInstructionIndex(*UseMI);
  }

  Register Dest = MRI.createVirtualRegister(OpRC);
  auto RS = BuildMI(*InsertBB, IP,
                    (IP != InsertBB->end() ? IP->getDebugLoc() : DebugLoc()),
                    TII.get(TargetOpcode::REG_SEQUENCE), Dest);

  SmallDenseSet<unsigned, 8> AddedSubIdxs;
  SmallDenseSet<LaneBitmask::Type, 8> AddedMasks;

  for (const LiveInterval::SubRange &SR : LI.subranges()) {
    if (!SR.getVNInfoAt(QueryIdx))
      continue;
    LaneBitmask Lane = SR.LaneMask;
    if (!AddedMasks.insert(Lane.getAsInteger()).second)
      continue;

    unsigned SubIdx = getSubRegIndexForLaneMask(Lane, &TRI);
    if (!SubIdx || !AddedSubIdxs.insert(SubIdx).second)
      continue;

    if (Lane == MaskToRewrite)
      RS.addReg(NewVR).addImm(SubIdx);
    else
      RS.addReg(OldVR, 0, SubIdx).addImm(SubIdx);

    LanesToExtend.push_back(Lane);
  }

  // Fallback: ensure at least the rewritten lane appears.
  if (AddedSubIdxs.empty()) {
    unsigned SubIdx = getSubRegIndexForLaneMask(MaskToRewrite, &TRI);
    RS.addReg(NewVR).addImm(SubIdx);
    LanesToExtend.push_back(MaskToRewrite);
  }

  LIS.InsertMachineInstrInMaps(*RS);
  OutIdx = LIS.getInstructionIndex(*RS);

  LLVM_DEBUG(dbgs() << "        Built REG_SEQUENCE: ");
  LLVM_DEBUG(RS->print(dbgs()));

  return Dest;
}

/// Extend LI (and only the specified subranges) at Idx.
void MachineLaneSSAUpdater::extendAt(LiveInterval &LI, SlotIndex Idx, 
                                     ArrayRef<LaneBitmask> Lanes) {
  SmallVector<SlotIndex, 1> P{Idx};
  LIS.extendToIndices(LI, P);
  for (auto &SR : LI.subranges())
    for (LaneBitmask L : Lanes)
      if (SR.LaneMask == L)
        LIS.extendToIndices(SR, P);
}

// Remove the old helper that's no longer needed
// LaneBitmask MachineLaneSSAUpdater::getLaneMaskForOperand(...) - REMOVED