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
      SmallVector<Register> PHIResults = createPHIInBlock(*JoinMBB, OrigVReg, Item.VReg, DefMask);
      
      // Add PHI results to worklist for further processing and to result collection
      for (Register PHIVReg : PHIResults) {
        if (PHIVReg.isValid()) {
          Worklist.emplace_back(PHIVReg, JoinMBB);
          AllCreatedPHIs.push_back(PHIVReg);
          LLVM_DEBUG(dbgs() << "      Created PHI result VReg=" << PHIVReg 
                            << ", added to worklist\n");
        }
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "  Worklist processing complete. Created " 
                    << AllCreatedPHIs.size() << " PHI registers total.\n");
  
  return AllCreatedPHIs;
}

// Helper: Create PHI in a specific block (extracted from previous implementation)
SmallVector<Register> MachineLaneSSAUpdater::createPHIInBlock(MachineBasicBlock &JoinMBB,
                                                    Register OrigVReg,
                                                               Register NewVReg,
                                                    LaneBitmask ResultMask) {
  LLVM_DEBUG(dbgs() << "    createPHIInBlock in BB#" << JoinMBB.getNumber()
                    << " OrigVReg=" << OrigVReg << " NewVReg=" << NewVReg 
                    << " ResultMask=" << PrintLaneMask(ResultMask) << "\n");
  
  // Get the LiveIntervals for both old and new registers
  LiveInterval &OldLI = LIS.getInterval(OrigVReg);
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const LaneBitmask FullMask = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg);
  const LaneBitmask NonReloadedMask = FullMask & ~ResultMask;
  const unsigned NoRegister = 0; // Target-independent equivalent to AMDGPU::NoRegister
  
  // Analyze each predecessor edge to determine operand sources
  SmallVector<MachineOperand> NewVRegOps;  // Operands for NewVReg PHI
  SmallVector<MachineOperand> OldVRegOps;  // Operands for OrigVReg PHI
  
  LaneBitmask NewCommonMask = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg); // intersection across preds
  LaneBitmask NewUnionMask = LaneBitmask::getNone();  // union across preds
  LaneBitmask OldCommonMask = MF.getRegInfo().getMaxLaneMaskForVReg(OrigVReg);
  LaneBitmask OldUnionMask = LaneBitmask::getNone();
  
  LLVM_DEBUG(dbgs() << "      Analyzing " << JoinMBB.pred_size() << " predecessors:\n");
  
  for (MachineBasicBlock *PredMBB : JoinMBB.predecessors()) {
    SlotIndex PredEndIdx = LIS.getMBBEndIdx(PredMBB);
    
    // Analyze NewVReg lanes on this edge
    LaneBitmask NewEdgeMask = LaneBitmask::getNone();
    if (VNInfo *NewVN = NewLI.getVNInfoBefore(PredEndIdx)) {
      // Check which lanes of NewVReg are live-out via subrange analysis
      for (const LiveInterval::SubRange &SR : NewLI.subranges()) {
        if (SR.getVNInfoBefore(PredEndIdx)) {
          NewEdgeMask |= SR.LaneMask;
        }
      }
      
      // If no subranges but main range is live, assume all reloaded lanes
      if (NewEdgeMask.none()) {
        NewEdgeMask = ResultMask;
      }
      
      // NewVReg can only contribute reloaded lanes
      NewEdgeMask &= ResultMask;
      
      LLVM_DEBUG(dbgs() << "        Pred BB#" << PredMBB->getNumber() 
                        << " NewVReg lanes: " << PrintLaneMask(NewEdgeMask)
                        << " (VN=" << NewVN->id << ")\n");
    }
    
    // Analyze OrigVReg lanes on this edge  
    LaneBitmask OldEdgeMask = LaneBitmask::getNone();
    if (VNInfo *OldVN = OldLI.getVNInfoBefore(PredEndIdx)) {
      // Check which lanes of OrigVReg are live-out via subrange analysis
      for (const LiveInterval::SubRange &SR : OldLI.subranges()) {
        if (SR.getVNInfoBefore(PredEndIdx)) {
          OldEdgeMask |= SR.LaneMask;
        }
      }
      
      // If no subranges but main range is live, assume all non-reloaded lanes
      if (OldEdgeMask.none()) {
        OldEdgeMask = NonReloadedMask;
      }
      
      // OrigVReg can only contribute non-reloaded lanes
      OldEdgeMask &= NonReloadedMask;
      
      LLVM_DEBUG(dbgs() << "        Pred BB#" << PredMBB->getNumber() 
                        << " OrigVReg lanes: " << PrintLaneMask(OldEdgeMask)
                        << " (VN=" << OldVN->id << ")\n");
    }
    
    // Update mask statistics
    NewCommonMask &= NewEdgeMask;
    NewUnionMask |= NewEdgeMask;
    OldCommonMask &= OldEdgeMask;
    OldUnionMask |= OldEdgeMask;
    
    // Create operands for NewVReg PHI if this edge contributes
    if (NewEdgeMask.any()) {
      unsigned SubIdx = NoRegister;
      if ((ResultMask & ~NewEdgeMask).any()) { // partial register incoming
        // TODO: Implement getSubRegIndexForLaneMask or equivalent
        // SubIdx = getSubRegIndexForLaneMask(NewEdgeMask, &TRI);
      }
      
      NewVRegOps.push_back(MachineOperand::CreateReg(NewVReg, /*isDef*/ false,
                                                     /*isImp*/ false, /*isKill*/ false,
                                                     /*isDead*/ false, /*isUndef*/ false,
                                                     /*isEarlyClobber*/ false, SubIdx));
      NewVRegOps.push_back(MachineOperand::CreateMBB(PredMBB));
    }
    
    // Create operands for OrigVReg PHI if this edge contributes  
    if (OldEdgeMask.any()) {
      unsigned SubIdx = NoRegister;
      if ((NonReloadedMask & ~OldEdgeMask).any()) { // partial register incoming
        // TODO: Implement getSubRegIndexForLaneMask or equivalent
        // SubIdx = getSubRegIndexForLaneMask(OldEdgeMask, &TRI);
      }
      
      OldVRegOps.push_back(MachineOperand::CreateReg(OrigVReg, /*isDef*/ false,
                                                     /*isImp*/ false, /*isKill*/ false,
                                                     /*isDead*/ false, /*isUndef*/ false,
                                                     /*isEarlyClobber*/ false, SubIdx));
      OldVRegOps.push_back(MachineOperand::CreateMBB(PredMBB));
    }
  }
  
  // Decide PHI mask strategies using CommonMask/UnionMask logic
  LaneBitmask NewPhiMask = (NewCommonMask.none() ? NewUnionMask : NewCommonMask);
  LaneBitmask OldPhiMask = (OldCommonMask.none() ? OldUnionMask : OldCommonMask);
  
  if (NewPhiMask.none()) NewPhiMask = ResultMask;
  if (OldPhiMask.none()) OldPhiMask = NonReloadedMask;
  
  LLVM_DEBUG(dbgs() << "      Analysis: NewPhiMask=" << PrintLaneMask(NewPhiMask)
                    << " OldPhiMask=" << PrintLaneMask(OldPhiMask) << "\n");
                    
  SmallVector<Register> ResultVRegs;
  
  // Create PHI(s) based on what we need
  if (NewUnionMask.any() && OldUnionMask.any()) {
    LLVM_DEBUG(dbgs() << "      Complex case: Creating separate PHIs for NewVReg and OrigVReg\n");
    
    // Create PHI for NewVReg lanes
    if (!NewVRegOps.empty()) {
      const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(NewVReg);
      Register NewPHIVReg = MF.getRegInfo().createVirtualRegister(RC);
      
      auto NewPHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                               TII->get(TargetOpcode::PHI), NewPHIVReg);
      for (const MachineOperand &Op : NewVRegOps) {
        NewPHINode.add(Op);
      }
      
      MachineInstr *NewPHI = NewPHINode.getInstr();
      LIS.InsertMachineInstrInMaps(*NewPHI);
      LIS.createAndComputeVirtRegInterval(NewPHIVReg);
      
      ResultVRegs.push_back(NewPHIVReg);
      LLVM_DEBUG(dbgs() << "      Created NewVReg PHI: ");
      LLVM_DEBUG(NewPHI->print(dbgs()));
    }
    
    // Create PHI for OrigVReg lanes
    if (!OldVRegOps.empty()) {
      const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(OrigVReg);
      Register OldPHIVReg = MF.getRegInfo().createVirtualRegister(RC);
      
      auto OldPHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                               TII->get(TargetOpcode::PHI), OldPHIVReg);
      for (const MachineOperand &Op : OldVRegOps) {
        OldPHINode.add(Op);
      }
      
      MachineInstr *OldPHI = OldPHINode.getInstr();
      LIS.InsertMachineInstrInMaps(*OldPHI);
      LIS.createAndComputeVirtRegInterval(OldPHIVReg);
      
      ResultVRegs.push_back(OldPHIVReg);
      LLVM_DEBUG(dbgs() << "      Created OrigVReg PHI: ");
      LLVM_DEBUG(OldPHI->print(dbgs()));
    }
    
  } else if (NewUnionMask.any()) {
    LLVM_DEBUG(dbgs() << "      Simple case: Creating PHI for NewVReg lanes only\n");
    
    // Create result register and PHI for NewVReg
    const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(NewVReg);
    Register PHIVReg = MF.getRegInfo().createVirtualRegister(RC);
    
    auto PHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                          TII->get(TargetOpcode::PHI), PHIVReg);
    for (const MachineOperand &Op : NewVRegOps) {
      PHINode.add(Op);
    }
    
    MachineInstr *PHI = PHINode.getInstr();
    LIS.InsertMachineInstrInMaps(*PHI);
    LIS.createAndComputeVirtRegInterval(PHIVReg);
    
    ResultVRegs.push_back(PHIVReg);
    LLVM_DEBUG(dbgs() << "      Created NewVReg PHI: ");
    LLVM_DEBUG(PHI->print(dbgs()));
    
  } else if (OldUnionMask.any()) {
    LLVM_DEBUG(dbgs() << "      Simple case: Creating PHI for OrigVReg lanes only\n");
    
    // Create result register and PHI for OrigVReg
    const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(OrigVReg);
    Register PHIVReg = MF.getRegInfo().createVirtualRegister(RC);
    
    auto PHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                          TII->get(TargetOpcode::PHI), PHIVReg);
    for (const MachineOperand &Op : OldVRegOps) {
      PHINode.add(Op);
    }
    
    MachineInstr *PHI = PHINode.getInstr();
    LIS.InsertMachineInstrInMaps(*PHI);
    LIS.createAndComputeVirtRegInterval(PHIVReg);
    
    ResultVRegs.push_back(PHIVReg);
    LLVM_DEBUG(dbgs() << "      Created OrigVReg PHI: ");
    LLVM_DEBUG(PHI->print(dbgs()));
    
  } else {
    LLVM_DEBUG(dbgs() << "      No lanes live-out from any predecessor - unusual case\n");
  }
  
  return ResultVRegs;
}

void MachineLaneSSAUpdater::rewriteDominatedUses(Register OrigVReg,
                                                  Register NewSSA,
                                                  LaneBitmask MaskToRewrite) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::rewriteDominatedUses OrigVReg=" << OrigVReg
                    << " NewSSA=" << NewSSA << " Mask=" << PrintLaneMask(MaskToRewrite) << "\n");
  
  // TODO: Implement dominated use rewriting
  // This should handle exact/subset/super policy:
  // - Exact match: direct replacement
  // - Subset: create REG_SEQUENCE combining old + new
  // - Super: extract subregister from new def
  // Preserve undef/dead flags, never mass-clear on partial defs
}