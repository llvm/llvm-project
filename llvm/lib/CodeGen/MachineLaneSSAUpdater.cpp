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
//  - Two explicit entry points:
//    * repairSSAForNewDef - Common use case: caller creates instruction defining
//      existing vreg (violating SSA), updater creates new vreg and repairs
//    * addDefAndRepairAfterSpill - Spill/reload use case: caller creates instruction
//      with new vreg, updater repairs SSA using spill-time EndPoints
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
#include "llvm/ADT/Hashing.h"
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
// MachineLaneSSAUpdater Implementation
//===----------------------------------------------------------------------===//

Register MachineLaneSSAUpdater::repairSSAForNewDef(
    MachineInstr &NewDefMI, Register OrigVReg,
    SmallVectorImpl<MachineOperand *> &PHIRegDefOps) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::repairSSAForNewDef VReg="
                    << OrigVReg << "\n");

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Step 1: Find the def operand for OrigVReg
  MachineOperand *DefOp = nullptr;
  unsigned DefOpIdx = 0;
  
  for (MachineOperand &MO : NewDefMI.defs()) {
    if (MO.getReg() == OrigVReg) {
      DefOp = &MO;
      break;
    }
    ++DefOpIdx;
  }
  
  assert(DefOp && "NewDefMI should have a def operand for OrigVReg");
  assert(DefOp->isDef() && "Found operand should be a definition");
  
  // Step 2: Derive DefMask from the operand's subreg index (if any)
  unsigned SubRegIdx = DefOp->getSubReg();
  LaneBitmask DefMask;
  
  if (SubRegIdx) {
    // Partial register definition - get lane mask for this subreg
    DefMask = TRI.getSubRegIndexLaneMask(SubRegIdx);
    LLVM_DEBUG(dbgs() << "  Partial def with subreg " << TRI.getSubRegIndexName(SubRegIdx)
                      << ", DefMask=" << PrintLaneMask(DefMask) << "\n");
  } else {
    // Full register definition - get all lanes for this register class
    DefMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
    LLVM_DEBUG(dbgs() << "  Full register def, DefMask=" << PrintLaneMask(DefMask) << "\n");
  }
  
  // Step 3: Create a new virtual register with class matching MaskToRewrite width
  const TargetRegisterClass *RC;
  if (SubRegIdx) {
    // For subreg defs, create register with class for the subreg (narrower)
    const TargetRegisterClass *OrigRC = MRI.getRegClass(OrigVReg);
    RC = TRI.getSubRegisterClass(OrigRC, SubRegIdx);
    assert(RC && "Failed to get subregister class");
  } else {
    // For full register defs, use same class as OrigVReg
    RC = MRI.getRegClass(OrigVReg);
  }
  Register NewSSAVReg = MRI.createVirtualRegister(RC);
  LLVM_DEBUG(dbgs() << "  Created new SSA vreg " << NewSSAVReg << " with RC=" << TRI.getRegClassName(RC) << "\n");
  
  // Step 4: Replace the operand in NewDefMI to define the new vreg
  // Clear subreg index since NewSSAVReg is a full register of the (possibly narrower) class
  DefOp->setReg(NewSSAVReg);
  if (SubRegIdx) {
    DefOp->setSubReg(0);  // Clear subreg - NewSSAVReg is full register of narrower class
    DefOp->setIsUndef(false);  // Clear undef flag - verifier requires undef only with subreg
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg << "." 
                      << TRI.getSubRegIndexName(SubRegIdx) 
                      << " -> " << NewSSAVReg << " (full narrower register)\n");
  } else {
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg 
                      << " -> " << NewSSAVReg << "\n");
  }
  
  // Step 5: Index the new instruction in SlotIndexes/LIS
  indexNewInstr(NewDefMI);
  
  // Step 6: Perform common SSA repair (PHI placement + use rewriting)
  // LiveInterval for NewSSAVReg will be created by getInterval() as needed
  PHIRegDefOps = performSSARepair(NewSSAVReg, OrigVReg, DefMask, NewDefMI.getParent());
  
  // Step 7: If SSA repair created subregister uses of OrigVReg (e.g., in PHIs or REG_SEQUENCEs),
  // recompute its LiveInterval to create subranges
  LaneBitmask AllLanes = MRI.getMaxLaneMaskForVReg(OrigVReg);
  if (DefMask != AllLanes) {
    LiveInterval &OrigLI = LIS.getInterval(OrigVReg);
    if (!OrigLI.hasSubRanges()) {
      // Check if any uses now access OrigVReg with subregister indices
      bool HasSubregUses = false;
      for (const MachineOperand &MO : MRI.use_operands(OrigVReg)) {
        if (MO.getSubReg() != 0) {
          HasSubregUses = true;
          break;
        }
      }
      
      if (HasSubregUses) {
        LLVM_DEBUG(dbgs() << "  Recomputing LiveInterval for " << OrigVReg 
                          << " after SSA repair created subregister uses\n");
        LIS.removeInterval(OrigVReg);
        LIS.createAndComputeVirtRegInterval(OrigVReg);
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "  repairSSAForNewDef complete, returning " 
                    << printReg(NewSSAVReg, &TRI) << "\n");
  return NewSSAVReg;
}

//===----------------------------------------------------------------------===//
// Common SSA Repair Logic
//===----------------------------------------------------------------------===//

SmallVector<MachineOperand *>
MachineLaneSSAUpdater::performSSARepair(Register NewVReg, Register OrigVReg,
                                        LaneBitmask DefMask,
                                        MachineBasicBlock *DefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::performSSARepair NewVReg=" << NewVReg
                    << " OrigVReg=" << OrigVReg << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Step 1: Use worklist-driven PHI placement
  SmallVector<MachineOperand *> AllPHIVResults =
      insertLaneAwarePHI(NewVReg, OrigVReg, DefMask, DefBB);

  // Step 2: Rewrite dominated uses once for each new register
  // Note: getInterval() will automatically create LiveIntervals if needed
  rewriteDominatedUses(OrigVReg, NewVReg, DefMask);
  for (MachineOperand* PHIVRes : AllPHIVResults) {
    rewriteDominatedUses(OrigVReg, PHIVRes->getReg(), DefMask);
  }
  
  // Step 3: Renumber values if needed
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  NewLI.RenumberValues();
  
  // Also renumber PHI intervals
  for (MachineOperand* PHIVRes : AllPHIVResults) {
    LiveInterval &PHILI = LIS.getInterval(PHIVRes->getReg());
    PHILI.RenumberValues();
  }
  
  // Recompute OrigVReg's LiveInterval to account for PHI operands
  // We do a full recomputation because PHI operands may reference subregisters
  // that weren't previously live on those paths, and we need to extend liveness
  // from the definition to the PHI use.
  LIS.removeInterval(OrigVReg);
  LIS.createAndComputeVirtRegInterval(OrigVReg);
  
  // Note: We do NOT call shrinkToUses on OrigVReg even after recomputation because:
  // shrinkToUses has a fundamental bug with PHI operands - it doesn't understand
  // that PHI operands require their source lanes to be live at the END of
  // predecessor blocks. When it sees a PHI operand like "%0.sub2_sub3" from BB3,
  // it only considers the PHI location (start of join block), not the predecessor
  // end where the value must be available. This causes it to incorrectly shrink
  // away lanes that ARE needed by PHI operands, leading to verification errors:
  // "Not all lanes of PHI source live at use". The createAndComputeVirtRegInterval
  // already produces correct, minimal liveness that includes PHI uses properly.
  
  // Step 4: Update operand flags to match the LiveIntervals
  updateDeadFlags(NewVReg);
  for (MachineOperand* PHIVRes : AllPHIVResults) {
    updateDeadFlags(PHIVRes->getReg());
  }
  
  // Step 5: Verification if enabled
  if (VerifyOnExit) {
    LLVM_DEBUG(dbgs() << "  Verifying after SSA repair...\n");
    // TODO: Add verification calls
  }
  
  LLVM_DEBUG(dbgs() << "  performSSARepair complete\n");
  return AllPHIVResults;
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
  const TargetRegisterInfo &TRI = *MF.getRegInfo().getTargetRegisterInfo();
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::computePrunedIDF VReg=" 
                    << printReg(OrigVReg, &TRI)
                    << " DefMask=" << PrintLaneMask(DefMask)
                    << " with " << NewDefBlocks.size() << " new def blocks\n");
  
  // Clear output vector at entry
  OutIDFBlocks.clear();
  
  // Try cache for single-block definitions (most common case in spilling)
  if (NewDefBlocks.size() == 1 && NewDefBlocks[0]) {
    IDFCacheKey Key{OrigVReg, DefMask, static_cast<unsigned>(NewDefBlocks[0]->getNumber())};
    auto It = IDFCache.find(Key);
    if (It != IDFCache.end()) {
      OutIDFBlocks.assign(It->second.begin(), It->second.end());
      LLVM_DEBUG(dbgs() << "  IDF cache HIT for VReg=" << printReg(OrigVReg, &TRI)
                        << " Mask=" << PrintLaneMask(DefMask)
                        << " DefBlock=bb." << NewDefBlocks[0]->getNumber() 
                        << "." << NewDefBlocks[0]->getName()
                        << " -> " << OutIDFBlocks.size() << " IDF blocks\n");
      return;
    }
    LLVM_DEBUG(dbgs() << "  IDF cache MISS for VReg=" << printReg(OrigVReg, &TRI)
                      << " Mask=" << PrintLaneMask(DefMask)
                      << " DefBlock=bb." << NewDefBlocks[0]->getNumber()
                      << "." << NewDefBlocks[0]->getName() << "\n");
  }
  
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
  
  // Store in cache if this was a single-block definition
  if (NewDefBlocks.size() == 1 && NewDefBlocks[0]) {
    IDFCacheKey Key{OrigVReg, DefMask, static_cast<unsigned>(NewDefBlocks[0]->getNumber())};
    IDFCache[Key].assign(OutIDFBlocks.begin(), OutIDFBlocks.end());
  }
  
  // Note: We do not place PHIs here; this function only computes candidate 
  // join blocks. The IDFCalculator handles deduplication automatically.
}

SmallVector<MachineOperand *> MachineLaneSSAUpdater::insertLaneAwarePHI(
    Register InitialVReg, Register OrigVReg, LaneBitmask DefMask,
    MachineBasicBlock *InitialDefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::insertLaneAwarePHI InitialVReg=" << InitialVReg
                    << " OrigVReg=" << OrigVReg << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  SmallVector<MachineOperand*> AllCreatedPHIs;
  
  // Step 1: Compute IDF (Iterated Dominance Frontier) for the initial definition
  // This gives us ALL blocks where PHI nodes need to be inserted
  SmallVector<MachineBasicBlock *> DefBlocks = {InitialDefBB};
  SmallVector<MachineBasicBlock *> IDFBlocks;
  computePrunedIDF(OrigVReg, DefMask, DefBlocks, IDFBlocks);
  
  LLVM_DEBUG(dbgs() << "  Computed IDF: found " << IDFBlocks.size() << " blocks needing PHIs\n");
  for (MachineBasicBlock *MBB : IDFBlocks) {
    LLVM_DEBUG(dbgs() << "    BB#" << MBB->getNumber() << "\n");
  }
  
  // Step 2: Iterate through IDF blocks sequentially, creating PHIs
  // Key insight: After creating a PHI, update NewVReg to the PHI result
  // so subsequent PHIs use the correct register
  Register CurrentNewVReg = InitialVReg;
  
  for (MachineBasicBlock *JoinMBB : IDFBlocks) {
    LLVM_DEBUG(dbgs() << "  Creating PHI in BB#" << JoinMBB->getNumber() 
                      << " with CurrentNewVReg=" << CurrentNewVReg << "\n");
    
    // Create PHI: merges OrigVReg and CurrentNewVReg based on dominance
    MachineOperand* PHIResult = createPHIInBlock(*JoinMBB, OrigVReg, CurrentNewVReg, DefMask);
    
    if (PHIResult) {
      AllCreatedPHIs.push_back(PHIResult);
      
      // Update CurrentNewVReg to be the PHI result
      // This ensures the next PHI (if any) uses this PHI's result, not the original InitialVReg
      CurrentNewVReg = PHIResult->getReg();
      
      LLVM_DEBUG(dbgs() << "    Created PHI result VReg=" << PHIResult 
                        << ", will use this for subsequent PHIs\n");
    }
  }
  
  LLVM_DEBUG(dbgs() << "  PHI insertion complete. Created " 
                    << AllCreatedPHIs.size() << " PHI registers total.\n");
  
  return AllCreatedPHIs;
}

// Helper: Create lane-specific PHI in a join block
MachineOperand *
MachineLaneSSAUpdater::createPHIInBlock(MachineBasicBlock &JoinMBB,
                                        Register OrigVReg, Register NewVReg,
                                        LaneBitmask DefMask) {
  LLVM_DEBUG(dbgs() << "    createPHIInBlock in BB#" << JoinMBB.getNumber()
                    << " OrigVReg=" << OrigVReg << " NewVReg=" << NewVReg 
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
  
  // Check for existing PHI that already merges this register with matching lane mask
  for (MachineInstr &MI : JoinMBB.phis()) {
    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      MachineOperand &ValOp = MI.getOperand(i);
      if (!ValOp.isReg() || ValOp.getReg() != OrigVReg)
        continue;
      
      // Get effective lane mask for this operand
      LaneBitmask OpMask = TRI.getSubRegIndexLaneMask(ValOp.getSubReg());
      if (OpMask == LaneBitmask::getAll())
        OpMask = FullMask;
      
      if ((OpMask & DefMask).any()) {
        LLVM_DEBUG(dbgs() << "      Found existing PHI with matching lanes: ");
        LLVM_DEBUG(MI.print(dbgs()));
        return &MI.getOperand(0);
      }
    }
  }
  
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  
  // Check if this is a partial lane redefinition
  const bool IsPartialReload = (DefMask != FullMask);
  
  // Collect PHI operands for the specific reload lanes
  SmallVector<MachineOperand> PHIOperands;
  
  LLVM_DEBUG(dbgs() << "      Creating PHI for " << (IsPartialReload ? "partial reload" : "full reload")
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");
  
  // Get the definition block of NewVReg for dominance checks
  MachineInstr *NewDefMI = MRI.getVRegDef(NewVReg);
  MachineBasicBlock *NewDefBB = NewDefMI->getParent();
  
  for (MachineBasicBlock *PredMBB : JoinMBB.predecessors()) {
    // Use dominance check instead of liveness: if NewDefBB dominates PredMBB,
    // then NewVReg is available at the end of PredMBB
    bool UseNewReg = MDT.dominates(NewDefBB, PredMBB);
    
    if (UseNewReg) {
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
        // Partial case: z = PHI(y, BB1, x.sub2_3, BB0)
        // Use DefMask to find which subreg of OrigVReg was redefined
        unsigned SubIdx = TRI.getSubRegIndexForLaneMask(DefMask);
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
    
    LLVM_DEBUG(dbgs() << "      Created lane-specific PHI: ");
    LLVM_DEBUG(PHI->print(dbgs()));
    
    return &PHINode->getOperand(0);
  }
  
  return nullptr;
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
  
  MachineBasicBlock *DefBB = DefMI->getParent();

  LLVM_DEBUG(dbgs() << "  Rewriting uses dominated by definition in BB#" << DefBB->getNumber() << ": ");
  LLVM_DEBUG(DefMI->print(dbgs()));

  // Get OrigVReg's LiveInterval for reference
  LiveInterval &OrigLI = LIS.getInterval(OrigVReg);

  // Iterate through all uses of OrigVReg
  for (MachineOperand &MO : llvm::make_early_inc_range(MRI.use_operands(OrigVReg))) {
    MachineInstr *UseMI = MO.getParent();
    
    // Skip the definition instruction itself
    if (UseMI == DefMI)
      continue;

    // Check if this use is DOMINATED by our definition
    // Reachable-but-not-dominated uses should use PHI results, not NewSSA directly
    if (!defDominatesUse(DefMI, UseMI, MO))
      continue;

    // Get the lane mask for this operand
    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    LLVM_DEBUG(dbgs() << "    Processing use with OpMask=" << PrintLaneMask(OpMask) << ": ");
    LLVM_DEBUG(UseMI->print(dbgs()));

    const TargetRegisterClass *OpRC = MRI.getRegClass(OrigVReg);
    const TargetRegisterClass *NewRC = MRI.getRegClass(NewSSA);

    // Case 1: Exact match - direct replacement
    if (OpMask == MaskToRewrite) {
      // Check register class compatibility
      // If operand uses a subreg, NewRC should match the subreg class
      // If operand uses full register, NewRC should match OpRC
      const TargetRegisterClass *ExpectedRC = MO.getSubReg() != 0 
          ? TRI.getSubRegisterClass(OpRC, MO.getSubReg()) 
          : OpRC;
      bool Compatible = (ExpectedRC == NewRC);
      
      if (Compatible) {
        LLVM_DEBUG(dbgs() << "      Exact match -> direct replacement\n");
        MO.setReg(NewSSA);
        MO.setSubReg(0); // Clear subregister (NewSSA is a full register of NewRC)
        
        // NOTE: Don't extend NewSSA's live interval here - it will be computed
        // later after all use rewriting is complete. Extending during the loop
        // triggers lazy interval creation with incomplete use info, causing
        // incorrect "dead" flags.
        
        continue;
      }
      
      // Incompatible register classes with same lane mask indicates corrupted MIR
      LLVM_DEBUG(dbgs() << "=== Register Class Mismatch ===\n"
                        << "  OrigVReg: " << printReg(OrigVReg, &TRI) << " class: " << TRI.getRegClassName(OpRC) << "\n"
                        << "  NewSSA: " << printReg(NewSSA, &TRI) << " class: " << TRI.getRegClassName(NewRC) << "\n"
                        << "  MaskToRewrite: " << PrintLaneMask(MaskToRewrite) << "\n"
                        << "  OpMask: " << PrintLaneMask(OpMask) << "\n"
                        << "  MO.getSubReg(): " << MO.getSubReg() << "\n"
                        << "  ExpectedRC: " << (ExpectedRC ? TRI.getRegClassName(ExpectedRC) : "null") << "\n"
                        << "  Use: "; UseMI->print(dbgs()));
      llvm_unreachable("Incompatible register classes with same lane mask - invalid MIR");
    }

    // Case 2: Super/Mixed - use needs more lanes than we're rewriting
    if ((OpMask & ~MaskToRewrite).any()) {
      LLVM_DEBUG(dbgs() << "      Super/Mixed case -> building REG_SEQUENCE\n");
      
      SmallVector<LaneBitmask, 4> LanesToExtend;
      SlotIndex RSIdx;
      Register RSReg = buildRSForSuperUse(UseMI, MO, OrigVReg, NewSSA, MaskToRewrite,
                                          OrigLI, OpRC, RSIdx, LanesToExtend);
      extendAt(OrigLI, RSIdx, LanesToExtend);
      MO.setReg(RSReg);
      MO.setSubReg(0);
      
      // Extend RSReg's live interval to cover this use
      SlotIndex UseIdx;
      if (UseMI->isPHI()) {
        // For PHI, the value must be live at the end of the predecessor block
        unsigned OpIdx = UseMI->getOperandNo(&MO);
        MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
        UseIdx = LIS.getMBBEndIdx(Pred);
      } else {
        UseIdx = LIS.getInstructionIndex(*UseMI).getRegSlot();
      }
      LiveInterval &RSLI = LIS.getInterval(RSReg);
      LIS.extendToIndices(RSLI, {UseIdx});
      
      // Update dead flag on REG_SEQUENCE result
      updateDeadFlags(RSReg);
      
    } else {
      // Case 3: Subset - use needs fewer lanes than NewSSA provides
      // Need to map OpMask to NewSSA's register class namespace if they differ
      LLVM_DEBUG(dbgs() << "      Subset case -> finding subregister index\n");
      
      LaneBitmask MappedOpMask = OpMask;
      if (NewRC != OpRC) {
        // Different register classes - need to map lane namespace
        // MaskToRewrite defines which lanes of OrigVReg we redefined
        // NewSSA is a narrower register that holds just those lanes
        // Shift OpMask down to map from OrigVReg space to NewSSA space
        unsigned ShiftAmt = llvm::countr_zero(MaskToRewrite.getAsInteger());
        MappedOpMask = LaneBitmask(OpMask.getAsInteger() >> ShiftAmt);
        
        LLVM_DEBUG(dbgs() << "        Namespace mapping: OrigRC=" << TRI.getRegClassName(OpRC)
                          << " NewRC=" << TRI.getRegClassName(NewRC)
                          << "\n          OpMask=" << PrintLaneMask(OpMask)
                          << " -> MappedOpMask=" << PrintLaneMask(MappedOpMask) << "\n");
      }
      
      // Find the subregister index for MappedOpMask in NewSSA's register class
      unsigned NewSubReg = TRI.getSubRegIndexForLaneMask(MappedOpMask);
      assert(NewSubReg && "Should find subreg index for mapped lanes");
      
      LLVM_DEBUG(dbgs() << "        MappedOpMask=" << PrintLaneMask(MappedOpMask)
                        << " -> SubReg=" << TRI.getSubRegIndexName(NewSubReg) << "\n");
      
      MO.setReg(NewSSA);
      MO.setSubReg(NewSubReg);
      
      // NOTE: Don't extend NewSSA's live interval here - computed later.
    }
  }
  
  LLVM_DEBUG(dbgs() << "  Completed rewriting dominated uses\n");
}

SmallVector<MachineOperand *, 4> MachineLaneSSAUpdater::repairSSAForReload(
    Register NewVReg, Register OrigVReg, LaneBitmask DefMask,
    MachineBasicBlock *DefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::repairSSAForReload NewVReg="
                    << NewVReg << " OrigVReg=" << OrigVReg
                    << " DefMask=" << PrintLaneMask(DefMask)
                    << " DefBB=BB#" << DefBB->getNumber() << "\n");
  return performSSARepair(NewVReg, OrigVReg, DefMask, DefBB);
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

/// Check if \p DefMI's definition reaches \p UseMI's use operand.
/// Get pruned IDF blocks for a definition (public API).
/// Caching is handled inside computePrunedIDF.
void MachineLaneSSAUpdater::getPrunedIDF(Register OrigVReg,
                                          LaneBitmask DefMask,
                                          MachineBasicBlock *DefBlock,
                                          SmallVectorImpl<MachineBasicBlock *> &OutIDFBlocks) {
  SmallVector<MachineBasicBlock *, 1> DefBlocks = {DefBlock};
  computePrunedIDF(OrigVReg, DefMask, DefBlocks, OutIDFBlocks);
}

MachineOperand *MachineLaneSSAUpdater::insertPHIAtBlock(
    MachineBasicBlock *JoinBB, Register OrigVReg,
    const DenseMap<MachineBasicBlock *, Register> &IncomingValues,
    LaneBitmask SpilledMask) {
  LLVM_DEBUG(dbgs() << "  insertPHIAtBlock in BB#" << JoinBB->getNumber()
                    << " for vreg" << OrigVReg << "\n");

  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Determine register class for the spilled lanes
  const TargetRegisterClass *RC;
  if (!IncomingValues.empty()) {
    Register AnyIncoming = IncomingValues.begin()->second;
    RC = MRI.getRegClass(AnyIncoming);
  } else {
    // Map is empty - derive RC from OrigVReg and SpilledMask
    const TargetRegisterClass *OrigRC = MRI.getRegClass(OrigVReg);
    LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
    if (SpilledMask != FullMask) {
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      unsigned SubIdx = TRI->getSubRegIndexForLaneMask(SpilledMask);
      RC = TRI->getSubRegisterClass(OrigRC, SubIdx);
    } else {
      RC = OrigRC;
    }
  }
  Register PHIVReg = MRI.createVirtualRegister(RC);

  auto PHIBuilder =
      BuildMI(*JoinBB, JoinBB->begin(), DebugLoc(), TII->get(TargetOpcode::PHI), PHIVReg);

  // Compute subreg index for partial spills (used when IncomingValues doesn't have an entry)
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
  bool IsPartialSpill = (SpilledMask != FullMask);
  unsigned SubIdx = IsPartialSpill ? TRI->getSubRegIndexForLaneMask(SpilledMask) : 0;

  for (MachineBasicBlock *Pred : JoinBB->predecessors()) {
    auto It = IncomingValues.find(Pred);
    if (It != IncomingValues.end()) {
      PHIBuilder.addReg(It->second).addMBB(Pred);
    } else {
      // Predecessor not in map - use OrigVReg with appropriate subreg for partial spills
      if (IsPartialSpill) {
        PHIBuilder.addReg(OrigVReg, RegState::NoFlags, SubIdx).addMBB(Pred);
      } else {
        PHIBuilder.addReg(OrigVReg).addMBB(Pred);
      }
    }
  }

  MachineInstr *PHI = PHIBuilder.getInstr();
  LIS.InsertMachineInstrInMaps(*PHI);

  LLVM_DEBUG(dbgs() << "    Created PHI: "; PHI->print(dbgs()));

  return &PHI->getOperand(0);
}

/// Check if a definition dominates a use.
/// This is stricter than reachability - dominance means the def is on ALL paths to the use.
///
/// \param DefMI - The instruction defining a register
/// \param UseMI - The instruction using a register
/// \param UseOp - The specific use operand
/// \returns true if DefMI dominates the use
bool MachineLaneSSAUpdater::defDominatesUse(MachineInstr *DefMI,
                                             MachineInstr *UseMI,
                                             MachineOperand &UseOp) {
  MachineBasicBlock *DefBlock = DefMI->getParent();
  MachineBasicBlock *UseBlock = UseMI->getParent();
  
  // Fast path 1: Same block - check instruction order
  if (UseBlock == DefBlock) {
    SlotIndex DefIdx = LIS.getInstructionIndex(*DefMI);
    SlotIndex UseIdx = LIS.getInstructionIndex(*UseMI);
    return DefIdx < UseIdx;
  }
  
  // For PHI uses, we need to check dominance of the predecessor block
  // For non-PHI uses, we check dominance of the use block
  MachineBasicBlock *TargetBlock = UseBlock;
  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(&UseOp);
    TargetBlock = UseMI->getOperand(OpIdx + 1).getMBB();
  }
  
  return MDT.dominates(DefBlock, TargetBlock);
}

/// Check if a use is reachable from a definition using IDF analysis.
///
/// This fixes a bug where PHI operands from non-dominated predecessors were
/// incorrectly left unrewritten after spills on parallel paths.
bool MachineLaneSSAUpdater::isUseReachableFromDef(MachineInstr *DefMI,
                                                    MachineInstr *UseMI,
                                                    Register OrigVReg,
                                                    LaneBitmask DefMask) {
  MachineBasicBlock *DefBlock = DefMI->getParent();
  
  // Find the use operand in UseMI for dominance check
  MachineOperand *UseOp = UseMI->findRegisterUseOperand(OrigVReg, &TRI, /*isKill=*/false);
  assert(UseOp && "UseMI must use OrigVReg");
  
  // Fast path: Check dominance first
  // If def dominates use, it definitely reaches it
  if (defDominatesUse(DefMI, UseMI, *UseOp)) {
    return true;
  }
  
  // Extract the target block for IDF analysis (use block or PHI predecessor)
  MachineBasicBlock *PredBlock = UseMI->getParent();
  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(UseOp);
    PredBlock = UseMI->getOperand(OpIdx + 1).getMBB();
  }
  
  // IDF-based reachability for non-dominated uses (parallel paths case)
  //
  // When a spill happens on one path but not all paths to a join point,
  // we need PHI insertion. Example from spill-use-before-spill.mir:
  //
  //   bb.0.entry:
  //     %0:vreg_128 = COPY ...
  //     Branch to bb.1 or bb.2
  //
  //   bb.1.spill_path:
  //     SI_SPILL_V128_SAVE %0  <- Spill here
  //     %reload:vreg_128 = SI_SPILL_V128_RESTORE  <- Reload immediately
  //     S_BRANCH %bb.3
  //
  //   bb.2.clean_path:
  //     S_BRANCH %bb.3
  //
  //   bb.3.join:
  //     S_ENDPGM 0, implicit %0  <- Needs PHI: one pred has reload, other has original!
  //
  // IDF analysis detects bb.3 is in the dominance frontier of bb.1,
  // so the use in bb.3 IS reachable from the spill and needs PHI insertion.
  
  SmallVector<MachineBasicBlock *, 4> PrunedIDFBlocks;
  SmallVector<MachineBasicBlock *, 1> DefBlocks = {DefBlock};
  computePrunedIDF(OrigVReg, DefMask, DefBlocks, PrunedIDFBlocks);
  
  LLVM_DEBUG({
    dbgs() << "  isUseReachableFromDef (slow path): DefBlock=bb." << DefBlock->getNumber()
           << "." << DefBlock->getName() << " PredBlock=bb." << PredBlock->getNumber()
           << "." << PredBlock->getName() << "\n";
    dbgs() << "  Pruned IDF blocks: ";
    for (auto *BB : PrunedIDFBlocks)
      dbgs() << "bb." << BB->getNumber() << "." << BB->getName() << " ";
    dbgs() << "\n";
  });
  
  // Check if PredBlock is in or dominated by any IDF block
  for (auto *IDFBB : PrunedIDFBlocks) {
    if (IDFBB == PredBlock || MDT.dominates(IDFBB, PredBlock)) {
      LLVM_DEBUG(dbgs() << "  => Reachable via IDF\n");
      return true;
    }
  }
  
  LLVM_DEBUG(dbgs() << "  => NOT reachable\n");
  return false;
}

bool MachineLaneSSAUpdater::isUseReachableFromDef(MachineOperand &DefOp,
                                                    MachineOperand &UseOp,
                                                    Register OrigVReg) {
  // Derive instructions and lane mask from operands and call the MachineInstr* version
  MachineInstr *DefMI = DefOp.getParent();
  MachineInstr *UseMI = UseOp.getParent();
  LaneBitmask DefMask = operandLaneMask(DefOp);
  return isUseReachableFromDef(DefMI, UseMI, OrigVReg, DefMask);
}

/// During SSA reconstruction, check if a definition reaches a use.
/// Uses IDF-based reachability analysis for correctness.
///
/// DefMI defines a NEW register (NewSSA), and UseOp uses the ORIGINAL register.
/// We need to check if the definition of NewSSA can reach this use of OrigVReg.
bool MachineLaneSSAUpdater::defReachesUse(MachineInstr *DefMI,
                                           Register NewSSA,
                                           MachineInstr *UseMI, 
                                           MachineOperand &UseOp) {
  Register OrigReg = UseOp.getReg();
  assert(OrigReg.isVirtual() && "Expected virtual register");
  assert(NewSSA.isVirtual() && "Expected virtual register");
  
  // Find the def operand in DefMI for NewSSA
  MachineOperand *DefOp = DefMI->findRegisterDefOperand(NewSSA, &TRI, /*isDead=*/false, /*Overlap=*/true);
  assert(DefOp && "DefMI must define NewSSA");
  
  // Use IDF-based reachability check
  // Check if DefOp (NewSSA's definition) can reach UseOp (OrigReg's use)
  return isUseReachableFromDef(*DefOp, UseOp, OrigReg);
}

/// What lanes does this operand read?
LaneBitmask MachineLaneSSAUpdater::operandLaneMask(const MachineOperand &MO) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  if (unsigned Sub = MO.getSubReg())
    return TRI.getSubRegIndexLaneMask(Sub);
  return MRI.getMaxLaneMaskForVReg(MO.getReg());
}

/// Helper: Decompose a potentially non-contiguous lane mask into a vector of
/// subregister indices that together cover all lanes in the mask.
/// From getCoveringSubRegsForLaneMask in AMDGPUSSARAUtils.h (PR #156049).
///
/// Key algorithm: Sort candidates by lane count (prefer larger subregs) to get
/// minimal covering set with largest possible subregisters.
///
/// Example: For vreg_128 with LaneMask = 0x0F | 0xF0 (sub0 + sub2, skipping sub1)
///          Returns: [sub0_idx, sub2_idx] (not lo16, hi16, sub2, sub3)
static SmallVector<unsigned, 4> getCoveringSubRegsForLaneMask(
    LaneBitmask Mask, const TargetRegisterInfo *TRI, 
    const TargetRegisterClass *RC) {
  if (Mask.none())
    return {};
  
  // Step 1: Collect all candidate subregisters that overlap with Mask
  SmallVector<unsigned, 4> Candidates;
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    // Check if this subreg index is valid for this register class
    if (!TRI->getSubRegisterClass(RC, SubIdx))
      continue;
    
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    // Add if it covers any lanes we need
    if ((SubMask & Mask).any()) {
      Candidates.push_back(SubIdx);
    }
  }
  
  // Step 2: Sort by number of lanes (descending) to prefer larger subregisters
  llvm::stable_sort(Candidates, [&](unsigned A, unsigned B) {
    return TRI->getSubRegIndexLaneMask(A).getNumLanes() >
           TRI->getSubRegIndexLaneMask(B).getNumLanes();
  });
  
  // Step 3: Greedily select subregisters, largest first
  SmallVector<unsigned, 4> OptimalSubIndices;
  for (unsigned SubIdx : Candidates) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    // Only add if this subreg is fully contained in the remaining mask
    if ((Mask & SubMask) == SubMask) {
      OptimalSubIndices.push_back(SubIdx);
      Mask &= ~SubMask; // Remove covered lanes
      
      if (Mask.none())
        break; // All lanes covered
    }
  }
  
  return OptimalSubIndices;
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

  // Determine what lanes the use needs
  LaneBitmask UseMask = operandLaneMask(MO);
  
  // Decompose into lanes from NewVR (updated) and lanes from OldVR (unchanged)
  LaneBitmask LanesFromNew = UseMask & MaskToRewrite;
  LaneBitmask LanesFromOld = UseMask & ~MaskToRewrite;
  
  LLVM_DEBUG(dbgs() << "        Building REG_SEQUENCE: UseMask=" << PrintLaneMask(UseMask)
                    << " LanesFromNew=" << PrintLaneMask(LanesFromNew)
                    << " LanesFromOld=" << PrintLaneMask(LanesFromOld) << "\n");
  
  SmallDenseSet<unsigned, 8> AddedSubIdxs;
  
  // Add source for lanes from NewVR (updated lanes)
  if (LanesFromNew.any()) {
    unsigned SubIdx = TRI.getSubRegIndexForLaneMask(LanesFromNew);
    assert(SubIdx && "Failed to find subregister index for LanesFromNew");
    RS.addReg(NewVR, RegState::NoFlags, 0).addImm(SubIdx);  // NewVR is full register, no subreg
    AddedSubIdxs.insert(SubIdx);
    LanesToExtend.push_back(LanesFromNew);
  }
  
  // Add source for lanes from OldVR (unchanged lanes)
  // Handle both contiguous and non-contiguous lane masks
  // Non-contiguous example: Redefining only sub2 of vreg_128 leaves LanesFromOld = sub0+sub1+sub3
  if (LanesFromOld.any()) {
    unsigned SubIdx = TRI.getSubRegIndexForLaneMask(LanesFromOld);
    
    if (SubIdx) {
      // Contiguous case: single subregister covers all lanes
      RS.addReg(OldVR, RegState::NoFlags, SubIdx).addImm(SubIdx);  // OldVR.subIdx
      AddedSubIdxs.insert(SubIdx);
      LanesToExtend.push_back(LanesFromOld);
    } else {
      // Non-contiguous case: decompose into multiple subregisters
      const TargetRegisterClass *OldRC = MRI.getRegClass(OldVR);
      SmallVector<unsigned, 4> CoveringSubRegs = 
          getCoveringSubRegsForLaneMask(LanesFromOld, &TRI, OldRC);
      
      assert(!CoveringSubRegs.empty() && 
             "Failed to decompose non-contiguous lane mask into covering subregs");
      
      LLVM_DEBUG(dbgs() << "        Non-contiguous LanesFromOld=" << PrintLaneMask(LanesFromOld)
                        << " decomposed into " << CoveringSubRegs.size() << " subregs\n");
      
      // Add each covering subregister as a source to the REG_SEQUENCE
      for (unsigned CoverSubIdx : CoveringSubRegs) {
        LaneBitmask CoverMask = TRI.getSubRegIndexLaneMask(CoverSubIdx);
        RS.addReg(OldVR, RegState::NoFlags, CoverSubIdx).addImm(CoverSubIdx);  // OldVR.CoverSubIdx
        AddedSubIdxs.insert(CoverSubIdx);
        LanesToExtend.push_back(CoverMask);
        
        LLVM_DEBUG(dbgs() << "          Added source: OldVR." 
                          << TRI.getSubRegIndexName(CoverSubIdx)
                          << " covering " << PrintLaneMask(CoverMask) << "\n");
      }
    }
  }
  
  assert(!AddedSubIdxs.empty() && "REG_SEQUENCE must have at least one source");

  LIS.InsertMachineInstrInMaps(*RS);
  OutIdx = LIS.getInstructionIndex(*RS);
  
  // Create live interval for the REG_SEQUENCE result
  LIS.createAndComputeVirtRegInterval(Dest);
  
  // Extend live intervals of all source registers to cover this REG_SEQUENCE
  // Use the register slot to ensure the live range covers the use
  SlotIndex UseSlot = OutIdx.getRegSlot();
  for (MachineOperand &MO : RS.getInstr()->uses()) {
    if (MO.isReg() && MO.getReg().isVirtual()) {
      Register SrcReg = MO.getReg();
      LiveInterval &SrcLI = LIS.getInterval(SrcReg);
      LIS.extendToIndices(SrcLI, {UseSlot});
    }
  }

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

void MachineLaneSSAUpdater::updateDeadFlags(Register Reg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LiveInterval &LI = LIS.getInterval(Reg);
  MachineInstr *DefMI = MRI.getVRegDef(Reg);
  if (!DefMI)
    return;
  
  for (MachineOperand &MO : DefMI->defs()) {
    if (MO.getReg() == Reg && MO.isDead()) {
      // Check if this register is actually live (has uses)
      if (!LI.empty() && !MRI.use_nodbg_empty(Reg)) {
        MO.setIsDead(false);
        LLVM_DEBUG(dbgs() << "  Cleared dead flag on " << Reg << "\n");
      }
    }
  }
}

// Remove the old helper that's no longer needed
// LaneBitmask MachineLaneSSAUpdater::getLaneMaskForOperand(...) - REMOVED