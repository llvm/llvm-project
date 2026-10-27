//===- MachineLaneSSAUpdater.cpp - SSA repair for Machine IR (lane-aware) ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lane-aware repair of inserted virtual-register definitions. The updater uses
// frozen LiveInterval reaching values to insert lane PHIs and rewrite uses,
// reconstructing mixed-lane values with REG_SEQUENCE where needed. The batch
// entry point finalizes coupled operands and recomputes affected intervals
// before reporting them to the caller. Debug uses are excluded from reaching
// queries, and external bundle operands share their reconstruction.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "machine-lane-ssa-updater"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineLaneSSAUpdater Implementation
//===----------------------------------------------------------------------===//

FrozenInterval MachineLaneSSAUpdater::freezeInterval(Register OrigVReg) const {
  // Deep copy of OrigVReg's current interval into a caller-owned allocator. Same
  // content the session self-freeze produces: assign copies the value numbers,
  // and each subrange is copied by lane mask. Read-only w.r.t. the live interval.
  FrozenInterval F;
  F.LI = std::make_unique<LiveInterval>(OrigVReg, 0.0f);
  if (LIS.hasInterval(OrigVReg)) {
    const LiveInterval &Src = LIS.getInterval(OrigVReg);
    F.LI->assign(Src, F.Alloc);
    for (const LiveInterval::SubRange &S : Src.subranges())
      F.LI->createSubRangeFrom(F.Alloc, S.LaneMask, S);
  }
  return F;
}

MachineLaneSSAUpdater::RepairResult
MachineLaneSSAUpdater::repairSSAForNewDefs(
    Register OrigVReg, ArrayRef<MachineInstr *> NewDefs,
    LivenessCallback OnChange) {
  assert(!AffectedRegs && "SSA repair notification is not reentrant");
  assert(OrigVReg.isVirtual() && LIS.hasInterval(OrigVReg) &&
         "batch requires indexed redefinitions and their reaching interval");

  SmallSetVector<Register, 8> Affected;
  AffectedRegs = &Affected;
  recordAffected(OrigVReg);
  resetSession();
  const FrozenInterval Oracle = freezeInterval(OrigVReg);
  bool InsertedPHI = false;
  for (MachineInstr *MI : NewDefs) {
    SmallVector<MachineOperand *, 4> PHIDefs;
    repairSSAForNewDef(*MI, OrigVReg, PHIDefs, &Oracle);
    InsertedPHI |= !PHIDefs.empty();
  }
  finalizeCoupledUses(OrigVReg);

  RepairResult Result;
  Result.Affected.append(Affected.begin(), Affected.end());
  llvm::sort(Result.Affected,
             [](Register A, Register B) { return A.id() < B.id(); });
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (Register VR : Result.Affected) {
    // All renames and operand rewrites are complete. Recomputing here is safe
    // even for the original partial-def value: no sibling lane is migrating.
    // It also accounts for PHI-edge uses and coupled-operand rewrites, which
    // cannot be described by simply truncating the old main range.
    if (LIS.hasInterval(VR))
      LIS.removeInterval(VR);
    if (MRI.def_empty(VR)) {
      if (llvm::any_of(MRI.use_nodbg_operands(VR),
                       [](const MachineOperand &MO) { return MO.readsReg(); }))
        report_fatal_error("SSA repair left a live value without a definition");
      // Undef-only operands still name this value. Keep an empty interval so
      // consumers can distinguish it from a fully retired register.
      if (!MRI.reg_nodbg_empty(VR))
        LIS.createEmptyInterval(VR);
      continue; // Keep its ID in the notification so old ownership is removed.
    }
    assert(MRI.hasOneDef(VR) && "batch did not restore SSA for affected value");
    LIS.createAndComputeVirtRegInterval(VR);
    updateDeadFlags(VR);
  }
  if (InsertedPHI)
    MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);

  // The external reaching oracle is local to this batch. A subsequent repair
  // must start a fresh session rather than observe the destroyed snapshot.
  ExternalFrozen = nullptr;
  resetSession();
  OnChange(Result.Affected);
  AffectedRegs = nullptr;
  return Result;
}

Register MachineLaneSSAUpdater::repairSSAForNewDef(
    MachineInstr &NewDefMI, Register OrigVReg,
    SmallVectorImpl<MachineOperand *> &PHIRegDefOps,
    const FrozenInterval *Oracle) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::repairSSAForNewDef VReg="
                    << OrigVReg << "\n");

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Scope the rename map to one OrigVReg. The driver processes all defs of an
  // OrigVReg before moving on, so clearing on change is sufficient.
  if (RenameSessionOrig != OrigVReg) {
    // Debug locations have no SlotIndex and cannot participate in reaching-value
    // repair. Retire them before renaming definitions rather than retain stale
    // references to the original register.
    MRI.markUsesInDebugValueAsUndef(OrigVReg);
    RenameSessionOrig = OrigVReg;
    DefInstrToRenamed.clear();
    LanePHIs.clear();
    SuperUseRSCache.clear();

    // Capture intra-instruction coupling groups BEFORE any rewrite. SSA repair
    // resolves each operand by its own reaching value, so two operands of ONE
    // non-PHI instruction that read OrigVReg (one a lane-subset of the other)
    // can land on different base registers. Some targets require the narrow
    // operand to stay a subregister of the wide one (e.g. AMDGPU V_MOVRELS src0
    // must be a subreg of the implicit vector use). For each use instruction,
    // scan ITS operands for others reading the same reg; a group of >=2 is
    // coupling to restore in finalizeCoupledUses.
    CoupledUseGroups.clear();
    for (MachineOperand &MO : MRI.use_nodbg_operands(OrigVReg)) {
      MachineInstr *UMI = MO.getParent();
      if (UMI->isPHI())
        continue;
      SmallVector<std::pair<unsigned, LaneBitmask>, 4> Group;
      for (unsigned I = 0, E = UMI->getNumOperands(); I < E; ++I) {
        const MachineOperand &Op = UMI->getOperand(I);
        if (Op.isReg() && Op.isUse() && Op.getReg() == OrigVReg)
          Group.push_back({I, operandLaneMask(Op)});
      }
      if (Group.size() >= 2)
        CoupledUseGroups[UMI] = std::move(Group); // idempotent on repeat visits
    }

    // Reaching-def oracle for this session: a frozen deep copy of OrigVReg's
    // interval whose VNInfos (def SlotIndex + isPHIDef) survive the in-place
    // renames done during repair (renaming strips a def's VNInfo from the live
    // interval when it is recomputed). Two sources:
    //  - EXTERNAL: the caller froze it once, AFTER emitting all redefs, and
    //    passes it in. Point at it; leave the self-frozen copy empty. This lets a
    //    caller emitting several redefs of one vreg share ONE snapshot without
    //    resetSession() churn between calls.
    //  - SELF: no oracle -> freeze here (default, standalone callers). Reset the
    //    old copy BEFORE its allocator, else the subranges (owned by FrozenAlloc)
    //    are double-freed.
    FrozenOrigLI.reset();
    FrozenAlloc.Reset();
    if (Oracle && Oracle->valid()) {
      ExternalFrozen = Oracle->LI.get();
    } else {
      ExternalFrozen = nullptr;
      FrozenOrigLI = std::make_unique<LiveInterval>(OrigVReg, 0.0f);
      if (LIS.hasInterval(OrigVReg)) {
        const LiveInterval &Src = LIS.getInterval(OrigVReg);
        FrozenOrigLI->assign(Src, FrozenAlloc);
        for (const LiveInterval::SubRange &S : Src.subranges())
          FrozenOrigLI->createSubRangeFrom(FrozenAlloc, S.LaneMask, S);
      }
    }
  }

  // Step 1: Find the def operand for OrigVReg. Iterate ALL operands filtered by
  // the isDef flag rather than NewDefMI.defs(): the range helper returns only
  // the leading explicit defs ([0, getNumExplicitDefs())), which is empty for
  // variadic instructions like INLINEASM -- their def operands sit after the asm
  // string and flag immediates, so defs() misses the def of OrigVReg entirely.
  MachineOperand *DefOp = nullptr;
  for (MachineOperand &MO : NewDefMI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() == OrigVReg) {
      DefOp = &MO;
      break;
    }
  }

  assert(DefOp && "NewDefMI should have a def operand for OrigVReg");
  assert(DefOp->isDef() && "Found operand should be a definition");

  // Step 2: Derive DefMask from the operand's subreg index (if any)
  unsigned SubRegIdx = DefOp->getSubReg();
  const bool UndefUnwrittenLanes = SubRegIdx && DefOp->isUndef();
  LaneBitmask DefMask;

  if (SubRegIdx) {
    // Partial register definition - get lane mask for this subreg
    DefMask = TRI.getSubRegIndexLaneMask(SubRegIdx);
    LLVM_DEBUG(dbgs() << "  Partial def with subreg "
                      << TRI.getSubRegIndexName(SubRegIdx)
                      << ", DefMask=" << PrintLaneMask(DefMask) << "\n");
  } else {
    // Full register definition - get all lanes for this register class
    DefMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
    LLVM_DEBUG(dbgs() << "  Full register def, DefMask="
                      << PrintLaneMask(DefMask) << "\n");
  }

  // Step 3: Create a new virtual register with class matching MaskToRewrite
  // width
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
  recordAffected(NewSSAVReg);
  LLVM_DEBUG(dbgs() << "  Created new SSA vreg " << NewSSAVReg
                    << " with RC=" << TRI.getRegClassName(RC) << "\n");

  // Step 4: Replace the operand in NewDefMI to define the new vreg
  // Clear subreg index since NewSSAVReg is a full register of the (possibly
  // narrower) class
  DefOp->setReg(NewSSAVReg);
  if (SubRegIdx) {
    DefOp->setSubReg(
        0); // Clear subreg - NewSSAVReg is full register of narrower class
    DefOp->setIsUndef(
        false); // Clear undef flag - verifier requires undef only with subreg
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg << "."
                      << TRI.getSubRegIndexName(SubRegIdx) << " -> "
                      << NewSSAVReg << " (full narrower register)\n");
  } else {
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg << " -> "
                      << NewSSAVReg << "\n");
  }

  // Step 5: Index the new instruction in SlotIndexes/LIS
  indexNewInstr(NewDefMI);

  // Make this rename available to later PHI construction in this session:
  // record the renamed def by its instruction (with the OrigVReg lanes it
  // covers), so reaching-VNInfo lookups can map a reaching real def to its
  // renamed vreg and rebase target lanes.
  DefInstrToRenamed[&NewDefMI] = {NewSSAVReg, DefMask,
                                  UndefUnwrittenLanes};

  // Step 6: Perform common SSA repair (PHI placement + use rewriting)
  // LiveInterval for NewSSAVReg will be created by getInterval() as needed
  PHIRegDefOps = performSSARepair(
      NewSSAVReg, OrigVReg, DefMask,
      LIS.getInstructionIndex(NewDefMI).getRegSlot(DefOp->isEarlyClobber()));

  // Step 7: If SSA repair created subregister uses of OrigVReg (e.g., in PHIs
  // or REG_SEQUENCEs), recompute its LiveInterval to create subranges
  LaneBitmask AllLanes = MRI.getMaxLaneMaskForVReg(OrigVReg);
  if (DefMask != AllLanes) {
    const LiveInterval &OrigLI = LIS.getInterval(OrigVReg);
    if (!OrigLI.hasSubRanges()) {
      // Check if any uses now access OrigVReg with subregister indices
      bool HasSubregUses = false;
      for (const MachineOperand &MO : MRI.use_nodbg_operands(OrigVReg)) {
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

void MachineLaneSSAUpdater::rebuildMainRangeFromSubranges(Register OrigVReg) {
  if (!LIS.hasInterval(OrigVReg))
    return;
  LiveInterval &LI = LIS.getInterval(OrigVReg);
  if (!LI.hasSubRanges())
    return;
  LLVM_DEBUG(dbgs() << "  rebuilding main range of " << printReg(OrigVReg, &TRI)
                    << " from subranges\n");
  LI.clear();
  LI.removeEmptySubRanges();
  LIS.constructMainRangeFromSubranges(LI);
}

//===----------------------------------------------------------------------===//
// Common SSA Repair Logic
//===----------------------------------------------------------------------===//

SmallVector<MachineOperand *>
MachineLaneSSAUpdater::performSSARepair(Register NewVReg, Register OrigVReg,
                                        LaneBitmask DefMask,
                                        SlotIndex MigratedDef) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::performSSARepair NewVReg="
                    << NewVReg << " OrigVReg=" << OrigVReg
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");

  // Step 1: Use worklist-driven PHI placement. Each PHI is paired with the
  // OrigVReg lane it covers.
  SmallVector<std::pair<MachineOperand *, LaneBitmask>> AllPHIVResults =
      insertLaneAwarePHI(OrigVReg, DefMask);

  // Step 2: Rewrite reaching uses once for each new register. Each PHI is
  // rewritten with ITS OWN lane (per-subrange in the reaching path; DefMask in
  // the legacy path) so subreg rebasing is correct.
  // Note: getInterval() will automatically create LiveIntervals if needed
  rewriteDominatedUses(OrigVReg, NewVReg, DefMask);
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    rewriteDominatedUses(OrigVReg, PHIVRes->getReg(), Lane);

  // Step 3: Renumber values if needed
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  NewLI.RenumberValues();

  // Also renumber PHI intervals
  for (auto &[PHIVRes, Lane] : AllPHIVResults) {
    LiveInterval &PHILI = LIS.getInterval(PHIVRes->getReg());
    PHILI.RenumberValues();
  }

  // Update only the lanes this repair touched. A whole-value recompute
  // (createAndComputeVirtRegInterval -> per-subrange extendToUses +
  // constructMainRangeFromSubranges) is invalid here: this repair moved the
  // DefMask lane's def to NewVReg, so OrigVReg's subranges no longer all belong to
  // OrigVReg. Recomputing every subrange re-derives a sibling lane whose def is
  // mid-migration, and constructMainRangeFromSubranges then aborts with "Use not
  // jointly dominated by defs" (physreg-pair partial-def case). Keep the updater
  // self-contained: drop the migrated def, refresh the DefMask subrange(s) from
  // OrigVReg's remaining uses, and leave sibling subranges untouched. The main
  // range is the union of the subranges and cannot be maintained while lanes are
  // split across vregs, so the driver rebuilds it once via
  // rebuildMainRangeFromSubranges() after its last incremental repair call.
  // The batch entry point instead recomputes affected intervals after all
  // renames and coupled-operand rewrites are complete.
  LiveInterval &OrigLI = LIS.getInterval(OrigVReg);
  if (OrigLI.hasSubRanges()) {
    for (LiveInterval::SubRange &S : OrigLI.subranges()) {
      if ((S.LaneMask & DefMask).none())
        continue;
      // The def at MigratedDef belongs to NewVReg now - its operand was
      // retargeted before this repair ran - so OrigVReg must lose that value.
      // shrinkToUses cannot do it: it trims segments but keeps dead defs, and
      // the surviving zero-length segment still reports as overlapping any
      // range that starts at the same slot.
      if (VNInfo *VNI = S.getVNInfoAt(MigratedDef))
        if (VNI->def.getBaseIndex() == MigratedDef.getBaseIndex())
          S.removeValNo(VNI);
      LIS.shrinkToUses(S, OrigVReg);
    }
    OrigLI.removeEmptySubRanges();
  } else {
    LIS.removeInterval(OrigVReg);
    LIS.createAndComputeVirtRegInterval(OrigVReg);
  }

  // No additional whole-register shrink is performed here. The subrange path
  // above updates only migrated lanes; the no-subrange path recomputes the
  // interval from the rewritten operands, including incoming PHI-edge uses.

  // Step 4: Update operand flags to match the LiveIntervals
  updateDeadFlags(NewVReg);
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    updateDeadFlags(PHIVRes->getReg());

  LLVM_DEBUG(dbgs() << "  performSSARepair complete\n");
  SmallVector<MachineOperand *> Results;
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    Results.push_back(PHIVRes);
  return Results;
}

//===----------------------------------------------------------------------===//
// Internal Helper Methods
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

SmallVector<std::pair<MachineOperand *, LaneBitmask>>
MachineLaneSSAUpdater::insertLaneAwarePHI(Register OrigVReg,
                                          LaneBitmask DefMask) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::insertLaneAwarePHI OrigVReg="
                    << OrigVReg << " DefMask=" << PrintLaneMask(DefMask)
                    << "\n");

  SmallVector<std::pair<MachineOperand *, LaneBitmask>> AllCreatedPHIs;

  // PHIs go exactly at the join points already recorded in OrigVReg's frozen
  // interval as PHI-def VNInfos. LiveIntervalCalc computed those when the
  // interval was built, so per lane they ARE the pruned iterated dominance
  // frontier -- no IDF recomputation needed. Dead lanes (no merge) have no
  // PHI-def VNInfo, so they correctly get no PHI. Operands are resolved from
  // the frozen reaching oracle (placeholder-then-patch).
  auto PlacePHIsFor = [&](const LiveRange &LR, LaneBitmask Lane) {
    for (const VNInfo *VNI : LR.valnos) {
      if (!VNI || VNI->isUnused() || !VNI->isPHIDef())
        continue;
      MachineBasicBlock *JoinMBB = LIS.getMBBFromIndex(VNI->def);
      if (!JoinMBB)
        continue;
      if (MachineOperand *PHIResult =
              createPHIInBlockReaching(*JoinMBB, OrigVReg, Lane))
        AllCreatedPHIs.push_back({PHIResult, Lane});
    }
  };
  if (const LiveInterval *Frozen = frozenLI()) {
    if (Frozen->hasSubRanges()) {
      for (const LiveInterval::SubRange &S : Frozen->subranges())
        if ((S.LaneMask & DefMask).any())
          PlacePHIsFor(S, S.LaneMask);
    } else {
      PlacePHIsFor(*Frozen, DefMask);
    }
  }
  LLVM_DEBUG(dbgs() << "  PHI insertion complete. Created "
                    << AllCreatedPHIs.size() << " PHI registers total.\n");
  return AllCreatedPHIs;
}

// Reaching-VNI path (approach A): one PHI for a single subrange-aligned lane
// group. Every predecessor operand has exactly one reaching piece -> a single
// source (renamed reaching def) or an OrigVReg.subIdx placeholder patched
// later.
MachineOperand *MachineLaneSSAUpdater::createPHIInBlockReaching(
    MachineBasicBlock &JoinMBB, Register OrigVReg, LaneBitmask Lane) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterClass *OrigRC = MRI.getRegClass(OrigVReg);
  const LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);

  // Dedup: reuse a PHI already built for this (block, lane) this session.
  auto Key = std::make_pair(&JoinMBB, Lane);
  if (Register Existing = LanePHIs.lookup(Key))
    return &MRI.getVRegDef(Existing)->getOperand(0);

  // PHI result class = subregister class for Lane (full reg when Lane == full).
  const TargetRegisterClass *RC = OrigRC;
  unsigned LaneSub =
      (Lane == FullMask) ? 0 : getSubRegIndexForLaneMask(Lane, &TRI);
  if (LaneSub)
    if (const TargetRegisterClass *SubRC =
            TRI.getSubRegisterClass(OrigRC, LaneSub))
      RC = SubRC;

  Register PHIVReg = MRI.createVirtualRegister(RC);
  recordAffected(PHIVReg);
  auto PHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                         TII->get(TargetOpcode::PHI), PHIVReg);
  LLVM_DEBUG(dbgs() << "    createPHIInBlockReaching in BB#"
                    << JoinMBB.getNumber()
                    << " OrigVReg=" << printReg(OrigVReg, &TRI) << " Lane="
                    << PrintLaneMask(Lane) << " -> " << PHIVReg << "\n");

  for (MachineBasicBlock *Pred : JoinMBB.predecessors()) {
    SlotIndex EndP = LIS.getMBBEndIdx(Pred);
    const VNInfo *V = reachingVNIForLaneGroup(*frozenLI(), Lane, EndP);

    if (const RenamedDef *RD = renamedForReachingVNI(V)) {
      // Extract Lane from the renamed vreg: rebase Lane into its namespace.
      LaneBitmask RLane = rebaseLaneMask(Lane, RD->OrigLanes);
      LaneBitmask RFull = MRI.getMaxLaneMaskForVReg(RD->VReg);
      unsigned Sub =
          (RLane == RFull) ? 0 : getSubRegIndexForLaneMask(RLane, &TRI);
      recordAffected(RD->VReg);
      PHINode.addReg(RD->VReg, {}, Sub);
    } else {
      // No renamed reaching def on this edge. Two cases:
      //  * V == null: the lane has no reaching def here (undef on this edge,
      //    e.g. a predecessor that establishes only the other lanes). Source it
      //    with the undef flag; a plain read would be "PHI operand is not
      //    live-out from predecessor".
      //  * V != null: a Root/final or not-yet-renamed value -> keep an OrigVReg
      //    placeholder (final, or patched later by its owner's rewrite).
      unsigned Sub =
          (Lane == FullMask) ? 0 : getSubRegIndexForLaneMask(Lane, &TRI);
      recordAffected(OrigVReg);
      PHINode.addReg(OrigVReg, V ? RegState{} : RegState::Undef, Sub);
    }
    PHINode.addMBB(Pred);
  }

  MachineInstr *PHI = PHINode.getInstr();
  LIS.InsertMachineInstrInMaps(*PHI);
  LanePHIs[Key] = PHIVReg;
  LLVM_DEBUG(dbgs() << "      Created: "; PHI->print(dbgs()));
  return &PHI->getOperand(0);
}

void MachineLaneSSAUpdater::rewriteUseReaching(
    Register OrigVReg, Register NewSSA, LaneBitmask MaskToRewrite,
    const MachineInstr *DefMI, MachineInstr *UseMI, MachineOperand &MO,
    LaneBitmask OpMask, LiveInterval &OrigLI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Point at which we query the reaching value: end of the predecessor for a
  // PHI operand, the use instruction otherwise.
  SlotIndex UsePt;
  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(&MO);
    MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
    UsePt = LIS.getMBBEndIdx(Pred);
  } else {
    // A tied use is read before the instruction's own def executes. For an
    // early-clobber two-address def (e.g. V_WMMA_*_twoaddr accumulator), that
    // def's VNInfo sits at getRegSlot(EC=true), i.e. at/adjacent to the plain
    // reg slot; querying the reaching value there resolves to the instruction's
    // OWN def instead of the incoming value, so the tied use is never rewritten
    // and is left reading the now-renamed OrigVReg ("Reading virtual register
    // without a def"). The base index precedes every def slot, so it yields the
    // incoming value. Non-tied uses are unaffected (no def of OrigVReg on the
    // use instruction to collide with).
    SlotIndex Idx = LIS.getInstructionIndex(*UseMI);
    UsePt = MO.isTied() ? Idx.getBaseIndex() : Idx.getRegSlot();
  }

  // Lanes (within MaskToRewrite) whose reaching value at the use IS this def.
  // A real (non-PHI) def matches the frozen VNInfo at its slot; an inserted PHI
  // result matches the frozen PHI-def VNInfo at the PHI's block (the frozen LI
  // predates our PHIs and records merges as block-boundary PHI-defs, so PHI
  // results can't be matched by instruction slot).
  const bool DefIsPHI = DefMI->isPHI();
  const MachineBasicBlock *DefBB = DefMI->getParent();
  // A real def's VNInfo sits at the early-clobber slot iff its def operand is
  // early-clobber (e.g. S_LOAD_..._ec); otherwise at the regular slot. Use the
  // exact slot so we match only THIS def, never another def on the same instr.
  SlotIndex DefSlot;
  if (!DefIsPHI) {
    bool EC = false;
    for (const MachineOperand &D : DefMI->defs())
      if (D.getReg() == NewSSA) {
        EC = D.isEarlyClobber();
        break;
      }
    DefSlot = LIS.getInstructionIndex(*DefMI).getRegSlot(EC);
  }
  SmallVector<std::pair<LaneBitmask, const VNInfo *>, 4> Pieces;
  collectReachingVNIs(*frozenLI(), OpMask & MaskToRewrite, UsePt, Pieces);
  LaneBitmask Owned = LaneBitmask::getNone();
  for (auto &[L, V] : Pieces) {
    if (!V)
      continue;
    bool Match = DefIsPHI
                     ? (V->isPHIDef() && LIS.getMBBFromIndex(V->def) == DefBB)
                     : (!V->isPHIDef() && V->def == DefSlot);
    if (Match)
      Owned |= L;
  }

  if (Owned.none())
    return; // this def provides none of the use's lanes; owner patches later

  LLVM_DEBUG(dbgs() << "    [reaching] use OpMask=" << PrintLaneMask(OpMask)
                    << " Owned=" << PrintLaneMask(Owned) << ": ";
             UseMI->print(dbgs()));

  if (Owned == OpMask) {
    // Whole operand provided by this def: use NewSSA with OpMask's subreg
    // (rebased into NewSSA's namespace; 0 when it is the full register).
    LaneBitmask NewLanes = rebaseLaneMask(OpMask, MaskToRewrite);
    LaneBitmask NewFull = MRI.getMaxLaneMaskForVReg(NewSSA);
    unsigned Sub =
        (NewLanes == NewFull) ? 0 : getSubRegIndexForLaneMask(NewLanes, &TRI);
    recordAffected(MO.getReg());
    recordAffected(NewSSA);
    MO.setReg(NewSSA);
    MO.setSubReg(Sub);
    return;
  }

  // Multiple operands of the same non-PHI instruction that read the same
  // OrigVReg lanes must resolve to ONE composed value, not a distinct
  // REG_SEQUENCE per operand -- otherwise a constant-bus- or src0==src1-
  // constrained instruction (e.g. V_MAX_F64 x, x, V_DIV_SCALE_F64 %v, %v) sees
  // two different registers. Reuse the super-use REG_SEQUENCE built for the
  // first such operand this session. PHIs are excluded: their operands read on
  // distinct edges, so a shared key would wrongly conflate different values.
  const bool ShareRS = !UseMI->isPHI();
  // A bundle header summarizes its external uses. Share the reconstructed
  // value with the matching operands on its members so both name one vreg.
  MachineInstr *UseUnit = UseMI;
  if (UseMI->isInsideBundle() && !MO.isInternalRead()) {
    auto BundleI =
        MachineBasicBlock::iterator::getAtBundleBegin(UseMI->getIterator());
    UseUnit = &*BundleI;
  }
  auto RSKey = std::make_pair(UseUnit, OpMask);
  if (ShareRS)
    if (Register Cached = SuperUseRSCache.lookup(RSKey)) {
      recordAffected(MO.getReg());
      recordAffected(Cached);
      MO.setReg(Cached);
      MO.setSubReg(0);
      return;
    }

  // Partial: compose Owned lanes from NewSSA; the remaining lanes stay OrigVReg
  // (Root/final or a placeholder patched by their owner). The REG_SEQUENCE
  // result is a fresh whole register that only needs to hold OpMask-many lanes.
  // The operand may name an UNALIGNED subregister of OrigVReg (e.g.
  // sub1_sub2_sub3 of an sgpr_128, formed while iteratively rebuilding a
  // partial-def chain), for which getSubRegisterClass(OpRC, MO.getSubReg()) is
  // null. Derive the class from the base-0 (rebased) lane mask instead, which
  // maps to an aligned, valid subregister index of OpRC.
  const TargetRegisterClass *OpRC = MRI.getRegClass(MO.getReg());
  LaneBitmask FullRC = MRI.getMaxLaneMaskForVReg(MO.getReg());
  LaneBitmask RebasedUse = rebaseLaneMask(OpMask, OpMask);
  unsigned UseSub =
      (RebasedUse == FullRC) ? 0 : getSubRegIndexForLaneMask(RebasedUse, &TRI);
  const TargetRegisterClass *UseRC =
      UseSub ? TRI.getSubRegisterClass(OpRC, UseSub) : OpRC;
  assert(UseRC && "use operand subreg has no register class");
  SmallVector<LaneBitmask, 4> LanesToExtend;
  SlotIndex RSIdx;
  Register RSReg = buildRSForSuperUse(UseMI, MO, OrigVReg, NewSSA, Owned, UseRC,
                                      RSIdx, LanesToExtend);
  if (ShareRS)
    SuperUseRSCache[RSKey] = RSReg;
  extendAt(OrigLI, RSIdx, LanesToExtend);
  recordAffected(MO.getReg());
  recordAffected(RSReg);
  MO.setReg(RSReg);
  MO.setSubReg(0);
  LIS.extendToIndices(LIS.getInterval(RSReg), {UsePt});
  updateDeadFlags(RSReg);
}

void MachineLaneSSAUpdater::rewriteDominatedUses(Register OrigVReg,
                                                 Register NewSSA,
                                                 LaneBitmask MaskToRewrite) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::rewriteDominatedUses OrigVReg="
                    << OrigVReg << " NewSSA=" << NewSSA
                    << " Mask=" << PrintLaneMask(MaskToRewrite) << "\n");

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Find the definition instruction for NewSSA
  const MachineInstr *DefMI = MRI.getVRegDef(NewSSA);
  if (!DefMI) {
    LLVM_DEBUG(dbgs() << "  No definition found for NewSSA, skipping\n");
    return;
  }

  const MachineBasicBlock *DefBB = DefMI->getParent();

  LLVM_DEBUG(dbgs() << "  Rewriting uses dominated by definition in BB#"
                    << DefBB->getNumber() << ": ");
  LLVM_DEBUG(DefMI->print(dbgs()));

  // Get OrigVReg's LiveInterval for reference
  LiveInterval &OrigLI = LIS.getInterval(OrigVReg);

  // Iterate through all uses of OrigVReg
  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI.use_nodbg_operands(OrigVReg))) {
    MachineInstr *UseMI = MO.getParent();

    // Skip the defining instruction itself -- its reads are the OLD value
    // (e.g. a read-modify-write redef "%new = OP %orig"). A PHI is the sole
    // exception: its incoming operands live on distinct edges and one may be a
    // legal loop self-reference to the PHI's own result (the back-edge lane).
    // Let those flow into rewriteUseReaching, which only claims lanes whose
    // reaching VNInfo is this PHI-def, so non-self operands are left untouched.
    if (UseMI == DefMI && !DefMI->isPHI())
      continue;

    // Get the lane mask for this operand
    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    // Reaching-VNI ownership: this def owns exactly the OpMask lanes whose
    // reaching value at the use is this def's own VNInfo (replaces the old
    // block-dominance filter, which over-/under-claimed lanes -- Bucket 3).
    rewriteUseReaching(OrigVReg, NewSSA, MaskToRewrite, DefMI, UseMI, MO,
                       OpMask, OrigLI);
  }

  LLVM_DEBUG(dbgs() << "  Completed rewriting dominated uses\n");
}

void MachineLaneSSAUpdater::finalizeCoupledUses(Register OrigVReg) {
  // Only act on the session that captured these groups.
  if (RenameSessionOrig != OrigVReg || CoupledUseGroups.empty())
    return;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  LaneBitmask FullOrig = MRI.getMaxLaneMaskForVReg(OrigVReg);

  for (auto &[UMI, Ops] : CoupledUseGroups) {
    // Coupling base = the operand that read the WHOLE OrigVReg. Whatever reg it
    // resolved to holds every lane in OrigVReg's layout, so any sibling subset
    // maps to a subreg of it with the same lane mask.
    int BaseI = -1;
    for (unsigned I = 0; I < Ops.size(); ++I)
      if (Ops[I].second == FullOrig) {
        BaseI = (int)I;
        break;
      }
    if (BaseI < 0)
      continue; // no whole-register read -> no anchor

    const MachineOperand &BaseMO = UMI->getOperand(Ops[BaseI].first);
    if (BaseMO.getSubReg())
      continue; // base no longer names a whole reg -> can't sub-index cleanly
    Register BaseReg = BaseMO.getReg();
    LaneBitmask BaseRegFull = MRI.getMaxLaneMaskForVReg(BaseReg);

    for (unsigned I = 0; I < Ops.size(); ++I) {
      if ((int)I == BaseI)
        continue;
      MachineOperand &MO = UMI->getOperand(Ops[I].first);
      if (MO.getReg() == BaseReg)
        continue; // already coupled (repair or SuperUseRSCache landed it here)

      // SubMask is in OrigVReg's layout, which equals BaseReg's (BaseReg is the
      // full-vector reconstruction), so it selects the right subreg of BaseReg.
      LaneBitmask SubMask = Ops[I].second;
      unsigned Sub = (SubMask == BaseRegFull)
                         ? 0
                         : getSubRegIndexForLaneMask(SubMask, &TRI);
      if (!Sub && SubMask != BaseRegFull)
        continue; // no aligned subreg for these lanes -> leave as-is (valid)

      recordAffected(MO.getReg());
      recordAffected(BaseReg);
      MO.setReg(BaseReg);
      MO.setSubReg(Sub);
    }
  }
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Decompose \p Mask by OrigVReg's subranges and append {laneSubmask, reaching
/// VNInfo at \p Idx} per covered piece. Pieces partition \p Mask (subranges
/// partition all lanes), so the caller uses a single value when one piece
/// covers all of Mask, else composes a REG_SEQUENCE. A null VNInfo means dead
/// lanes. A reg without subranges yields one {Mask, mainVNI}.
void MachineLaneSSAUpdater::collectReachingVNIs(
    const LiveInterval &OrigLI, LaneBitmask Mask, SlotIndex Idx,
    SmallVectorImpl<std::pair<LaneBitmask, const VNInfo *>> &Out) {
  if (OrigLI.hasSubRanges()) {
    for (const LiveInterval::SubRange &S : OrigLI.subranges()) {
      LaneBitmask Piece = S.LaneMask & Mask;
      if (Piece.none())
        continue;
      Out.push_back({Piece, S.getVNInfoBefore(Idx)});
    }
    return;
  }
  Out.push_back({Mask, OrigLI.getVNInfoBefore(Idx)});
}

const VNInfo *MachineLaneSSAUpdater::reachingVNIForLaneGroup(
    const LiveInterval &OrigLI, LaneBitmask Lane, SlotIndex Idx) {
  SmallVector<std::pair<LaneBitmask, const VNInfo *>, 2> Pieces;
  collectReachingVNIs(OrigLI, Lane, Idx, Pieces);
  assert(Pieces.size() <= 1 &&
         "reachingVNIForLaneGroup: lane group must lie within one subrange");
  return Pieces.empty() ? nullptr : Pieces.front().second;
}

/// Map a reaching VNInfo to the RenamedDef its def was renamed to this session.
/// Returns null when the value is a PHI-def merge (resolved via placeholder-
/// then-patch) or when the def has not (yet) been renamed, so the caller keeps
/// an OrigVReg placeholder that a later repair patches.
const MachineLaneSSAUpdater::RenamedDef *
MachineLaneSSAUpdater::renamedForReachingVNI(const VNInfo *V) const {
  if (!V || V->isUnused() || V->isPHIDef())
    return nullptr;
  const MachineInstr *DefMI = LIS.getInstructionFromIndex(V->def);
  if (!DefMI)
    return nullptr;
  auto It = DefInstrToRenamed.find(DefMI);
  return It != DefInstrToRenamed.end() ? &It->second : nullptr;
}

/// Check whether \p UseMI (a use of \p OrigVReg) is reachable from \p DefMI in
/// the CFG. Since OrigVReg is SSA, the value is live along every def->use path,
/// so plain CFG reachability exactly answers "is this use downstream of the
/// def" -- no dominance-frontier needed.
bool MachineLaneSSAUpdater::isUseReachableFromDef(const MachineInstr *DefMI,
                                                  const MachineInstr *UseMI,
                                                  Register OrigVReg) const {
  const MachineBasicBlock *DefBlock = DefMI->getParent();
  const MachineOperand *UseOp =
      UseMI->findRegisterUseOperand(OrigVReg, &TRI, /*isKill=*/false);
  assert(UseOp && "UseMI must use OrigVReg");

  // Target of the query: the use's block, or -- for a PHI use -- the block the
  // value flows in from.
  const bool IsPHI = UseMI->isPHI();
  const MachineBasicBlock *Target = UseMI->getParent();
  if (IsPHI) {
    unsigned OpIdx = UseMI->getOperandNo(UseOp);
    Target = UseMI->getOperand(OpIdx + 1).getMBB();
  }

  if (Target == DefBlock) {
    // A PHI incoming value leaves DefBlock after the def, so it is downstream.
    // A non-PHI same-block use is downstream iff the def precedes it; a use
    // *before* the def is reachable only via a back-edge (handled by the walk).
    if (IsPHI ||
        LIS.getInstructionIndex(*DefMI) < LIS.getInstructionIndex(*UseMI))
      return true;
  } else if (MDT.dominates(DefBlock, Target)) {
    // Fast path: dominance implies reachability.
    return true;
  }

  // General CFG reachability from DefBlock's successors. A block is its own
  // successor only through a back-edge, which correctly captures loop-carried
  // uses and excludes acyclic use-before-def.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  SmallVector<const MachineBasicBlock *, 16> Worklist(DefBlock->succ_begin(),
                                                      DefBlock->succ_end());
  while (!Worklist.empty()) {
    const MachineBasicBlock *BB = Worklist.pop_back_val();
    if (BB == Target)
      return true;
    if (Visited.insert(BB).second)
      Worklist.append(BB->succ_begin(), BB->succ_end());
  }
  return false;
}

/// What lanes does this operand read?
LaneBitmask
MachineLaneSSAUpdater::operandLaneMask(const MachineOperand &MO) const {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  if (unsigned Sub = MO.getSubReg())
    return TRI.getSubRegIndexLaneMask(Sub);
  return MRI.getMaxLaneMaskForVReg(MO.getReg());
}

/// Helper: Decompose a potentially non-contiguous lane mask into a vector of
/// subregister indices that together cover all lanes in the mask.
///
/// Key algorithm: Sort candidates by lane count (prefer larger subregs) to get
/// minimal covering set with largest possible subregisters.
///
/// Example: For vreg_128 with LaneMask = 0x0F | 0xF0 (sub0 + sub2, skipping
/// sub1)
///          Returns: [sub0_idx, sub2_idx] (not lo16, hi16, sub2, sub3)
static SmallVector<unsigned, 4>
getCoveringSubRegsForLaneMask(LaneBitmask Mask, const TargetRegisterInfo *TRI,
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
/// Inserts at the PHI predecessor terminator (for PHI uses) or before the
/// instruction or bundle containing the use. Returns the new full-width vreg,
/// the RS index via OutIdx, and the lane masks to extend to that point.
Register MachineLaneSSAUpdater::buildRSForSuperUse(
    MachineInstr *UseMI, const MachineOperand &MO, Register OldVR,
    Register NewVR, LaneBitmask MaskToRewrite, const TargetRegisterClass *UseRC,
    SlotIndex &OutIdx, SmallVectorImpl<LaneBitmask> &LanesToExtend) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineBasicBlock *InsertBB = UseMI->getParent();
  MachineBasicBlock::iterator IP =
      MachineBasicBlock::iterator::getAtBundleBegin(UseMI->getIterator());
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

  // Lanes the use needs. UseRC is already sized to exactly these lanes, so the
  // caller reads Dest whole.
  LaneBitmask UseMask = operandLaneMask(MO);

  // REG_SEQUENCE destination indices are in Dest's (narrower) namespace, while
  // the lane masks below are in OrigVReg's; re-base to convert.
  auto DestSub = [&](LaneBitmask Lanes) {
    return getSubRegIndexForLaneMask(rebaseLaneMask(Lanes, UseMask), &TRI);
  };

  Register Dest = MRI.createVirtualRegister(UseRC);
  recordAffected(Dest);
  auto RS = BuildMI(*InsertBB, IP,
                    (IP != InsertBB->end() ? IP->getDebugLoc() : DebugLoc()),
                    TII.get(TargetOpcode::REG_SEQUENCE), Dest);

  // Decompose into lanes from NewVR (updated) and lanes from OldVR (unchanged)
  LaneBitmask LanesFromNew = UseMask & MaskToRewrite;
  LaneBitmask LanesFromOld = UseMask & ~MaskToRewrite;

  LLVM_DEBUG(dbgs() << "        Building REG_SEQUENCE: UseMask="
                    << PrintLaneMask(UseMask)
                    << " LanesFromNew=" << PrintLaneMask(LanesFromNew)
                    << " LanesFromOld=" << PrintLaneMask(LanesFromOld) << "\n");

  // A well-formed REG_SEQUENCE defines each of Dest's lanes exactly once, so
  // track what the sources added so far cover. Lanes is in OrigVReg's namespace;
  // DestSub() re-bases it to the destination slice actually written.
  const LaneBitmask DestFull = MRI.getMaxLaneMaskForVReg(Dest);
  LaneBitmask DestCovered = LaneBitmask::getNone();
  auto AddSrc = [&](Register SrcReg, RegState Flags, unsigned SrcSub,
                    LaneBitmask Lanes) {
    unsigned DestIdx = DestSub(Lanes);
    // A zero index writes the whole destination (no slice narrower than Dest).
    LaneBitmask Slice = DestIdx ? TRI.getSubRegIndexLaneMask(DestIdx) : DestFull;
    assert((DestCovered & Slice).none() &&
           "REG_SEQUENCE defines the same destination lanes twice");
    DestCovered |= Slice;
    RS.addReg(SrcReg, Flags, SrcSub).addImm(DestIdx);
  };

  // Add source for lanes from NewVR (updated lanes)
  if (getSubRegIndexForLaneMask(LanesFromNew, &TRI)) {
    AddSrc(NewVR, {}, 0, LanesFromNew); // NewVR whole
    LanesToExtend.push_back(LanesFromNew);
  } else if (LanesFromNew.any()) {
    // The group has no subregister index of its own -- e.g. sub1..sub15 of a
    // vreg_512, formed when sub0 is established by a separate partial def. It
    // can be neither read as one slice of NewVR nor written to one slice of
    // Dest, so split it into covering subregisters, as the old lanes are below.
    // A group without an index also means NewVR kept OrigVReg's register class,
    // so each covering index addresses NewVR directly, with no rebasing.
    for (unsigned CoverSubIdx : getCoveringSubRegsForLaneMask(
             LanesFromNew, &TRI, MRI.getRegClass(NewVR))) {
      LaneBitmask CovLanes = TRI.getSubRegIndexLaneMask(CoverSubIdx);
      AddSrc(NewVR, {}, CoverSubIdx, CovLanes);
      LanesToExtend.push_back(CovLanes);
    }
  }

  // Source the old (unchanged) lanes per their reaching VNI at the use point
  // (getVNInfoBefore via collectReachingVNIs) -- the value the original whole-
  // register use actually read there:
  //   * null  -> the lane is dead/undef at this use (e.g. a poison buffer
  //              descriptor base read whole by a later store) -> undef source.
  //              A plain read would be "Reading virtual register without a def".
  //   * a def renamed this session -> its renamed vreg. Resolve now so the
  //     result is independent of session/patch order (a placeholder would never
  //     be patched if that lane's rewrite already ran).
  //   * a live, not-yet/never-renamed def -> an OrigVReg placeholder, patched by
  //     that def's own rewrite or kept as the Root/final value.
  // Per-use-point liveness (NOT the global "has a subrange" test) is required:
  // OrigVReg may own a subrange for a lane that is nonetheless dead here.
  auto AddOldGroup = [&](LaneBitmask PieceLanes, const VNInfo *V) {
    Register SrcReg = OldVR;
    LaneBitmask SrcBase = LaneBitmask::getNone();
    bool Undef = false, Live = false;
    if (!V)
      Undef = true;
    else if (const RenamedDef *RD = renamedForReachingVNI(V)) {
      SrcReg = RD->VReg;
      SrcBase = RD->OrigLanes;
    } else
      Live = true;

    auto EmitCover = [&](LaneBitmask CovLanes) {
      RegState Flags = Undef ? RegState::Undef : RegState{};
      unsigned Sub;
      if (SrcReg == OldVR)
        Sub = getSubRegIndexForLaneMask(CovLanes, &TRI);
      else {
        LaneBitmask RLane = rebaseLaneMask(CovLanes, SrcBase);
        LaneBitmask RFull = MRI.getMaxLaneMaskForVReg(SrcReg);
        Sub = (RLane == RFull) ? 0 : getSubRegIndexForLaneMask(RLane, &TRI);
      }
      AddSrc(SrcReg, Flags, Sub, CovLanes);
      if (Live)
        LanesToExtend.push_back(CovLanes);
    };

    // Emit covering subregisters so a group without a single class-supported
    // index -- non-contiguous (e.g. sub0+sub1+sub3) or an unaligned slice the
    // class has no index for (e.g. sub10_..._sub15 of an sgpr_512) -- is split
    // into valid, class-supported subregister sources.
    unsigned DirectIdx = getSubRegIndexForLaneMask(PieceLanes, &TRI);
    if (DirectIdx &&
        TRI.getSubClassWithSubReg(MRI.getRegClass(OldVR), DirectIdx))
      EmitCover(PieceLanes);
    else
      for (unsigned CoverSubIdx : getCoveringSubRegsForLaneMask(
               PieceLanes, &TRI, MRI.getRegClass(OldVR)))
        EmitCover(TRI.getSubRegIndexLaneMask(CoverSubIdx));
  };

  // Restrict the reaching query to OrigVReg lanes that have a REAL establishing
  // definition. When a value is placed in an oversized register class (e.g. a
  // 320-bit value held in an sgpr_512), LiveIntervalCalc fabricates a subrange
  // over the never-defined padding lanes whose only values are undef (printed
  // @x, i.e. VNInfo::isUnused) inputs joined into a PHI-def. Querying such a
  // subrange returns that PHI-def VNInfo, which has no renamed/real def to
  // resolve to, so the padding lanes would be emitted as a live OrigVReg
  // placeholder that is never patched -> dangling use + illegal subregister
  // index. Excluding them routes the padding lanes to the undef NoSub piece.
  LaneBitmask RealDef = LaneBitmask::getNone();
  {
    auto HasRealDef = [](const LiveRange &LR) {
      for (const VNInfo *V : LR.valnos)
        if (V && !V->isUnused() && !V->isPHIDef())
          return true;
      return false;
    };
    const LiveInterval *Frozen = frozenLI();
    if (Frozen->hasSubRanges()) {
      for (const LiveInterval::SubRange &S : Frozen->subranges())
        if (HasRealDef(S))
          RealDef |= S.LaneMask;
    } else if (HasRealDef(*Frozen))
      RealDef = MRI.getMaxLaneMaskForVReg(OldVR);
  }

  SmallVector<std::pair<LaneBitmask, const VNInfo *>, 4> OldPieces;
  collectReachingVNIs(*frozenLI(), LanesFromOld & RealDef, QueryIdx, OldPieces);
  // collectReachingVNIs only covers lanes that have a subrange; any remaining
  // old lanes have no establishing def and are undef here.
  LaneBitmask Covered = LaneBitmask::getNone();
  for (auto &[L, V] : OldPieces)
    Covered |= L;
  if (LaneBitmask NoSub = LanesFromOld & ~Covered; NoSub.any())
    OldPieces.push_back({NoSub, nullptr});

  for (auto &[PieceLanes, V] : OldPieces) {
    if (PieceLanes.none())
      continue;

    // Without subranges, the frozen main range associates every lane with one
    // VNInfo. If that value came from an `undef` partial definition, only the
    // definition's explicit lanes exist; its unwritten lanes are undef. Split
    // that otherwise-indivisible main-range piece before selecting RS sources.
    if (const RenamedDef *RD = renamedForReachingVNI(V);
        RD && RD->UndefUnwrittenLanes) {
      LaneBitmask Defined = PieceLanes & RD->OrigLanes;
      LaneBitmask Undef = PieceLanes & ~RD->OrigLanes;
      if (Defined.any())
        AddOldGroup(Defined, V);
      if (Undef.any())
        AddOldGroup(Undef, nullptr);
      continue;
    }
    AddOldGroup(PieceLanes, V);
  }

  assert(DestCovered == DestFull &&
         "REG_SEQUENCE must define every destination lane exactly once");

  LIS.InsertMachineInstrInMaps(*RS);
  OutIdx = LIS.getInstructionIndex(*RS);

  // Create live interval for the REG_SEQUENCE result
  LIS.createAndComputeVirtRegInterval(Dest);

  // Extend live intervals of all source registers to cover this REG_SEQUENCE
  // Use the register slot to ensure the live range covers the use
  SlotIndex UseSlot = OutIdx.getRegSlot();
  for (const MachineOperand &MO : RS.getInstr()->uses()) {
    if (MO.isReg() && MO.getReg().isVirtual()) {
      Register SrcReg = MO.getReg();
      recordAffected(SrcReg);
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
  recordAffected(LI.reg());
  SmallVector<SlotIndex, 1> P{Idx};
  LIS.extendToIndices(LI, P);
  for (auto &SR : LI.subranges())
    for (LaneBitmask L : Lanes)
      if (SR.LaneMask == L)
        LIS.extendToIndices(SR, P);
}

void MachineLaneSSAUpdater::updateDeadFlags(Register Reg) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const LiveInterval &LI = LIS.getInterval(Reg);
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