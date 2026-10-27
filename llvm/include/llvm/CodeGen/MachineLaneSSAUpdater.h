//===- MachineLaneSSAUpdater.h - SSA repair for Machine IR (lane-aware) -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// === MachineLaneSSAUpdater Design Notes ===
//

#ifndef LLVM_CODEGEN_MACHINELANESSAUPDATER_H
#define LLVM_CODEGEN_MACHINELANESSAUPDATER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"      // SmallVector
#include "llvm/ADT/bit.h"              // countr_zero
#include "llvm/CodeGen/LiveInterval.h" // LiveRange
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Register.h"           // Register
#include "llvm/CodeGen/SlotIndexes.h"        // SlotIndex
#include "llvm/CodeGen/TargetRegisterInfo.h" // For inline function
#include "llvm/MC/LaneBitmask.h"             // LaneBitmask
#include "llvm/Support/Allocator.h" // BumpPtrAllocator (VNInfo::Allocator)
#include <memory>                   // std::unique_ptr

namespace llvm {

// Forward declarations to avoid heavy includes in the header.
class MachineFunction;
class MachineBasicBlock;
class MachineInstr;
class LiveIntervals;
class LiveRange;
class MachineDominatorTree;

//===----------------------------------------------------------------------===//
// Lane-aware SSA repair after inserting definitions of an existing virtual
// register. repairSSAForNewDefs repairs a complete batch and reports affected
// values after finalizing their liveness. repairSSAForNewDef provides the
// incremental operation with caller-managed session and liveness finalization.
// Partial definitions are renamed to registers of the corresponding subregister
// class. Reaching values come from a frozen LiveInterval snapshot.
//===----------------------------------------------------------------------===//

/// A frozen deep copy of a vreg's LiveInterval plus the allocator backing its
/// subrange VNInfos. The two are a lifetime pair: the subranges live in Alloc,
/// so LI must be destroyed before Alloc — the member order below guarantees it
/// (members destroy in reverse declaration order). Produced by
/// MachineLaneSSAUpdater::freezeInterval() and passed back to
/// repairSSAForNewDef() so several incremental redefs of the same vreg share ONE
/// snapshot without re-freezing. Move-only (it owns the allocator).
struct FrozenInterval {
  BumpPtrAllocator Alloc;
  std::unique_ptr<LiveInterval> LI; // declared after Alloc -> destroyed first
  FrozenInterval() = default;
  FrozenInterval(FrozenInterval &&) = default;
  FrozenInterval &operator=(FrozenInterval &&) = default;
  FrozenInterval(const FrozenInterval &) = delete;
  FrozenInterval &operator=(const FrozenInterval &) = delete;
  bool valid() const { return LI != nullptr; }
};

class MachineLaneSSAUpdater {
public:
  MachineLaneSSAUpdater(MachineFunction &MF, LiveIntervals &LIS,
                        const MachineDominatorTree &MDT,
                        const TargetRegisterInfo &TRI)
      : MF(MF), LIS(LIS), MDT(MDT), TRI(TRI) {}

  struct RepairResult {
    /// Sorted, unique IDs of values whose definitions, uses or liveness were
    /// touched, including new PHI/reconstruction results and retired values.
    /// An affected value need not still have an interval at completion.
    SmallVector<Register, 8> Affected;
  };

  using LivenessCallback = function_ref<void(ArrayRef<Register>)>;

  /// Repair a complete batch of already-inserted redefinitions of OrigVReg.
  /// Instructions must be indexed and OrigVReg's current interval must include
  /// all new definitions. NewDefs contains no duplicates and
  /// leaves at most one original definition after renaming. A one-copy split
  /// supplies the instruction "OrigVReg = COPY OrigVReg" as its single entry.
  /// An empty batch finalizes OrigVReg after edits needing no new definition.
  ///
  /// Uses the existing reaching-def repair, then finalizes coupled operands
  /// and recomputes only affected intervals from the completed SSA form.
  /// OnChange is called synchronously once, after this finalization and before
  /// return. The same affected IDs are returned to the caller for allocation.
  ///
  /// OnChange may reconcile external ownership using its stored before-state
  /// and the repaired LIS. It must not mutate MIR/LIS or reenter this updater.
  /// No allocation query is permitted between individual repairs in the batch.
  /// The callback is not retained. No physical home is chosen by the updater;
  /// failure to place a repaired value does not undo the completed mutation.
  RepairResult repairSSAForNewDefs(Register OrigVReg,
                                  ArrayRef<MachineInstr *> NewDefs,
                                  LivenessCallback OnChange);

  // Repair SSA for a new definition that violates SSA form.
  //
  // Before taking the reaching snapshot, index all new definitions and
  // recompute OrigVReg's interval to include them. This applies both to an
  // explicit Oracle and to the internal snapshot taken on the first call.
  // Adding definitions to an existing session requires refreshing the
  // interval and resetting the session before taking a new snapshot.
  // Incremental repair leaves session and liveness finalization to the caller;
  // use repairSSAForNewDefs for a complete batch with finalized intervals.
  //
  // Parameters:
  //   NewDefMI: Instruction with a def operand that currently defines OrigVReg
  //   (violating SSA)
  //   OrigVReg: The virtual register being redefined
  //   PHIRegDefOps: A vector of def operands for the PHI registers that were
  //   created
  // This function will:
  //   1. Find the def operand in NewDefMI that defines OrigVReg
  //   2. Derive the lane mask from the operand's subreg index (if any)
  //   3. Create a new virtual register, using the subregister class for a
  //      partial definition and the original class for a full definition
  //   4. Retarget the definition to the new register; partial definitions
  //      become whole-register definitions with subregister and undef cleared
  //   5. Perform SSA repair (insert PHIs, rewrite uses)
  //
  // Returns: The newly created SSA-repaired virtual register
  //
  //   Oracle: optional caller-owned frozen reaching snapshot (see
  //   freezeInterval). When non-null it is used as the reaching oracle instead
  //   of the internal self-freeze — the caller guarantees it was taken AFTER
  //   all redefs of OrigVReg in this session were emitted. When null (default)
  //   the updater self-freezes on the first call per OrigVReg. The snapshot is
  //   read-only and must outlive the repair session.
  Register repairSSAForNewDef(MachineInstr &NewDefMI, Register OrigVReg,
                              SmallVectorImpl<MachineOperand *> &PHIRegDefOps,
                              const FrozenInterval *Oracle = nullptr);

  /// Take a frozen deep copy of \p OrigVReg's current LiveInterval (value numbers
  /// and subranges preserved) with its own backing allocator. The caller owns the
  /// result and passes it to repairSSAForNewDef() as the reaching oracle.
  /// Index all inserted definitions and refresh OrigVReg's interval before
  /// freezing; this method copies the interval without recomputing it. This is
  /// the same content the updater would self-freeze — factored out so a caller
  /// emitting several redefs can freeze ONCE, after placement, and skip
  /// resetSession() churn. Read-only w.r.t. the live interval (deep copy).
  FrozenInterval freezeInterval(Register OrigVReg) const;

  /// Invalidate the per-OrigVReg repair session so the next repairSSAForNewDef
  /// re-freezes OrigVReg's interval. Required when new defs (e.g. spiller
  /// reloads) are added incrementally between repair calls for the same
  /// OrigVReg; otherwise the cached FrozenOrigLI is stale and reaching
  /// resolution misses the new def.
  void resetSession() { RenameSessionOrig = Register(); }

  /// Check whether \p UseMI (a use of \p OrigVReg) is reachable from \p DefMI
  /// in the CFG -- i.e. whether the use lies downstream of the def (for a PHI
  /// use, the target is the incoming predecessor block). Used by the spiller to
  /// decide which uses a spill at \p DefMI affects. Since OrigVReg is SSA the
  /// value is live along every def->use path, so plain CFG reachability is
  /// exact -- no dominance-frontier needed.
  bool isUseReachableFromDef(const MachineInstr *DefMI,
                             const MachineInstr *UseMI,
                             Register OrigVReg) const;

  /// Rewrite lanes of OrigVReg whose frozen reaching value is defined by
  /// NewSSA. Reconstruct mixed-lane uses with REG_SEQUENCE when needed.
  void rewriteDominatedUses(Register OrigVReg, Register NewSSA,
                            LaneBitmask MaskToRewrite);

  /// After all repairSSAForNewDef calls for \p OrigVReg, restore intra-
  /// instruction register coupling. SSA repair resolves each operand by its own
  /// reaching value, so two operands of one non-PHI instruction that read
  /// OrigVReg (one a lane-subset of the other) can land on different base regs.
  /// Targets like AMDGPU V_MOVRELS require the narrow src to stay a subregister
  /// of the wide (implicit-vector) use. Re-point each subset operand onto the
  /// full-read sibling's resolved reg, sub-indexed to its lanes. No-op unless a
  /// use instruction read OrigVReg in >=2 operands.
  void finalizeCoupledUses(Register OrigVReg);

  /// After all repairSSAForNewDef calls for \p OrigVReg, rebuild its main range
  /// as the union of its subranges. Repair keeps each subrange accurate for the
  /// lanes it migrated, but the main range spans lanes that now live in several
  /// vregs and cannot be maintained incrementally, so it is reconstructed once
  /// the driver has renamed every def it intends to. Leaving it stale makes the
  /// value claim registers it no longer occupies: readers that consult the main
  /// range (LiveInterval::overlaps, point-liveness queries) then disagree with
  /// readers that consult the subranges. No-op when OrigVReg has no subranges -
  /// that path recomputes the whole interval during repair.
  void rebuildMainRangeFromSubranges(Register OrigVReg);

  /// Reaching-VNI path: rewrite a single use operand \p MO. Owns exactly the
  /// \p OpMask lanes whose reaching value at the use is this def (\p DefMI /
  /// \p NewSSA covering \p MaskToRewrite). Owned==OpMask -> direct NewSSA
  /// subreg; partial -> REG_SEQUENCE (owned from NewSSA, rest OrigVReg
  /// placeholder); none -> no change. DefMI is read-only; MO is rewritten and
  /// OrigLI is extended when a reconstruction is inserted before UseMI (or on
  /// its incoming PHI edge).
  void rewriteUseReaching(Register OrigVReg, Register NewSSA,
                          LaneBitmask MaskToRewrite, const MachineInstr *DefMI,
                          MachineInstr *UseMI, MachineOperand &MO,
                          LaneBitmask OpMask, LiveInterval &OrigLI);

private:
  // Non-null only for the synchronous batch entry point. Legacy incremental
  // callers retain their existing repair/finalization contract.
  SmallSetVector<Register, 8> *AffectedRegs = nullptr;
  void recordAffected(Register VR) {
    if (AffectedRegs && VR.isVirtual())
      AffectedRegs->insert(VR);
  }

  // Common SSA repair logic
  // Returns a vector of MachineOperand pointers to the PHI result registers
  SmallVector<MachineOperand *> performSSARepair(Register NewVReg,
                                                 Register OrigVReg,
                                                 LaneBitmask DefMask,
                                                 SlotIndex MigratedDef);

  // --- Internal helpers ---

  // Index MI in SlotIndexes / LIS maps immediately after insertion.
  // Returns the SlotIndex assigned to the instruction.
  SlotIndex indexNewInstr(MachineInstr &MI);

  // Insert lane-aware Machine PHIs at the join points recorded as PHI-def
  // VNInfos in OrigVReg's frozen interval (per lane, the pruned IDF that
  // LiveIntervalCalc already computed). Returns each PHI result operand paired
  // with the OrigVReg lane it covers, so the caller can rewrite that PHI's uses
  // with the correct per-PHI lane (one PHI per subrange).
  SmallVector<std::pair<MachineOperand *, LaneBitmask>>
  insertLaneAwarePHI(Register OrigVReg, LaneBitmask DefMask);

  // Create a PHI for a single lane group
  // (subrange-aligned \p Lane) at \p JoinMBB, resolving each predecessor
  // operand from the frozen OrigVReg reaching oracle (single source or OrigVReg
  // placeholder). Dedups via LanePHIs. Returns the PHI result operand.
  MachineOperand *createPHIInBlockReaching(MachineBasicBlock &JoinMBB,
                                           Register OrigVReg, LaneBitmask Lane);

  // Per-OrigVReg repair session key: reset when repairSSAForNewDef is first
  // called for a different OrigVReg (the driver processes all defs of an
  // OrigVReg before moving on).
  Register RenameSessionOrig;

  // A renamed def: the fresh SSA vreg plus the OrigVReg-namespace lanes it
  // covers. OrigLanes lets us rebase a target lane into VReg's own namespace to
  // extract the right subreg when VReg is wider than the queried lane (e.g. a
  // full def feeding a sub0 PHI operand).
  struct RenamedDef {
    Register VReg;
    LaneBitmask OrigLanes;
    // The original subregister def did not read the complementary lanes.
    bool UndefUnwrittenLanes = false;
  };

  // Per-OrigVReg repair session: maps each renamed real (non-PHI) def
  // instruction to its RenamedDef. PHI-operand and use resolution map a
  // reaching VNInfo to its def instruction and look it up here. Only real defs
  // are stored; PHI-def reaching VNIs (block-boundary merges) are resolved via
  // placeholder-then-patch, so a MachineInstr* key suffices and is cheaper than
  // SlotIndex (no DenseMapInfo<SlotIndex>, O(1), no node allocs). Reset at the
  // start of each session.
  DenseMap<const MachineInstr *, RenamedDef> DefInstrToRenamed;

  // Frozen deep copy of OrigVReg's LiveInterval, taken at session start before
  // any rename: the STABLE reaching-def oracle. Renames done during the session
  // strip renamed defs' VNInfos from the live LIS interval, so reaching queries
  // must run against this frozen copy instead. Rebuilt each session; VNInfos
  // are owned by FrozenAlloc (def SlotIndexes and isPHIDef are preserved by the
  // copy, which is all the lookup needs).
  BumpPtrAllocator FrozenAlloc;
  std::unique_ptr<LiveInterval> FrozenOrigLI;

  // When a caller passes an external FrozenInterval oracle to repairSSAForNewDef,
  // this points at its LiveInterval and the self-frozen FrozenOrigLI is left
  // empty. Reaching queries read frozenLI() which returns this when set. Null =>
  // self-freeze path (FrozenOrigLI). Reset per session.
  const LiveInterval *ExternalFrozen = nullptr;

  // The reaching-def oracle in effect this session: the caller-provided external
  // snapshot when set, else the self-frozen copy. All reaching queries go here.
  const LiveInterval *frozenLI() const {
    return ExternalFrozen ? ExternalFrozen : FrozenOrigLI.get();
  }

  // Dedup for the reaching-VNI path's per-subrange PHIs (approach A): reuse the
  // PHI already built for a (join block, lane group) instead of creating a
  // duplicate when several defs share an IDF block. Reset per session.
  DenseMap<std::pair<MachineBasicBlock *, LaneBitmask>, Register> LanePHIs;

  // Within one OrigVReg session, share a super-use REG_SEQUENCE across multiple
  // operands of the SAME non-PHI instruction that read the same OrigVReg lanes.
  // Without this, an instruction reading a value in two operands (e.g.
  // V_MAX_F64 x, x or V_DIV_SCALE_F64 %v, %v) would get a distinct composed
  // vreg per operand, presenting two different registers to a constant-bus- or
  // src0==src1-constrained instruction. External bundle-member uses share
  // the header's key so its aggregate operands name the same reconstruction.
  // Keyed by (use instruction or bundle header, operand lane mask); reset per
  // session.
  DenseMap<std::pair<MachineInstr *, LaneBitmask>, Register> SuperUseRSCache;

  // Captured at session start (before any rewrite): per non-PHI instruction
  // that reads OrigVReg in >=2 operands, the (operand index, original
  // OrigVReg-namespace lane mask) of each such operand. Consumed by
  // finalizeCoupledUses to keep a narrow operand a subreg of its full-read
  // sibling. Reset per session, like the caches above.
  DenseMap<MachineInstr *, SmallVector<std::pair<unsigned, LaneBitmask>, 4>>
      CoupledUseGroups;

  // Reaching-def oracle (sound successor to dominance+order for lane
  // decisions). collectReachingVNIs: decompose \p Mask by OrigVReg's (old,
  // still-valid) LiveInterval subranges and append, for each covered piece,
  // {laneSubmask, reaching VNInfo at \p Idx}. The pieces partition \p Mask, so
  // a caller can use a single value when one piece covers all of Mask, or build
  // a REG_SEQUENCE across pieces otherwise. A null VNInfo means that piece's
  // lanes are dead at \p Idx. For a reg without subranges, returns one {Mask,
  // mainVNI}.
  static void collectReachingVNIs(
      const LiveInterval &OrigLI, LaneBitmask Mask, SlotIndex Idx,
      SmallVectorImpl<std::pair<LaneBitmask, const VNInfo *>> &Out);
  // Convenience wrapper for a single subrange-aligned lane group: returns the
  // one reaching VNInfo at \p Idx (null if the lane is dead there). Asserts if
  // \p Lane straddles subranges (which would yield more than one piece).
  static const VNInfo *reachingVNIForLaneGroup(const LiveInterval &OrigLI,
                                               LaneBitmask Lane, SlotIndex Idx);
  // renamedForReachingVNI: map a reaching VNInfo to the RenamedDef its def was
  // renamed to this session, or null if it is a PHI-def merge or has not been
  // renamed (caller then leaves an OrigVReg placeholder to be patched later).
  const RenamedDef *renamedForReachingVNI(const VNInfo *V) const;

  // Internal helper methods for use rewriting
  LaneBitmask operandLaneMask(const MachineOperand &MO) const;
  // Read MO to build a reconstruction before UseMI (or on its PHI edge).
  // Mutates MIR/LIS and appends source lanes/sets the index in the outputs.
  Register buildRSForSuperUse(MachineInstr *UseMI, const MachineOperand &MO,
                              Register OldVR, Register NewVR,
                              LaneBitmask MaskToRewrite,
                              const TargetRegisterClass *UseRC,
                              SlotIndex &OutIdx,
                              SmallVectorImpl<LaneBitmask> &LanesToExtend);
  void extendAt(LiveInterval &LI, SlotIndex Idx, ArrayRef<LaneBitmask> Lanes);
  void updateDeadFlags(Register Reg);

  // --- Data members ---
  MachineFunction &MF;
  LiveIntervals &LIS;
  const MachineDominatorTree &MDT;
  const TargetRegisterInfo &TRI;
};

/// Get the subregister index that corresponds to the given lane mask.
/// \param Mask The lane mask to convert to a subregister index
/// \param TRI The target register info (provides target-specific subregister
/// mapping) \return The subregister index, or 0 if no single subregister
/// matches
inline unsigned getSubRegIndexForLaneMask(LaneBitmask Mask,
                                          const TargetRegisterInfo *TRI) {
  if (Mask.none())
    return 0; // No subregister

  // Iterate through all subregister indices to find a match
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    if (SubMask == Mask) {
      return SubIdx;
    }
  }

  // No exact match found - this might be a composite mask requiring
  // REG_SEQUENCE
  return 0;
}

/// Re-express a lane mask from a super-register's namespace into the namespace
/// of a sub-register. \p Base is the sub-register's lane mask (in the super's
/// namespace); the result shifts \p Lanes down so Base's first lane sits at
/// bit 0. Identity when Base already starts at lane 0.
inline LaneBitmask rebaseLaneMask(LaneBitmask Lanes, LaneBitmask Base) {
  return LaneBitmask(Lanes.getAsInteger() >>
                     llvm::countr_zero(Base.getAsInteger()));
}

// DenseMapInfo specialization for LaneBitmask
template <> struct DenseMapInfo<LaneBitmask> {
  static inline LaneBitmask getEmptyKey() {
    // Use a specific bit pattern for empty key
    return LaneBitmask(~0U - 1);
  }

  static inline LaneBitmask getTombstoneKey() {
    // Use a different bit pattern for tombstone
    return LaneBitmask(~0U);
  }

  static unsigned getHashValue(const LaneBitmask &Val) {
    return (unsigned)Val.getAsInteger();
  }

  static bool isEqual(const LaneBitmask &LHS, const LaneBitmask &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINELANESSAUPDATER_H