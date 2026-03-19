//=====-- Rematerializer.h - MIR rematerialization support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// MIR-level target-independent rematerialization helpers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REMATERIALIZER_H
#define LLVM_CODEGEN_REMATERIALIZER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <iterator>

namespace llvm {

/// MIR-level target-independent rematerializer. Provides an API to identify and
/// rematerialize registers within a machine function.
///
/// At the moment this supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual and has a single defining instruction.
/// 2. The single defining instruction is deemed rematerializable by the TII and
///    doesn't have any physical register use that is both non-constant and
///    non-ignorable.
/// 3. The register has at least one non-debug use that is inside or at a region
///    boundary (see below for what we consider to be a region).
///
/// Rematerializable registers (represented by \ref Rematerializer::Reg) form a
/// DAG of their own, with every register having incoming edges from all
/// rematerializable registers which are read by the instruction defining it. It
/// is possible to rematerialize registers with unrematerializable dependencies;
/// however the latter are not considered part of this DAG since their
/// position/identity never change and therefore do not require the same level
/// of tracking.
///
/// Each register has a "dependency DAG" which is defined as the subset of nodes
/// in the overall DAG that have at least one path to the register, which is
/// called the "root" register in this context. Semantically, these nodes are
/// the registers which are involved into the computation of the root register
/// i.e., all of its transitive dependencies. We use the term "root" because all
/// paths within the dependency DAG of a register terminate at it; however,
/// there may be multiple paths between a non-root node and the root node, so a
/// dependency DAG is not always a tree.
///
/// The API uses dense unsigned integers starting at 0 to reference
/// rematerializable registers. These indices are immutable i.e., even when
/// registers are deleted their respective integer handle remain valid. Method
/// which perform actual rematerializations should however be assumed to
/// invalidate addresses to \ref Rematerializer::Reg objects.
///
/// The rematerializer tracks def/use points of registers based on regions.
/// These are alike the regions the machine scheduler works on. A region is
/// simply a pair on MBB iterators encoding a range of machine instructions. The
/// first iterator (beginning of the region) is inclusive whereas the second
/// iterator (end of the region) is exclusive and can either point to a MBB's
/// end sentinel or an actual MI (not necessarily a terminator). Regions must be
/// non-empty, cannot overlap, and cannot contain terminators. However, they do
/// not have to cover the whole function.
///
/// The API uses dense unsigned integers starting at 0 to reference regions.
/// These map directly to the indices of the corresponding regions in the region
/// vector passed during construction.
///
/// The rematerializer supports rematerializing arbitrary complex DAGs of
/// registers to regions where these registers are used, with the option of
/// re-using non-root registers or their previous rematerializations instead of
/// rematerializing them again. It also optionally supports rolling back
/// previous rematerializations (set during analysis phase, see \ref
/// Rematerializer::analyze) to restore the MIR state to what it was
/// pre-rematerialization. When enabled, machine instructions defining
/// rematerializable registers that no longer have any uses following previous
/// rematerializations will not be deleted from the MIR; their opcode will
/// instead be set to a DEBUG_VALUE and their read register operands set to the
/// null register. This maintains their position in the MIR and keeps the
/// original register alive for potential rollback while allowing other
/// passes/analyzes (e.g., machine scheduler, live-interval analysis) to ignore
/// them. \ref Rematerializer::commitRematerializations actually deletes those
/// instructions when rollback is deemed unnecessary.
///
/// Throughout its lifetime, the rematerializer tracks new registers it creates
/// (which are rematerializable by construction) and their relations to other
/// registers. It performs DAG updates immediately on rematerialization but
/// defers/batches all necessary live interval updates to reduce the number of
/// expensive LIS queries when successively rematerializing many registers. \ref
/// Rematerializer::updateLiveIntervals performs all currently batched live
/// interval updates.
///
/// In its nomenclature, the rematerializer differentiates between "original
/// registers" (registers that were present when it analyzed the function) and
/// rematerializations of these original registers. Rematerializations have an
/// "origin" which is the index of the original regiser they were rematerialized
/// from (transitivity applies; a rematerialization and all of its own
/// rematerializations have the same origin). Semantically, only original
/// registers have rematerializations.
class Rematerializer {
public:
  /// Index type for rematerializable registers.
  using RegisterIdx = unsigned;

  /// A rematerializable register defined by a single machine instruction.
  ///
  /// A rematerializable register has a set of dependencies, which correspond
  /// to the unique read register operands of its defining instruction.
  /// They are identified by their machine operand index, and can themselves be
  /// rematerializable. Operand indices corresponding to unrematerializable
  /// dependencies are managed by and queried from the rematerializer.
  ///
  /// A rematerializable register also has an arbitrary number of users in an
  /// arbitrary number of regions, potentially including its own defining
  /// region. When rematerializations lead to operand changes in users, a
  /// register may find itself without any user left, at which point the
  /// rematerializer marks it for deletion. Its defining instruction either
  /// becomes nullptr (without rollback support) or its opcode is set to
  /// TargetOpcode::DBG_VALUE (with rollback support) until \ref
  /// Rematerializer::commitRematerializations is called.
  struct Reg {
    /// Single MI defining the rematerializable register.
    MachineInstr *DefMI;
    /// Defining region of \p DefMI.
    unsigned DefRegion;
    /// The rematerializable register's lane bitmask.
    LaneBitmask Mask;

    using RegionUsers = SmallDenseSet<MachineInstr *, 4>;
    /// Uses of the register, mapped by region.
    SmallDenseMap<unsigned, RegionUsers, 2> Uses;

    /// A read register operand of \p DefMI that is rematerializable (according
    /// to the rematerializer).
    struct Dependency {
      /// The register's machine operand index in \p DefMI.
      unsigned MOIdx;
      /// The corresponding register's index in the rematerializer.
      RegisterIdx RegIdx;

      Dependency(unsigned MOIdx, RegisterIdx RegIdx)
          : MOIdx(MOIdx), RegIdx(RegIdx) {}
    };
    /// This register's rematerializable dependencies, one per unique
    /// rematerializable register operand.
    SmallVector<Dependency, 2> Dependencies;

    /// Returns the rematerializable register from its defining instruction.
    inline Register getDefReg() const {
      assert(DefMI && "defining instruction was deleted");
      assert(DefMI->getOperand(0).isDef() && "not a register def");
      return DefMI->getOperand(0).getReg();
    }

    bool hasUsersInDefRegion() const {
      return !Uses.empty() && Uses.contains(DefRegion);
    }

    bool hasUsersOutsideDefRegion() const {
      if (Uses.empty())
        return false;
      return Uses.size() > 1 || Uses.begin()->first != DefRegion;
    }

    /// Returns the first and last user of the register in region \p UseRegion.
    /// If the register has no user in the region, returns a pair of nullptr's.
    std::pair<MachineInstr *, MachineInstr *>
    getRegionUseBounds(unsigned UseRegion, const LiveIntervals &LIS) const;

    bool isAlive() const {
      return DefMI && DefMI->getOpcode() != TargetOpcode::DBG_VALUE;
    }

  private:
    void addUser(MachineInstr *MI, unsigned Region);
    void addUsers(const RegionUsers &NewUsers, unsigned Region);
    void eraseUser(MachineInstr *MI, unsigned Region);

    friend Rematerializer;
  };

  /// Error value for register indices.
  static constexpr unsigned NoReg = ~0;

  /// A region's boundaries i.e. a pair of instruction bundle iterators. The
  /// lower boundary is inclusive, the upper boundary is exclusive.
  using RegionBoundaries =
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

  /// Simply initializes some internal state, does not identify
  /// rematerialization candidates.
  Rematerializer(MachineFunction &MF,
                 SmallVectorImpl<RegionBoundaries> &Regions,
                 LiveIntervals &LIS);

  /// Goes through the whole MF and identifies all rematerializable registers.
  /// When \p SupportRollback is set, rematerializations of original registers
  /// can be rolled back and original registers are maintained in the IR even
  /// when they longer have any users. Returns whether there is any
  /// rematerializable register in regions.
  bool analyze(bool SupportRollback);

  inline const Reg &getReg(RegisterIdx RegIdx) const {
    assert(RegIdx < Regs.size() && "out of bounds");
    return Regs[RegIdx];
  };
  inline ArrayRef<Reg> getRegs() const { return Regs; };
  inline unsigned getNumRegs() const { return Regs.size(); };

  inline const RegionBoundaries &getRegion(RegisterIdx RegionIdx) {
    assert(RegionIdx < Regions.size() && "out of bounds");
    return Regions[RegionIdx];
  }
  inline unsigned getNumRegions() const { return Regions.size(); }

  /// Whether register \p RegIdx is a rematerialization of some original
  /// register.
  inline bool isRematerializedRegister(RegisterIdx RegIdx) const {
    assert(RegIdx < Regs.size() && "out of bounds");
    return RegIdx >= UnrematableOprds.size();
  }
  /// Returns the origin index of rematerializable register \p RegIdx.
  inline RegisterIdx getOriginOf(RegisterIdx RematRegIdx) const {
    assert(isRematerializedRegister(RematRegIdx) && "not a rematerialization");
    return Origins[RematRegIdx - UnrematableOprds.size()];
  }
  /// If \p RegIdx is a rematerialization, returns its origin's index. If it is
  /// an original register's index, returns the same index.
  inline RegisterIdx getOriginOrSelf(RegisterIdx RegIdx) const {
    if (isRematerializedRegister(RegIdx))
      return getOriginOf(RegIdx);
    return RegIdx;
  }
  /// Returns operand indices corresponding to unrematerializable operands for
  /// any register \p RegIdx.
  inline ArrayRef<unsigned> getUnrematableOprds(unsigned RegIdx) const {
    return UnrematableOprds[getOriginOrSelf(RegIdx)];
  }

  /// When rematerializating a register (called the "root" register in this
  /// context) to a given position, we must decide what to do with all its
  /// rematerializable dependencies (for unrematerializable dependencies, we
  /// have no choice but to re-use the same register). For each rematerializable
  /// dependency we can either
  /// 1. rematerialize it along with the register,
  /// 2. re-use it as-is, or
  /// 3. re-use a pre-existing rematerialization of it.
  /// In case 1, the same decision needs to be made for all of the dependency's
  /// dependencies. In cases 2 and 3, the dependency's dependencies need not be
  /// examined.
  ///
  /// This struct allows to encode decisions of types (2) and (3) when
  /// rematerialization of all of the root's dependency DAG is undesirable.
  /// During rematerialization, registers in the root's dependency DAG which
  /// have a path to the root made up exclusively of non-re-used registers will
  /// be rematerialized along with the root.
  struct DependencyReuseInfo {
    /// Keys and values are rematerializable register indices.
    ///
    /// Before rematerialization, this only contains entries for non-root
    /// registers of the root's dependency DAG which should not be
    /// rematerialized i.e., for which an existing register should be used
    /// instead. These map each such non-root register to either the same
    /// register (case 2, \ref DependencyReuseInfo::reuse) or to a
    /// rematerialization of the key register (case 3, \ref
    /// DependencyReuseInfo::useRemat).
    ///
    /// After rematerialization, this contains additional entries for non-root
    /// registers of the root's dependency DAG that needed to be rematerialized
    /// along the root. These map each such non-root register to their
    /// corresponding new rematerialization that is used in the rematerialized
    /// root's dependency DAG. It follows that the difference in map size before
    /// and after rematerialization indicates the number of non-root registers
    /// that were rematerialized along the root.
    SmallDenseMap<RegisterIdx, RegisterIdx, 4> DependencyMap;

    DependencyReuseInfo &reuse(RegisterIdx DepIdx) {
      DependencyMap.insert({DepIdx, DepIdx});
      return *this;
    }
    DependencyReuseInfo &useRemat(RegisterIdx DepIdx, RegisterIdx DepRematIdx) {
      DependencyMap.insert({DepIdx, DepRematIdx});
      return *this;
    }
    DependencyReuseInfo &clear() {
      DependencyMap.clear();
      return *this;
    }
  };

  /// Rematerializes register \p RootIdx just before its first user inside
  /// region \p UseRegion, transfers all its users in the region to the new
  /// register, and returns the latter's index. The root's dependency DAG is
  /// rematerialized or re-used according to \p DRI.
  ///
  /// When the method returns, \p DRI contains additional entries for non-root
  /// registers of the root's dependency DAG that needed to be rematerialized
  /// along the root. References to \ref Rematerializer::Reg should be
  /// considered invalidated by calls to this method.
  RegisterIdx rematerializeToRegion(RegisterIdx RootIdx, unsigned UseRegion,
                                    DependencyReuseInfo &DRI);

  /// Rematerializes register \p RootIdx before position \p InsertPos and
  /// returns the new register's index. The root's dependency DAG is
  /// rematerialized or re-used according to \p DRI.
  ///
  /// When the method returns, \p DRI contains additional entries for non-root
  /// registers of the root's dependency DAG that needed to be rematerialized
  /// along the root. References to \ref Rematerializer::Reg should be
  /// considered invalidated by calls to this method.
  RegisterIdx rematerializeToPos(RegisterIdx RootIdx,
                                 MachineBasicBlock::iterator InsertPos,
                                 DependencyReuseInfo &DRI);

  /// Rolls back all rematerializations of original register \p RootIdx,
  /// transfering all their users back to it and permanently deleting them from
  /// the MIR. The root register is revived if it was fully rematerialized (this
  /// requires that rollback support was set at that time). Transitive
  /// dependencies of the root register that were fully rematerialized are
  /// re-vived at their original positions; this requires that rollback support
  /// was set when they were rematerialized.
  void rollbackRematsOf(RegisterIdx RootIdx);

  /// Rolls back register \p RematIdx (which must be a rematerialization)
  /// transfering all its users back to its origin. The latter is revived if it
  /// was fully rematerialized (this requires that rollback support was set at
  /// that time).
  void rollback(RegisterIdx RematIdx);

  /// Revives original register \p RootIdx at its original position in the MIR
  /// if it was fully rematerialized with rollback support set. Transitive
  /// dependencies of the root register that were fully rematerialized are
  /// revived at their original positions; this requires that rollback support
  /// was set when they were themselves rematerialized.
  void reviveRegIfDead(RegisterIdx RootIdx);

  /// Transfers all users of register \p FromRegIdx in region \p UseRegion to \p
  /// ToRegIdx, the latter of which must be a rematerialization of the former or
  /// have the same origin register. Users in \p UseRegion must be reachable
  /// from \p ToRegIdx.
  void transferRegionUsers(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                           unsigned UseRegion);

  /// Transfers user \p UserMI from register \p FromRegIdx to \p ToRegIdx,
  /// the latter of which must be a rematerialization of the former or have the
  /// same origin register. \p UserMI must be a direct user of \p FromRegIdx. \p
  /// UserMI must be reachable from \p ToRegIdx.
  void transferUser(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                    MachineInstr &UserMI);

  /// Recomputes all live intervals that have changed as a result of previous
  /// rematerializations/rollbacks.
  void updateLiveIntervals();

  /// Deletes unused rematerialized registers that were left in the MIR to
  /// support rollback.
  void commitRematerializations();

  /// Determines whether (sub-)register operand \p MO has the same value at
  /// all \p Uses as at \p MO. This implies that it is also available at all \p
  /// Uses according to its current live interval.
  bool isMOIdenticalAtUses(MachineOperand &MO, ArrayRef<SlotIndex> Uses) const;

  /// Finds the closest rematerialization of register \p RegIdx in region \p
  /// Region that exists before slot \p Before. If no such rematerialization
  /// exists, returns \ref Rematerializer::NoReg.
  RegisterIdx findRematInRegion(RegisterIdx RegIdx, unsigned Region,
                                SlotIndex Before) const;

  Printable printDependencyDAG(RegisterIdx RootIdx) const;
  Printable printID(RegisterIdx RegIdx) const;
  Printable printRematReg(RegisterIdx RegIdx, bool SkipRegions = false) const;
  Printable printRegUsers(RegisterIdx RegIdx) const;
  Printable printUser(const MachineInstr *MI) const;

private:
  SmallVectorImpl<RegionBoundaries> &Regions;
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  /// Rematerializable registers identified since the rematerializer's creation,
  /// both dead and alive, originals and rematerializations. No register is ever
  /// deleted. Indices inside this vector serve as handles for rematerializable
  /// registers.
  SmallVector<Reg> Regs;
  /// For each original register, stores indices of its read register operands
  /// which are unrematerializable. This doesn't change after the initial
  /// collection period, so the size of the vector indicates the number of
  /// original registers.
  SmallVector<SmallVector<unsigned, 2>> UnrematableOprds;
  /// Indicates the original register index of each rematerialization, in the
  /// order in which they are created. The size of the vector indicates the
  /// total number of rematerializations ever created, including those that were
  /// deleted or rolled back.
  SmallVector<RegisterIdx> Origins;
  using RematsOf = SmallDenseSet<RegisterIdx, 4>;
  /// Maps original register indices to their currently alive
  /// rematerializations. In practice most registers don't have
  /// rematerializations so this is represented as a map to lower memory cost.
  DenseMap<RegisterIdx, RematsOf> Rematerializations;

  /// Registers mapped to the index of their corresponding rematerialization
  /// data in the \ref Regs vector. This includes registers that no longer exist
  /// in the MIR.
  DenseMap<Register, RegisterIdx> RegToIdx;
  /// Maps all MIs to their parent region. Region terminators are considered
  /// part of the region they terminate.
  DenseMap<MachineInstr *, unsigned> MIRegion;
  /// Set of registers whose live-range may have changed during past
  /// rematerializations/rollbacks.
  DenseSet<RegisterIdx> LISUpdates;
  /// Keys are fully rematerialized registers whose rematerializations are
  /// currently rollback-able. Values map register machine operand indices to
  /// their original register.
  DenseMap<RegisterIdx, DenseMap<unsigned, Register>> Revivable;
  /// Whether all rematerializations of registers identified during the last
  /// analysis phase will be rollback-able.
  bool SupportRollback = false;

  /// During the analysis phase, creates a \ref Rematerializer::Reg object for
  /// virtual register \p VirtRegIdx if it
  void addRegIfRematerializable(unsigned VirtRegIdx, BitVector &SeenRegs);

  /// Determines whether \p MI is considered rematerializable. This further
  /// restricts constraints imposed by the TII on rematerializable instructions,
  /// requiring for example that the defined register is virtual and only
  /// defined once.
  bool isMIRematerializable(const MachineInstr &MI) const;

  /// Rematerializes register \p RegIdx at \p InsertPos, adding the new
  /// rematerializable register to the backing vector \ref Regs and returning
  /// its index inside the vector. Sets the new registers' rematerializable
  /// dependencies to \p Dependencies (these are assumed to already exist in the
  /// MIR) and its unrematerializable dependencies to the same as \p RegIdx. The
  /// new register initially has no user. Since the method appends to \ref Regs,
  /// references to elements within it should be considered invalidated across
  /// calls to this method unless the vector can be guaranteed to have enough
  /// space for an extra element.
  RegisterIdx rematerializeReg(RegisterIdx RegIdx,
                               MachineBasicBlock::iterator InsertPos,
                               SmallVectorImpl<Reg::Dependency> &&Dependencies);

  /// Implementation of \ref Rematerializer::transferUser that doesn't update
  /// register users.
  void transferUserImpl(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                        MachineInstr &UserMI);

  /// Deletes register \p RootIdx if it no longer has any user. If the register
  /// is deleted, recursively deletes any of its transitive rematerializable
  /// dependencies that no longer have users as a result.
  void deleteRegIfUnused(RegisterIdx RootIdx);

  /// Deletes rematerializable register \p RegIdx from the DAG and relevant
  /// internal state.
  void deleteReg(RegisterIdx RegIdx);

  /// If \p MI's first operand defines a register and that register is a
  /// rematerializable register tracked by the rematerializer, returns its
  /// index in the \ref Regs vector. Otherwise returns \ref
  /// Rematerializer::NoReg.
  RegisterIdx getDefRegIdx(const MachineInstr &MI) const;
};

} // namespace llvm

#endif // LLVM_CODEGEN_REMATERIALIZER_H
