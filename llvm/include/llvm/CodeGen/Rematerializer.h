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

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {

/// MIR-level target-independent rematerializer. Provides an API to identify and
/// rematerialize registers within a machine function.
///
/// At the moment this supports rematerializing registers that meet all of the
/// following constraints.
/// 1. The register is virtual.
/// 2. The register is defined within a single region---potentially over
///    multiple MIs---and isn't used by a MI that is not defining part of the
///    register before its last defining MI.
/// 3. All defining instructions are deemed rematerializable by the TII and
///    don't have any physical register use that is both non-constant and
///    non-ignorable.
/// 4. The register has at least one non-debug use that is inside or at a region
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
/// rematerializing them again.
///
/// Throughout its lifetime, the rematerializer tracks new registers it creates
/// (which are rematerializable by construction) and their relations to other
/// registers. It performs DAG and live interval updates immediately on
/// rematerialization and/or user transfer. Importantly, missing dead flags on
/// partial definitions of unrematerializable registers can yield dead
/// definitions when rematerializing their users. They are deleted to preserve
/// live interval validity. These deletions can cascade to other
/// (un)rematerializable registers that also become dead as a result.
///
/// In its nomenclature, the rematerializer differentiates between "original
/// registers" (registers that were present when it analyzed the function) and
/// rematerializations of these original registers. Rematerializations have an
/// "origin" which is the index of the original register they were
/// rematerialized from (transitivity applies; a rematerialization and all of
/// its own rematerializations have the same origin). Semantically, only
/// original registers have rematerializations.
///
/// Dealing with sub-registers is complicated, we have to handle dead-defs,
/// undef flags, and connected components
class Rematerializer {
public:
  /// Index type for rematerializable registers.
  using RegisterIdx = unsigned;

  /// A rematerializable register, potentially defined by multiple instructions.
  ///
  /// A rematerializable register has a set of dependencies, which correspond
  /// to the unique read register operands of its defining instruction(s) and
  /// which can themselves be rematerializable. Operands of defining
  /// instructions corresponding to unrematerializable dependencies are managed
  /// by and queried from the rematerializer, whereas rematerializable ones are
  /// part of this struct and identified through their register index.
  ///
  /// A rematerializable register also has an arbitrary number of users in an
  /// arbitrary number of regions, potentially including its own defining
  /// region. When rematerializations lead to operand changes in users, a
  /// register may find itself without any user left, at which point the
  /// rematerializer deletes it (emptying \ref Reg::Defs).
  struct Reg {
    /// All instructions that define the register, in program order.
    SmallVector<MachineInstr *, 1> Defs;
    /// Defining region of the register.
    unsigned DefRegion;
    /// The rematerializable register's lane bitmask.
    LaneBitmask Mask;

    using RegionUsers = SmallDenseSet<MachineInstr *, 4>;
    /// Uses of the register, mapped by region. Users that also define a part of
    /// the register are considered defs and not accounted for here.
    SmallDenseMap<unsigned, RegionUsers, 2> Uses;

    /// This register's rematerializable dependencies, one per unique
    /// rematerializable register operand over all definitions.
    SmallVector<RegisterIdx, 2> Dependencies;

    MachineInstr *getFirstDef() const { return Defs.front(); }
    MachineInstr *getLastDef() const { return Defs.back(); }

    /// Returns the rematerializable register from one of its defining
    /// instructions.
    Register getDefReg() const {
      const MachineInstr *DefMI = getFirstDef();
      assert(DefMI && "defining instruction(s) were deleted");
      if (!DefMI->getOperand(0).isDef())
        dbgs() << *DefMI;
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

    /// Returns the index of \p DefMI in the register's definitions order.
    /// Returns the number of definitions if \p DefMI is not a definition of the
    /// register.
    unsigned getDefIdx(MachineInstr *DefMI) const {
      return std::distance(Defs.begin(), find(Defs, DefMI));
    }

    /// Returns the first and last user of the register in region \p UseRegion.
    /// If the register has no user in the region, returns a pair of nullptr's.
    std::pair<MachineInstr *, MachineInstr *>
    getRegionUseBounds(unsigned UseRegion, const LiveIntervals &LIS) const;

    bool isAlive() const { return !Defs.empty(); }

  private:
    void addUser(MachineInstr *MI, unsigned Region);
    void addUsers(const RegionUsers &NewUsers, unsigned Region);
    void eraseUser(MachineInstr *MI, unsigned Region);

    /// Erases user \p MI from region \p Region if it exists. Returns whether \p
    /// MI was actually deleted.
    bool tryEraseUser(MachineInstr *MI, unsigned Region);

    friend Rematerializer;
  };

  /// Rematerializer listener. Defines overridable hooks that allow to catch
  /// specific events inside the rematerializer. All hooks do nothing by
  /// default. Listeners can be added or removed at any time during the
  /// rematerializer's lifetime.
  class Listener {
  public:
    using RegisterIdx = Rematerializer::RegisterIdx;

    /// Called just after register \p NewRegIdx is created (following a
    /// rematerialization). At this point the rematerialization exists in the \p
    /// Remater state and the MIR but does not yet have any user.
    virtual void rematerializerNoteRegCreated(const Rematerializer &Remater,
                                              RegisterIdx NewRegIdx) {}

    /// Called just before register \p RegIdx is deleted from the MIR. At this
    /// point the register still exists in the MIR but no longer has any user.
    virtual void
    rematerializerNoteRegWillBeDeleted(const Rematerializer &Remater,
                                       RegisterIdx RegIdx) {}

    /// Called just before unrematerializable instruction \p MI is deleted from
    /// the MIR because it has become a dead definition.
    virtual void
    rematerializerNoteMIWillBeDeleted(const Rematerializer &Remater,
                                      MachineInstr &MI) {}

    virtual ~Listener() = default;

  private:
    virtual void anchor();
  };

  /// Error value for register indices.
  static constexpr unsigned NoReg = ~0;

  /// A region's boundaries i.e. a pair of instruction bundle iterators. The
  /// lower boundary is inclusive, the upper boundary is exclusive.
  using RegionBoundaries =
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>;

  using RematsOf = SmallDenseSet<RegisterIdx, 4>;

  /// Simply initializes some internal state, does not identify
  /// rematerialization candidates.
  Rematerializer(MachineFunction &MF,
                 SmallVectorImpl<RegionBoundaries> &Regions,
                 LiveIntervals &LIS);

  /// Goes through the whole MF and identifies all rematerializable registers.
  /// Returns whether there is any rematerializable register in regions.
  bool analyze();

  /// Adds a new listener to the rematerializer.
  void addListener(Listener *Listen) {
    assert(Listen && "null listener");
    if (!Listeners.insert(Listen).second)
      llvm_unreachable("duplicate listener");
  }

  /// Removes a listener from the rematerializer.
  void removeListener(Listener *Listen) {
    if (!Listeners.erase(Listen))
      llvm_unreachable("unknown listener");
  }

  /// Removes all listeners from the rematerializer.
  void clearListeners() { Listeners.clear(); }

  const Reg &getReg(RegisterIdx RegIdx) const {
    assert(RegIdx < Regs.size() && "out of bounds");
    return Regs[RegIdx];
  };
  ArrayRef<Reg> getRegs() const { return Regs; };
  unsigned getNumRegs() const { return Regs.size(); };

  /// Determines whether register \p RegIdx fully disappeared from the MIR. This
  /// may happen when it was only used by instructions which became dead during
  /// the rematerializer's lifetime.
  bool isPermanentlyDead(RegisterIdx RegIdx) const {
    RegisterIdx OrigIdx = getOriginOrSelf(RegIdx);
    return !getReg(OrigIdx).isAlive() && !Rematerializations.contains(OrigIdx);
  }

  const RegionBoundaries &getRegion(RegisterIdx RegionIdx) const {
    assert(RegionIdx < Regions.size() && "out of bounds");
    return Regions[RegionIdx];
  }
  unsigned getNumRegions() const { return Regions.size(); }

  /// Whether register \p RegIdx is an original register.
  bool isOriginalRegister(RegisterIdx RegIdx) const {
    return !isRematerializedRegister(RegIdx);
  }
  /// Whether register \p RegIdx is a rematerialization of some original
  /// register.
  bool isRematerializedRegister(RegisterIdx RegIdx) const {
    assert(RegIdx < Regs.size() && "out of bounds");
    return RegIdx >= UnrematableDeps.size();
  }
  /// Returns the origin index of rematerializable register \p RegIdx.
  RegisterIdx getOriginOf(RegisterIdx RematRegIdx) const {
    assert(isRematerializedRegister(RematRegIdx) && "not a rematerialization");
    return Origins[RematRegIdx - UnrematableDeps.size()];
  }
  /// If \p RegIdx is a rematerialization, returns its origin's index. If it is
  /// an original register's index, returns the same index.
  RegisterIdx getOriginOrSelf(RegisterIdx RegIdx) const {
    if (isRematerializedRegister(RegIdx))
      return getOriginOf(RegIdx);
    return RegIdx;
  }
  /// Returns unreamaterializable read lanes of register operands for
  /// register \p RegIdx.
  ArrayRef<std::pair<Register, LaneBitmask>>
  getUnrematableDeps(RegisterIdx RegIdx) const {
    return UnrematableDeps[getOriginOrSelf(RegIdx)];
  }

  /// If \p MI's first operand defines a register and that register is a
  /// rematerializable register tracked by the rematerializer, returns its
  /// index in the \ref Regs vector. Otherwise returns \ref
  /// Rematerializer::NoReg.
  RegisterIdx getDefRegIdx(const MachineInstr &MI) const;

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
  /// region \p UseRegion (or at the end of the region if it has no user),
  /// transfers all its users in the region to the new register, and returns the
  /// latter's index. The root's dependency DAG is rematerialized or re-used
  /// according to \p DRI.
  ///
  /// When the method returns, \p DRI contains additional entries for non-root
  /// registers of the root's dependency DAG that needed to be rematerialized
  /// along the root. References to \ref Rematerializer::Reg should be
  /// considered invalidated by calls to this method.
  RegisterIdx rematerializeToRegion(RegisterIdx RootIdx, unsigned UseRegion,
                                    DependencyReuseInfo &DRI);

  /// Rematerializes register \p RootIdx before position \p InsertPos in \p
  /// UseRegion and returns the new register's index. The root's dependency DAG
  /// is rematerialized or re-used according to \p DRI.
  ///
  /// When the method returns, \p DRI contains additional entries for non-root
  /// registers of the root's dependency DAG that needed to be rematerialized
  /// along the root. References to \ref Rematerializer::Reg should be
  /// considered invalidated by calls to this method.
  RegisterIdx rematerializeToPos(RegisterIdx RootIdx, unsigned UseRegion,
                                 MachineBasicBlock::iterator InsertPos,
                                 DependencyReuseInfo &DRI);

  /// Rematerializes register \p RegIdx before \p InsertPos in \p UseRegion,
  /// adding the new rematerializable register to the backing vector \ref Regs
  /// and returning its index inside the vector. Sets the new register's
  /// rematerializable dependencies to \p Dependencies (these are assumed to
  /// already exist in the MIR) and its unrematerializable dependencies to the
  /// same as \p RegIdx. The new register initially has no user. Since the
  /// method appends to \ref Regs, references to elements within it should be
  /// considered invalidated across calls to this method unless the vector can
  /// be guaranteed to have enough space for an extra element.
  RegisterIdx rematerializeReg(RegisterIdx RegIdx, unsigned UseRegion,
                               MachineBasicBlock::iterator InsertPos,
                               SmallVectorImpl<RegisterIdx> &&Dependencies);

  /// Re-creates each defining instruction of a previously deleted register \p
  /// RegIdx before each position in \p Positions (one position per defining
  /// instruction, in the same order). Positions must be in the same region as
  /// the deleted register, and earlier than all uses of the register in the
  /// region. \p DefReg must be the original virtual register that \p RegIdx
  /// used to define. Rematerializable dependencies are assumed to already exist
  /// in the MIR.
  void recreateReg(RegisterIdx RegIdx,
                   ArrayRef<MachineBasicBlock::iterator> Positions,
                   Register DefReg);

  /// Transfers all users of register \p FromRegIdx in region \p UseRegion to \p
  /// ToRegIdx, the latter of which must be a rematerialization of the former or
  /// have the same origin register. Users in \p UseRegion must be reachable
  /// from \p ToRegIdx.
  void transferRegionUsers(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                           unsigned UseRegion);

  /// Transfers user \p UserMI in region \p UserRegion from register \p
  /// FromRegIdx to \p ToRegIdx, the latter of which must be a rematerialization
  /// of the former or have the same origin register. \p UserMI must be a direct
  /// user of \p FromRegIdx. \p UserMI must be reachable from \p ToRegIdx.
  void transferUser(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                    unsigned UserRegion, MachineInstr &UserMI);

  /// Transfers all users of register \p FromRegIdx to register \p ToRegIdx, the
  /// latter of which must be a rematerialization of the former or have the same
  /// origin register. Users of \p FromRegIdx must be reachable from \p
  /// ToRegIdx.
  void transferAllUsers(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx);

  /// Determines whether (sub-)register operand \p MO has the same value at
  /// all \p Uses as at \p MO. This implies that it is also available at all \p
  /// Uses according to its current live interval.
  bool isMOIdenticalAtUses(MachineOperand &MO, ArrayRef<SlotIndex> Uses) const;

  /// Determines whether lanes \p Mask of register \p Reg habe the same value at
  /// all \p Uses as at \p RefSlot. This implies that it is also available at
  /// all \p Uses according to its current live interval.
  bool isRegIdenticalAtUses(Register Reg, LaneBitmask Mask, SlotIndex RefSlot,
                            ArrayRef<SlotIndex> Uses) const;

  /// Finds the closest rematerialization of register \p RegIdx in region \p
  /// Region that exists before slot \p Before. If no such rematerialization
  /// exists, returns \ref Rematerializer::NoReg.
  RegisterIdx findRematInRegion(RegisterIdx RegIdx, unsigned Region,
                                SlotIndex Before) const;

  Printable printDependencyDAG(RegisterIdx RootIdx) const;
  Printable printID(RegisterIdx RegIdx) const;
  Printable printRematReg(RegisterIdx RegIdx, bool SkipRegions = false,
                          unsigned DefIdx = 0) const;
  Printable printRegUsers(RegisterIdx RegIdx) const;
  Printable printUser(const MachineInstr *MI,
                      std::optional<unsigned> UseRegion = std::nullopt) const;

private:
  struct DeadDefDelegate : LiveRangeEdit::Delegate {
    Rematerializer &Remater;
    DeadDefDelegate(Rematerializer &Remater) : Remater(Remater) {}
    void LRE_WillEraseInstruction(MachineInstr *MI) override;
  };

  SmallVectorImpl<RegionBoundaries> &Regions;
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  SmallPtrSet<Listener *, 1> Listeners;

  void noteRegCreated(RegisterIdx RegIdx) const {
    for (Listener *Listen : Listeners)
      Listen->rematerializerNoteRegCreated(*this, RegIdx);
  }

  void noteRegWillBeDeleted(RegisterIdx RegIdx) const {
    for (Listener *Listen : Listeners)
      Listen->rematerializerNoteRegWillBeDeleted(*this, RegIdx);
  }

  void noteMIWillBeDeleted(MachineInstr &MI) const {
    for (Listener *Listen : Listeners)
      Listen->rematerializerNoteMIWillBeDeleted(*this, MI);
  }

  /// Rematerializable registers identified since the rematerializer's creation,
  /// both dead and alive, originals and rematerializations. No register is ever
  /// deleted. Indices inside this vector serve as handles for rematerializable
  /// registers.
  SmallVector<Reg> Regs;
  /// For each original register, stores unrematerializable read lanes of
  /// register operands. This doesn't change after the initial collection
  /// period, so the size of the vector indicates the number of original
  /// registers.
  SmallVector<SmallVector<std::pair<Register, LaneBitmask>, 2>> UnrematableDeps;
  /// Indicates the original register index of each rematerialization, in the
  /// order in which they are created. The size of the vector indicates the
  /// total number of rematerializations ever created, including those that were
  /// deleted.
  SmallVector<RegisterIdx> Origins;
  /// Maps original register indices to their currently alive
  /// rematerializations. In practice most registers don't have
  /// rematerializations so this is represented as a map to lower memory cost.
  DenseMap<RegisterIdx, RematsOf> Rematerializations;

  /// Registers mapped to the index of their corresponding rematerialization
  /// data in the \ref Regs vector. This includes registers that no longer exist
  /// in the MIR.
  DenseMap<Register, RegisterIdx> RegToIdx;
  /// Parent block of each region, in order.
  SmallVector<MachineBasicBlock *> RegionMBB;

  /// Common post-processing step after creating a new register \p RematRegIdx
  /// based on register \p ModelRegIdx.
  void postRematerialization(RegisterIdx ModelRegIdx, RegisterIdx RematRegIdx);

  /// Common pre-processing step before deleting a register \p DeleteRegIdx. The
  /// register must still have alive definitions.
  void preDeletion(RegisterIdx DeleteRegIdx);

  /// Extends \p LI over \p Mask to be live at \p UdeIdx.
  void extendInterval(LiveInterval &LI, LaneBitmask Mask,
                      SlotIndex UseIdx) const;

  /// Extends the live interval of rematerializable register \p RegIdx to be
  /// live at the register slot of all MIs in \p NewUsers. Creates and/or
  /// refines the interval's sub-ranges as needed. Updates the register's
  /// defining instruction's dead flag as needed.
  void extendToNewUsers(RegisterIdx RegIdx,
                        ArrayRef<MachineInstr *> NewUsers) const;

  /// Shrinks the live interval of rematerializable register \p RegIdx to its
  /// current uses. If the register has no users, deletes it along with
  /// registers in its dependency DAG that no longer have users as a result.
  void shrinkToUses(RegisterIdx RegIdx);

  /// Shrinks the live interval of unrematerializable register \p Reg to its
  /// current uses. The interval is split if necessary, creating new
  /// unrematerializable registers and updating register dependencies as needed.
  void shrinkToUsesUnremat(Register Reg);

  /// During the analysis phase, creates a \ref Rematerializer::Reg object for
  /// virtual register \p VirtRegIdx if it is rematerializable. \p MIRegion maps
  /// all MIs to their parent region. Set bits in \p SeenRegs indicate virtual
  /// register indices that have already been visited.
  void
  addRegIfRematerializable(unsigned VirtRegIdx,
                           const DenseMap<MachineInstr *, unsigned> &MIRegion,
                           BitVector &SeenRegs);

  /// Determines whether \p MI is considered rematerializable. This further
  /// restricts constraints imposed by the TII on rematerializable instructions,
  /// requiring for example that the defined register is virtual.
  bool isMIRematerializable(const MachineInstr &MI) const;

  /// Implementation of \ref Rematerializer::transferUser that doesn't update
  /// register users.
  void transferUserImpl(RegisterIdx FromRegIdx, RegisterIdx ToRegIdx,
                        MachineInstr &UserMI);

  /// Deletes register \p RootIdx, which must not have any users left. If the
  /// register is deleted, recursively deletes any of its transitive
  /// rematerializable dependencies that no longer have users as a result. In
  /// case of recursive deletion, all of a register's users are always deleted
  /// before the register itself.
  void deleteReg(RegisterIdx RootIdx);
};

/// Rematerializer listener with the ability to re-create deleted registers and
/// rollback rematerializations. Starts recording register deletions and
/// rematerializations as soon as it is attached to the rematerializer.
class Rollbacker : public Rematerializer::Listener {
public:
  Rollbacker() = default;

  /// Re-creates all deleted registers and rolls back all rematerializations
  /// that were recorded.
  void rollback(Rematerializer &Remater);

  void rematerializerNoteRegCreated(const Rematerializer &Remater,
                                    RegisterIdx RegIdx) override;

  void rematerializerNoteRegWillBeDeleted(const Rematerializer &Remater,
                                          RegisterIdx RegIdx) override;

  void rematerializerNoteMIWillBeDeleted(const Rematerializer &Remater,
                                         MachineInstr &MI) override;

private:
  struct DeadReg {
    /// Register index.
    RegisterIdx Idx;
    /// Original register.
    Register DefReg;
    /// Original definitions of the register. The underlying MIs no longer exist
    /// at rollback time, but may be referenced as re-creation positions for
    /// previously deleted registers.
    SmallVector<MachineInstr *, 1> Defs;

    DeadReg(RegisterIdx Idx, const Rematerializer &Remater)
        : Idx(Idx), DefReg(Remater.getReg(Idx).getDefReg()),
          Defs(Remater.getReg(Idx).Defs) {}
  };

  /// An insertion position in the MIR. The pointer should be interpreted as:
  /// - a MachineInstr* if the int is 0/false (insert before the MI).
  /// - a MachineBasicBlock* if the int is 1/true (insert at the MBB's end).
  using InsertBeforePos = PointerIntPair<void *, 1, bool>;

  /// Original registers that have been deleted, in order of deletion.
  SmallVector<DeadReg> DeadRegs;
  /// Re-creation positions for all original registers that have been deleted,
  /// one per defining instruction, in program order for any given register and
  /// in register deletion order overall. A position is either a MachineInstr*
  /// that existed in the MIR at the time the rollbacker was attached to the
  /// rematerializer, or a MachineBasicBlock*.
  SmallVector<InsertBeforePos> Positions;
  /// Maps all re-creation positions that exist in \ref Positions to the indices
  /// of elements holding that position in the vector.
  DenseMap<InsertBeforePos, SmallDenseSet<unsigned, 1>> PosToIdx;
  /// Registers which have been rematerialized (from original index to
  /// rematerialized index).
  DenseMap<RegisterIdx, Rematerializer::RematsOf> Rematerializations;
  /// Used to block further recording of events whenver we are actively rolling
  /// back.
  bool RollingBack = false;

  InsertBeforePos makePos(MachineInstr *MI) const {
    return InsertBeforePos(MI, false);
  }
  InsertBeforePos makePos(MachineBasicBlock *MBB) const {
    return InsertBeforePos(MBB, true);
  }
  InsertBeforePos makePos(MachineBasicBlock::iterator It,
                          MachineBasicBlock *MBB) const {
    if (It == MBB->end())
      return makePos(MBB);
    return makePos(&*It);
  }

  /// Whether \p MI would be deleted if we were to rollback later. These are MIs
  /// defining rematerializable registers whose creation has been recorded by
  /// the rollbacker.
  bool isRollbackableMI(const MachineInstr &MI,
                        const Rematerializer &Remater) const;

  /// Switches all positions that point to \p MI to \p It in the \ref Positions
  /// vector, and updates \ref PosToIdx accordingly. This is used when it
  /// becomes known that \p MI is about to be permanently deleted from the MIR
  /// and thus becomes an invalid re-creation position.
  void invalidatePosition(MachineInstr *MI, MachineBasicBlock::iterator It);
};

} // namespace llvm

#endif // LLVM_CODEGEN_REMATERIALIZER_H
