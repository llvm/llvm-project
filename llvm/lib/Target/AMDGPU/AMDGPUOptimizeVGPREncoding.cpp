//===-- AMDGPUOptimizeVGPREncoding.cpp --------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass is meant to run between register allocation and
/// virtual-to-physical register rewriting. On subtargets where the MSBs of some
/// VGPRs come from the processor's MODE, the pass tries to modify the existing
/// virtual-to-physical register mappings to reduce the number of MODE-setting
/// instructions that will need to be inserted in the program to honor MSB group
/// differences between physical VGPRs. The pass cannot cause extra spilling to
/// occur.
///
/// In the future, the intent is for this pass to also try to minimize VGPR bank
/// conflicts on subtarget where it is relevant.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOptimizeVGPREncoding.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <functional>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-optimize-vgpr-encoding"

namespace {

/// MSB group, identified by an unsigned ID in [0, NumMSBGroups).
using MSBGroup = unsigned;
static constexpr unsigned NumMSBGroups = 4;
static constexpr unsigned DefaultGroup = 0;

/// Operand type where the MSB group is relevant, identified by an unsigned ID
/// in [0, NumOprdTypes).
using OprdType = unsigned;
static constexpr unsigned NumOprdTypes = 4;

/// An instruction that has at least one VGPR operand whose MSBs are provided by
/// MODE. Instructions are part of a list and refer to other "neighbor"
/// instructions through their respective index in this list. Two instructions
/// are neighbors if they have at least one VGPR operand in the same operand
/// type and no other instruction with such an operand in between them. Two
/// neighbor instructions whose respective VGPR in a shared operand type differ
/// in MSB group require at least one MODE-setting instruction to be placed
/// somewhere in between them.
struct ModeInstr {
  /// Sentinel value in previous/next index arrays to indicate the non-existence
  /// of a previous/next instruction.
  static constexpr unsigned NoIdx = ~0U;

  /// For each operand type, virtual of physical VGPR operand used by the
  /// instruction. A null register indicates the instruction has no VGPR operand
  /// of that type. For VOPD instructions, this holds VGPRs for the first of the
  /// two instruction which define a VGPR of each operand type.
  std::array<Register, NumOprdTypes> Oprds;
  /// For each operand type, indices of previous/next neighbor instructions with
  /// defined operands in the instruction list this instruction is a part of.
  /// \ref NoIdx indicates that there is no such previous or next instruction.
  std::array<unsigned, NumOprdTypes> Prev, Next;

  ModeInstr() {
    Oprds.fill(Register());
    Prev.fill(NoIdx);
    Next.fill(NoIdx);
  }
};

/// Summarizes MSB bits usage over a machine basic block.
class MBBModeUsage {
public:
  /// The machine basic block.
  const MachineBasicBlock &MBB;

  /// Iterates over \p MBB's instructions to find those for which MSB bits
  /// provided by MODE are relevant. Indices of virtual registers used at least
  /// once in an operand reading MSB bits are set in \p OptVirtRegs. Indices of
  /// those that cannot change MSB group throughout optimization are set in \p
  /// PinnedVirtRegs.
  MBBModeUsage(const MachineBasicBlock &MBB, BitVector &OptVirtRegs,
               BitVector &PinnedVirtRegs);

  /// Returns the list of MODE-using instructions in the block.
  ArrayRef<ModeInstr> getInstructions() const { return Instructions; }

  /// Returns the MODE-using instruction in the block at index \p Idx.
  const ModeInstr &getInstruction(unsigned Idx) const {
    assert(Idx < Instructions.size() && "out of bounds");
    return Instructions[Idx];
  }

  /// Returns the index of the first instruction with a MODE-reading operand of
  /// type \p Oprd in the block, or \ref ModeInstr::NoIdx if none exists.
  unsigned getFirstOprd(OprdType Oprd) const {
    return Instructions.empty() ? ModeInstr::NoIdx : getFirstInstrFrom(0, Oprd);
  }

  /// Returns the index of the last instruction with a MODE-reading operand of
  /// type \p Oprd in the block, or \ref ModeInstr::NoIdx if none exists.
  unsigned getLastOprd(OprdType Oprd) const {
    return Instructions.empty()
               ? ModeInstr::NoIdx
               : getLastInstrUntil(Instructions.size() - 1, Oprd);
  }

  /// Returns the index of the first instruction from \p InstrIdx (included)
  /// with a MODE-reading operand of type \p Oprd in the block, or \ref
  /// ModeInstr::NoIdx if none exists.
  unsigned getFirstInstrFrom(unsigned InstrIdx, OprdType Oprd) const {
    return getInstrIdxImpl<false, false>(InstrIdx, Oprd);
  }

  /// Returns the index of the first instruction after \p InstrIdx (excluded)
  /// with a MODE-reading operand of type \p Oprd in the block, or \ref
  /// ModeInstr::NoIdx if none exists.
  unsigned getFirstInstrAfter(unsigned InstrIdx, OprdType Oprd) const {
    return getInstrIdxImpl<false, true>(InstrIdx, Oprd);
  }

  /// Returns the index of the last instruction until \p InstrIdx (included)
  /// with a MODE-reading operand of type \p Oprd in the block, or \ref
  /// ModeInstr::NoIdx if none exists.
  unsigned getLastInstrUntil(unsigned InstrIdx, OprdType Oprd) const {
    return getInstrIdxImpl<true, false>(InstrIdx, Oprd);
  }

  /// Returns the index of the last instruction before \p InstrIdx (excluded)
  /// with a MODE-reading operand of type \p Oprd in the block, or \ref
  /// ModeInstr::NoIdx if none exists.
  unsigned getLastInstrBefore(unsigned InstrIdx, OprdType Oprd) const {
    return getInstrIdxImpl<true, true>(InstrIdx, Oprd);
  }

#ifndef NDEBUG
  Printable print(const VirtRegMap &VRM) const;
#endif

private:
  /// List of all instructions in the machine basic block which have at least
  /// one operand for which MSB bits provided by MODE are relevant, in program
  /// order.
  SmallVector<ModeInstr> Instructions;

  template <bool UsePrev, bool SkipCurrent>
  unsigned getInstrIdxImpl(unsigned InstrIdx, OprdType Oprd) const;
};

/// A virtual register whose assigned physical register's MSB group can be
/// optimized for. An optimizable register maintains a per-group score encoding
/// its current level of "neighboring-ness" to other registers in each MSB
/// group, with higher scores indicating a higher number of "neighbor" registers
/// currently in the corresponding MSB group. Two registers (virtual or
/// physical) are initially considered neighbors when they are used as operands
/// of the same type in two neighboring instructions (c.f. \ref ModeInstr). A
/// register's neighborhood---and thefore its score---changes throughout the
/// pass's lifetime to reflect the simulataed placement of MODE-setting
/// instructions.
///
/// An optimizable register is said to be "pinned" when the MSB group of its
/// assigned physical register cannot change. Once a register is pinned it never
/// becomes unpinned. Score contributions from pinned and unpinned neighbors are
/// kept separate to enable identification of unoptimizable MSB group conflicts.
class OptReg {
public:
  using WeightedNeighbors = SmallDenseMap<OptReg *, unsigned, 4>;

  /// Abstract coordinates for an occurence of this register.
  struct Coordinates {
    /// This index of the MBB.
    unsigned MBBIndex;
    /// The index of the MODE-using instruction.
    unsigned InstrIdx;
    /// The operand type.
    OprdType Oprd;
  };

  /// Creates a neighbor-less optimizable register for register \p VirtReg.
  /// Neighboring relations with other optimizable registers are
  /// created/destroyed through class methods.
  OptReg(Register VirtReg, const VirtRegMap &VRM);

  /// Returns the total number of occurences of pinned neighbor registers in \p
  /// Group.
  unsigned getGroupPinnedScore(MSBGroup Group) const {
    return PinnedScore[Group];
  }

  /// Returns the total number of occurences of neighbor registers in \p Group.
  unsigned getGroupScore(MSBGroup Group) const {
    return PinnedScore[Group] + Score[Group];
  }

  /// Returns the total number of occurences of neighbor registers in this
  /// register's current MSB group.
  unsigned getCurrentGroupScore() const { return getGroupScore(MSB); }

  /// Returns a bitvector the size of the number of MSB groups whose set bits
  /// indicate the MSB groups in which this register currently has at least one
  /// pinned neighbor.
  SmallBitVector getPinGroups() const;

  /// Returns the register's current neighbors.
  const WeightedNeighbors &getNeighbors() const { return Neighbors; }

  /// Returns the list of coordinates corresponding to this register's
  /// occurences.
  ArrayRef<Coordinates> getOccurences() const { return Occurences; }

  /// Returns the underlying virtual register.
  Register getVirt() const { return VirtReg; }

  /// Returns the underlying virtual register's index.
  unsigned getVirtIndex() const { return VirtReg.virtRegIndex(); }

  /// Returns the MSB group of this register's currently assigned physical
  /// register.
  MSBGroup getMSB() const { return MSB; }

  /// Returns whether the register is pinned.
  bool isPinned() const { return IsPinned; }

  // addOccurence and record* methods used by OptimizableRegs to initialize the
  // occurences and neighborhood of all optimizable registers at the beginning.

  /// Adds an occurence of this register in operand type \p Oprd of instruction
  /// \p InstrIdx of MBB \p MBBIdx.
  void addOccurence(unsigned MBBIndex, unsigned InstrIdx, OprdType Oprd) {
    Occurences.push_back({MBBIndex, InstrIdx, Oprd});
  }

  /// Records an occurence of \p NeighborReg as a neighbor.
  void recordNeighborOccurence(OptReg &NeighborReg);

  /// Records an occurence of physical register \p PhysReg as a neighbor.
  void recordPhysNeighborOccurence(Register PhysReg, const VirtRegMap &VRM);

  /// Records an occurence of this register at a block boundary. This adds a
  /// "pinned occurence" of the default MSB group in which all MBBs start and
  /// end.
  void recordBlockBoundaryPin() { ++PinnedScore[DefaultGroup]; }

  // pinMSBGroup and remove* methods used by ModeSetOptimizer to progressively
  // simplify/destroy the neighborhood of all optimizable registers as it
  // simulates placement of MODE-setting instructions. remove* methods mirror
  // record* methods 1-to-1.

  /// Pins this register's to the MSB group of its currently assigned physical
  /// register.
  void pinMSBGroup();

  /// Removes an occurence of \p NeighborReg as a neighbor.
  void removeNeighborOccurence(OptReg &NeighborReg);

  /// Removes an occurence of physical register \p PhysReg as a neighbor.
  void removePhysNeighborOccurence(Register PhysReg, const VirtRegMap &VRM);

  /// Removes an occurence of this register at a block boundary. This removes a
  /// "pinned occurence" of the default MSB group in which all MBBs start and
  /// end.
  void removeBlockBoundaryPin() {
    assert(PinnedScore[DefaultGroup] > 0 && "underflow");
    --PinnedScore[DefaultGroup];
  }

  /// Notifies the optimizable register that its assigned physcial register has
  /// changed and that its new assignment belongs to MSB group \p NewGroup. It
  /// is illegal to change the MSB group of a pinned register.
  void notifyPhysAssignmentChanged(MSBGroup NewGroup);

#ifndef NDEBUG
  Printable print(const VirtRegMap &VRM) const;
#endif

private:
  /// The virtual register.
  Register VirtReg;
  /// MSB group of the virtual register's current physcial register assignment.
  MSBGroup MSB;
  /// Per-MSB group score, counting the number of occurences of neighbor
  /// registers in each group, separated between occurences of unpinned
  /// optimizable registers from the others (physical registers, pinned
  /// optimizable registers, and block boundaries). Reflects the register's
  /// current neighborhood.
  std::array<unsigned, NumMSBGroups> Score, PinnedScore;
  /// Maps neighboring optimizable registers to the number of times they occur
  /// in the neighborhood of one of this register's occurences. Neighbors can be
  /// added or removed at will after construction, impacting the score.
  WeightedNeighbors Neighbors;
  /// Occurences of this register in the function under consideration.
  /// Occurences can be added after construction but cannot be removed.
  SmallVector<Coordinates> Occurences;
  /// Whether the register is pinned to \ref MSB.
  bool IsPinned = false;
};

/// Manages all optimizable virtual registers for a function.
class OptimizableRegs {
public:
  /// Creates an \ref OptReg for each virtual register whose index is set in \p
  /// OptVirtRegs, immediately pinning those whose index is set in \p
  /// PinnedVirtRegs. Then derives neighborhood of all optimizable registers
  /// from MODE-using instructions in each block of \p ModeUsage.
  OptimizableRegs(const BitVector &OptVirtRegs, const BitVector &PinnedVirtRegs,
                  ArrayRef<MBBModeUsage> ModeUsage, const VirtRegMap &VRM);

  OptReg *operator[](Register Reg) {
    if (Reg.isPhysical())
      return nullptr;
    unsigned Idx = VirtRegToStorageIdx[Reg.virtRegIndex()];
    return Idx == NoIdx ? nullptr : &Storage[Idx];
  }
  OptReg &operator[](unsigned VirtRegIdx) {
    assert(OptVirtRegs.test(VirtRegIdx) && "invalid index");
    return Storage[VirtRegToStorageIdx[VirtRegIdx]];
  }

  /// Returns a bitvector whose set bits indicate indices of virtual registers
  /// which are optimizable i.e., for which (*this)[VirtReg] returns a valid
  /// \ref OptReg.
  const BitVector &getAllOptVirtRegs() const { return OptVirtRegs; }

  /// Returns the number of virtual registers, as reported by the MRI.
  unsigned getNumVirtRegs() const { return OptVirtRegs.size(); }

  using iterator = SmallVector<OptReg>::iterator;
  using const_iterator = SmallVector<OptReg>::const_iterator;
  iterator begin() { return Storage.begin(); }
  iterator end() { return Storage.end(); }
  const_iterator begin() const { return Storage.begin(); }
  const_iterator end() const { return Storage.end(); }

private:
  /// Sentinel value in \p VirtRegToStorageIdx to indicate the non-existence of
  /// a corresponding \ref OptReg.
  static constexpr unsigned NoIdx = ~0;

  /// Set bits indicate indices of virtual registers which are optimizable.
  BitVector OptVirtRegs;
  /// Backing storage for optimizable registers.
  SmallVector<OptReg, 0> Storage;
  /// Works as a map from virtual register indices to the index of the
  /// corresponding \ref OptReg in \ref Storage. A virtual register that is not
  /// optimizable "maps to" \ref NoIdx.
  SmallVector<unsigned, 0> VirtRegToStorageIdx;
};

/// Simulates placement of MODE-setting instructions as unoptimizable MSB groups
/// conflicts are detected, driving optimization forward by progressively
/// pruning register neighborhoods and pinning optimizable registers once they
/// reach an "ideal" MSB group.
///
/// The detection and resolution of unoptimizable MSB group conflicts is this
/// class's main purpose. An optimizable register with non-null score
/// contributions from pinned neighbors in more that one MSB group will
/// necessarily require MODE-setting instructions around its occurences that
/// neighbor pinned registers in all but one of those MSB groups. This is a
/// conflict in the sense that we would need the register to be in multiple MSB
/// groups at the same time to not need MODE-setting instructions. It is
/// unoptimizable by the pass because pinned neighbors are not allowed to change
/// MSB group, so no amount of register re-assignment can resolve it. The
/// objective is to detect those situations early so that no effort is made
/// attempting to optimize MSB conflits at code locations where we are
/// guaranteed to be unable to solve them. The class resolves such conflicts by
/// simulating the placement of MODE-setting instructions around problematic
/// registers, effectively "breaking" their relationships with some pinned
/// neighbors until any remaining MSB group conflict becomes optimizable again,
/// at the known cost of "placed" MODE-setting instructions.
///
/// Resolving conflicts strictly lowers the per-group score of optimizable
/// registers that neighbor "placed" MODE-setting instructions. This ensures
/// forward progress (scores are lower bounded at 0) and can uncover new
/// optimization opportunities as register neighborhoods become smaller and some
/// registers reach an "ideal" MSB group that they can be pinned to.
///
/// FIXME: The current approach to determine where we place MODE-setting
/// instructions to resolve conflicts is correct however when there are multiple
/// possible locations to choose from it does not attempt to analyze the
/// expected benefit of each. Picking the best location in such cases should
/// improve overall pass performance.
class ModeSetOptimizer {
public:
  /// Initializes the optimizer with all optimizable registers in \p OptRegs and
  /// all blocks in \p ModeUsage. Performs a first round of register pinning and
  /// conflict resolution on all relevant registers.
  ModeSetOptimizer(OptimizableRegs &OptRegs, ArrayRef<MBBModeUsage> ModeUsage,
                   const VirtRegMap &VRM);

  /// Iteratively resolves MSB group conflicts by simulating placement of
  /// MODE-setting instructions and pins newly eligible registers until reaching
  /// a fixed-point. By the end there are no unoptimizable MSB group conflicts
  /// and all registers that would be eligible for pinning are pinned. Returns
  /// whether any register changed state.
  bool resolveConflictsAndPinRegs();

  /// Notifies the optimizer that \p Reg changed MSB group. Sets bits in \p
  /// ScoreChanged for all virtual register indices whose score was affected by
  /// the move. Tracks which registers can become eligible for pinning or may
  /// exhibit a conflict as a result of the change.
  void regChangedGroup(OptReg &Reg, BitVector &ScoreChanged);

private:
  /// Result of BitVector::find* methods when no bit was found.
  static constexpr int NoBit = -1;

  /// Optimizable registers under consideration.
  OptimizableRegs &OptRegs;
  /// Set bits indicate registers which may be pinnable.
  BitVector CheckShouldBePinned;
  /// Set bits indicate registers which may have neighbors pinned in more than
  /// one MSB group.
  BitVector CheckResolveConflict;

  /// Simulated MODE-setting instruction placement in each machine basic block,
  /// in the same order as \ref ModeUsage. For any bitvector, a bit at position
  /// Idx means that there is a mode set placed in between instructions Idx - 1
  /// (or block entry when Idx == 0) and instruction Idx (or block exit when Idx
  /// == MBB.Instructions.size()).
  SmallVector<BitVector> ModeSetPlacement;
  /// MODE usage in all MBBs.
  ArrayRef<MBBModeUsage> ModeUsage;
  const VirtRegMap &VRM;

  /// Resolve conflicts for \p Reg, if any, and returns whether the register had
  /// conflicts.
  bool resolveConflictingPins(OptReg &Reg);

  /// Returns whether we consider that \p Reg should be pinned. Registers whose
  /// only remaining neighbors (pinned or not) are all currently in a single MSB
  /// group are in the perfect MSB group and should never change group again.
  bool shouldBePinned(const OptReg &Reg) const;

  /// Pins \p Reg if it is eligible according to \ref shouldbePinned. Returns
  /// whether the register was newly pinned.
  bool pinIfEligible(OptReg &Reg);

  /// Determines whether a MODE-setting instruction was already placed in block
  /// \p MBBIdx between the beginning of the block and MODE-using instruction \p
  /// InstrIdx.
  bool hasModeSetBefore(unsigned MBBIdx, unsigned InstrIdx) const {
    int FirstIdx = ModeSetPlacement[MBBIdx].find_first();
    return FirstIdx == NoBit ? false
                             : static_cast<unsigned>(FirstIdx) <= InstrIdx;
  }

  /// Determines whether a MODE-setting instruction was already placed in block
  /// \p MBBIdx between MODE-using instruction \p InstrIdx and the end of the
  /// block.
  bool hasModeSetAfter(unsigned MBBIdx, unsigned InstrIdx) const {
    int LastIdx = ModeSetPlacement[MBBIdx].find_last();
    return LastIdx == NoBit ? false : static_cast<unsigned>(LastIdx) > InstrIdx;
  }

  /// Determines whether a MODE-setting instruction was already placed in block
  /// \p MBBIdx between MODE-using instructions \p AfterIdx and \p BeforeIdx.
  bool hasModeSetBetween(unsigned MBBIdx, unsigned AfterIdx,
                         unsigned BeforeIdx) const {
    assert(AfterIdx < BeforeIdx && "inconsistent indices");
    // This looks in [AfterIdx + 1, BeforeIdx + 1) == [AfterIdx + 1, BeforeIdx].
    // This will therefore detect a bit before AfterIdx + 1 (equivalently, after
    // AfterIdx) and a bit before BeforeIdx.
    return ModeSetPlacement[MBBIdx].find_first_in(AfterIdx + 1,
                                                  BeforeIdx + 1) != NoBit;
  }

  /// Among the MSB groups with non-zero pin score of \p Reg, selects the most
  /// desirable one in which we would eventually like the register to end up.
  MSBGroup selectPreferredMSBGroup(const OptReg &Reg) const;

  /// Places a MODE-setting instruction just before MODE-using instruction \p
  /// InstrIdx in block \p MBBIdx.
  void placeJustBefore(unsigned MBBIdx, unsigned InstrIdx);

  /// Places a MODE-setting instruction just after MODE-using instruction \p
  /// InstrIdx in block \p MBBIdx.
  void placeJustAfter(unsigned MBBIdx, unsigned InstrIdx);

  /// Breaks neighbor relationship between MODE-using instructions \p AfterIdx
  /// and \p BeforeIdx in block \p MBBIdx and for operand type \p Oprd. Both \p
  /// AfterIdx and \p BeforeIdx can be \ref ModeInstr::NoIdx in which case they
  /// encode the default MSB group pin at, respectively, the entry and exit of
  /// the block.
  void breakNeighborRelationship(unsigned MBBIdx, unsigned AfterIdx,
                                 unsigned BeforeIdx, OprdType Oprd);
};

/// An optimizable virtual register we consider for re-assignment to a different
/// MSB group with higher score. Register candidates are weakly ordered,
/// "larger" candidates being considered more profitable to re-assign to target
/// MSB groups.
class OptRegCandidate {
public:
  /// Starting epoch when a candidate is created.
  static constexpr unsigned StartEpoch = 0;

  enum class PinState {
    /// Single pin in non-target group.
    PinInBadGroup = 0,
    /// No pins.
    NoPin = 1,
    /// Single pin in target group.
    TargetIsPin = 2
  };

  /// The optimizable register.
  OptReg &Reg;

  /// Set bits indicate MSB groups with the highest score and more desirable
  /// than the current one the register is in i.e., target groups. No targets
  /// mean the register is in the best group already.
  SmallBitVector Targets;

  /// State of pinned neighbors around the register. Only relevant when the
  /// candidate has at least one target.
  PinState NeighboringPins;

  /// The overall estimated benefit in re-assigning the register to one of the
  /// target groups. Higher is better. A negative benefit is still desirable.
  /// Only relevant when the candidate has at least one target.
  int Benefit;

  /// Creates the candidate for \p Reg.
  OptRegCandidate(OptReg &Reg) : Reg(Reg), Targets(NumMSBGroups) {
    assert(!Reg.isPinned() && "register cannot be pinned initially");
    update();
  }

  /// Re-computes the candidate's target groups and potential benefit.
  void update();

  /// Whether the candidate has a profitable re-assignment to any MSB group
  /// i.e., a re-assigment that will strictly increase the combined score of all
  /// optimizable registers.
  bool isProfitable() const { return Targets.any(); }

  /// Returns the current epoch.
  unsigned getEpoch() const { return Epoch; }

  /// Bumps the epoch to \p NewEpoch, which must be higher than the current one.
  void bumpEpoch(unsigned NewEpoch) {
    assert(Epoch < NewEpoch && "epoch must increase");
    Epoch = NewEpoch;
  }

  bool operator<(const OptRegCandidate &Other) const;

#ifndef NDEBUG
  Printable print(const VirtRegMap &VRM) const;
#endif

private:
  /// Last epoch at which a re-assignment took place and during which the
  /// candidate was evaluated.
  unsigned Epoch = StartEpoch;
};

/// A minimal binary max-heap for \ref OptRegCandidate. All tree elements are
/// created at the beginning and never removed.
class MaxHeap {
public:
  /// Constructs the heap, creating a candidate for each unpinned optimizable
  /// register tracked by \p OptRegs.
  MaxHeap(OptimizableRegs &OptRegs);

  /// Returns the most profitable candidate, if any is profitable at all.
  OptRegCandidate *getMostProfitable() {
    if (HeapToSlot.empty())
      return nullptr;
    OptRegCandidate &TopCand = Slots[HeapToSlot.front()].Cand;
    return TopCand.isProfitable() ? &TopCand : nullptr;
  }

  /// If \p Reg is a candidate, update its score and re-sort the heap.
  void reorderIfExists(const OptReg &Reg);

  /// Returns whether the heap is empty.
  bool empty() const { return HeapToSlot.empty(); }

#ifndef NDEBUG
  Printable print(const VirtRegMap &VRM, const LiveIntervals &LIS) const;
#endif

private:
  /// A register candidate along with its current heap index.
  struct Slot {
    OptRegCandidate Cand;
    unsigned HeapIdx;
    Slot(OptReg &Reg, unsigned HeapIdx) : Cand(Reg), HeapIdx(HeapIdx) {}
  };

  /// Storage for register candidates under consideration.
  SmallVector<Slot> Slots;
  /// The heap itself: a complete binary tree, breadth-first, holding indices in
  /// \ref Slots. The first element is the maximum.
  SmallVector<unsigned> HeapToSlot;
  /// Maps registers with a corresponding candidate to the latter's index in
  /// \ref Slots.
  DenseMap<const OptReg *, unsigned> OptRegToSlotIdx;

  /// Places slot with index \p SlotIdx at heap position \p HeapIdx, keeping the
  /// reverse mapping in sync.
  void place(unsigned HeapIdx, unsigned SlotIdx) {
    HeapToSlot[HeapIdx] = SlotIdx;
    Slots[SlotIdx].HeapIdx = HeapIdx;
  }

  /// Sifts the element at \p HeapIdx towards the root while it outranks its
  /// parent. Returns true if it moved at all.
  bool siftUp(unsigned HeapIdx);

  /// Sifts the element at \p HeapIdx towards the leaves while it is outranged
  /// by its children.
  bool siftDown(unsigned HeapIdx);
};

/// Handles re-assignment of virtual registers to physical registers.
class VirtRegReMap {
public:
  VirtRegReMap(LiveRegMatrix &LRM, LiveIntervals &LIS,
               const MachineFunction &MF, const VirtRegMap &VRM);

  /// Attempts to find an available physical register in MSB group \p Dst that
  /// virtual register \p Reg can be assigned to. Returns the first such
  /// register it finds, or the sentinel register if none could be found.
  MCRegister tryAssignInGroup(Register Reg, MSBGroup Group);

  /// Attempts to re-assign virtual register \p Reg to an available physical
  /// register in any of the MSB groups indicated by set bits in \p Targets.
  /// Returns whether any re-assignment took place.
  bool tryAssignToTargetGroups(OptReg &Reg, const SmallBitVector &Targets);

private:
  LiveRegMatrix &LRM;
  LiveIntervals &LIS;
  RegisterClassInfo RCI;
  const VirtRegMap &VRM;
  const SIRegisterInfo &TRI;
  const BitVector ReservedRegs;
};

class AMDGPUOptimizeVGPREncoding {
public:
  AMDGPUOptimizeVGPREncoding(VirtRegMap &VRM, LiveIntervals &LIS,
                             LiveRegMatrix &LRM)
      : VRM(VRM), LIS(LIS), LRM(LRM){};

  bool run(MachineFunction &MF);

private:
  VirtRegMap &VRM;
  LiveIntervals &LIS;
  LiveRegMatrix &LRM;
};

} // namespace

#ifndef NDEBUG
static std::string printGroup(MSBGroup Group) {
  return "MSB#" + std::to_string(Group);
}

static std::string printOprdType(OprdType Oprd) {
  return "OpTy" + std::to_string(Oprd);
}
#endif

/// Returns \p Reg's MSB group. When \p Reg is virtual, \p VRM is used to
/// determine its current physical assignment.
static MSBGroup getVGPRGroup(Register Reg, const VirtRegMap &VRM) {
  const auto &TRI =
      *static_cast<const SIRegisterInfo *>(&VRM.getTargetRegInfo());
  MCRegister PhysReg = Reg.isVirtual() ? VRM.getPhys(Reg) : Reg.asMCReg();
  return TRI.getHWRegIndex(PhysReg) >> 8;
}

/// Returns true if \p RC is confined to the first 256 VGPRs i.e., every
/// register it contains has a hardware index below 256 (MSB group 0).
static bool isLo256VGPRClass(const TargetRegisterClass *RC,
                             const SIRegisterInfo &TRI) {
  const TargetRegisterClass *Lo256RC =
      TRI.getAlignedLo256VGPRClassForBitWidth(TRI.getRegSizeInBits(*RC));
  return (Lo256RC && Lo256RC->hasSubClassEq(RC)) ||
         Lo256RC == &AMDGPU::VS_32_Lo256RegClass ||
         Lo256RC == &AMDGPU::VS_64_Lo256RegClass;
}

MBBModeUsage::MBBModeUsage(const MachineBasicBlock &MBB, BitVector &OptVirtRegs,
                           BitVector &PinnedVirtRegs)
    : MBB(MBB) {

  const MachineFunction &MF = *MBB.getParent();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();

  std::array<unsigned, NumOprdTypes> PrevOprdIndex;
  PrevOprdIndex.fill(ModeInstr::NoIdx);

  // Identify instructions in the block for which VGPR MSBs are relevant.
  for (const MachineInstr &MI : MBB) {
    const MCInstrDesc &Desc = MI.getDesc();
    const auto [Table, VOPDTable] = AMDGPU::getVGPRLoweringOperandTables(Desc);
    if (!Table)
      continue;

    // Determines whether the MI's Name operand has its MSBs provided by MODE.
    // Returns the underlying machine operand (representing a VGPR) if it is
    // relevant, otherwise nullptr.
    auto GetRelevantMO =
        [&](const AMDGPU::OpName &Name) -> const MachineOperand * {
      if (Name == AMDGPU::OpName::NUM_OPERAND_NAMES)
        return nullptr;

      const MachineOperand *MO = TII.getNamedOperand(MI, Name);
      if (!MO || !MO->isReg())
        return nullptr;

      Register Reg = MO->getReg();
      const TargetRegisterClass *RC = TRI.getRegClassForReg(MRI, Reg);
      return (RC && SIRegisterInfo::isVGPRClass(RC)) ? MO : nullptr;
    };

    const unsigned InstrIdx = Instructions.size();

    // Sets the register as the operand of a particular type for the current MI.
    auto SetVGPROprd = [&](Register Reg, OprdType Oprd) -> void {
      // We need to create the MODE-using instruction if this is the first
      // MODE-using operand we are seeing for it.
      if (Instructions.size() == InstrIdx) {
        ModeInstr &Instr = Instructions.emplace_back();
        Instr.Prev = PrevOprdIndex;
        Instr.Oprds[Oprd] = Reg;
      } else {
        Instructions.back().Oprds[Oprd] = Reg;
      }

      // Update next index for all previous instructions that do not use that
      // operand type. The previous index for this operand type becomes ours for
      // further instructions.
      unsigned &PrevOprdIdx = PrevOprdIndex[Oprd];
      unsigned I = (PrevOprdIdx == ModeInstr::NoIdx) ? 0 : PrevOprdIdx;
      for (; I < InstrIdx; ++I)
        Instructions[I].Next[Oprd] = InstrIdx;
      PrevOprdIdx = InstrIdx;
    };

    for (OprdType Oprd : seq(NumOprdTypes)) {
      const MachineOperand *MO = GetRelevantMO(Table[Oprd]);
      if (!MO)
        continue;
      Register Reg = MO->getReg();

      // Tied src2 uses of VOP2 and 32-bit-encoded VOP3 only depend on the vdst
      // bit and are handled with the def, so they are not their own operand for
      // MSB group purposes.
      if (Table[Oprd] == AMDGPU::OpName::src2 && !MO->isDef() && MO->isTied() &&
          (SIInstrInfo::isVOP2(MI) ||
           (SIInstrInfo::isVOP3(MI) &&
            TII.hasVALU32BitEncoding(MI.getOpcode()))))
        continue;

      SetVGPROprd(Reg, Oprd);
      if (Reg.isPhysical())
        continue;

      const unsigned VirtRegIdx = Reg.virtRegIndex();
      OptVirtRegs.set(VirtRegIdx);

      // For VOPD instructions, we have to take into account that every operand
      // type would need to fall within the same group in both instructions.
      // This is outside current modelling capabilities so we just make those
      // registers pinned by default.
      if (VOPDTable || isLo256VGPRClass(TRI.getRegClassForReg(MRI, Reg), TRI))
        PinnedVirtRegs.set(VirtRegIdx);
    }

    // Pin MODE-using registers for VOPD instructions.
    if (!VOPDTable)
      continue;
    for (OprdType Oprd : seq(NumOprdTypes)) {
      const MachineOperand *MO = GetRelevantMO(Table[Oprd]);
      if (!MO)
        continue;
      Register Reg = MO->getReg();

      // In case the first operation did not use a MODE-using register for this
      // operand but the second operation does it is still useful to set the
      // operand for the current VOPD instruction. This lets other optimizable
      // registers know there is a pinned register in a particular MSB group at
      // this location.
      if (Instructions.size() == InstrIdx || !Instructions.back().Oprds[Oprd])
        SetVGPROprd(Reg, Oprd);

      if (Reg.isPhysical())
        continue;
      const unsigned VirtRegIdx = Reg.virtRegIndex();
      OptVirtRegs.set(VirtRegIdx);
      PinnedVirtRegs.set(VirtRegIdx);
    }
  }
}

template <bool UsePrev, bool SkipCurrent>
unsigned MBBModeUsage::getInstrIdxImpl(unsigned InstrIdx, OprdType Oprd) const {
  assert(InstrIdx < Instructions.size() && "out of bounds");
  const ModeInstr &Instr = Instructions[InstrIdx];
  if constexpr (!SkipCurrent) {
    if (Instr.Oprds[Oprd])
      return InstrIdx;
  }

  unsigned Idx;
  if constexpr (UsePrev)
    Idx = Instr.Prev[Oprd];
  else
    Idx = Instr.Next[Oprd];

  if (Idx == ModeInstr::NoIdx)
    return ModeInstr::NoIdx;
  assert(Instructions[Idx].Oprds[Oprd] && "reg must exist");
  return Idx;
}

OptReg::OptReg(Register VirtReg, const VirtRegMap &VRM)
    : VirtReg(VirtReg), MSB(getVGPRGroup(VirtReg, VRM)) {
  assert(VirtReg.isVirtual() && "optimizable register should be virtual");
  PinnedScore.fill(0);
  Score.fill(0);
}

SmallBitVector OptReg::getPinGroups() const {
  SmallBitVector Groups(NumMSBGroups);
  for (const auto [Group, Score] : enumerate(PinnedScore)) {
    if (Score != 0)
      Groups.set(Group);
  }
  return Groups;
}

void OptReg::pinMSBGroup() {
  if (IsPinned)
    return;

  IsPinned = true;
  for (const auto &[NeighborReg, NumOcc] : Neighbors) {
    assert(NeighborReg->Score[MSB] >= NumOcc);
    NeighborReg->Score[MSB] -= NumOcc;
    NeighborReg->PinnedScore[MSB] += NumOcc;
  }
}

void OptReg::recordNeighborOccurence(OptReg &NeighborReg) {
  assert(&NeighborReg != this && "cannot be neighbor with itself");
  ++Neighbors.insert({&NeighborReg, 0}).first->getSecond();
  if (NeighborReg.isPinned())
    ++PinnedScore[NeighborReg.MSB];
  else
    ++Score[NeighborReg.MSB];
}

void OptReg::removeNeighborOccurence(OptReg &NeighborReg) {
  // Update neighbors.
  auto Neighbor = Neighbors.find(&NeighborReg);
  assert(Neighbor != Neighbors.end() && "neighbor must exist");
  if (--Neighbor->getSecond() == 0)
    Neighbors.erase(&NeighborReg);

  // Update score.
  if (NeighborReg.isPinned()) {
    assert(PinnedScore[NeighborReg.MSB] > 0 && "underflow");
    --PinnedScore[NeighborReg.MSB];
  } else {
    assert(Score[NeighborReg.MSB] > 0 && "underflow");
    --Score[NeighborReg.MSB];
  }
}

void OptReg::recordPhysNeighborOccurence(Register PhysReg,
                                         const VirtRegMap &VRM) {
  assert(PhysReg.isPhysical() && "must be physical register");
  ++PinnedScore[getVGPRGroup(PhysReg, VRM)];
}

void OptReg::removePhysNeighborOccurence(Register PhysReg,
                                         const VirtRegMap &VRM) {
  assert(PhysReg.isPhysical() && "must be physical register");
  MSBGroup PhysGroup = getVGPRGroup(PhysReg, VRM);
  assert(PinnedScore[PhysGroup] > 0 && "underflow");
  --PinnedScore[PhysGroup];
}

void OptReg::notifyPhysAssignmentChanged(MSBGroup NewGroup) {
  if (MSB == NewGroup)
    return;
  assert(!IsPinned && "pinned register cannot change MSB group");

  for (const auto &[NeighborReg, NumOcc] : Neighbors) {
    assert(NeighborReg->Score[MSB] >= NumOcc);
    NeighborReg->Score[MSB] -= NumOcc;
    NeighborReg->Score[NewGroup] += NumOcc;
  }
  MSB = NewGroup;
}

OptimizableRegs::OptimizableRegs(const BitVector &OptVirtRegs,
                                 const BitVector &PinnedVirtRegs,
                                 ArrayRef<MBBModeUsage> ModeUsage,
                                 const VirtRegMap &VRM)
    : OptVirtRegs(OptVirtRegs), VirtRegToStorageIdx(OptVirtRegs.size(), NoIdx) {
  assert(OptVirtRegs.size() == PinnedVirtRegs.size() &&
         "inconsistent bitvector sizes");

  // Create initial tracking data for all optimizable virtual registers.
  Storage.reserve(OptVirtRegs.size());
  for (unsigned VirtRegIdx : OptVirtRegs.set_bits()) {
    VirtRegToStorageIdx[VirtRegIdx] = Storage.size();

    Register VirtReg = Register::index2VirtReg(VirtRegIdx);
    OptReg &Reg = Storage.emplace_back(VirtReg, VRM);
    if (PinnedVirtRegs.test(VirtRegIdx))
      Reg.pinMSBGroup();
  }

  // Idnetify the neighborhood and occurences of each register. This initializes
  // the score of all optimizable registers.
  for (const auto &[MBBIdx, BlockUsage] : enumerate(ModeUsage)) {
    ArrayRef<ModeInstr> Instructions = BlockUsage.getInstructions();
    if (Instructions.empty())
      continue;

    for (OprdType Oprd : seq(NumOprdTypes)) {
      OptReg *PreviousOptReg = nullptr;
      unsigned InstrIdx = BlockUsage.getFirstOprd(Oprd);

      while (InstrIdx != ModeInstr::NoIdx) {
        const ModeInstr &CurrentInstr = Instructions[InstrIdx];
        Register Reg = CurrentInstr.Oprds[Oprd];

        OptReg *CurrentOptReg = (*this)[Reg];
        if (CurrentOptReg) {
          assert(Reg.isVirtual() && "only virtregs are optimizable");
          CurrentOptReg->addOccurence(MBBIdx, InstrIdx, Oprd);

          if (!PreviousOptReg) {
            unsigned PrevIdx = CurrentInstr.Prev[Oprd];
            if (PrevIdx == ModeInstr::NoIdx) {
              // This is the first operand of that type in the block. Every
              // block starts with all operand types in the default group. We
              // model this by incrementing the default's group pinned score for
              // the first register.
              CurrentOptReg->recordBlockBoundaryPin();
            } else {
              // The register immediately preceeding this was a physical one.
              Register PhysReg = Instructions[PrevIdx].Oprds[Oprd];
              CurrentOptReg->recordPhysNeighborOccurence(PhysReg, VRM);
            }
          } else if (CurrentOptReg != PreviousOptReg) {
            // The two virtual registers are neighbors.
            CurrentOptReg->recordNeighborOccurence(*PreviousOptReg);
            PreviousOptReg->recordNeighborOccurence(*CurrentOptReg);
          }
        } else if (PreviousOptReg) {
          // We have a virtual register followed by a physical one on the same
          // operand stream. We just need to update the former's pin score.
          PreviousOptReg->recordPhysNeighborOccurence(Reg, VRM);
          PreviousOptReg = nullptr;
        }

        PreviousOptReg = CurrentOptReg;
        InstrIdx = CurrentInstr.Next[Oprd];
      }

      if (PreviousOptReg) {
        // This is the last operand of that type in the block. Every block
        // ends with all operand types in the default group. We model this
        // by incrementing the default's group pinned score for the last
        // register.
        PreviousOptReg->recordBlockBoundaryPin();
      }
    }
  }
}

ModeSetOptimizer::ModeSetOptimizer(OptimizableRegs &OptRegs,
                                   ArrayRef<MBBModeUsage> ModeUsage,
                                   const VirtRegMap &VRM)
    : OptRegs(OptRegs), CheckShouldBePinned(OptRegs.getAllOptVirtRegs()),
      CheckResolveConflict(OptRegs.getNumVirtRegs()),
      ModeSetPlacement(ModeUsage.size()), ModeUsage(ModeUsage), VRM(VRM) {
  for (const auto &[MBB, Placement] : zip_equal(ModeUsage, ModeSetPlacement))
    Placement.resize(MBB.getInstructions().size() + 1);

  // We initially check all registers for pin-eligibility, and those that are
  // neighbors of already pinned registers for conflicting pins.
  for (const OptReg &Reg : OptRegs) {
    if (!Reg.isPinned())
      continue;
    for (const auto &[NeighborReg, _] : Reg.getNeighbors())
      CheckResolveConflict.set(NeighborReg->getVirtIndex());
  }
  resolveConflictsAndPinRegs();
}

MSBGroup ModeSetOptimizer::selectPreferredMSBGroup(const OptReg &Reg) const {
  // Initially favor the current group the register is in. This is guaranteed to
  // change if the register has no pinned neighbors in this MSB group.
  MSBGroup BestGroup = Reg.getMSB();
  unsigned MaxPinnedScore = Reg.getGroupPinnedScore(BestGroup),
           MaxTotalScore = Reg.getGroupScore(BestGroup);

  for (MSBGroup Group : seq(NumMSBGroups)) {
    if (Group == Reg.getMSB())
      continue;

    // Select the MSB group with the highest number of pinned neighbors.
    unsigned PinnedScore = Reg.getGroupPinnedScore(Group);
    if (PinnedScore < MaxPinnedScore)
      continue;

    // Ammong MSB groups with the same number of pinned neighbors, favor the one
    // with highest number of unpinned neighbors.
    unsigned TotalScore = Reg.getGroupScore(Group);
    if (PinnedScore == MaxPinnedScore && MaxTotalScore > TotalScore)
      continue;

    BestGroup = Group;
    MaxPinnedScore = PinnedScore;
    MaxTotalScore = TotalScore;
  }
  return BestGroup;
}

void ModeSetOptimizer::placeJustBefore(unsigned MBBIdx, unsigned InstrIdx) {
  const MBBModeUsage &MBB = ModeUsage[MBBIdx];
  for (OprdType Oprd : seq(NumOprdTypes)) {
    breakNeighborRelationship(MBBIdx, MBB.getLastInstrBefore(InstrIdx, Oprd),
                              MBB.getFirstInstrFrom(InstrIdx, Oprd), Oprd);
  }
  ModeSetPlacement[MBBIdx].set(InstrIdx);
}

void ModeSetOptimizer::placeJustAfter(unsigned MBBIdx, unsigned InstrIdx) {
  const MBBModeUsage &MBB = ModeUsage[MBBIdx];
  for (OprdType Oprd : seq(NumOprdTypes)) {
    breakNeighborRelationship(MBBIdx, MBB.getLastInstrUntil(InstrIdx, Oprd),
                              MBB.getFirstInstrAfter(InstrIdx, Oprd), Oprd);
  }
  ModeSetPlacement[MBBIdx].set(InstrIdx + 1);
}

void ModeSetOptimizer::breakNeighborRelationship(unsigned MBBIdx,
                                                 unsigned AfterIdx,
                                                 unsigned BeforeIdx,
                                                 OprdType Oprd) {
  if (AfterIdx == ModeInstr::NoIdx && BeforeIdx == ModeInstr::NoIdx)
    return;

  ArrayRef<ModeInstr> Instructions = ModeUsage[MBBIdx].getInstructions();
  if (AfterIdx == ModeInstr::NoIdx) {
    // This may break the relationship between an optimizable register and the
    // entry block pin.
    Register BeforeReg = Instructions[BeforeIdx].Oprds[Oprd];
    if (OptRegs[BeforeReg] && !hasModeSetBefore(MBBIdx, BeforeIdx))
      OptRegs[BeforeReg]->removeBlockBoundaryPin();
    return;
  }
  if (BeforeIdx == ModeInstr::NoIdx) {
    // This may break the relationship between an optimizable register and the
    // exit block pin.
    Register AfterReg = Instructions[AfterIdx].Oprds[Oprd];
    if (OptRegs[AfterReg] && !hasModeSetAfter(MBBIdx, AfterIdx))
      OptRegs[AfterReg]->removeBlockBoundaryPin();
    return;
  }
  assert(AfterIdx < BeforeIdx && "incoherent indices");

  // The two neighbor registers on that operand lane are already not neighbors
  // if there is a MODE-setting instruction between them.
  if (hasModeSetBetween(MBBIdx, AfterIdx, BeforeIdx))
    return;

  // A register is never considered a neighbor to itself.
  Register AfterReg = Instructions[AfterIdx].Oprds[Oprd];
  Register BeforeReg = Instructions[BeforeIdx].Oprds[Oprd];
  assert(AfterReg && BeforeReg && "register operands must exist");
  if (AfterReg == BeforeReg)
    return;

  // Notify registers that one occurence of their neighborhood relationship is
  // broken. Optimizable registers which have their score affected by the
  // break may become pinnable.
  OptReg *AferOptReg = OptRegs[AfterReg];
  OptReg *BeforeOptReg = OptRegs[BeforeReg];
  if (AferOptReg && BeforeOptReg) {
    AferOptReg->removeNeighborOccurence(*BeforeOptReg);
    BeforeOptReg->removeNeighborOccurence(*AferOptReg);
    CheckShouldBePinned.set(AfterReg.virtRegIndex());
    CheckShouldBePinned.set(BeforeReg.virtRegIndex());
  } else if (AferOptReg) {
    AferOptReg->removePhysNeighborOccurence(BeforeReg, VRM);
    CheckShouldBePinned.set(AfterReg.virtRegIndex());
  } else if (BeforeOptReg) {
    BeforeOptReg->removePhysNeighborOccurence(AfterReg, VRM);
    CheckShouldBePinned.set(BeforeReg.virtRegIndex());
  }
}

bool ModeSetOptimizer::resolveConflictingPins(OptReg &Reg) {
  if (Reg.getPinGroups().count() <= 1)
    return false;

  LLVM_DEBUG(dbgs() << "  Resolving conflicts for " << Reg.print(VRM) << '\n');

  // Only one MSB group with pinned neighbors must remain.
  MSBGroup PreferredGroup = selectPreferredMSBGroup(Reg);
  assert(PreferredGroup < NumMSBGroups && "invalid group");

  LLVM_DEBUG(dbgs() << "    Preferred group is " << printGroup(PreferredGroup)
                    << '\n');

  // Conflicts with block boundary pins may need to be resolved when the default
  // group is not the preferred one.
  if (PreferredGroup != DefaultGroup &&
      Reg.getGroupPinnedScore(DefaultGroup) > 0) {
    for (const auto &[MBBIdx, InstrIdx, Oprd] : Reg.getOccurences()) {
      const MBBModeUsage &MBB = ModeUsage[MBBIdx];
      if (InstrIdx == MBB.getFirstOprd(Oprd) &&
          !hasModeSetBefore(MBBIdx, InstrIdx)) {
        LLVM_DEBUG(dbgs() << "    Resolving default entry pin "
                          << printOprdType(Oprd) << " in MBB#"
                          << MBB.MBB.getNumber() << '\n');
        placeJustBefore(MBBIdx, InstrIdx);
      }
      if (InstrIdx == MBB.getLastOprd(Oprd) &&
          !hasModeSetAfter(MBBIdx, InstrIdx)) {
        LLVM_DEBUG(dbgs() << "    Resolving default exit pin "
                          << printOprdType(Oprd) << " in MBB#"
                          << MBB.MBB.getNumber() << '\n');
        placeJustAfter(MBBIdx, InstrIdx);
      }
    }
  }

  // Conflicts with occurences of pinned neighbors in the non-preferred MSB
  // group need to be resolved. Iterate over a copy of the list of neighbors
  // because they will be modified as we simulate placement of MODE-setting
  // instructions.
  OptReg::WeightedNeighbors Neighbors(Reg.getNeighbors());
  for (const auto &[NeighborReg, _] : Neighbors) {
    // Early exit when we know we are not gonna find conflicting occurences.
    if (!NeighborReg->isPinned() || NeighborReg->getMSB() == PreferredGroup ||
        Reg.getGroupPinnedScore(NeighborReg->getMSB()) == 0)
      continue;

    // Look through occurences for conflicts.
    for (const auto &[MBBIdx, InstrIdx, Oprd] : NeighborReg->getOccurences()) {
      const MBBModeUsage &MBB = ModeUsage[MBBIdx];
      ArrayRef<ModeInstr> Instructions = MBB.getInstructions();

      // Look at the operand immediately before this neighbor occurence. If it
      // matches the register for which we are currently resolving conflicts,
      // then it is one of the neighborhood relationship to break.
      unsigned PrevIdx = MBB.getLastInstrBefore(InstrIdx, Oprd);
      if (PrevIdx != ModeInstr::NoIdx &&
          Instructions[PrevIdx].Oprds[Oprd] == Reg.getVirt() &&
          !hasModeSetBetween(MBBIdx, PrevIdx, InstrIdx)) {

        LLVM_DEBUG(dbgs() << "    Resolving with next neighbor "
                          << NeighborReg->print(VRM) << " for operand type "
                          << printOprdType(Oprd) << " in MBB#"
                          << MBB.MBB.getNumber() << '\n');
        placeJustBefore(MBBIdx, InstrIdx);
      }

      // Look at the operand immediately after this neighbor occurence. If it
      // matches the register for which we are currently resolving conflicts,
      // then it is one of the neighborhood relationship to break.
      unsigned NextIdx = MBB.getFirstInstrAfter(InstrIdx, Oprd);
      if (NextIdx != ModeInstr::NoIdx &&
          Instructions[NextIdx].Oprds[Oprd] == Reg.getVirt() &&
          !hasModeSetBetween(MBBIdx, InstrIdx, NextIdx)) {
        LLVM_DEBUG(dbgs() << "    Resolving with previous neighbor "
                          << NeighborReg->print(VRM) << " for operand type "
                          << printOprdType(Oprd) << " in MBB#"
                          << MBB.MBB.getNumber() << '\n');
        placeJustBefore(MBBIdx, NextIdx);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  | Updated register: " << Reg.print(VRM) << '\n');

  // It is possible that the number of pinned groups was reduced to zero when
  // all of the preferred group's pins were close to other groups' respective
  // pins.
  assert(Reg.getPinGroups().count() <= 1 && "at most one pin");
  return true;
}

bool ModeSetOptimizer::shouldBePinned(const OptReg &Reg) const {
  for (MSBGroup Group : seq(NumMSBGroups)) {
    if (Reg.getGroupScore(Group) == 0)
      continue;
    if (Group != Reg.getMSB())
      return false;
  }
  return true;
}

bool ModeSetOptimizer::pinIfEligible(OptReg &Reg) {
  if (Reg.isPinned() || !shouldBePinned(Reg))
    return false;
  Reg.pinMSBGroup();
  LLVM_DEBUG(dbgs() << "  Pinned " << Reg.print(VRM) << " to "
                    << printGroup(Reg.getMSB()) << '\n');
  return true;
}

void ModeSetOptimizer::regChangedGroup(OptReg &Reg, BitVector &ScoreChanged) {
  // The candidate register and its neighbors had their score changed; they may
  // also be eligible for pinning.
  pinIfEligible(Reg);
  ScoreChanged.set(Reg.getVirtIndex());

  for (const auto &[NeighborReg, _] : Reg.getNeighbors()) {
    ScoreChanged.set(NeighborReg->getVirtIndex());
    if (!pinIfEligible(*NeighborReg))
      continue;
    // Pinning a register impacts its neighbors' respective score. However, they
    // cannot become newly pinnable themselves because pin-eligibility is
    // independent of whether score contributions come from pinned or unpinned
    // neighbors.
    for (const auto &[SecondDegNeighbor, _] : NeighborReg->getNeighbors())
      ScoreChanged.set(SecondDegNeighbor->getVirtIndex());
  }

  // A score change may mean the register now has a conflict.
  CheckResolveConflict |= ScoreChanged;
}

bool ModeSetOptimizer::resolveConflictsAndPinRegs() {
  LLVM_DEBUG(dbgs() << "* Resolving conflicts and pins:\n");
  bool AnyChange = false;
  do {
    // Conflict resolution.
    for (unsigned VirtRegIdx : CheckResolveConflict.set_bits())
      AnyChange |= resolveConflictingPins(OptRegs[VirtRegIdx]);
    CheckResolveConflict.reset();

    // Pin eligible registers.
    for (unsigned VirtRegIdx : CheckShouldBePinned.set_bits()) {
      OptReg &Reg = OptRegs[VirtRegIdx];
      if (!pinIfEligible(Reg))
        continue;
      AnyChange = true;

      // Neighbors of registers which became pinned may now have conflicts.
      for (const auto &[NeighborReg, _] : Reg.getNeighbors())
        CheckResolveConflict.set(NeighborReg->getVirtIndex());
    }
    CheckShouldBePinned.reset();
  } while (CheckResolveConflict.any());
  return AnyChange;
}

bool OptRegCandidate::operator<(const OptRegCandidate &Other) const {
  // Registers without targets are not useful candidates. Pinned registers have
  // no targets by construction so they get caught here as well.
  if (Other.Targets.none())
    return Targets.none();
  if (Targets.none())
    return true;

  // Earlier/Smaller epoch wins.
  if (Epoch != Other.Epoch)
    return Other.Epoch < Epoch;

  // Bigger pin-state wins. This generally favors MSB group changes toward
  // groups with bigger number of pinned neighbors in.
  if (NeighboringPins != Other.NeighboringPins)
    return NeighboringPins < Other.NeighboringPins;

  // Higher benefit wins.
  if (Benefit < Other.Benefit)
    return Benefit < Other.Benefit;

  // Break ties with unique virtual register index.
  return Reg.getVirtIndex() < Other.Reg.getVirtIndex();
}

void OptRegCandidate::update() {
  Targets.reset();
  if (Reg.isPinned())
    return;

  SmallBitVector PinGroups = Reg.getPinGroups();
  if (PinGroups.count() > 1) {
    // Registers with unresolved conflicts should not be re-assigned until we
    // have resolved them.
    return;
  }
  const unsigned CurrentScore = Reg.getCurrentGroupScore();

  // The overall benefit of moving MSB group reflects the distribution of scores
  // over all MSB groups. We want to favor registers with highly unbalanced
  // per-group score, at the extreme registers whose neighborhood is entirely
  // assigned to a single MSB group. On the contrary, registers whose
  // neighborhood is roughly evenly split between all MSB groups are not that
  // profitable to re-assign; even if a MSB group with higher score than the
  // current one the register is in exists.
  Benefit = CurrentScore;
  unsigned MaxGroupScore = CurrentScore;
  for (MSBGroup Group : seq(NumMSBGroups)) {
    if (Group == Reg.getMSB())
      continue;
    const unsigned GroupScore = Reg.getGroupScore(Group);
    if (GroupScore > MaxGroupScore) {
      // The current best groups' score was counted positively once before, but
      // we now want to count it negatively.
      Benefit -= MaxGroupScore * 2;
      // The new highest group score is counted positively.
      Benefit += GroupScore;
      // This group becomes the only target.
      Targets.reset();
      Targets.set(Group);
      MaxGroupScore = GroupScore;
      continue;
    }
    // GroupScore <= MaxGroupScore

    Benefit -= GroupScore;
    // This group is a good target only when it has the highest score we have
    // seen so far and that score is above the current group's score.
    if (GroupScore != CurrentScore && GroupScore == MaxGroupScore)
      Targets.set(Group);
  }
  assert(!Targets.test(Reg.getMSB()) && "current group cannot be target");
  if (Targets.none())
    return;

  // We want to favor registers which would benefit from being re-assigned to
  // the single MSB group in which they haved pinned neighbors (if there is
  // one), as this is likely to resolve conflicts in the future without having
  // to place MODE-setting instructions.
  if (PinGroups.none()) {
    NeighboringPins = PinState::NoPin;
  } else {
    const MSBGroup SinglePin = PinGroups.find_first();
    NeighboringPins = (SinglePin == Reg.getMSB() || !Targets.test(SinglePin))
                          ? PinState::PinInBadGroup
                          : PinState::TargetIsPin;
  }
}

MaxHeap::MaxHeap(OptimizableRegs &OptRegs) {
  for (OptReg &Reg : OptRegs) {
    if (Reg.isPinned())
      continue;

    const unsigned SlotIdx = Slots.size();
    const unsigned HeapIdx = HeapToSlot.size();
    Slots.emplace_back(Reg, HeapIdx);
    HeapToSlot.push_back(SlotIdx);
    OptRegToSlotIdx.insert({&Reg, SlotIdx});
    siftUp(HeapIdx);
  }
}

void MaxHeap::reorderIfExists(const OptReg &Reg) {
  auto Cand = OptRegToSlotIdx.find(&Reg);
  if (Cand == OptRegToSlotIdx.end())
    return;
  Slot &S = Slots[Cand->second];
  S.Cand.update();

  // Re-order the tree around the updated slot.
  if (!siftUp(S.HeapIdx))
    siftDown(S.HeapIdx);
}

bool MaxHeap::siftUp(unsigned HeapIdx) {
  unsigned S = HeapToSlot[HeapIdx];
  bool Moved = false;
  while (HeapIdx != 0) {
    unsigned Parent = (HeapIdx - 1) / 2;
    if (!(Slots[HeapToSlot[Parent]].Cand < Slots[S].Cand))
      break;
    place(HeapIdx, HeapToSlot[Parent]);
    HeapIdx = Parent;
    Moved = true;
  }
  if (Moved)
    place(HeapIdx, S);
  return Moved;
}

bool MaxHeap::siftDown(unsigned HeapIdx) {
  unsigned S = HeapToSlot[HeapIdx];
  unsigned N = HeapToSlot.size();
  bool Moved = false;
  while (true) {
    unsigned Left = 2 * HeapIdx + 1;
    unsigned Right = Left + 1;
    unsigned Largest = HeapIdx;
    if (Left < N && Slots[S].Cand < Slots[HeapToSlot[Left]].Cand)
      Largest = Left;
    if (Right < N &&
        Slots[HeapToSlot[Largest]].Cand < Slots[HeapToSlot[Right]].Cand)
      Largest = Right;
    if (Largest == HeapIdx)
      break;
    place(HeapIdx, HeapToSlot[Largest]);
    HeapIdx = Largest;
    Moved = true;
  }
  if (Moved)
    place(HeapIdx, S);
  return Moved;
}

VirtRegReMap::VirtRegReMap(LiveRegMatrix &LRM, LiveIntervals &LIS,
                           const MachineFunction &MF, const VirtRegMap &VRM)
    : LRM(LRM), LIS(LIS), VRM(VRM),
      TRI(*static_cast<const SIRegisterInfo *>(&VRM.getTargetRegInfo())),
      ReservedRegs(TRI.getReservedRegs(MF)) {
  RCI.runOnMachineFunction(MF);
}

static bool overflowsLastGroup(unsigned HWRegIdx, unsigned NumLanes,
                               MSBGroup Group) {
  if (Group != NumMSBGroups - 1)
    return false;
  return ((HWRegIdx + NumLanes - 1) >> 8) != NumMSBGroups - 1;
}

MCRegister VirtRegReMap::tryAssignInGroup(Register VirtReg, MSBGroup Group) {
  assert(VirtReg.isVirtual() && "expected virtreg");
  const LiveInterval &RegLI = LIS.getInterval(VirtReg);
  const MachineRegisterInfo &MRI = VRM.getRegInfo();
  const TargetRegisterClass &RC = *MRI.getRegClass(VirtReg);
  const unsigned NumLanes = divideCeil(TRI.getRegSizeInBits(RC), 32);
  for (MCPhysReg CandPhysReg : RCI.getOrder(&RC)) {
    if (ReservedRegs[CandPhysReg] || getVGPRGroup(CandPhysReg, VRM) != Group ||
        overflowsLastGroup(TRI.getHWRegIndex(CandPhysReg), NumLanes, Group))
      continue;
    if (LRM.checkInterference(RegLI, CandPhysReg) == LiveRegMatrix::IK_Free)
      return CandPhysReg;
  }
  return MCPhysReg();
}

bool VirtRegReMap::tryAssignToTargetGroups(OptReg &Reg,
                                           const SmallBitVector &Targets) {
  Register VirtReg = Reg.getVirt();
  const LiveInterval &LI = LIS.getInterval(VirtReg);

  MCRegister OriginalPhys = VRM.getPhys(VirtReg);
  LRM.unassign(LI);

  for (MSBGroup Target : Targets.set_bits()) {
    assert(Target != Reg.getMSB() && "target is current MSB group");
    MCRegister NewPhysReg = tryAssignInGroup(VirtReg, Target);
    if (NewPhysReg) {
      LRM.assign(LI, NewPhysReg);
      Reg.notifyPhysAssignmentChanged(Target);
      return true;
    }
  }
  // We failed to find a register, re-assign the original one.
  LRM.assign(LI, OriginalPhys);
  return false;
}

bool AMDGPUOptimizeVGPREncoding::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs())
    return false;

  LLVM_DEBUG(dbgs() << "*** AMDGPUOptimizeVGPREncoding on " << MF.getName()
                    << " ***\n");
  const MachineRegisterInfo &MRI = VRM.getRegInfo();

  // Analyze MODE-usage in each block, keeping track of which virtual registers
  // are used in relevant instruction operands.
  BitVector OptVirtRegs(MRI.getNumVirtRegs()), Pinned(MRI.getNumVirtRegs());
  SmallVector<MBBModeUsage> ModeUsage;
  ModeUsage.reserve(MF.getNumBlockIDs());
  for (const MachineBasicBlock &MBB : MF)
    ModeUsage.emplace_back(MBB, OptVirtRegs, Pinned);

  // Initialize all optimizable registers.
  OptimizableRegs OptRegs(OptVirtRegs, Pinned, ModeUsage, VRM);

  LLVM_DEBUG({
    dbgs() << "* Per-block MODE-usage:\n";
    for (const MBBModeUsage &MBB : ModeUsage) {
      dbgs() << "  MBB #" << MBB.MBB.getNumber() << ":\n" << MBB.print(VRM);
    }
    dbgs() << "* Neighborhoods:\n";
    for (const OptReg &Reg : OptRegs) {
      dbgs() << "  " << Reg.print(VRM) << '\n';
      for (const auto &[NeighborReg, NumOcc] : Reg.getNeighbors()) {
        dbgs() << "    [" << NumOcc << "] " << NeighborReg->print(VRM) << '\n';
      }
    }
  });

  ModeSetOptimizer Optimizer(OptRegs, ModeUsage, VRM);
  VirtRegReMap VRRM(LRM, LIS, MF, VRM);

  bool Changed = false;
  do {
    // Constructs a max-heap with all remaining unpinned registers that are
    // candidates for re-assignment.
    //
    /// FIXME: There is no need to re-construct the heap every time, we can just
    /// let pin registers fall to the bottom of it since they have no target MSB
    /// group by construction.
    MaxHeap AllCandidates(OptRegs);
    LLVM_DEBUG(dbgs() << AllCandidates.print(VRM, LIS));

    // Incremented at each successful register re-assignment, which may unblock
    // previously failed attempts at re-assigning.
    unsigned Epoch = OptRegCandidate::StartEpoch + 1;

    // Keeps track of which optimizable virtual registers have their score
    // changed by re-assignments.
    BitVector ScoreChanged(OptRegs.getNumVirtRegs());

    // Only accepting profitable candidates guarantees forward progress
    // because re-assigning to target groups increase the combined score, which
    // is upper-bounded by the number of neighboring relationships between all
    // registers (itself only decreasing after initialization).
    while (OptRegCandidate *Candidate = AllCandidates.getMostProfitable()) {
      LLVM_DEBUG({
        dbgs() << "| Attempting re-assignment of " << Candidate->print(VRM)
               << '\n';
        for (const auto &[NeighborReg, _] : Candidate->Reg.getNeighbors())
          dbgs() << "    " << NeighborReg->print(VRM) << '\n';
      });

      // The epoch check catches a second evaluation of the same candidate under
      // the same exact conditions, which would fail again.
      if (Candidate->getEpoch() == Epoch) {
        LLVM_DEBUG(dbgs() << "  | No more useful candidates!\n");
        break;
      }
      Candidate->bumpEpoch(Epoch);
      ScoreChanged.reset();

      // The candidate's epoch changed.
      ScoreChanged.set(Candidate->Reg.getVirtIndex());

      // Attempts re-assignment to a better MSB group.
      if (VRRM.tryAssignToTargetGroups(Candidate->Reg, Candidate->Targets)) {
        LLVM_DEBUG(dbgs() << "  | SUCCESS: Assigned to free physical register "
                          << VRM.getPhys(Candidate->Reg.getVirt()) << '\n');
        ++Epoch;
        Optimizer.regChangedGroup(Candidate->Reg, ScoreChanged);
      }

      // Optimizable registers whose score changed need to be re-ordered within
      // the max-heap.
      for (unsigned VirtRegIdx : ScoreChanged.set_bits())
        AllCandidates.reorderIfExists(OptRegs[VirtRegIdx]);
    }
    Changed |= (Epoch != OptRegCandidate::StartEpoch + 1);
  } while (Optimizer.resolveConflictsAndPinRegs());

  return Changed;
}

#ifndef NDEBUG

Printable MBBModeUsage::print(const VirtRegMap &VRM) const {
  // 16 characters is enough for a virtual register with 9 digits, or to
  // display the beginning of a wide physical register.
  static constexpr unsigned ColumnWidth = 16;

  return Printable([&](raw_ostream &OS) {
    const MachineRegisterInfo &MRI = VRM.getRegInfo();

    std::array<std::optional<MSBGroup>, NumOprdTypes> GroupPerOprd;
    GroupPerOprd.fill(0);

    // Display operand type names.
    OS << "    [  ";
    for (OprdType Oprd : seq(NumOprdTypes)) {
      SmallString<ColumnWidth> Buf;
      raw_svector_ostream OprdStream(Buf);
      OprdStream << printOprdType(Oprd);
      StringRef OprdStr = OprdStream.str();
      OprdStr = OprdStr.take_front(ColumnWidth - 2);
      OS << OprdStr << indent(ColumnWidth - OprdStr.size());
    }
    OS << "  ]\n";

    for (const ModeInstr &Instr : Instructions) {
      OS << "    [  ";
      for (const auto &[Oprd, Reg] : enumerate(Instr.Oprds)) {
        if (!Reg) {
          OS << '|' << indent(ColumnWidth - 1);
          // No operand of that type for that instruction means no MSB
          // requirement.
          continue;
        }

        SmallString<ColumnWidth> Buf;
        raw_svector_ostream RegStream(Buf);
        RegStream << printReg(Reg, &VRM.getTargetRegInfo(), 0, &MRI);
        StringRef RegStr = RegStream.str();
        RegStr = RegStr.take_front(ColumnWidth);
        OS << RegStr << indent(ColumnWidth - RegStr.size());

        MSBGroup OprdGroup = getVGPRGroup(Reg, VRM);
        std::optional<MSBGroup> &CurrentOprdGroup = GroupPerOprd[Oprd];
        if (CurrentOprdGroup.has_value()) {
          if (*CurrentOprdGroup != OprdGroup) {
            // Different group as previously used for that operand.
            CurrentOprdGroup = OprdGroup;
          }
        } else {
          // First operand in the sequence. This defines the initial desired
          // group for that operand in the instruction sequence, so we do not
          // account for the set MSB here.
          CurrentOprdGroup = OprdGroup;
        }
      }
      OS << "  ]\n";
    }
  });
}

Printable OptReg::print(const VirtRegMap &VRM) const {
  return Printable([&](raw_ostream &OS) {
    const auto &TRI =
        *static_cast<const SIRegisterInfo *>(&VRM.getTargetRegInfo());
    const MachineRegisterInfo &MRI = VRM.getRegInfo();
    OS << printReg(VirtReg, &TRI, 0, &MRI) << '/'
       << printReg(VRM.getPhys(VirtReg), &TRI, 0, &MRI) << '/'
       << printGroup(getVGPRGroup(VirtReg, VRM));
    if (isPinned())
      OS << " (pinned)";
    OS << " [";
    for (MSBGroup Group : seq(NumMSBGroups)) {
      OS << (Group == MSB ? '+' : '-') << Score[Group] << '('
         << PinnedScore[Group] << ')';
      if (Group != NumMSBGroups - 1)
        OS << ", ";
    }
    OS << ']';
  });
}

Printable OptRegCandidate::print(const VirtRegMap &VRM) const {
  return Printable([&](raw_ostream &OS) {
    OS << Reg.print(VRM);
    if (Targets.none()) {
      OS << " has no targets";
      return;
    }

    auto PrintMoveScore = [&](MSBGroup MSB) {
      OS << printGroup(MSB) << "(+"
         << Reg.getGroupScore(MSB) - Reg.getCurrentGroupScore() << ')';
    };

    OS << " move to ";
    for (MSBGroup Dst : drop_end(Targets.set_bits())) {
      PrintMoveScore(Dst);
      OS << " / ";
    }
    PrintMoveScore(Targets.find_last());
    OS << " with benefit " << Benefit;

    OS << " (";
    switch (NeighboringPins) {
    case PinState::PinInBadGroup:
      OS << "bad pin";
      break;
    case PinState::NoPin:
      OS << "no pin";
      break;
    case PinState::TargetIsPin:
      OS << "move to pin";
      break;
    }
    OS << ") @ " << "epoch " << Epoch;
  });
}

Printable MaxHeap::print(const VirtRegMap &VRM,
                         const LiveIntervals &LIS) const {
  return Printable([&](raw_ostream &OS) {
    auto PrintRegsList = [&](ArrayRef<const OptRegCandidate *> Cands,
                             StringRef Name) {
      OS << "      " << Name << " (" << Cands.size() << " registers):\n";
      for (const OptRegCandidate *Cand : Cands)
        OS << "        " << Cand->print(VRM) << '\n';
    };

    auto PrintRegs = [&](ArrayRef<const OptRegCandidate *> Cands) -> void {
      using PinState = OptRegCandidate::PinState;
      // Group candidates per pin state.
      SmallDenseMap<PinState, SmallVector<const OptRegCandidate *>, 4>
          CandsPerPinState;
      SmallVector<const OptRegCandidate *> NoTargets;
      for (const OptRegCandidate *Cand : Cands) {
        if (Cand->Targets.any())
          CandsPerPinState[Cand->NeighboringPins].push_back(Cand);
        else
          NoTargets.push_back(Cand);
      }
      PrintRegsList(CandsPerPinState[PinState::TargetIsPin], "Good pin");
      PrintRegsList(CandsPerPinState[PinState::NoPin], "No pin");
      PrintRegsList(CandsPerPinState[PinState::PinInBadGroup], "Bad pin");
      PrintRegsList(NoTargets, "No target");
    };

    // Group candidates per MBB for displaying.
    DenseMap<const MachineBasicBlock *, SmallVector<const OptRegCandidate *>>
        CandsPerMBB;
    for (const Slot &S : Slots) {
      const OptRegCandidate &Cand = S.Cand;
      const MachineBasicBlock *MBB =
          LIS.intervalIsInOneMBB(LIS.getInterval(Cand.Reg.getVirt()));
      CandsPerMBB[MBB].push_back(&Cand);
    }

    OS << "* Starting re-assignment phase with registers:\n| Candidates are:\n";
    OS << "    Global (" << CandsPerMBB[nullptr].size() << " registers):\n";
    PrintRegs(CandsPerMBB[nullptr]);
    for (const auto &[MBB, Cands] : CandsPerMBB) {
      if (!MBB)
        continue;
      OS << "    MBB #" << MBB->getNumber() << " (" << Cands.size()
         << " registers)\n";
      PrintRegs(Cands);
    }
  });
}

#endif

namespace {
class AMDGPUOptimizeVGPREncodingLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUOptimizeVGPREncodingLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Optimize VGPR Encoding";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveRegMatrixWrapperLegacy>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

INITIALIZE_PASS_BEGIN(AMDGPUOptimizeVGPREncodingLegacy, DEBUG_TYPE,
                      "AMDGPU Optimize VGPR Encoding", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_END(AMDGPUOptimizeVGPREncodingLegacy, DEBUG_TYPE,
                    "AMDGPU Optimize VGPR Encoding", false, false)

char AMDGPUOptimizeVGPREncodingLegacy::ID = 0;

char &llvm::SIAMDGPUOptimizeVGPREncodingLegacyID =
    AMDGPUOptimizeVGPREncodingLegacy::ID;

bool AMDGPUOptimizeVGPREncodingLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  VirtRegMap &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  LiveRegMatrix &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  return AMDGPUOptimizeVGPREncoding(VRM, LIS, LRM).run(MF);
}

PreservedAnalyses
AMDGPUOptimizeVGPREncodingPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {

  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  LiveRegMatrix &LRM = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  if (!AMDGPUOptimizeVGPREncoding(VRM, LIS, LRM).run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
