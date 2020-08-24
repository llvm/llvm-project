//===- InstrRefBasedImpl.cpp - Tracking Debug Value MIs -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file InstrRefBasedImpl.cpp
///
/// This is a separate implementation of LiveDebugValues, see
/// LiveDebugValues.cpp and VarLocBasedImpl.cpp for more information.
///
/// This pass propagates variable locations between basic blocks, resolving
/// control flow conflicts between them. The problem is much like SSA
/// construction, where each DBG_VALUE instruction assigns the *value* that
/// a variable has, and every instruction where the variable is in scope uses
/// that variable. The resulting map of instruction-to-value is then translated
/// into a register (or spill) location for each variable over each instruction.
///
/// This pass determines which DBG_VALUE dominates which instructions, or if
/// none do, where values must be merged (like PHI nodes). The added
/// complication is that because codegen has already finished, a PHI node may
/// be needed for a variable location to be correct, but no register or spill
/// slot merges the necessary values. In these circumstances, the variable
/// location is dropped.
///
/// What makes this analysis non-trivial is loops: we cannot tell in advance
/// whether a variable location is live throughout a loop, or whether its
/// location is clobbered (or redefined by another DBG_VALUE), without
/// exploring all the way through.
///
/// To make this simpler we perform two kinds of analysis. First, we identify
/// every value defined by every instruction (ignoring those that only move
/// another value), then compute a map of which values are available for each
/// instruction. This is stronger than a reaching-def analysis, as we create
/// PHI values where other values merge.
///
/// Secondly, for each variable, we effectively re-construct SSA using each
/// DBG_VALUE as a def. The DBG_VALUEs read a value-number computed by the
/// first analysis from the location they refer to. We can then compute the
/// dominance frontiers of where a variable has a value, and create PHI nodes
/// where they merge.
/// This isn't precisely SSA-construction though, because the function shape
/// is pre-defined. If a variable location requires a PHI node, but no
/// PHI for the relevant values is present in the function (as computed by the
/// first analysis), the location must be dropped.
///
/// Once both are complete, we can pass back over all instructions knowing:
///  * What _value_ each variable should contain, either defined by an
///    instruction or where control flow merges
///  * What the location of that value is (if any).
/// Allowing us to create appropriate live-in DBG_VALUEs, and DBG_VALUEs when
/// a value moves location. After this pass runs, all variable locations within
/// a block should be specified by DBG_VALUEs within that block, allowing
/// DbgEntityHistoryCalculator to focus on individual blocks.
///
/// This pass is able to go fast because the size of the first
/// reaching-definition analysis is proportional to the working-set size of
/// the function, which the compiler tries to keep small. (It's also
/// proportional to the number of blocks). Additionally, we repeatedly perform
/// the second reaching-definition analysis with only the variables and blocks
/// in a single lexical scope, exploiting their locality.
///
/// Determining where PHIs happen is trickier with this approach, and it comes
/// to a head in the major problem for LiveDebugValues: is a value live-through
/// a loop, or not? Your garden-variety dataflow analysis aims to build a set of
/// facts about a function, however this analysis needs to generate new value
/// numbers at joins.
///
/// To do this, consider a lattice of all definition values, from instructions
/// and from PHIs. Each PHI is characterised by the RPO number of the block it
/// occurs in. Each value pair A, B can be ordered by RPO(A) < RPO(B):
/// with non-PHI values at the top, and any PHI value in the last block (by RPO
/// order) at the bottom.
///
/// (Awkwardly: lower-down-the _lattice_ means a greater RPO _number_. Below,
/// "rank" always refers to the former).
///
/// At any join, for each register, we consider:
///  * All incoming values, and
///  * The PREVIOUS live-in value at this join.
/// If all incoming values agree: that's the live-in value. If they do not, the
/// incoming values are ranked according to the partial order, and the NEXT
/// LOWEST rank after the PREVIOUS live-in value is picked (multiple values of
/// the same rank are ignored as conflicting). If there are no candidate values,
/// or if the rank of the live-in would be lower than the rank of the current
/// blocks PHIs, create a new PHI value.
///
/// Intuitively: if it's not immediately obvious what value a join should result
/// in, we iteratively descend from instruction-definitions down through PHI
/// values, getting closer to the current block each time. If the current block
/// is a loop head, this ordering is effectively searching outer levels of
/// loops, to find a value that's live-through the current loop.
///
/// If there is no value that's live-through this loop, a PHI is created for
/// this location instead. We can't use a lower-ranked PHI because by definition
/// it doesn't dominate the current block. We can't create a PHI value any
/// earlier, because we risk creating a PHI value at a location where values do
/// not in fact merge, thus misrepresenting the truth, and not making the true
/// live-through value for variable locations.
///
/// This algorithm applies to both calculating the availability of values in
/// the first analysis, and the location of variables in the second. However
/// for the second we add an extra dimension of pain: creating a variable
/// location PHI is only valid if, for each incoming edge,
///  * There is a value for the variable on the incoming edge, and
///  * All the edges have that value in the same register.
/// Or put another way: we can only create a variable-location PHI if there is
/// a matching machine-location PHI, each input to which is the variables value
/// in the predecessor block.
///
/// To accomodate this difference, each point on the lattice is split in
/// two: a "proposed" PHI and "definite" PHI. Any PHI that can immediately
/// have a location determined are "definite" PHIs, and no further work is
/// needed. Otherwise, a location that all non-backedge predecessors agree
/// on is picked and propagated as a "proposed" PHI value. If that PHI value
/// is truly live-through, it'll appear on the loop backedges on the next
/// dataflow iteration, after which the block live-in moves to be a "definite"
/// PHI. If it's not truly live-through, the variable value will be downgraded
/// further as we explore the lattice, or remains "proposed" and is considered
/// invalid once dataflow completes.
///
/// ### Terminology
///
/// A machine location is a register or spill slot, a value is something that's
/// defined by an instruction or PHI node, while a variable value is the value
/// assigned to a variable. A variable location is a machine location, that must
/// contain the appropriate variable value. A value that is a PHI node is
/// occasionally called an mphi.
///
/// The first dataflow problem is the "machine value location" problem,
/// because we're determining which machine locations contain which values.
/// The "locations" are constant: what's unknown is what value they contain.
///
/// The second dataflow problem (the one for variables) is the "variable value
/// problem", because it's determining what values a variable has, rather than
/// what location those values are placed in. Unfortunately, it's not that
/// simple, because producing a PHI value always involves picking a location.
/// This is an imperfection that we just have to accept, at least for now.
///
/// TODO:
///   Overlapping fragments
///   Entry values
///   Add back DEBUG statements for debugging this
///   Collect statistics
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>
#include <limits.h>
#include <limits>

#include "LiveDebugValues.h"

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");
STATISTIC(NumRemoved, "Number of DBG_VALUE instructions removed");

// Act more like the VarLoc implementation, by propagating some locations too
// far and ignoring some transfers.
static cl::opt<bool> EmulateOldLDV("emulate-old-livedebugvalues", cl::Hidden,
                                   cl::desc("Act like old LiveDebugValues did"),
                                   cl::init(false));

// Rely on isStoreToStackSlotPostFE and similar to observe all stack spills.
static cl::opt<bool>
    ObserveAllStackops("observe-all-stack-ops", cl::Hidden,
                       cl::desc("Allow non-kill spill and restores"),
                       cl::init(false));

namespace {

// The location at which a spilled value resides. It consists of a register and
// an offset.
struct SpillLoc {
  unsigned SpillBase;
  int SpillOffset;
  bool operator==(const SpillLoc &Other) const {
    return std::tie(SpillBase, SpillOffset) ==
           std::tie(Other.SpillBase, Other.SpillOffset);
  }
  bool operator<(const SpillLoc &Other) const {
    return std::tie(SpillBase, SpillOffset) <
           std::tie(Other.SpillBase, Other.SpillOffset);
  }
};

class LocIdx {
  unsigned Location;

  // Default constructor is private, initializing to an illegal location number.
  // Use only for "not an entry" elements in IndexedMaps.
  LocIdx() : Location(UINT_MAX) { }

public:
  #define NUM_LOC_BITS 24
  LocIdx(unsigned L) : Location(L) {
    assert(L < (1 << NUM_LOC_BITS) && "Machine locations must fit in 24 bits");
  }

  static LocIdx MakeIllegalLoc() {
    return LocIdx();
  }

  bool isIllegal() const {
    return Location == UINT_MAX;
  }

  uint64_t asU64() const {
    return Location;
  }

  bool operator==(unsigned L) const {
    return Location == L;
  }

  bool operator==(const LocIdx &L) const {
    return Location == L.Location;
  }

  bool operator!=(unsigned L) const {
    return !(*this == L);
  }

  bool operator!=(const LocIdx &L) const {
    return !(*this == L);
  }

  bool operator<(const LocIdx &Other) const {
    return Location < Other.Location;
  }
};

class LocIdxToIndexFunctor {
public:
  using argument_type = LocIdx;
  unsigned operator()(const LocIdx &L) const {
    return L.asU64();
  }
};

/// Unique identifier for a value defined by an instruction, as a value type.
/// Casts back and forth to a uint64_t. Probably replacable with something less
/// bit-constrained. Each value identifies the instruction and machine location
/// where the value is defined, although there may be no corresponding machine
/// operand for it (ex: regmasks clobbering values). The instructions are
/// one-based, and definitions that are PHIs have instruction number zero.
///
/// The obvious limits of a 1M block function or 1M instruction blocks are
/// problematic; but by that point we should probably have bailed out of
/// trying to analyse the function.
class ValueIDNum {
  uint64_t BlockNo : 20;         /// The block where the def happens.
  uint64_t InstNo : 20;          /// The Instruction where the def happens.
                                 /// One based, is distance from start of block.
  uint64_t LocNo : NUM_LOC_BITS; /// The machine location where the def happens.

public:
  // XXX -- temporarily enabled while the live-in / live-out tables are moved
  // to something more type-y
  ValueIDNum() : BlockNo(0xFFFFF),
                 InstNo(0xFFFFF),
                 LocNo(0xFFFFFF) { }

  ValueIDNum(uint64_t Block, uint64_t Inst, uint64_t Loc)
    : BlockNo(Block), InstNo(Inst), LocNo(Loc) { }

  ValueIDNum(uint64_t Block, uint64_t Inst, LocIdx Loc)
    : BlockNo(Block), InstNo(Inst), LocNo(Loc.asU64()) { }

  uint64_t getBlock() const { return BlockNo; }
  uint64_t getInst() const { return InstNo; }
  uint64_t getLoc() const { return LocNo; }
  bool isPHI() const { return InstNo == 0; }

  uint64_t asU64() const {
    uint64_t TmpBlock = BlockNo;
    uint64_t TmpInst = InstNo;
    return TmpBlock << 44ull | TmpInst << NUM_LOC_BITS | LocNo;
  }

  static ValueIDNum fromU64(uint64_t v) {
    uint64_t L = (v & 0x3FFF);
    return {v >> 44ull, ((v >> NUM_LOC_BITS) & 0xFFFFF), L};
  }

  bool operator<(const ValueIDNum &Other) const {
    return asU64() < Other.asU64();
  }

  bool operator==(const ValueIDNum &Other) const {
    return std::tie(BlockNo, InstNo, LocNo) ==
           std::tie(Other.BlockNo, Other.InstNo, Other.LocNo);
  }

  bool operator!=(const ValueIDNum &Other) const { return !(*this == Other); }

  std::string asString(const std::string &mlocname) const {
    return Twine("bb ")
        .concat(Twine(BlockNo).concat(Twine(" inst ").concat(
            Twine(InstNo).concat(Twine(" loc ").concat(Twine(mlocname))))))
        .str();
  }

  static ValueIDNum EmptyValue;
};

} // end anonymous namespace

namespace {

/// Meta qualifiers for a value. Pair of whatever expression is used to qualify
/// the the value, and Boolean of whether or not it's indirect.
class DbgValueProperties {
public:
  DbgValueProperties(const DIExpression *DIExpr, bool Indirect)
      : DIExpr(DIExpr), Indirect(Indirect) {}

  /// Extract properties from an existing DBG_VALUE instruction.
  DbgValueProperties(const MachineInstr &MI) {
    assert(MI.isDebugValue());
    DIExpr = MI.getDebugExpression();
    Indirect = MI.getOperand(1).isImm();
  }

  bool operator==(const DbgValueProperties &Other) const {
    return std::tie(DIExpr, Indirect) == std::tie(Other.DIExpr, Other.Indirect);
  }

  bool operator!=(const DbgValueProperties &Other) const {
    return !(*this == Other);
  }

  const DIExpression *DIExpr;
  bool Indirect;
};

/// Tracker for what values are in machine locations. Listens to the Things
/// being Done by various instructions, and maintains a table of what machine
/// locations have what values (as defined by a ValueIDNum).
///
/// There are potentially a much larger number of machine locations on the
/// target machine than the actual working-set size of the function. On x86 for
/// example, we're extremely unlikely to want to track values through control
/// or debug registers. To avoid doing so, MLocTracker has several layers of
/// indirection going on, with two kinds of ``location'':
///  * A LocID uniquely identifies a register or spill location, with a
///    predictable value.
///  * A LocIdx is a key (in the database sense) for a LocID and a ValueIDNum.
/// Whenever a location is def'd or used by a MachineInstr, we automagically
/// create a new LocIdx for a location, but not otherwise. This ensures we only
/// account for locations that are actually used or defined. The cost is another
/// vector lookup (of LocID -> LocIdx) over any other implementation. This is
/// fairly cheap, and the compiler tries to reduce the working-set at any one
/// time in the function anyway.
///
/// Register mask operands completely blow this out of the water; I've just
/// piled hacks on top of hacks to get around that.
class MLocTracker {
public:
  MachineFunction &MF;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const TargetLowering &TLI;

  /// IndexedMap type, mapping from LocIdx to ValueIDNum.
  typedef IndexedMap<ValueIDNum, LocIdxToIndexFunctor> LocToValueType;

  /// Map of LocIdxes to the ValueIDNums that they store. This is tightly
  /// packed, entries only exist for locations that are being tracked.
  LocToValueType LocIdxToIDNum;

  /// "Map" of machine location IDs (i.e., raw register or spill number) to the
  /// LocIdx key / number for that location. There are always at least as many
  /// as the number of registers on the target -- if the value in the register
  /// is not being tracked, then the LocIdx value will be zero. New entries are
  /// appended if a new spill slot begins being tracked.
  /// This, and the corresponding reverse map persist for the analysis of the
  /// whole function, and is necessarying for decoding various vectors of
  /// values.
  std::vector<LocIdx> LocIDToLocIdx;

  /// Inverse map of LocIDToLocIdx.
  IndexedMap<unsigned, LocIdxToIndexFunctor> LocIdxToLocID;

  /// Unique-ification of spill slots. Used to number them -- their LocID
  /// number is the index in SpillLocs minus one plus NumRegs.
  UniqueVector<SpillLoc> SpillLocs;

  // If we discover a new machine location, assign it an mphi with this
  // block number.
  unsigned CurBB;

  /// Cached local copy of the number of registers the target has.
  unsigned NumRegs;

  /// Collection of register mask operands that have been observed. Second part
  /// of pair indicates the instruction that they happened in. Used to
  /// reconstruct where defs happened if we start tracking a location later
  /// on.
  SmallVector<std::pair<const MachineOperand *, unsigned>, 32> Masks;

  /// Iterator for locations and the values they contain. Dereferencing
  /// produces a struct/pair containing the LocIdx key for this location,
  /// and a reference to the value currently stored. Simplifies the process
  /// of seeking a particular location.
  class MLocIterator {
    LocToValueType &ValueMap;
    LocIdx Idx;

  public:
    class value_type {
      public:
      value_type(LocIdx Idx, ValueIDNum &Value) : Idx(Idx), Value(Value) { }
      const LocIdx Idx;  /// Read-only index of this location.
      ValueIDNum &Value; /// Reference to the stored value at this location.
    };

    MLocIterator(LocToValueType &ValueMap, LocIdx Idx)
      : ValueMap(ValueMap), Idx(Idx) { }

    bool operator==(const MLocIterator &Other) const {
      assert(&ValueMap == &Other.ValueMap);
      return Idx == Other.Idx;
    }

    bool operator!=(const MLocIterator &Other) const {
      return !(*this == Other);
    }

    void operator++() {
      Idx = LocIdx(Idx.asU64() + 1);
    }

    value_type operator*() {
      return value_type(Idx, ValueMap[LocIdx(Idx)]);
    }
  };

  MLocTracker(MachineFunction &MF, const TargetInstrInfo &TII,
              const TargetRegisterInfo &TRI, const TargetLowering &TLI)
      : MF(MF), TII(TII), TRI(TRI), TLI(TLI),
        LocIdxToIDNum(ValueIDNum::EmptyValue),
        LocIdxToLocID(0) {
    NumRegs = TRI.getNumRegs();
    reset();
    LocIDToLocIdx.resize(NumRegs, LocIdx::MakeIllegalLoc());
    assert(NumRegs < (1u << NUM_LOC_BITS)); // Detect bit packing failure

    // Always track SP. This avoids the implicit clobbering caused by regmasks
    // from affectings its values. (LiveDebugValues disbelieves calls and
    // regmasks that claim to clobber SP).
    unsigned SP = TLI.getStackPointerRegisterToSaveRestore();
    if (SP) {
      unsigned ID = getLocID(SP, false);
      (void)lookupOrTrackRegister(ID);
    }
  }

  /// Produce location ID number for indexing LocIDToLocIdx. Takes the register
  /// or spill number, and flag for whether it's a spill or not.
  unsigned getLocID(unsigned RegOrSpill, bool isSpill) {
    return (isSpill) ? RegOrSpill + NumRegs - 1 : RegOrSpill;
  }

  /// Accessor for reading the value at Idx.
  ValueIDNum getNumAtPos(LocIdx Idx) const {
    assert(Idx.asU64() < LocIdxToIDNum.size());
    return LocIdxToIDNum[Idx];
  }

  unsigned getNumLocs(void) const { return LocIdxToIDNum.size(); }

  /// Reset all locations to contain a PHI value at the designated block. Used
  /// sometimes for actual PHI values, othertimes to indicate the block entry
  /// value (before any more information is known).
  void setMPhis(unsigned NewCurBB) {
    CurBB = NewCurBB;
    for (auto Location : locations())
      Location.Value = {CurBB, 0, Location.Idx};
  }

  /// Load values for each location from array of ValueIDNums. Take current
  /// bbnum just in case we read a value from a hitherto untouched register.
  void loadFromArray(ValueIDNum *Locs, unsigned NewCurBB) {
    CurBB = NewCurBB;
    // Iterate over all tracked locations, and load each locations live-in
    // value into our local index.
    for (auto Location : locations())
      Location.Value = Locs[Location.Idx.asU64()];
  }

  /// Wipe any un-necessary location records after traversing a block.
  void reset(void) {
    // We could reset all the location values too; however either loadFromArray
    // or setMPhis should be called before this object is re-used. Just
    // clear Masks, they're definitely not needed.
    Masks.clear();
  }

  /// Clear all data. Destroys the LocID <=> LocIdx map, which makes most of
  /// the information in this pass uninterpretable.
  void clear(void) {
    reset();
    LocIDToLocIdx.clear();
    LocIdxToLocID.clear();
    LocIdxToIDNum.clear();
    //SpillLocs.reset(); XXX UniqueVector::reset assumes a SpillLoc casts from 0
    SpillLocs = decltype(SpillLocs)();

    LocIDToLocIdx.resize(NumRegs, LocIdx::MakeIllegalLoc());
  }

  /// Set a locaiton to a certain value.
  void setMLoc(LocIdx L, ValueIDNum Num) {
    assert(L.asU64() < LocIdxToIDNum.size());
    LocIdxToIDNum[L] = Num;
  }

  /// Create a LocIdx for an untracked register ID. Initialize it to either an
  /// mphi value representing a live-in, or a recent register mask clobber.
  LocIdx trackRegister(unsigned ID) {
    assert(ID != 0);
    LocIdx NewIdx = LocIdx(LocIdxToIDNum.size());
    LocIdxToIDNum.grow(NewIdx);
    LocIdxToLocID.grow(NewIdx);

    // Default: it's an mphi.
    ValueIDNum ValNum = {CurBB, 0, NewIdx};
    // Was this reg ever touched by a regmask?
    for (const auto &MaskPair : reverse(Masks)) {
      if (MaskPair.first->clobbersPhysReg(ID)) {
        // There was an earlier def we skipped.
        ValNum = {CurBB, MaskPair.second, NewIdx};
        break;
      }
    }

    LocIdxToIDNum[NewIdx] = ValNum;
    LocIdxToLocID[NewIdx] = ID;
    return NewIdx;
  }

  LocIdx lookupOrTrackRegister(unsigned ID) {
    LocIdx &Index = LocIDToLocIdx[ID];
    if (Index.isIllegal())
      Index = trackRegister(ID);
    return Index;
  }

  /// Record a definition of the specified register at the given block / inst.
  /// This doesn't take a ValueIDNum, because the definition and its location
  /// are synonymous.
  void defReg(Register R, unsigned BB, unsigned Inst) {
    unsigned ID = getLocID(R, false);
    LocIdx Idx = lookupOrTrackRegister(ID);
    ValueIDNum ValueID = {BB, Inst, Idx};
    LocIdxToIDNum[Idx] = ValueID;
  }

  /// Set a register to a value number. To be used if the value number is
  /// known in advance.
  void setReg(Register R, ValueIDNum ValueID) {
    unsigned ID = getLocID(R, false);
    LocIdx Idx = lookupOrTrackRegister(ID);
    LocIdxToIDNum[Idx] = ValueID;
  }

  ValueIDNum readReg(Register R) {
    unsigned ID = getLocID(R, false);
    LocIdx Idx = lookupOrTrackRegister(ID);
    return LocIdxToIDNum[Idx];
  }

  /// Reset a register value to zero / empty. Needed to replicate the
  /// VarLoc implementation where a copy to/from a register effectively
  /// clears the contents of the source register. (Values can only have one
  ///  machine location in VarLocBasedImpl).
  void wipeRegister(Register R) {
    unsigned ID = getLocID(R, false);
    LocIdx Idx = LocIDToLocIdx[ID];
    LocIdxToIDNum[Idx] = ValueIDNum::EmptyValue;
  }

  /// Determine the LocIdx of an existing register.
  LocIdx getRegMLoc(Register R) {
    unsigned ID = getLocID(R, false);
    return LocIDToLocIdx[ID];
  }

  /// Record a RegMask operand being executed. Defs any register we currently
  /// track, stores a pointer to the mask in case we have to account for it
  /// later.
  void writeRegMask(const MachineOperand *MO, unsigned CurBB, unsigned InstID) {
    // Ensure SP exists, so that we don't override it later.
    unsigned SP = TLI.getStackPointerRegisterToSaveRestore();

    // Def any register we track have that isn't preserved. The regmask
    // terminates the liveness of a register, meaning its value can't be
    // relied upon -- we represent this by giving it a new value.
    for (auto Location : locations()) {
      unsigned ID = LocIdxToLocID[Location.Idx];
      // Don't clobber SP, even if the mask says it's clobbered.
      if (ID < NumRegs && ID != SP && MO->clobbersPhysReg(ID))
        defReg(ID, CurBB, InstID);
    }
    Masks.push_back(std::make_pair(MO, InstID));
  }

  /// Find LocIdx for SpillLoc \p L, creating a new one if it's not tracked.
  LocIdx getOrTrackSpillLoc(SpillLoc L) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0) {
      SpillID = SpillLocs.insert(L);
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIdx(LocIdxToIDNum.size()); // New idx
      LocIdxToIDNum.grow(Idx);
      LocIdxToLocID.grow(Idx);
      LocIDToLocIdx.push_back(Idx);
      LocIdxToLocID[Idx] = L;
      return Idx;
    } else {
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIDToLocIdx[L];
      return Idx;
    }
  }

  /// Set the value stored in a spill slot.
  void setSpill(SpillLoc L, ValueIDNum ValueID) {
    LocIdx Idx = getOrTrackSpillLoc(L);
    LocIdxToIDNum[Idx] = ValueID;
  }

  /// Read whatever value is in a spill slot, or None if it isn't tracked.
  Optional<ValueIDNum> readSpill(SpillLoc L) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0)
      return None;

    unsigned LocID = getLocID(SpillID, true);
    LocIdx Idx = LocIDToLocIdx[LocID];
    return LocIdxToIDNum[Idx];
  }

  /// Determine the LocIdx of a spill slot. Return None if it previously
  /// hasn't had a value assigned.
  Optional<LocIdx> getSpillMLoc(SpillLoc L) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0)
      return None;
    unsigned LocNo = getLocID(SpillID, true);
    return LocIDToLocIdx[LocNo];
  }

  /// Return true if Idx is a spill machine location.
  bool isSpill(LocIdx Idx) const {
    return LocIdxToLocID[Idx] >= NumRegs;
  }

  MLocIterator begin() {
    return MLocIterator(LocIdxToIDNum, 0);
  }

  MLocIterator end() {
    return MLocIterator(LocIdxToIDNum, LocIdxToIDNum.size());
  }

  /// Return a range over all locations currently tracked.
  iterator_range<MLocIterator> locations() {
    return llvm::make_range(begin(), end());
  }

  std::string LocIdxToName(LocIdx Idx) const {
    unsigned ID = LocIdxToLocID[Idx];
    if (ID >= NumRegs)
      return Twine("slot ").concat(Twine(ID - NumRegs)).str();
    else
      return TRI.getRegAsmName(ID).str();
  }

  std::string IDAsString(const ValueIDNum &Num) const {
    std::string DefName = LocIdxToName(Num.getLoc());
    return Num.asString(DefName);
  }

  LLVM_DUMP_METHOD
  void dump() {
    for (auto Location : locations()) {
      std::string MLocName = LocIdxToName(Location.Value.getLoc());
      std::string DefName = Location.Value.asString(MLocName);
      dbgs() << LocIdxToName(Location.Idx) << " --> " << DefName << "\n";
    }
  }

  LLVM_DUMP_METHOD
  void dump_mloc_map() {
    for (auto Location : locations()) {
      std::string foo = LocIdxToName(Location.Idx);
      dbgs() << "Idx " << Location.Idx.asU64() << " " << foo << "\n";
    }
  }

  /// Create a DBG_VALUE based on  machine location \p MLoc. Qualify it with the
  /// information in \pProperties, for variable Var. Don't insert it anywhere,
  /// just return the builder for it.
  MachineInstrBuilder emitLoc(Optional<LocIdx> MLoc, const DebugVariable &Var,
                              const DbgValueProperties &Properties) {
    DebugLoc DL =
        DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII.get(TargetOpcode::DBG_VALUE));

    const DIExpression *Expr = Properties.DIExpr;
    if (!MLoc) {
      // No location -> DBG_VALUE $noreg
      MIB.addReg(0, RegState::Debug);
      MIB.addReg(0, RegState::Debug);
    } else if (LocIdxToLocID[*MLoc] >= NumRegs) {
      unsigned LocID = LocIdxToLocID[*MLoc];
      const SpillLoc &Spill = SpillLocs[LocID - NumRegs + 1];
      Expr = DIExpression::prepend(Expr, DIExpression::ApplyOffset,
                                   Spill.SpillOffset);
      unsigned Base = Spill.SpillBase;
      MIB.addReg(Base, RegState::Debug);
      MIB.addImm(0);
    } else {
      unsigned LocID = LocIdxToLocID[*MLoc];
      MIB.addReg(LocID, RegState::Debug);
      if (Properties.Indirect)
        MIB.addImm(0);
      else
        MIB.addReg(0, RegState::Debug);
    }

    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Expr);
    return MIB;
  }
};

/// Class recording the (high level) _value_ of a variable. Identifies either
/// the value of the variable as a ValueIDNum, or a constant MachineOperand.
/// This class also stores meta-information about how the value is qualified.
/// Used to reason about variable values when performing the second
/// (DebugVariable specific) dataflow analysis.
class DbgValue {
public:
  union {
    /// If Kind is Def, the value number that this value is based on.
    ValueIDNum ID;
    /// If Kind is Const, the MachineOperand defining this value.
    MachineOperand MO;
    /// For a NoVal DbgValue, which block it was generated in.
    unsigned BlockNo;
  };
  /// Qualifiers for the ValueIDNum above.
  DbgValueProperties Properties;

  typedef enum {
    Undef,     // Represents a DBG_VALUE $noreg in the transfer function only.
    Def,       // This value is defined by an inst, or is a PHI value.
    Const,     // A constant value contained in the MachineOperand field.
    Proposed,  // This is a tentative PHI value, which may be confirmed or
               // invalidated later.
    NoVal      // Empty DbgValue, generated during dataflow. BlockNo stores
               // which block this was generated in.
   } KindT;
  /// Discriminator for whether this is a constant or an in-program value.
  KindT Kind;

  DbgValue(const ValueIDNum &Val, const DbgValueProperties &Prop, KindT Kind)
    : ID(Val), Properties(Prop), Kind(Kind) {
    assert(Kind == Def || Kind == Proposed);
  }

  DbgValue(unsigned BlockNo, const DbgValueProperties &Prop, KindT Kind)
    : BlockNo(BlockNo), Properties(Prop), Kind(Kind) {
    assert(Kind == NoVal);
  }

  DbgValue(const MachineOperand &MO, const DbgValueProperties &Prop, KindT Kind)
    : MO(MO), Properties(Prop), Kind(Kind) {
    assert(Kind == Const);
  }

  DbgValue(const DbgValueProperties &Prop, KindT Kind)
    : Properties(Prop), Kind(Kind) {
    assert(Kind == Undef &&
           "Empty DbgValue constructor must pass in Undef kind");
  }

  void dump(const MLocTracker *MTrack) const {
    if (Kind == Const) {
      MO.dump();
    } else if (Kind == NoVal) {
      dbgs() << "NoVal(" << BlockNo << ")";
    } else if (Kind == Proposed) {
      dbgs() << "VPHI(" << MTrack->IDAsString(ID) << ")";
    } else {
      assert(Kind == Def);
      dbgs() << MTrack->IDAsString(ID);
    }
    if (Properties.Indirect)
      dbgs() << " indir";
    if (Properties.DIExpr)
      dbgs() << " " << *Properties.DIExpr;
  }

  bool operator==(const DbgValue &Other) const {
    if (std::tie(Kind, Properties) != std::tie(Other.Kind, Other.Properties))
      return false;
    else if (Kind == Proposed && ID != Other.ID)
      return false;
    else if (Kind == Def && ID != Other.ID)
      return false;
    else if (Kind == NoVal && BlockNo != Other.BlockNo)
      return false;
    else if (Kind == Const)
      return MO.isIdenticalTo(Other.MO);

    return true;
  }

  bool operator!=(const DbgValue &Other) const { return !(*this == Other); }
};

/// Types for recording sets of variable fragments that overlap. For a given
/// local variable, we record all other fragments of that variable that could
/// overlap it, to reduce search time.
using FragmentOfVar =
    std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
using OverlapMap =
    DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

/// Collection of DBG_VALUEs observed when traversing a block. Records each
/// variable and the value the DBG_VALUE refers to. Requires the machine value
/// location dataflow algorithm to have run already, so that values can be
/// identified.
class VLocTracker {
public:
  /// Map DebugVariable to the latest Value it's defined to have.
  /// Needs to be a MapVector because we determine order-in-the-input-MIR from
  /// the order in this container.
  /// We only retain the last DbgValue in each block for each variable, to
  /// determine the blocks live-out variable value. The Vars container forms the
  /// transfer function for this block, as part of the dataflow analysis. The
  /// movement of values between locations inside of a block is handled at a
  /// much later stage, in the TransferTracker class.
  MapVector<DebugVariable, DbgValue> Vars;
  DenseMap<DebugVariable, const DILocation *> Scopes;
  MachineBasicBlock *MBB;

public:
  VLocTracker() {}

  void defVar(const MachineInstr &MI, Optional<ValueIDNum> ID) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    DbgValueProperties Properties(MI);
    DbgValue Rec = (ID) ? DbgValue(*ID, Properties, DbgValue::Def)
                        : DbgValue(Properties, DbgValue::Undef);

    // Attempt insertion; overwrite if it's already mapped.
    auto Result = Vars.insert(std::make_pair(Var, Rec));
    if (!Result.second)
      Result.first->second = Rec;
    Scopes[Var] = MI.getDebugLoc().get();
  }

  void defVar(const MachineInstr &MI, const MachineOperand &MO) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    DbgValueProperties Properties(MI);
    DbgValue Rec = DbgValue(MO, Properties, DbgValue::Const);

    // Attempt insertion; overwrite if it's already mapped.
    auto Result = Vars.insert(std::make_pair(Var, Rec));
    if (!Result.second)
      Result.first->second = Rec;
    Scopes[Var] = MI.getDebugLoc().get();
  }
};

/// Tracker for converting machine value locations and variable values into
/// variable locations (the output of LiveDebugValues), recorded as DBG_VALUEs
/// specifying block live-in locations and transfers within blocks.
///
/// Operating on a per-block basis, this class takes a (pre-loaded) MLocTracker
/// and must be initialized with the set of variable values that are live-in to
/// the block. The caller then repeatedly calls process(). TransferTracker picks
/// out variable locations for the live-in variable values (if there _is_ a
/// location) and creates the corresponding DBG_VALUEs. Then, as the block is
/// stepped through, transfers of values between machine locations are
/// identified and if profitable, a DBG_VALUE created.
///
/// This is where debug use-before-defs would be resolved: a variable with an
/// unavailable value could materialize in the middle of a block, when the
/// value becomes available. Or, we could detect clobbers and re-specify the
/// variable in a backup location. (XXX these are unimplemented).
class TransferTracker {
public:
  const TargetInstrInfo *TII;
  /// This machine location tracker is assumed to always contain the up-to-date
  /// value mapping for all machine locations. TransferTracker only reads
  /// information from it. (XXX make it const?)
  MLocTracker *MTracker;
  MachineFunction &MF;

  /// Record of all changes in variable locations at a block position. Awkwardly
  /// we allow inserting either before or after the point: MBB != nullptr
  /// indicates it's before, otherwise after.
  struct Transfer {
    MachineBasicBlock::iterator Pos; /// Position to insert DBG_VALUes
    MachineBasicBlock *MBB;          /// non-null if we should insert after.
    SmallVector<MachineInstr *, 4> Insts; /// Vector of DBG_VALUEs to insert.
  };

  typedef struct {
    LocIdx Loc;
    DbgValueProperties Properties;
  } LocAndProperties;

  /// Collection of transfers (DBG_VALUEs) to be inserted.
  SmallVector<Transfer, 32> Transfers;

  /// Local cache of what-value-is-in-what-LocIdx. Used to identify differences
  /// between TransferTrackers view of variable locations and MLocTrackers. For
  /// example, MLocTracker observes all clobbers, but TransferTracker lazily
  /// does not.
  std::vector<ValueIDNum> VarLocs;

  /// Map from LocIdxes to which DebugVariables are based that location.
  /// Mantained while stepping through the block. Not accurate if
  /// VarLocs[Idx] != MTracker->LocIdxToIDNum[Idx].
  std::map<LocIdx, SmallSet<DebugVariable, 4>> ActiveMLocs;

  /// Map from DebugVariable to it's current location and qualifying meta
  /// information. To be used in conjunction with ActiveMLocs to construct
  /// enough information for the DBG_VALUEs for a particular LocIdx.
  DenseMap<DebugVariable, LocAndProperties> ActiveVLocs;

  /// Temporary cache of DBG_VALUEs to be entered into the Transfers collection.
  SmallVector<MachineInstr *, 4> PendingDbgValues;

  const TargetRegisterInfo &TRI;
  const BitVector &CalleeSavedRegs;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *MTracker,
                  MachineFunction &MF, const TargetRegisterInfo &TRI,
                  const BitVector &CalleeSavedRegs)
      : TII(TII), MTracker(MTracker), MF(MF), TRI(TRI),
        CalleeSavedRegs(CalleeSavedRegs) {}

  /// Load object with live-in variable values. \p mlocs contains the live-in
  /// values in each machine location, while \p vlocs the live-in variable
  /// values. This method picks variable locations for the live-in variables,
  /// creates DBG_VALUEs and puts them in #Transfers, then prepares the other
  /// object fields to track variable locations as we step through the block.
  /// FIXME: could just examine mloctracker instead of passing in \p mlocs?
  void loadInlocs(MachineBasicBlock &MBB, ValueIDNum *MLocs,
                  SmallVectorImpl<std::pair<DebugVariable, DbgValue>> &VLocs,
                  unsigned NumLocs) {
    ActiveMLocs.clear();
    ActiveVLocs.clear();
    VarLocs.clear();
    VarLocs.reserve(NumLocs);

    auto isCalleeSaved = [&](LocIdx L) {
      unsigned Reg = MTracker->LocIdxToLocID[L];
      if (Reg >= MTracker->NumRegs)
        return false;
      for (MCRegAliasIterator RAI(Reg, &TRI, true); RAI.isValid(); ++RAI)
        if (CalleeSavedRegs.test(*RAI))
          return true;
      return false;
    };

    // Map of the preferred location for each value.
    std::map<ValueIDNum, LocIdx> ValueToLoc;

    // Produce a map of value numbers to the current machine locs they live
    // in. When emulating VarLocBasedImpl, there should only be one
    // location; when not, we get to pick.
    for (auto Location : MTracker->locations()) {
      LocIdx Idx = Location.Idx;
      ValueIDNum &VNum = MLocs[Idx.asU64()];
      VarLocs.push_back(VNum);
      auto it = ValueToLoc.find(VNum);
      // In order of preference, pick:
      //  * Callee saved registers,
      //  * Other registers,
      //  * Spill slots.
      if (it == ValueToLoc.end() || MTracker->isSpill(it->second) ||
          (!isCalleeSaved(it->second) && isCalleeSaved(Idx.asU64()))) {
        // Insert, or overwrite if insertion failed.
        auto PrefLocRes = ValueToLoc.insert(std::make_pair(VNum, Idx));
        if (!PrefLocRes.second)
          PrefLocRes.first->second = Idx;
      }
    }

    // Now map variables to their picked LocIdxes.
    for (auto Var : VLocs) {
      if (Var.second.Kind == DbgValue::Const) {
        PendingDbgValues.push_back(
            emitMOLoc(Var.second.MO, Var.first, Var.second.Properties));
        continue;
      }

      // If the value has no location, we can't make a variable location.
      auto ValuesPreferredLoc = ValueToLoc.find(Var.second.ID);
      if (ValuesPreferredLoc == ValueToLoc.end())
        continue;

      LocIdx M = ValuesPreferredLoc->second;
      auto NewValue = LocAndProperties{M, Var.second.Properties};
      auto Result = ActiveVLocs.insert(std::make_pair(Var.first, NewValue));
      if (!Result.second)
        Result.first->second = NewValue;
      ActiveMLocs[M].insert(Var.first);
      PendingDbgValues.push_back(
          MTracker->emitLoc(M, Var.first, Var.second.Properties));
    }
    flushDbgValues(MBB.begin(), &MBB);
  }

  /// Helper to move created DBG_VALUEs into Transfers collection.
  void flushDbgValues(MachineBasicBlock::iterator Pos, MachineBasicBlock *MBB) {
    if (PendingDbgValues.size() > 0) {
      Transfers.push_back({Pos, MBB, PendingDbgValues});
      PendingDbgValues.clear();
    }
  }

  /// Handle a DBG_VALUE within a block. Terminate the variables current
  /// location, and record the value its DBG_VALUE refers to, so that we can
  /// detect location transfers later on.
  void redefVar(const MachineInstr &MI) {
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    const MachineOperand &MO = MI.getOperand(0);

    // Erase any previous location,
    auto It = ActiveVLocs.find(Var);
    if (It != ActiveVLocs.end()) {
      ActiveMLocs[It->second.Loc].erase(Var);
    }

    // Insert a new variable location. Ignore non-register locations, we don't
    // transfer those, and can't currently describe spill locs independently of
    // regs.
    // (This is because a spill location is a DBG_VALUE of the stack pointer).
    if (!MO.isReg() || MO.getReg() == 0) {
      if (It != ActiveVLocs.end())
        ActiveVLocs.erase(It);
      return;
    }

    Register Reg = MO.getReg();
    LocIdx MLoc = MTracker->getRegMLoc(Reg);
    DbgValueProperties Properties(MI);

    // Check whether our local copy of values-by-location in #VarLocs is out of
    // date. Wipe old tracking data for the location if it's been clobbered in
    // the meantime.
    if (MTracker->getNumAtPos(MLoc) != VarLocs[MLoc.asU64()]) {
      for (auto &P : ActiveMLocs[MLoc.asU64()]) {
        ActiveVLocs.erase(P);
      }
      ActiveMLocs[MLoc].clear();
      VarLocs[MLoc.asU64()] = MTracker->getNumAtPos(MLoc);
    }

    ActiveMLocs[MLoc].insert(Var);
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(std::make_pair(Var, LocAndProperties{MLoc, Properties}));
    } else {
      It->second.Loc = MLoc;
      It->second.Properties = Properties;
    }
  }

  /// Explicitly terminate variable locations based on \p mloc. Creates undef
  /// DBG_VALUEs for any variables that were located there, and clears
  /// #ActiveMLoc / #ActiveVLoc tracking information for that location.
  void clobberMloc(LocIdx MLoc, MachineBasicBlock::iterator Pos) {
    assert(MTracker->isSpill(MLoc));
    auto ActiveMLocIt = ActiveMLocs.find(MLoc);
    if (ActiveMLocIt == ActiveMLocs.end())
      return;

    VarLocs[MLoc.asU64()] = ValueIDNum::EmptyValue;

    for (auto &Var : ActiveMLocIt->second) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      // Create an undef. We can't feed in a nullptr DIExpression alas,
      // so use the variables last expression. Pass None as the location.
      const DIExpression *Expr = ActiveVLocIt->second.Properties.DIExpr;
      DbgValueProperties Properties(Expr, false);
      PendingDbgValues.push_back(MTracker->emitLoc(None, Var, Properties));
      ActiveVLocs.erase(ActiveVLocIt);
    }
    flushDbgValues(Pos, nullptr);

    ActiveMLocIt->second.clear();
  }

  /// Transfer variables based on \p Src to be based on \p Dst. This handles
  /// both register copies as well as spills and restores. Creates DBG_VALUEs
  /// describing the movement.
  void transferMlocs(LocIdx Src, LocIdx Dst, MachineBasicBlock::iterator Pos) {
    // Does Src still contain the value num we expect? If not, it's been
    // clobbered in the meantime, and our variable locations are stale.
    if (VarLocs[Src.asU64()] != MTracker->getNumAtPos(Src))
      return;

    // assert(ActiveMLocs[Dst].size() == 0);
    //^^^ Legitimate scenario on account of un-clobbered slot being assigned to?
    ActiveMLocs[Dst] = ActiveMLocs[Src];
    VarLocs[Dst.asU64()] = VarLocs[Src.asU64()];

    // For each variable based on Src; create a location at Dst.
    for (auto &Var : ActiveMLocs[Src]) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      assert(ActiveVLocIt != ActiveVLocs.end());
      ActiveVLocIt->second.Loc = Dst;

      assert(Dst != 0);
      MachineInstr *MI =
          MTracker->emitLoc(Dst, Var, ActiveVLocIt->second.Properties);
      PendingDbgValues.push_back(MI);
    }
    ActiveMLocs[Src].clear();
    flushDbgValues(Pos, nullptr);

    // XXX XXX XXX "pretend to be old LDV" means dropping all tracking data
    // about the old location.
    if (EmulateOldLDV)
      VarLocs[Src.asU64()] = ValueIDNum::EmptyValue;
  }

  MachineInstrBuilder emitMOLoc(const MachineOperand &MO,
                                const DebugVariable &Var,
                                const DbgValueProperties &Properties) {
    DebugLoc DL =
        DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE));
    MIB.add(MO);
    if (Properties.Indirect)
      MIB.addImm(0);
    else
      MIB.addReg(0);
    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Properties.DIExpr);
    return MIB;
  }
};

class InstrRefBasedLDV : public LDVImpl {
private:
  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// Machine location/value transfer function, a mapping of which locations
  // are assigned which new values.
  typedef std::map<LocIdx, ValueIDNum> MLocTransferMap;

  /// Live in/out structure for the variable values: a per-block map of
  /// variables to their values. XXX, better name?
  typedef DenseMap<const MachineBasicBlock *,
                   DenseMap<DebugVariable, DbgValue> *>
      LiveIdxT;

  typedef std::pair<DebugVariable, DbgValue> VarAndLoc;

  /// Type for a live-in value: the predecessor block, and its value.
  typedef std::pair<MachineBasicBlock *, DbgValue *> InValueT;

  /// Vector (per block) of a collection (inner smallvector) of live-ins.
  /// Used as the result type for the variable value dataflow problem.
  typedef SmallVector<SmallVector<VarAndLoc, 8>, 8> LiveInsT;

  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;
  TargetPassConfig *TPC;

  /// Object to track machine locations as we step through a block. Could
  /// probably be a field rather than a pointer, as it's always used.
  MLocTracker *MTracker;

  /// Number of the current block LiveDebugValues is stepping through.
  unsigned CurBB;

  /// Number of the current instruction LiveDebugValues is evaluating.
  unsigned CurInst;

  /// Variable tracker -- listens to DBG_VALUEs occurring as InstrRefBasedImpl
  /// steps through a block. Reads the values at each location from the
  /// MLocTracker object.
  VLocTracker *VTracker;

  /// Tracker for transfers, listens to DBG_VALUEs and transfers of values
  /// between locations during stepping, creates new DBG_VALUEs when values move
  /// location.
  TransferTracker *TTracker;

  /// Blocks which are artificial, i.e. blocks which exclusively contain
  /// instructions without DebugLocs, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  // Mapping of blocks to and from their RPOT order.
  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  DenseMap<unsigned, unsigned> BBNumToRPO;

  // Map of overlapping variable fragments.
  OverlapMap OverlapFragments;
  VarToFragments SeenFragments;

  /// Tests whether this instruction is a spill to a stack slot.
  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF);

  /// Decide if @MI is a spill instruction and return true if it is. We use 2
  /// criteria to make this decision:
  /// - Is this instruction a store to a spill slot?
  /// - Is there a register operand that is both used and killed?
  /// TODO: Store optimization can fold spills into other stores (including
  /// other spills). We do not handle this yet (more than one memory operand).
  bool isLocationSpill(const MachineInstr &MI, MachineFunction *MF,
                       unsigned &Reg);

  /// If a given instruction is identified as a spill, return the spill slot
  /// and set \p Reg to the spilled register.
  Optional<SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                          MachineFunction *MF, unsigned &Reg);

  /// Given a spill instruction, extract the register and offset used to
  /// address the spill slot in a target independent way.
  SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);

  /// Observe a single instruction while stepping through a block.
  void process(MachineInstr &MI);

  /// Examines whether \p MI is a DBG_VALUE and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferDebugValue(const MachineInstr &MI);

  /// Examines whether \p MI is copy instruction, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferRegisterCopy(MachineInstr &MI);

  /// Examines whether \p MI is stack spill or restore  instruction, and
  /// notifies trackers. \returns true if MI was recognized and processed.
  bool transferSpillOrRestoreInst(MachineInstr &MI);

  /// Examines \p MI for any registers that it defines, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  void transferRegisterDef(MachineInstr &MI);

  /// Copy one location to the other, accounting for movement of subregisters
  /// too.
  void performCopy(Register Src, Register Dst);

  void accumulateFragmentMap(MachineInstr &MI);

  /// Step through the function, recording register definitions and movements
  /// in an MLocTracker. Convert the observations into a per-block transfer
  /// function in \p MLocTransfer, suitable for using with the machine value
  /// location dataflow problem. Do the same with VLoc trackers in \p VLocs,
  /// although the precise machine value numbers can't be known until after
  /// the machine value number problem is solved.
  void produceTransferFunctions(MachineFunction &MF,
                                SmallVectorImpl<MLocTransferMap> &MLocTransfer,
                                unsigned MaxNumBlocks,
                                SmallVectorImpl<VLocTracker> &VLocs);

  /// Solve the machine value location dataflow problem. Takes as input the
  /// transfer functions in \p MLocTransfer. Writes the output live-in and
  /// live-out arrays to the (initialized to zero) multidimensional arrays in
  /// \p MInLocs and \p MOutLocs. The outer dimension is indexed by block
  /// number, the inner by LocIdx.
  void mlocDataflow(ValueIDNum **MInLocs, ValueIDNum **MOutLocs,
                    SmallVectorImpl<MLocTransferMap> &MLocTransfer);

  /// Perform a control flow join (lattice value meet) of the values in machine
  /// locations at \p MBB. Follows the algorithm described in the file-comment,
  /// reading live-outs of predecessors from \p OutLocs, the current live ins
  /// from \p InLocs, and assigning the newly computed live ins back into
  /// \p InLocs. \returns two bools -- the first indicates whether a change
  /// was made, the second whether a lattice downgrade occurred. If the latter
  /// is true, revisiting this block is necessary.
  std::tuple<bool, bool>
  mlocJoin(MachineBasicBlock &MBB,
           SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
           ValueIDNum **OutLocs, ValueIDNum *InLocs);

  /// Solve the variable value dataflow problem, for a single lexical scope.
  /// Uses the algorithm from the file comment to resolve control flow joins,
  /// although there are extra hacks, see vlocJoin. Reads the
  /// locations of values from the \p MInLocs and \p MOutLocs arrays (see
  /// mlocDataflow) and reads the variable values transfer function from
  /// \p AllTheVlocs. Live-in and Live-out variable values are stored locally,
  /// with the live-ins permanently stored to \p Output once the fixedpoint is
  /// reached.
  /// \p VarsWeCareAbout contains a collection of the variables in \p Scope
  /// that we should be tracking.
  /// \p AssignBlocks contains the set of blocks that aren't in \p Scope, but
  /// which do contain DBG_VALUEs, which VarLocBasedImpl tracks locations
  /// through.
  void vlocDataflow(const LexicalScope *Scope, const DILocation *DILoc,
                    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
                    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
                    LiveInsT &Output, ValueIDNum **MOutLocs,
                    ValueIDNum **MInLocs,
                    SmallVectorImpl<VLocTracker> &AllTheVLocs);

  /// Compute the live-ins to a block, considering control flow merges according
  /// to the method in the file comment. Live out and live in variable values
  /// are stored in \p VLOCOutLocs and \p VLOCInLocs. The live-ins for \p MBB
  /// are computed and stored into \p VLOCInLocs. \returns true if the live-ins
  /// are modified.
  /// \p InLocsT Output argument, storage for calculated live-ins.
  /// \returns two bools -- the first indicates whether a change
  /// was made, the second whether a lattice downgrade occurred. If the latter
  /// is true, revisiting this block is necessary.
  std::tuple<bool, bool>
  vlocJoin(MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs, LiveIdxT &VLOCInLocs,
           SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited,
           unsigned BBNum, const SmallSet<DebugVariable, 4> &AllVars,
           ValueIDNum **MOutLocs, ValueIDNum **MInLocs,
           SmallPtrSet<const MachineBasicBlock *, 8> &InScopeBlocks,
           SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
           DenseMap<DebugVariable, DbgValue> &InLocsT);

  /// Continue exploration of the variable-value lattice, as explained in the
  /// file-level comment. \p OldLiveInLocation contains the current
  /// exploration position, from which we need to descend further. \p Values
  /// contains the set of live-in values, \p CurBlockRPONum the RPO number of
  /// the current block, and \p CandidateLocations a set of locations that
  /// should be considered as PHI locations, if we reach the bottom of the
  /// lattice. \returns true if we should downgrade; the value is the agreeing
  /// value number in a non-backedge predecessor.
  bool vlocDowngradeLattice(const MachineBasicBlock &MBB,
                            const DbgValue &OldLiveInLocation,
                            const SmallVectorImpl<InValueT> &Values,
                            unsigned CurBlockRPONum);

  /// For the given block and live-outs feeding into it, try to find a
  /// machine location where they all join. If a solution for all predecessors
  /// can't be found, a location where all non-backedge-predecessors join
  /// will be returned instead. While this method finds a join location, this
  /// says nothing as to whether it should be used.
  /// \returns Pair of value ID if found, and true when the correct value
  /// is available on all predecessor edges, or false if it's only available
  /// for non-backedge predecessors.
  std::tuple<Optional<ValueIDNum>, bool>
  pickVPHILoc(MachineBasicBlock &MBB, const DebugVariable &Var,
              const LiveIdxT &LiveOuts, ValueIDNum **MOutLocs,
              ValueIDNum **MInLocs,
              const SmallVectorImpl<MachineBasicBlock *> &BlockOrders);

  /// Given the solutions to the two dataflow problems, machine value locations
  /// in \p MInLocs and live-in variable values in \p SavedLiveIns, runs the
  /// TransferTracker class over the function to produce live-in and transfer
  /// DBG_VALUEs, then inserts them. Groups of DBG_VALUEs are inserted in the
  /// order given by AllVarsNumbering -- this could be any stable order, but
  /// right now "order of appearence in function, when explored in RPO", so
  /// that we can compare explictly against VarLocBasedImpl.
  void emitLocations(MachineFunction &MF, LiveInsT SavedLiveIns,
                     ValueIDNum **MInLocs,
                     DenseMap<DebugVariable, unsigned> &AllVarsNumbering);

  /// Boilerplate computation of some initial sets, artifical blocks and
  /// RPOT block ordering.
  void initialSetup(MachineFunction &MF);

  bool ExtendRanges(MachineFunction &MF, TargetPassConfig *TPC) override;

public:
  /// Default construct and initialize the pass.
  InstrRefBasedLDV();

  LLVM_DUMP_METHOD
  void dump_mloc_transfer(const MLocTransferMap &mloc_transfer) const;

  bool isCalleeSaved(LocIdx L) {
    unsigned Reg = MTracker->LocIdxToLocID[L];
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

ValueIDNum ValueIDNum::EmptyValue = {UINT_MAX, UINT_MAX, UINT_MAX};

/// Default construct and initialize the pass.
InstrRefBasedLDV::InstrRefBasedLDV() {}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
// Something to restore in the future.
// void InstrRefBasedLDV::printVarLocInMBB(..)
#endif

SpillLoc
InstrRefBasedLDV::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  Register Reg;
  int Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
bool InstrRefBasedLDV::transferDebugValue(const MachineInstr &MI) {
  if (!MI.isDebugValue())
    return false;

  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);

  // If there are no instructions in this lexical scope, do no location tracking
  // at all, this variable shouldn't get a legitimate location range.
  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // handled it; by doing nothing

  const MachineOperand &MO = MI.getOperand(0);

  // MLocTracker needs to know that this register is read, even if it's only
  // read by a debug inst.
  if (MO.isReg() && MO.getReg() != 0)
    (void)MTracker->readReg(MO.getReg());

  // If we're preparing for the second analysis (variables), the machine value
  // locations are already solved, and we report this DBG_VALUE and the value
  // it refers to to VLocTracker.
  if (VTracker) {
    if (MO.isReg()) {
      // Feed defVar the new variable location, or if this is a
      // DBG_VALUE $noreg, feed defVar None.
      if (MO.getReg())
        VTracker->defVar(MI, MTracker->readReg(MO.getReg()));
      else
        VTracker->defVar(MI, None);
    } else if (MI.getOperand(0).isImm() || MI.getOperand(0).isFPImm() ||
               MI.getOperand(0).isCImm()) {
      VTracker->defVar(MI, MI.getOperand(0));
    }
  }

  // If performing final tracking of transfers, report this variable definition
  // to the TransferTracker too.
  if (TTracker)
    TTracker->redefVar(MI);
  return true;
}

void InstrRefBasedLDV::transferRegisterDef(MachineInstr &MI) {
  // Meta Instructions do not affect the debug liveness of any register they
  // define.
  if (MI.isImplicitDef()) {
    // Except when there's an implicit def, and the location it's defining has
    // no value number. The whole point of an implicit def is to announce that
    // the register is live, without be specific about it's value. So define
    // a value if there isn't one already.
    ValueIDNum Num = MTracker->readReg(MI.getOperand(0).getReg());
    // Has a legitimate value -> ignore the implicit def.
    if (Num.getLoc() != 0)
      return;
    // Otherwise, def it here.
  } else if (MI.isMetaInstruction())
    return;

  MachineFunction *MF = MI.getMF();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();

  // Find the regs killed by MI, and find regmasks of preserved regs.
  // Max out the number of statically allocated elements in `DeadRegs`, as this
  // prevents fallback to std::set::count() operations.
  SmallSet<uint32_t, 32> DeadRegs;
  SmallVector<const uint32_t *, 4> RegMasks;
  SmallVector<const MachineOperand *, 4> RegMaskPtrs;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        Register::isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MO.getReg() == SP)) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        // FIXME: Can we break out of this loop early if no insertion occurs?
        DeadRegs.insert(*RAI);
    } else if (MO.isRegMask()) {
      RegMasks.push_back(MO.getRegMask());
      RegMaskPtrs.push_back(&MO);
    }
  }

  // Tell MLocTracker about all definitions, of regmasks and otherwise.
  for (uint32_t DeadReg : DeadRegs)
    MTracker->defReg(DeadReg, CurBB, CurInst);

  for (auto *MO : RegMaskPtrs)
    MTracker->writeRegMask(MO, CurBB, CurInst);
}

void InstrRefBasedLDV::performCopy(Register SrcRegNum, Register DstRegNum) {
  ValueIDNum SrcValue = MTracker->readReg(SrcRegNum);

  MTracker->setReg(DstRegNum, SrcValue);

  // In all circumstances, re-def the super registers. It's definitely a new
  // value now. This doesn't uniquely identify the composition of subregs, for
  // example, two identical values in subregisters composed in different
  // places would not get equal value numbers.
  for (MCSuperRegIterator SRI(DstRegNum, TRI); SRI.isValid(); ++SRI)
    MTracker->defReg(*SRI, CurBB, CurInst);

  // If we're emulating VarLocBasedImpl, just define all the subregisters.
  // DBG_VALUEs of them will expect to be tracked from the DBG_VALUE, not
  // through prior copies.
  if (EmulateOldLDV) {
    for (MCSubRegIndexIterator DRI(DstRegNum, TRI); DRI.isValid(); ++DRI)
      MTracker->defReg(DRI.getSubReg(), CurBB, CurInst);
    return;
  }

  // Otherwise, actually copy subregisters from one location to another.
  // XXX: in addition, any subregisters of DstRegNum that don't line up with
  // the source register should be def'd.
  for (MCSubRegIndexIterator SRI(SrcRegNum, TRI); SRI.isValid(); ++SRI) {
    unsigned SrcSubReg = SRI.getSubReg();
    unsigned SubRegIdx = SRI.getSubRegIndex();
    unsigned DstSubReg = TRI->getSubReg(DstRegNum, SubRegIdx);
    if (!DstSubReg)
      continue;

    // Do copy. There are two matching subregisters, the source value should
    // have been def'd when the super-reg was, the latter might not be tracked
    // yet.
    // This will force SrcSubReg to be tracked, if it isn't yet.
    (void)MTracker->readReg(SrcSubReg);
    LocIdx SrcL = MTracker->getRegMLoc(SrcSubReg);
    assert(SrcL.asU64());
    (void)MTracker->readReg(DstSubReg);
    LocIdx DstL = MTracker->getRegMLoc(DstSubReg);
    assert(DstL.asU64());
    (void)DstL;
    ValueIDNum CpyValue = {SrcValue.getBlock(), SrcValue.getInst(), SrcL};

    MTracker->setReg(DstSubReg, CpyValue);
  }
}

bool InstrRefBasedLDV::isSpillInstruction(const MachineInstr &MI,
                                          MachineFunction *MF) {
  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  return true;
}

bool InstrRefBasedLDV::isLocationSpill(const MachineInstr &MI,
                                       MachineFunction *MF, unsigned &Reg) {
  if (!isSpillInstruction(MI, MF))
    return false;

  // XXX FIXME: On x86, isStoreToStackSlotPostFE returns '1' instead of an
  // actual register number.
  if (ObserveAllStackops) {
    int FI;
    Reg = TII->isStoreToStackSlotPostFE(MI, FI);
    return Reg != 0;
  }

  auto isKilledReg = [&](const MachineOperand MO, unsigned &Reg) {
    if (!MO.isReg() || !MO.isUse()) {
      Reg = 0;
      return false;
    }
    Reg = MO.getReg();
    return MO.isKill();
  };

  for (const MachineOperand &MO : MI.operands()) {
    // In a spill instruction generated by the InlineSpiller the spilled
    // register has its kill flag set.
    if (isKilledReg(MO, Reg))
      return true;
    if (Reg != 0) {
      // Check whether next instruction kills the spilled register.
      // FIXME: Current solution does not cover search for killed register in
      // bundles and instructions further down the chain.
      auto NextI = std::next(MI.getIterator());
      // Skip next instruction that points to basic block end iterator.
      if (MI.getParent()->end() == NextI)
        continue;
      unsigned RegNext;
      for (const MachineOperand &MONext : NextI->operands()) {
        // Return true if we came across the register from the
        // previous spill instruction that is killed in NextI.
        if (isKilledReg(MONext, RegNext) && RegNext == Reg)
          return true;
      }
    }
  }
  // Return false if we didn't find spilled register.
  return false;
}

Optional<SpillLoc>
InstrRefBasedLDV::isRestoreInstruction(const MachineInstr &MI,
                                       MachineFunction *MF, unsigned &Reg) {
  if (!MI.hasOneMemOperand())
    return None;

  // FIXME: Handle folded restore instructions with more than one memory
  // operand.
  if (MI.getRestoreSize(TII)) {
    Reg = MI.getOperand(0).getReg();
    return extractSpillBaseRegAndOffset(MI);
  }
  return None;
}

bool InstrRefBasedLDV::transferSpillOrRestoreInst(MachineInstr &MI) {
  // XXX -- it's too difficult to implement VarLocBasedImpl's  stack location
  // limitations under the new model. Therefore, when comparing them, compare
  // versions that don't attempt spills or restores at all.
  if (EmulateOldLDV)
    return false;

  MachineFunction *MF = MI.getMF();
  unsigned Reg;
  Optional<SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, terminate that variable location. The value in memory
  // will have changed. DbgEntityHistoryCalculator doesn't try to detect this.
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);

    if (TTracker) {
      Optional<LocIdx> MLoc = MTracker->getSpillMLoc(*Loc);
      if (MLoc)
        TTracker->clobberMloc(*MLoc, MI.getIterator());
    }
  }

  // Try to recognise spill and restore instructions that may transfer a value.
  if (isLocationSpill(MI, MF, Reg)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    auto ValueID = MTracker->readReg(Reg);

    // If the location is empty, produce a phi, signify it's the live-in value.
    if (ValueID.getLoc() == 0)
      ValueID = {CurBB, 0, MTracker->getRegMLoc(Reg)};

    MTracker->setSpill(*Loc, ValueID);
    auto OptSpillLocIdx = MTracker->getSpillMLoc(*Loc);
    assert(OptSpillLocIdx && "Spill slot set but has no LocIdx?");
    LocIdx SpillLocIdx = *OptSpillLocIdx;

    // Tell TransferTracker about this spill, produce DBG_VALUEs for it.
    if (TTracker)
      TTracker->transferMlocs(MTracker->getRegMLoc(Reg), SpillLocIdx,
                              MI.getIterator());

    // VarLocBasedImpl would, at this point, stop tracking the source
    // register of the store.
    if (EmulateOldLDV) {
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);
    }
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return false;

    // Is there a value to be restored?
    auto OptValueID = MTracker->readSpill(*Loc);
    if (OptValueID) {
      ValueIDNum ValueID = *OptValueID;
      LocIdx SpillLocIdx = *MTracker->getSpillMLoc(*Loc);
      // XXX -- can we recover sub-registers of this value? Until we can, first
      // overwrite all defs of the register being restored to.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Now override the reg we're restoring to.
      MTracker->setReg(Reg, ValueID);

      // Report this restore to the transfer tracker too.
      if (TTracker)
        TTracker->transferMlocs(SpillLocIdx, MTracker->getRegMLoc(Reg),
                                MI.getIterator());
    } else {
      // There isn't anything in the location; not clear if this is a code path
      // that still runs. Def this register anyway just in case.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Force the spill slot to be tracked.
      LocIdx L = MTracker->getOrTrackSpillLoc(*Loc);

      // Set the restored value to be a machine phi number, signifying that it's
      // whatever the spills live-in value is in this block. Definitely has
      // a LocIdx due to the setSpill above.
      ValueIDNum ValueID = {CurBB, 0, L};
      MTracker->setReg(Reg, ValueID);
      MTracker->setSpill(*Loc, ValueID);
    }
  }
  return true;
}

bool InstrRefBasedLDV::transferRegisterCopy(MachineInstr &MI) {
  auto DestSrc = TII->isCopyInstr(MI);
  if (!DestSrc)
    return false;

  const MachineOperand *DestRegOp = DestSrc->Destination;
  const MachineOperand *SrcRegOp = DestSrc->Source;

  auto isCalleeSavedReg = [&](unsigned Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  Register SrcReg = SrcRegOp->getReg();
  Register DestReg = DestRegOp->getReg();

  // Ignore identity copies. Yep, these make it as far as LiveDebugValues.
  if (SrcReg == DestReg)
    return true;

  // For emulating VarLocBasedImpl:
  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register, which is callee saved, is
  // going to stay unclobbered longer, even if it is killed.
  //
  // For InstrRefBasedImpl, we can track multiple locations per value, so
  // ignore this condition.
  if (EmulateOldLDV && !isCalleeSavedReg(DestReg))
    return false;

  // InstrRefBasedImpl only followed killing copies.
  if (EmulateOldLDV && !SrcRegOp->isKill())
    return false;

  // Copy MTracker info, including subregs if available.
  InstrRefBasedLDV::performCopy(SrcReg, DestReg);

  // Only produce a transfer of DBG_VALUE within a block where old LDV
  // would have. We might make use of the additional value tracking in some
  // other way, later.
  if (TTracker && isCalleeSavedReg(DestReg) && SrcRegOp->isKill())
    TTracker->transferMlocs(MTracker->getRegMLoc(SrcReg),
                            MTracker->getRegMLoc(DestReg), MI.getIterator());

  // VarLocBasedImpl would quit tracking the old location after copying.
  if (EmulateOldLDV && SrcReg != DestReg)
    MTracker->defReg(SrcReg, CurBB, CurInst);

  return true;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
void InstrRefBasedLDV::accumulateFragmentMap(MachineInstr &MI) {
  DebugVariable MIVar(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
  FragmentInfo ThisFragment = MIVar.getFragmentOrDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVariable());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVariable(), OneFragment});

    OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
  if (!IsInOLapMap.second)
    return;

  auto &ThisFragmentsOverlaps = IsInOLapMap.first->second;
  auto &AllSeenFragments = SeenIt->second;

  // Otherwise, examine all other seen fragments for this variable, with "this"
  // fragment being a previously unseen fragment. Record any pair of
  // overlapping fragments.
  for (auto &ASeenFragment : AllSeenFragments) {
    // Does this previously seen fragment overlap?
    if (DIExpression::fragmentsOverlap(ThisFragment, ASeenFragment)) {
      // Yes: Mark the current fragment as being overlapped.
      ThisFragmentsOverlaps.push_back(ASeenFragment);
      // Mark the previously seen fragment as being overlapped by the current
      // one.
      auto ASeenFragmentsOverlaps =
          OverlapFragments.find({MIVar.getVariable(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlapFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

void InstrRefBasedLDV::process(MachineInstr &MI) {
  // Try to interpret an MI as a debug or transfer instruction. Only if it's
  // none of these should we interpret it's register defs as new value
  // definitions.
  if (transferDebugValue(MI))
    return;
  if (transferRegisterCopy(MI))
    return;
  if (transferSpillOrRestoreInst(MI))
    return;
  transferRegisterDef(MI);
}

void InstrRefBasedLDV::produceTransferFunctions(
    MachineFunction &MF, SmallVectorImpl<MLocTransferMap> &MLocTransfer,
    unsigned MaxNumBlocks, SmallVectorImpl<VLocTracker> &VLocs) {
  // Because we try to optimize around register mask operands by ignoring regs
  // that aren't currently tracked, we set up something ugly for later: RegMask
  // operands that are seen earlier than the first use of a register, still need
  // to clobber that register in the transfer function. But this information
  // isn't actively recorded. Instead, we track each RegMask used in each block,
  // and accumulated the clobbered but untracked registers in each block into
  // the following bitvector. Later, if new values are tracked, we can add
  // appropriate clobbers.
  SmallVector<BitVector, 32> BlockMasks;
  BlockMasks.resize(MaxNumBlocks);

  // Reserve one bit per register for the masks described above.
  unsigned BVWords = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  for (auto &BV : BlockMasks)
    BV.resize(TRI->getNumRegs(), true);

  // Step through all instructions and inhale the transfer function.
  for (auto &MBB : MF) {
    // Object fields that are read by trackers to know where we are in the
    // function.
    CurBB = MBB.getNumber();
    CurInst = 1;

    // Set all machine locations to a PHI value. For transfer function
    // production only, this signifies the live-in value to the block.
    MTracker->reset();
    MTracker->setMPhis(CurBB);

    VTracker = &VLocs[CurBB];
    VTracker->MBB = &MBB;

    // Step through each instruction in this block.
    for (auto &MI : MBB) {
      process(MI);
      // Also accumulate fragment map.
      if (MI.isDebugValue())
        accumulateFragmentMap(MI);
      ++CurInst;
    }

    // Produce the transfer function, a map of machine location to new value. If
    // any machine location has the live-in phi value from the start of the
    // block, it's live-through and doesn't need recording in the transfer
    // function.
    for (auto Location : MTracker->locations()) {
      LocIdx Idx = Location.Idx;
      ValueIDNum &P = Location.Value;
      if (P.isPHI() && P.getLoc() == Idx.asU64())
        continue;

      // Insert-or-update.
      auto &TransferMap = MLocTransfer[CurBB];
      auto Result = TransferMap.insert(std::make_pair(Idx.asU64(), P));
      if (!Result.second)
        Result.first->second = P;
    }

    // Accumulate any bitmask operands into the clobberred reg mask for this
    // block.
    for (auto &P : MTracker->Masks) {
      BlockMasks[CurBB].clearBitsNotInMask(P.first->getRegMask(), BVWords);
    }
  }

  // Compute a bitvector of all the registers that are tracked in this block.
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  BitVector UsedRegs(TRI->getNumRegs());
  for (auto Location : MTracker->locations()) {
    unsigned ID = MTracker->LocIdxToLocID[Location.Idx];
    if (ID >= TRI->getNumRegs() || ID == SP)
      continue;
    UsedRegs.set(ID);
  }

  // Check that any regmask-clobber of a register that gets tracked, is not
  // live-through in the transfer function. It needs to be clobbered at the
  // very least.
  for (unsigned int I = 0; I < MaxNumBlocks; ++I) {
    BitVector &BV = BlockMasks[I];
    BV.flip();
    BV &= UsedRegs;
    // This produces all the bits that we clobber, but also use. Check that
    // they're all clobbered or at least set in the designated transfer
    // elem.
    for (unsigned Bit : BV.set_bits()) {
      unsigned ID = MTracker->getLocID(Bit, false);
      LocIdx Idx = MTracker->LocIDToLocIdx[ID];
      auto &TransferMap = MLocTransfer[I];

      // Install a value representing the fact that this location is effectively
      // written to in this block. As there's no reserved value, instead use
      // a value number that is never generated. Pick the value number for the
      // first instruction in the block, def'ing this location, which we know
      // this block never used anyway.
      ValueIDNum NotGeneratedNum = ValueIDNum(I, 1, Idx);
      auto Result =
        TransferMap.insert(std::make_pair(Idx.asU64(), NotGeneratedNum));
      if (!Result.second) {
        ValueIDNum &ValueID = Result.first->second;
        if (ValueID.getBlock() == I && ValueID.isPHI())
          // It was left as live-through. Set it to clobbered.
          ValueID = NotGeneratedNum;
      }
    }
  }
}

std::tuple<bool, bool>
InstrRefBasedLDV::mlocJoin(MachineBasicBlock &MBB,
                           SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
                           ValueIDNum **OutLocs, ValueIDNum *InLocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;
  bool DowngradeOccurred = false;

  // Collect predecessors that have been visited. Anything that hasn't been
  // visited yet is a backedge on the first iteration, and the meet of it's
  // lattice value for all locations will be unaffected.
  SmallVector<const MachineBasicBlock *, 8> BlockOrders;
  for (auto Pred : MBB.predecessors()) {
    if (Visited.count(Pred)) {
      BlockOrders.push_back(Pred);
    }
  }

  // Visit predecessors in RPOT order.
  auto Cmp = [&](const MachineBasicBlock *A, const MachineBasicBlock *B) {
    return BBToOrder.find(A)->second < BBToOrder.find(B)->second;
  };
  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);

  // Skip entry block.
  if (BlockOrders.size() == 0)
    return std::tuple<bool, bool>(false, false);

  // Step through all machine locations, then look at each predecessor and
  // detect disagreements.
  unsigned ThisBlockRPO = BBToOrder.find(&MBB)->second;
  for (auto Location : MTracker->locations()) {
    LocIdx Idx = Location.Idx;
    // Pick out the first predecessors live-out value for this location. It's
    // guaranteed to be not a backedge, as we order by RPO.
    ValueIDNum BaseVal = OutLocs[BlockOrders[0]->getNumber()][Idx.asU64()];

    // Some flags for whether there's a disagreement, and whether it's a
    // disagreement with a backedge or not.
    bool Disagree = false;
    bool NonBackEdgeDisagree = false;

    // Loop around everything that wasn't 'base'.
    for (unsigned int I = 1; I < BlockOrders.size(); ++I) {
      auto *MBB = BlockOrders[I];
      if (BaseVal != OutLocs[MBB->getNumber()][Idx.asU64()]) {
        // Live-out of a predecessor disagrees with the first predecessor.
        Disagree = true;

        // Test whether it's a disagreemnt in the backedges or not.
        if (BBToOrder.find(MBB)->second < ThisBlockRPO) // might be self b/e
          NonBackEdgeDisagree = true;
      }
    }

    bool OverRide = false;
    if (Disagree && !NonBackEdgeDisagree) {
      // Only the backedges disagree. Consider demoting the livein
      // lattice value, as per the file level comment. The value we consider
      // demoting to is the value that the non-backedge predecessors agree on.
      // The order of values is that non-PHIs are \top, a PHI at this block
      // \bot, and phis between the two are ordered by their RPO number.
      // If there's no agreement, or we've already demoted to this PHI value
      // before, replace with a PHI value at this block.

      // Calculate order numbers: zero means normal def, nonzero means RPO
      // number.
      unsigned BaseBlockRPONum = BBNumToRPO[BaseVal.getBlock()] + 1;
      if (!BaseVal.isPHI())
        BaseBlockRPONum = 0;

      ValueIDNum &InLocID = InLocs[Idx.asU64()];
      unsigned InLocRPONum = BBNumToRPO[InLocID.getBlock()] + 1;
      if (!InLocID.isPHI())
        InLocRPONum = 0;

      // Should we ignore the disagreeing backedges, and override with the
      // value the other predecessors agree on (in "base")?
      unsigned ThisBlockRPONum = BBNumToRPO[MBB.getNumber()] + 1;
      if (BaseBlockRPONum > InLocRPONum && BaseBlockRPONum < ThisBlockRPONum) {
        // Override.
        OverRide = true;
        DowngradeOccurred = true;
      }
    }
    // else: if we disagree in the non-backedges, then this is definitely
    // a control flow merge where different values merge. Make it a PHI.

    // Generate a phi...
    ValueIDNum PHI = {(uint64_t)MBB.getNumber(), 0, Idx};
    ValueIDNum NewVal = (Disagree && !OverRide) ? PHI : BaseVal;
    if (InLocs[Idx.asU64()] != NewVal) {
      Changed |= true;
      InLocs[Idx.asU64()] = NewVal;
    }
  }

  // Uhhhhhh, reimplement NumInserted and NumRemoved pls.
  return std::tuple<bool, bool>(Changed, DowngradeOccurred);
}

void InstrRefBasedLDV::mlocDataflow(
    ValueIDNum **MInLocs, ValueIDNum **MOutLocs,
    SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  // We track what is on the current and pending worklist to avoid inserting
  // the same thing twice. We could avoid this with a custom priority queue,
  // but this is probably not worth it.
  SmallPtrSet<MachineBasicBlock *, 16> OnPending, OnWorklist;

  // Initialize worklist with every block to be visited.
  for (unsigned int I = 0; I < BBToOrder.size(); ++I) {
    Worklist.push(I);
    OnWorklist.insert(OrderToBB[I]);
  }

  MTracker->reset();

  // Set inlocs for entry block -- each as a PHI at the entry block. Represents
  // the incoming value to the function.
  MTracker->setMPhis(0);
  for (auto Location : MTracker->locations())
    MInLocs[0][Location.Idx.asU64()] = Location.Value;

  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // Vector for storing the evaluated block transfer function.
    SmallVector<std::pair<LocIdx, ValueIDNum>, 32> ToRemap;

    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      CurBB = MBB->getNumber();
      Worklist.pop();

      // Join the values in all predecessor blocks.
      bool InLocsChanged, DowngradeOccurred;
      std::tie(InLocsChanged, DowngradeOccurred) =
          mlocJoin(*MBB, Visited, MOutLocs, MInLocs[CurBB]);
      InLocsChanged |= Visited.insert(MBB).second;

      // If a downgrade occurred, book us in for re-examination on the next
      // iteration.
      if (DowngradeOccurred && OnPending.insert(MBB).second)
        Pending.push(BBToOrder[MBB]);

      // Don't examine transfer function if we've visited this loc at least
      // once, and inlocs haven't changed.
      if (!InLocsChanged)
        continue;

      // Load the current set of live-ins into MLocTracker.
      MTracker->loadFromArray(MInLocs[CurBB], CurBB);

      // Each element of the transfer function can be a new def, or a read of
      // a live-in value. Evaluate each element, and store to "ToRemap".
      ToRemap.clear();
      for (auto &P : MLocTransfer[CurBB]) {
        if (P.second.getBlock() == CurBB && P.second.isPHI()) {
          // This is a movement of whatever was live in. Read it.
          ValueIDNum NewID = MTracker->getNumAtPos(P.second.getLoc());
          ToRemap.push_back(std::make_pair(P.first, NewID));
        } else {
          // It's a def. Just set it.
          assert(P.second.getBlock() == CurBB);
          ToRemap.push_back(std::make_pair(P.first, P.second));
        }
      }

      // Commit the transfer function changes into mloc tracker, which
      // transforms the contents of the MLocTracker into the live-outs.
      for (auto &P : ToRemap)
        MTracker->setMLoc(P.first, P.second);

      // Now copy out-locs from mloc tracker into out-loc vector, checking
      // whether changes have occurred. These changes can have come from both
      // the transfer function, and mlocJoin.
      bool OLChanged = false;
      for (auto Location : MTracker->locations()) {
        OLChanged |= MOutLocs[CurBB][Location.Idx.asU64()] != Location.Value;
        MOutLocs[CurBB][Location.Idx.asU64()] = Location.Value;
      }

      MTracker->reset();

      // No need to examine successors again if out-locs didn't change.
      if (!OLChanged)
        continue;

      // All successors should be visited: put any back-edges on the pending
      // list for the next dataflow iteration, and any other successors to be
      // visited this iteration, if they're not going to be already.
      for (auto s : MBB->successors()) {
        // Does branching to this successor represent a back-edge?
        if (BBToOrder[s] > BBToOrder[MBB]) {
          // No: visit it during this dataflow iteration.
          if (OnWorklist.insert(s).second)
            Worklist.push(BBToOrder[s]);
        } else {
          // Yes: visit it on the next iteration.
          if (OnPending.insert(s).second)
            Pending.push(BBToOrder[s]);
        }
      }
    }

    Worklist.swap(Pending);
    std::swap(OnPending, OnWorklist);
    OnPending.clear();
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // Once all the live-ins don't change on mlocJoin(), we've reached a
  // fixedpoint.
}

bool InstrRefBasedLDV::vlocDowngradeLattice(
    const MachineBasicBlock &MBB, const DbgValue &OldLiveInLocation,
    const SmallVectorImpl<InValueT> &Values, unsigned CurBlockRPONum) {
  // Ranking value preference: see file level comment, the highest rank is
  // a plain def, followed by PHI values in reverse post-order. Numerically,
  // we assign all defs the rank '0', all PHIs their blocks RPO number plus
  // one, and consider the lowest value the highest ranked.
  int OldLiveInRank = BBNumToRPO[OldLiveInLocation.ID.getBlock()] + 1;
  if (!OldLiveInLocation.ID.isPHI())
    OldLiveInRank = 0;

  // Allow any unresolvable conflict to be over-ridden.
  if (OldLiveInLocation.Kind == DbgValue::NoVal) {
    // Although if it was an unresolvable conflict from _this_ block, then
    // all other seeking of downgrades and PHIs must have failed before hand.
    if (OldLiveInLocation.BlockNo == (unsigned)MBB.getNumber())
      return false;
    OldLiveInRank = INT_MIN;
  }

  auto &InValue = *Values[0].second;

  if (InValue.Kind == DbgValue::Const || InValue.Kind == DbgValue::NoVal)
    return false;

  unsigned ThisRPO = BBNumToRPO[InValue.ID.getBlock()];
  int ThisRank = ThisRPO + 1;
  if (!InValue.ID.isPHI())
    ThisRank = 0;

  // Too far down the lattice?
  if (ThisRPO >= CurBlockRPONum)
    return false;

  // Higher in the lattice than what we've already explored?
  if (ThisRank <= OldLiveInRank)
    return false;

  return true;
}

std::tuple<Optional<ValueIDNum>, bool> InstrRefBasedLDV::pickVPHILoc(
    MachineBasicBlock &MBB, const DebugVariable &Var, const LiveIdxT &LiveOuts,
    ValueIDNum **MOutLocs, ValueIDNum **MInLocs,
    const SmallVectorImpl<MachineBasicBlock *> &BlockOrders) {
  // Collect a set of locations from predecessor where its live-out value can
  // be found.
  SmallVector<SmallVector<LocIdx, 4>, 8> Locs;
  unsigned NumLocs = MTracker->getNumLocs();
  unsigned BackEdgesStart = 0;

  for (auto p : BlockOrders) {
    // Pick out where backedges start in the list of predecessors. Relies on
    // BlockOrders being sorted by RPO.
    if (BBToOrder[p] < BBToOrder[&MBB])
      ++BackEdgesStart;

    // For each predecessor, create a new set of locations.
    Locs.resize(Locs.size() + 1);
    unsigned ThisBBNum = p->getNumber();
    auto LiveOutMap = LiveOuts.find(p);
    if (LiveOutMap == LiveOuts.end())
      // This predecessor isn't in scope, it must have no live-in/live-out
      // locations.
      continue;

    auto It = LiveOutMap->second->find(Var);
    if (It == LiveOutMap->second->end())
      // There's no value recorded for this variable in this predecessor,
      // leave an empty set of locations.
      continue;

    const DbgValue &OutVal = It->second;

    if (OutVal.Kind == DbgValue::Const || OutVal.Kind == DbgValue::NoVal)
      // Consts and no-values cannot have locations we can join on.
      continue;

    assert(OutVal.Kind == DbgValue::Proposed || OutVal.Kind == DbgValue::Def);
    ValueIDNum ValToLookFor = OutVal.ID;

    // Search the live-outs of the predecessor for the specified value.
    for (unsigned int I = 0; I < NumLocs; ++I) {
      if (MOutLocs[ThisBBNum][I] == ValToLookFor)
        Locs.back().push_back(LocIdx(I));
    }
  }

  // If there were no locations at all, return an empty result.
  if (Locs.empty())
    return std::tuple<Optional<ValueIDNum>, bool>(None, false);

  // Lambda for seeking a common location within a range of location-sets.
  typedef SmallVector<SmallVector<LocIdx, 4>, 8>::iterator LocsIt;
  auto SeekLocation =
      [&Locs](llvm::iterator_range<LocsIt> SearchRange) -> Optional<LocIdx> {
    // Starting with the first set of locations, take the intersection with
    // subsequent sets.
    SmallVector<LocIdx, 4> base = Locs[0];
    for (auto &S : SearchRange) {
      SmallVector<LocIdx, 4> new_base;
      std::set_intersection(base.begin(), base.end(), S.begin(), S.end(),
                            std::inserter(new_base, new_base.begin()));
      base = new_base;
    }
    if (base.empty())
      return None;

    // We now have a set of LocIdxes that contain the right output value in
    // each of the predecessors. Pick the lowest; if there's a register loc,
    // that'll be it.
    return *base.begin();
  };

  // Search for a common location for all predecessors. If we can't, then fall
  // back to only finding a common location between non-backedge predecessors.
  bool ValidForAllLocs = true;
  auto TheLoc = SeekLocation(Locs);
  if (!TheLoc) {
    ValidForAllLocs = false;
    TheLoc =
        SeekLocation(make_range(Locs.begin(), Locs.begin() + BackEdgesStart));
  }

  if (!TheLoc)
    return std::tuple<Optional<ValueIDNum>, bool>(None, false);

  // Return a PHI-value-number for the found location.
  LocIdx L = *TheLoc;
  ValueIDNum PHIVal = {(unsigned)MBB.getNumber(), 0, L};
  return std::tuple<Optional<ValueIDNum>, bool>(PHIVal, ValidForAllLocs);
}

std::tuple<bool, bool> InstrRefBasedLDV::vlocJoin(
    MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs, LiveIdxT &VLOCInLocs,
    SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited, unsigned BBNum,
    const SmallSet<DebugVariable, 4> &AllVars, ValueIDNum **MOutLocs,
    ValueIDNum **MInLocs,
    SmallPtrSet<const MachineBasicBlock *, 8> &InScopeBlocks,
    SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
    DenseMap<DebugVariable, DbgValue> &InLocsT) {
  bool DowngradeOccurred = false;

  // To emulate VarLocBasedImpl, process this block if it's not in scope but
  // _does_ assign a variable value. No live-ins for this scope are transferred
  // in though, so we can return immediately.
  if (InScopeBlocks.count(&MBB) == 0 && !ArtificialBlocks.count(&MBB)) {
    if (VLOCVisited)
      return std::tuple<bool, bool>(true, false);
    return std::tuple<bool, bool>(false, false);
  }

  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Find any live-ins computed in a prior iteration.
  auto ILSIt = VLOCInLocs.find(&MBB);
  assert(ILSIt != VLOCInLocs.end());
  auto &ILS = *ILSIt->second;

  // Order predecessors by RPOT order, for exploring them in that order.
  SmallVector<MachineBasicBlock *, 8> BlockOrders;
  for (auto p : MBB.predecessors())
    BlockOrders.push_back(p);

  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);

  unsigned CurBlockRPONum = BBToOrder[&MBB];

  // Force a re-visit to loop heads in the first dataflow iteration.
  // FIXME: if we could "propose" Const values this wouldn't be needed,
  // because they'd need to be confirmed before being emitted.
  if (!BlockOrders.empty() &&
      BBToOrder[BlockOrders[BlockOrders.size() - 1]] >= CurBlockRPONum &&
      VLOCVisited)
    DowngradeOccurred = true;

  auto ConfirmValue = [&InLocsT](const DebugVariable &DV, DbgValue VR) {
    auto Result = InLocsT.insert(std::make_pair(DV, VR));
    (void)Result;
    assert(Result.second);
  };

  auto ConfirmNoVal = [&ConfirmValue, &MBB](const DebugVariable &Var, const DbgValueProperties &Properties) {
    DbgValue NoLocPHIVal(MBB.getNumber(), Properties, DbgValue::NoVal);

    ConfirmValue(Var, NoLocPHIVal);
  };

  // Attempt to join the values for each variable.
  for (auto &Var : AllVars) {
    // Collect all the DbgValues for this variable.
    SmallVector<InValueT, 8> Values;
    bool Bail = false;
    unsigned BackEdgesStart = 0;
    for (auto p : BlockOrders) {
      // If the predecessor isn't in scope / to be explored, we'll never be
      // able to join any locations.
      if (BlocksToExplore.find(p) == BlocksToExplore.end()) {
        Bail = true;
        break;
      }

      // Don't attempt to handle unvisited predecessors: they're implicitly
      // "unknown"s in the lattice.
      if (VLOCVisited && !VLOCVisited->count(p))
        continue;

      // If the predecessors OutLocs is absent, there's not much we can do.
      auto OL = VLOCOutLocs.find(p);
      if (OL == VLOCOutLocs.end()) {
        Bail = true;
        break;
      }

      // No live-out value for this predecessor also means we can't produce
      // a joined value.
      auto VIt = OL->second->find(Var);
      if (VIt == OL->second->end()) {
        Bail = true;
        break;
      }

      // Keep track of where back-edges begin in the Values vector. Relies on
      // BlockOrders being sorted by RPO.
      unsigned ThisBBRPONum = BBToOrder[p];
      if (ThisBBRPONum < CurBlockRPONum)
        ++BackEdgesStart;

      Values.push_back(std::make_pair(p, &VIt->second));
    }

    // If there were no values, or one of the predecessors couldn't have a
    // value, then give up immediately. It's not safe to produce a live-in
    // value.
    if (Bail || Values.size() == 0)
      continue;

    // Enumeration identifying the current state of the predecessors values.
    enum {
      Unset = 0,
      Agreed,       // All preds agree on the variable value.
      PropDisagree, // All preds agree, but the value kind is Proposed in some.
      BEDisagree,   // Only back-edges disagree on variable value.
      PHINeeded,    // Non-back-edge predecessors have conflicing values.
      NoSolution    // Conflicting Value metadata makes solution impossible.
    } OurState = Unset;

    // All (non-entry) blocks have at least one non-backedge predecessor.
    // Pick the variable value from the first of these, to compare against
    // all others.
    const DbgValue &FirstVal = *Values[0].second;
    const ValueIDNum &FirstID = FirstVal.ID;

    // Scan for variable values that can't be resolved: if they have different
    // DIExpressions, different indirectness, or are mixed constants /
    // non-constants.
    for (auto &V : Values) {
      if (V.second->Properties != FirstVal.Properties)
        OurState = NoSolution;
      if (V.second->Kind == DbgValue::Const && FirstVal.Kind != DbgValue::Const)
        OurState = NoSolution;
    }

    // Flags diagnosing _how_ the values disagree.
    bool NonBackEdgeDisagree = false;
    bool DisagreeOnPHINess = false;
    bool IDDisagree = false;
    bool Disagree = false;
    if (OurState == Unset) {
      for (auto &V : Values) {
        if (*V.second == FirstVal)
          continue; // No disagreement.

        Disagree = true;

        // Flag whether the value number actually diagrees.
        if (V.second->ID != FirstID)
          IDDisagree = true;

        // Distinguish whether disagreement happens in backedges or not.
        // Relies on Values (and BlockOrders) being sorted by RPO.
        unsigned ThisBBRPONum = BBToOrder[V.first];
        if (ThisBBRPONum < CurBlockRPONum)
          NonBackEdgeDisagree = true;

        // Is there a difference in whether the value is definite or only
        // proposed?
        if (V.second->Kind != FirstVal.Kind &&
            (V.second->Kind == DbgValue::Proposed ||
             V.second->Kind == DbgValue::Def) &&
            (FirstVal.Kind == DbgValue::Proposed ||
             FirstVal.Kind == DbgValue::Def))
          DisagreeOnPHINess = true;
      }

      // Collect those flags together and determine an overall state for
      // what extend the predecessors agree on a live-in value.
      if (!Disagree)
        OurState = Agreed;
      else if (!IDDisagree && DisagreeOnPHINess)
        OurState = PropDisagree;
      else if (!NonBackEdgeDisagree)
        OurState = BEDisagree;
      else
        OurState = PHINeeded;
    }

    // An extra indicator: if we only disagree on whether the value is a
    // Def, or proposed, then also flag whether that disagreement happens
    // in backedges only.
    bool PropOnlyInBEs = Disagree && !IDDisagree && DisagreeOnPHINess &&
                         !NonBackEdgeDisagree && FirstVal.Kind == DbgValue::Def;

    const auto &Properties = FirstVal.Properties;

    auto OldLiveInIt = ILS.find(Var);
    const DbgValue *OldLiveInLocation =
        (OldLiveInIt != ILS.end()) ? &OldLiveInIt->second : nullptr;

    bool OverRide = false;
    if (OurState == BEDisagree && OldLiveInLocation) {
      // Only backedges disagree: we can consider downgrading. If there was a
      // previous live-in value, use it to work out whether the current
      // incoming value represents a lattice downgrade or not.
      OverRide =
          vlocDowngradeLattice(MBB, *OldLiveInLocation, Values, CurBlockRPONum);
    }

    // Use the current state of predecessor agreement and other flags to work
    // out what to do next. Possibilities include:
    //  * Accept a value all predecessors agree on, or accept one that
    //    represents a step down the exploration lattice,
    //  * Use a PHI value number, if one can be found,
    //  * Propose a PHI value number, and see if it gets confirmed later,
    //  * Emit a 'NoVal' value, indicating we couldn't resolve anything.
    if (OurState == Agreed) {
      // Easiest solution: all predecessors agree on the variable value.
      ConfirmValue(Var, FirstVal);
    } else if (OurState == BEDisagree && OverRide) {
      // Only backedges disagree, and the other predecessors have produced
      // a new live-in value further down the exploration lattice.
      DowngradeOccurred = true;
      ConfirmValue(Var, FirstVal);
    } else if (OurState == PropDisagree) {
      // Predecessors agree on value, but some say it's only a proposed value.
      // Propagate it as proposed: unless it was proposed in this block, in
      // which case we're able to confirm the value.
      if (FirstID.getBlock() == (uint64_t)MBB.getNumber() && FirstID.isPHI()) {
        ConfirmValue(Var, DbgValue(FirstID, Properties, DbgValue::Def));
      } else if (PropOnlyInBEs) {
        // If only backedges disagree, a higher (in RPO) block confirmed this
        // location, and we need to propagate it into this loop.
        ConfirmValue(Var, DbgValue(FirstID, Properties, DbgValue::Def));
      } else {
        // Otherwise; a Def meeting a Proposed is still a Proposed.
        ConfirmValue(Var, DbgValue(FirstID, Properties, DbgValue::Proposed));
      }
    } else if ((OurState == PHINeeded || OurState == BEDisagree)) {
      // Predecessors disagree and can't be downgraded: this can only be
      // solved with a PHI. Use pickVPHILoc to go look for one.
      Optional<ValueIDNum> VPHI;
      bool AllEdgesVPHI = false;
      std::tie(VPHI, AllEdgesVPHI) =
          pickVPHILoc(MBB, Var, VLOCOutLocs, MOutLocs, MInLocs, BlockOrders);

      if (VPHI && AllEdgesVPHI) {
        // There's a PHI value that's valid for all predecessors -- we can use
        // it. If any of the non-backedge predecessors have proposed values
        // though, this PHI is also only proposed, until the predecessors are
        // confirmed.
        DbgValue::KindT K = DbgValue::Def;
        for (unsigned int I = 0; I < BackEdgesStart; ++I)
          if (Values[I].second->Kind == DbgValue::Proposed)
            K = DbgValue::Proposed;

        ConfirmValue(Var, DbgValue(*VPHI, Properties, K));
      } else if (VPHI) {
        // There's a PHI value, but it's only legal for backedges. Leave this
        // as a proposed PHI value: it might come back on the backedges,
        // and allow us to confirm it in the future.
        DbgValue NoBEValue = DbgValue(*VPHI, Properties, DbgValue::Proposed);
        ConfirmValue(Var, NoBEValue);
      } else {
        ConfirmNoVal(Var, Properties);
      }
    } else {
      // Otherwise: we don't know. Emit a "phi but no real loc" phi.
      ConfirmNoVal(Var, Properties);
    }
  }

  // Store newly calculated in-locs into VLOCInLocs, if they've changed.
  Changed = ILS != InLocsT;
  if (Changed)
    ILS = InLocsT;

  return std::tuple<bool, bool>(Changed, DowngradeOccurred);
}

void InstrRefBasedLDV::vlocDataflow(
    const LexicalScope *Scope, const DILocation *DILoc,
    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks, LiveInsT &Output,
    ValueIDNum **MOutLocs, ValueIDNum **MInLocs,
    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
  // This method is much like mlocDataflow: but focuses on a single
  // LexicalScope at a time. Pick out a set of blocks and variables that are
  // to have their value assignments solved, then run our dataflow algorithm
  // until a fixedpoint is reached.
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;
  SmallPtrSet<MachineBasicBlock *, 16> OnWorklist, OnPending;

  // The set of blocks we'll be examining.
  SmallPtrSet<const MachineBasicBlock *, 8> BlocksToExplore;

  // The order in which to examine them (RPO).
  SmallVector<MachineBasicBlock *, 8> BlockOrders;

  // RPO ordering function.
  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  LS.getMachineBasicBlocks(DILoc, BlocksToExplore);

  // A separate container to distinguish "blocks we're exploring" versus
  // "blocks that are potentially in scope. See comment at start of vlocJoin.
  SmallPtrSet<const MachineBasicBlock *, 8> InScopeBlocks = BlocksToExplore;

  // Old LiveDebugValues tracks variable locations that come out of blocks
  // not in scope, where DBG_VALUEs occur. This is something we could
  // legitimately ignore, but lets allow it for now.
  if (EmulateOldLDV)
    BlocksToExplore.insert(AssignBlocks.begin(), AssignBlocks.end());

  // We also need to propagate variable values through any artificial blocks
  // that immediately follow blocks in scope.
  DenseSet<const MachineBasicBlock *> ToAdd;

  // Helper lambda: For a given block in scope, perform a depth first search
  // of all the artificial successors, adding them to the ToAdd collection.
  auto AccumulateArtificialBlocks =
      [this, &ToAdd, &BlocksToExplore,
       &InScopeBlocks](const MachineBasicBlock *MBB) {
        // Depth-first-search state: each node is a block and which successor
        // we're currently exploring.
        SmallVector<std::pair<const MachineBasicBlock *,
                              MachineBasicBlock::const_succ_iterator>,
                    8>
            DFS;

        // Find any artificial successors not already tracked.
        for (auto *succ : MBB->successors()) {
          if (BlocksToExplore.count(succ) || InScopeBlocks.count(succ))
            continue;
          if (!ArtificialBlocks.count(succ))
            continue;
          DFS.push_back(std::make_pair(succ, succ->succ_begin()));
          ToAdd.insert(succ);
        }

        // Search all those blocks, depth first.
        while (!DFS.empty()) {
          const MachineBasicBlock *CurBB = DFS.back().first;
          MachineBasicBlock::const_succ_iterator &CurSucc = DFS.back().second;
          // Walk back if we've explored this blocks successors to the end.
          if (CurSucc == CurBB->succ_end()) {
            DFS.pop_back();
            continue;
          }

          // If the current successor is artificial and unexplored, descend into
          // it.
          if (!ToAdd.count(*CurSucc) && ArtificialBlocks.count(*CurSucc)) {
            DFS.push_back(std::make_pair(*CurSucc, (*CurSucc)->succ_begin()));
            ToAdd.insert(*CurSucc);
            continue;
          }

          ++CurSucc;
        }
      };

  // Search in-scope blocks and those containing a DBG_VALUE from this scope
  // for artificial successors.
  for (auto *MBB : BlocksToExplore)
    AccumulateArtificialBlocks(MBB);
  for (auto *MBB : InScopeBlocks)
    AccumulateArtificialBlocks(MBB);

  BlocksToExplore.insert(ToAdd.begin(), ToAdd.end());
  InScopeBlocks.insert(ToAdd.begin(), ToAdd.end());

  // Single block scope: not interesting! No propagation at all. Note that
  // this could probably go above ArtificialBlocks without damage, but
  // that then produces output differences from original-live-debug-values,
  // which propagates from a single block into many artificial ones.
  if (BlocksToExplore.size() == 1)
    return;

  // Picks out relevants blocks RPO order and sort them.
  for (auto *MBB : BlocksToExplore)
    BlockOrders.push_back(const_cast<MachineBasicBlock *>(MBB));

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);
  unsigned NumBlocks = BlockOrders.size();

  // Allocate some vectors for storing the live ins and live outs. Large.
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> LiveIns, LiveOuts;
  LiveIns.resize(NumBlocks);
  LiveOuts.resize(NumBlocks);

  // Produce by-MBB indexes of live-in/live-outs, to ease lookup within
  // vlocJoin.
  LiveIdxT LiveOutIdx, LiveInIdx;
  LiveOutIdx.reserve(NumBlocks);
  LiveInIdx.reserve(NumBlocks);
  for (unsigned I = 0; I < NumBlocks; ++I) {
    LiveOutIdx[BlockOrders[I]] = &LiveOuts[I];
    LiveInIdx[BlockOrders[I]] = &LiveIns[I];
  }

  for (auto *MBB : BlockOrders) {
    Worklist.push(BBToOrder[MBB]);
    OnWorklist.insert(MBB);
  }

  // Iterate over all the blocks we selected, propagating variable values.
  bool FirstTrip = true;
  SmallPtrSet<const MachineBasicBlock *, 16> VLOCVisited;
  while (!Worklist.empty() || !Pending.empty()) {
    while (!Worklist.empty()) {
      auto *MBB = OrderToBB[Worklist.top()];
      CurBB = MBB->getNumber();
      Worklist.pop();

      DenseMap<DebugVariable, DbgValue> JoinedInLocs;

      // Join values from predecessors. Updates LiveInIdx, and writes output
      // into JoinedInLocs.
      bool InLocsChanged, DowngradeOccurred;
      std::tie(InLocsChanged, DowngradeOccurred) = vlocJoin(
          *MBB, LiveOutIdx, LiveInIdx, (FirstTrip) ? &VLOCVisited : nullptr,
          CurBB, VarsWeCareAbout, MOutLocs, MInLocs, InScopeBlocks,
          BlocksToExplore, JoinedInLocs);

      auto &VTracker = AllTheVLocs[MBB->getNumber()];
      bool FirstVisit = VLOCVisited.insert(MBB).second;

      // Always explore transfer function if inlocs changed, or if we've not
      // visited this block before.
      InLocsChanged |= FirstVisit;

      // If a downgrade occurred, book us in for re-examination on the next
      // iteration.
      if (DowngradeOccurred && OnPending.insert(MBB).second)
        Pending.push(BBToOrder[MBB]);

      // Patch up the variable value transfer function to use the live-in
      // machine values, now that that problem is solved.
      if (FirstVisit) {
        for (auto &Transfer : VTracker.Vars) {
          if (Transfer.second.Kind == DbgValue::Def &&
              Transfer.second.ID.getBlock() == CurBB &&
              Transfer.second.ID.isPHI()) {
            LocIdx Loc = Transfer.second.ID.getLoc();
            Transfer.second.ID = MInLocs[CurBB][Loc.asU64()];
          }
        }
      }

      if (!InLocsChanged)
        continue;

      // Do transfer function.
      for (auto &Transfer : VTracker.Vars) {
        // Is this var we're mangling in this scope?
        if (VarsWeCareAbout.count(Transfer.first)) {
          // Erase on empty transfer (DBG_VALUE $noreg).
          if (Transfer.second.Kind == DbgValue::Undef) {
            JoinedInLocs.erase(Transfer.first);
          } else {
            // Insert new variable value; or overwrite.
            auto NewValuePair = std::make_pair(Transfer.first, Transfer.second);
            auto Result = JoinedInLocs.insert(NewValuePair);
            if (!Result.second)
              Result.first->second = Transfer.second;
          }
        }
      }

      // Did the live-out locations change?
      bool OLChanged = JoinedInLocs != *LiveOutIdx[MBB];

      // If they haven't changed, there's no need to explore further.
      if (!OLChanged)
        continue;

      // Commit to the live-out record.
      *LiveOutIdx[MBB] = JoinedInLocs;

      // We should visit all successors. Ensure we'll visit any non-backedge
      // successors during this dataflow iteration; book backedge successors
      // to be visited next time around.
      for (auto s : MBB->successors()) {
        // Ignore out of scope / not-to-be-explored successors.
        if (LiveInIdx.find(s) == LiveInIdx.end())
          continue;

        if (BBToOrder[s] > BBToOrder[MBB]) {
          if (OnWorklist.insert(s).second)
            Worklist.push(BBToOrder[s]);
        } else if (OnPending.insert(s).second && (FirstTrip || OLChanged)) {
          Pending.push(BBToOrder[s]);
        }
      }
    }
    Worklist.swap(Pending);
    std::swap(OnWorklist, OnPending);
    OnPending.clear();
    assert(Pending.empty());
    FirstTrip = false;
  }

  // Dataflow done. Now what? Save live-ins. Ignore any that are still marked
  // as being variable-PHIs, because those did not have their machine-PHI
  // value confirmed. Such variable values are places that could have been
  // PHIs, but are not.
  for (auto *MBB : BlockOrders) {
    auto &VarMap = *LiveInIdx[MBB];
    for (auto &P : VarMap) {
      if (P.second.Kind == DbgValue::Proposed ||
          P.second.Kind == DbgValue::NoVal)
        continue;
      Output[MBB->getNumber()].push_back(P);
    }
  }

  BlockOrders.clear();
  BlocksToExplore.clear();
}

void InstrRefBasedLDV::dump_mloc_transfer(
    const MLocTransferMap &mloc_transfer) const {
  for (auto &P : mloc_transfer) {
    std::string foo = MTracker->LocIdxToName(P.first);
    std::string bar = MTracker->IDAsString(P.second);
    dbgs() << "Loc " << foo << " --> " << bar << "\n";
  }
}

void InstrRefBasedLDV::emitLocations(
    MachineFunction &MF, LiveInsT SavedLiveIns, ValueIDNum **MInLocs,
    DenseMap<DebugVariable, unsigned> &AllVarsNumbering) {
  TTracker = new TransferTracker(TII, MTracker, MF, *TRI, CalleeSavedRegs);
  unsigned NumLocs = MTracker->getNumLocs();

  // For each block, load in the machine value locations and variable value
  // live-ins, then step through each instruction in the block. New DBG_VALUEs
  // to be inserted will be created along the way.
  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    MTracker->reset();
    MTracker->loadFromArray(MInLocs[bbnum], bbnum);
    TTracker->loadInlocs(MBB, MInLocs[bbnum], SavedLiveIns[MBB.getNumber()],
                         NumLocs);

    CurBB = bbnum;
    CurInst = 1;
    for (auto &MI : MBB) {
      process(MI);
      ++CurInst;
    }
  }

  // We have to insert DBG_VALUEs in a consistent order, otherwise they appeaer
  // in DWARF in different orders. Use the order that they appear when walking
  // through each block / each instruction, stored in AllVarsNumbering.
  auto OrderDbgValues = [&](const MachineInstr *A,
                            const MachineInstr *B) -> bool {
    DebugVariable VarA(A->getDebugVariable(), A->getDebugExpression(),
                       A->getDebugLoc()->getInlinedAt());
    DebugVariable VarB(B->getDebugVariable(), B->getDebugExpression(),
                       B->getDebugLoc()->getInlinedAt());
    return AllVarsNumbering.find(VarA)->second <
           AllVarsNumbering.find(VarB)->second;
  };

  // Go through all the transfers recorded in the TransferTracker -- this is
  // both the live-ins to a block, and any movements of values that happen
  // in the middle.
  for (auto &P : TTracker->Transfers) {
    // Sort them according to appearance order.
    llvm::sort(P.Insts.begin(), P.Insts.end(), OrderDbgValues);
    // Insert either before or after the designated point...
    if (P.MBB) {
      MachineBasicBlock &MBB = *P.MBB;
      for (auto *MI : P.Insts) {
        MBB.insert(P.Pos, MI);
      }
    } else {
      MachineBasicBlock &MBB = *P.Pos->getParent();
      for (auto *MI : P.Insts) {
        MBB.insertAfter(P.Pos, MI);
      }
    }
  }
}

void InstrRefBasedLDV::initialSetup(MachineFunction &MF) {
  // Build some useful data structures.
  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  // Collect a set of all the artificial blocks.
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  // Compute mappings of block <=> RPO order.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    BBNumToRPO[(*RI)->getNumber()] = RPONumber;
    ++RPONumber;
  }
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool InstrRefBasedLDV::ExtendRanges(MachineFunction &MF,
                                    TargetPassConfig *TPC) {
  // No subprogram means this function contains no debuginfo.
  if (!MF.getFunction().getSubprogram())
    return false;

  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");
  this->TPC = TPC;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->getCalleeSaves(MF, CalleeSavedRegs);
  LS.initialize(MF);

  MTracker =
      new MLocTracker(MF, *TII, *TRI, *MF.getSubtarget().getTargetLowering());
  VTracker = nullptr;
  TTracker = nullptr;

  SmallVector<MLocTransferMap, 32> MLocTransfer;
  SmallVector<VLocTracker, 8> vlocs;
  LiveInsT SavedLiveIns;

  int MaxNumBlocks = -1;
  for (auto &MBB : MF)
    MaxNumBlocks = std::max(MBB.getNumber(), MaxNumBlocks);
  assert(MaxNumBlocks >= 0);
  ++MaxNumBlocks;

  MLocTransfer.resize(MaxNumBlocks);
  vlocs.resize(MaxNumBlocks);
  SavedLiveIns.resize(MaxNumBlocks);

  initialSetup(MF);

  produceTransferFunctions(MF, MLocTransfer, MaxNumBlocks, vlocs);

  // Allocate and initialize two array-of-arrays for the live-in and live-out
  // machine values. The outer dimension is the block number; while the inner
  // dimension is a LocIdx from MLocTracker.
  ValueIDNum **MOutLocs = new ValueIDNum *[MaxNumBlocks];
  ValueIDNum **MInLocs = new ValueIDNum *[MaxNumBlocks];
  unsigned NumLocs = MTracker->getNumLocs();
  for (int i = 0; i < MaxNumBlocks; ++i) {
    MOutLocs[i] = new ValueIDNum[NumLocs];
    MInLocs[i] = new ValueIDNum[NumLocs];
  }

  // Solve the machine value dataflow problem using the MLocTransfer function,
  // storing the computed live-ins / live-outs into the array-of-arrays. We use
  // both live-ins and live-outs for decision making in the variable value
  // dataflow problem.
  mlocDataflow(MInLocs, MOutLocs, MLocTransfer);

  // Number all variables in the order that they appear, to be used as a stable
  // insertion order later.
  DenseMap<DebugVariable, unsigned> AllVarsNumbering;

  // Map from one LexicalScope to all the variables in that scope.
  DenseMap<const LexicalScope *, SmallSet<DebugVariable, 4>> ScopeToVars;

  // Map from One lexical scope to all blocks in that scope.
  DenseMap<const LexicalScope *, SmallPtrSet<MachineBasicBlock *, 4>>
      ScopeToBlocks;

  // Store a DILocation that describes a scope.
  DenseMap<const LexicalScope *, const DILocation *> ScopeToDILocation;

  // To mirror old LiveDebugValues, enumerate variables in RPOT order. Otherwise
  // the order is unimportant, it just has to be stable.
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    auto *MBB = OrderToBB[I];
    auto *VTracker = &vlocs[MBB->getNumber()];
    // Collect each variable with a DBG_VALUE in this block.
    for (auto &idx : VTracker->Vars) {
      const auto &Var = idx.first;
      const DILocation *ScopeLoc = VTracker->Scopes[Var];
      assert(ScopeLoc != nullptr);
      auto *Scope = LS.findLexicalScope(ScopeLoc);

      // No insts in scope -> shouldn't have been recorded.
      assert(Scope != nullptr);

      AllVarsNumbering.insert(std::make_pair(Var, AllVarsNumbering.size()));
      ScopeToVars[Scope].insert(Var);
      ScopeToBlocks[Scope].insert(VTracker->MBB);
      ScopeToDILocation[Scope] = ScopeLoc;
    }
  }

  // OK. Iterate over scopes: there might be something to be said for
  // ordering them by size/locality, but that's for the future. For each scope,
  // solve the variable value problem, producing a map of variables to values
  // in SavedLiveIns.
  for (auto &P : ScopeToVars) {
    vlocDataflow(P.first, ScopeToDILocation[P.first], P.second,
                 ScopeToBlocks[P.first], SavedLiveIns, MOutLocs, MInLocs,
                 vlocs);
  }

  // Using the computed value locations and variable values for each block,
  // create the DBG_VALUE instructions representing the extended variable
  // locations.
  emitLocations(MF, SavedLiveIns, MInLocs, AllVarsNumbering);

  for (int Idx = 0; Idx < MaxNumBlocks; ++Idx) {
    delete[] MOutLocs[Idx];
    delete[] MInLocs[Idx];
  }
  delete[] MOutLocs;
  delete[] MInLocs;

  // Did we actually make any changes? If we created any DBG_VALUEs, then yes.
  bool Changed = TTracker->Transfers.size() != 0;

  delete MTracker;
  VTracker = nullptr;
  TTracker = nullptr;

  ArtificialBlocks.clear();
  OrderToBB.clear();
  BBToOrder.clear();
  BBNumToRPO.clear();

  return Changed;
}

LDVImpl *llvm::makeInstrRefBasedLiveDebugValues() {
  return new InstrRefBasedLDV();
}
