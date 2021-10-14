//===- InstrRefBasedImpl.h - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_INSTRREFBASEDLDV_H
#define LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_INSTRREFBASEDLDV_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "LiveDebugValues.h"

class TransferTracker;

// Forward dec of unit test class, so that we can peer into the LDV object.
class InstrRefLDVTest;

namespace LiveDebugValues {

class MLocTracker;

using namespace llvm;

/// Handle-class for a particular "location". This value-type uniquely
/// symbolises a register or stack location, allowing manipulation of locations
/// without concern for where that location is. Practically, this allows us to
/// treat the state of the machine at a particular point as an array of values,
/// rather than a map of values.
class LocIdx {
  unsigned Location;

  // Default constructor is private, initializing to an illegal location number.
  // Use only for "not an entry" elements in IndexedMaps.
  LocIdx() : Location(UINT_MAX) {}

public:
#define NUM_LOC_BITS 24
  LocIdx(unsigned L) : Location(L) {
    assert(L < (1 << NUM_LOC_BITS) && "Machine locations must fit in 24 bits");
  }

  static LocIdx MakeIllegalLoc() { return LocIdx(); }

  bool isIllegal() const { return Location == UINT_MAX; }

  uint64_t asU64() const { return Location; }

  bool operator==(unsigned L) const { return Location == L; }

  bool operator==(const LocIdx &L) const { return Location == L.Location; }

  bool operator!=(unsigned L) const { return !(*this == L); }

  bool operator!=(const LocIdx &L) const { return !(*this == L); }

  bool operator<(const LocIdx &Other) const {
    return Location < Other.Location;
  }
};

// The location at which a spilled value resides. It consists of a register and
// an offset.
struct SpillLoc {
  unsigned SpillBase;
  StackOffset SpillOffset;
  bool operator==(const SpillLoc &Other) const {
    return std::make_pair(SpillBase, SpillOffset) ==
           std::make_pair(Other.SpillBase, Other.SpillOffset);
  }
  bool operator<(const SpillLoc &Other) const {
    return std::make_tuple(SpillBase, SpillOffset.getFixed(),
                           SpillOffset.getScalable()) <
           std::make_tuple(Other.SpillBase, Other.SpillOffset.getFixed(),
                           Other.SpillOffset.getScalable());
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
  // Default-initialize to EmptyValue. This is necessary to make IndexedMaps
  // of values to work.
  ValueIDNum() : BlockNo(0xFFFFF), InstNo(0xFFFFF), LocNo(0xFFFFFF) {}

  ValueIDNum(uint64_t Block, uint64_t Inst, uint64_t Loc)
      : BlockNo(Block), InstNo(Inst), LocNo(Loc) {}

  ValueIDNum(uint64_t Block, uint64_t Inst, LocIdx Loc)
      : BlockNo(Block), InstNo(Inst), LocNo(Loc.asU64()) {}

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
    return Twine("Value{bb: ")
        .concat(Twine(BlockNo).concat(
            Twine(", inst: ")
                .concat((InstNo ? Twine(InstNo) : Twine("live-in"))
                            .concat(Twine(", loc: ").concat(Twine(mlocname)))
                            .concat(Twine("}")))))
        .str();
  }

  static ValueIDNum EmptyValue;
};

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

/// Class recording the (high level) _value_ of a variable. Identifies either
/// the value of the variable as a ValueIDNum, or a constant MachineOperand.
/// This class also stores meta-information about how the value is qualified.
/// Used to reason about variable values when performing the second
/// (DebugVariable specific) dataflow analysis.
class DbgValue {
public:
  /// If Kind is Def, the value number that this value is based on. VPHIs set
  /// this field to EmptyValue if there is no machine-value for this VPHI, or
  /// the corresponding machine-value if there is one.
  ValueIDNum ID;
  /// If Kind is Const, the MachineOperand defining this value.
  Optional<MachineOperand> MO;
  /// For a NoVal or VPHI DbgValue, which block it was generated in.
  int BlockNo;

  /// Qualifiers for the ValueIDNum above.
  DbgValueProperties Properties;

  typedef enum {
    Undef, // Represents a DBG_VALUE $noreg in the transfer function only.
    Def,   // This value is defined by an inst, or is a PHI value.
    Const, // A constant value contained in the MachineOperand field.
    VPHI,  // Incoming values to BlockNo differ, those values must be joined by
           // a PHI in this block.
    NoVal, // Empty DbgValue indicating an unknown value. Used as initializer,
           // before dominating blocks values are propagated in.
  } KindT;
  /// Discriminator for whether this is a constant or an in-program value.
  KindT Kind;

  DbgValue(const ValueIDNum &Val, const DbgValueProperties &Prop, KindT Kind)
      : ID(Val), MO(None), BlockNo(0), Properties(Prop), Kind(Kind) {
    assert(Kind == Def);
  }

  DbgValue(unsigned BlockNo, const DbgValueProperties &Prop, KindT Kind)
      : ID(ValueIDNum::EmptyValue), MO(None), BlockNo(BlockNo),
        Properties(Prop), Kind(Kind) {
    assert(Kind == NoVal || Kind == VPHI);
  }

  DbgValue(const MachineOperand &MO, const DbgValueProperties &Prop, KindT Kind)
      : ID(ValueIDNum::EmptyValue), MO(MO), BlockNo(0), Properties(Prop),
        Kind(Kind) {
    assert(Kind == Const);
  }

  DbgValue(const DbgValueProperties &Prop, KindT Kind)
    : ID(ValueIDNum::EmptyValue), MO(None), BlockNo(0), Properties(Prop),
      Kind(Kind) {
    assert(Kind == Undef &&
           "Empty DbgValue constructor must pass in Undef kind");
  }

#ifndef NDEBUG
  void dump(const MLocTracker *MTrack) const;
#endif

  bool operator==(const DbgValue &Other) const {
    if (std::tie(Kind, Properties) != std::tie(Other.Kind, Other.Properties))
      return false;
    else if (Kind == Def && ID != Other.ID)
      return false;
    else if (Kind == NoVal && BlockNo != Other.BlockNo)
      return false;
    else if (Kind == Const)
      return MO->isIdenticalTo(*Other.MO);
    else if (Kind == VPHI && BlockNo != Other.BlockNo)
      return false;
    else if (Kind == VPHI && ID != Other.ID)
      return false;

    return true;
  }

  bool operator!=(const DbgValue &Other) const { return !(*this == Other); }
};

class LocIdxToIndexFunctor {
public:
  using argument_type = LocIdx;
  unsigned operator()(const LocIdx &L) const { return L.asU64(); }
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
  using LocToValueType = IndexedMap<ValueIDNum, LocIdxToIndexFunctor>;

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

  /// When clobbering register masks, we chose to not believe the machine model
  /// and don't clobber SP. Do the same for SP aliases, and for efficiency,
  /// keep a set of them here.
  SmallSet<Register, 8> SPAliases;

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
      value_type(LocIdx Idx, ValueIDNum &Value) : Idx(Idx), Value(Value) {}
      const LocIdx Idx;  /// Read-only index of this location.
      ValueIDNum &Value; /// Reference to the stored value at this location.
    };

    MLocIterator(LocToValueType &ValueMap, LocIdx Idx)
        : ValueMap(ValueMap), Idx(Idx) {}

    bool operator==(const MLocIterator &Other) const {
      assert(&ValueMap == &Other.ValueMap);
      return Idx == Other.Idx;
    }

    bool operator!=(const MLocIterator &Other) const {
      return !(*this == Other);
    }

    void operator++() { Idx = LocIdx(Idx.asU64() + 1); }

    value_type operator*() { return value_type(Idx, ValueMap[LocIdx(Idx)]); }
  };

  MLocTracker(MachineFunction &MF, const TargetInstrInfo &TII,
              const TargetRegisterInfo &TRI, const TargetLowering &TLI);

  /// Produce location ID number for indexing LocIDToLocIdx. Takes the register
  /// or spill number, and flag for whether it's a spill or not.
  unsigned getLocID(Register RegOrSpill, bool isSpill) {
    return (isSpill) ? RegOrSpill.id() + NumRegs - 1 : RegOrSpill.id();
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
    // SpillLocs.reset(); XXX UniqueVector::reset assumes a SpillLoc casts from
    // 0
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
  LocIdx trackRegister(unsigned ID);

  LocIdx lookupOrTrackRegister(unsigned ID) {
    LocIdx &Index = LocIDToLocIdx[ID];
    if (Index.isIllegal())
      Index = trackRegister(ID);
    return Index;
  }

  /// Is register R currently tracked by MLocTracker?
  bool isRegisterTracked(Register R) {
    LocIdx &Index = LocIDToLocIdx[R];
    return !Index.isIllegal();
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
  void writeRegMask(const MachineOperand *MO, unsigned CurBB, unsigned InstID);

  /// Find LocIdx for SpillLoc \p L, creating a new one if it's not tracked.
  LocIdx getOrTrackSpillLoc(SpillLoc L);

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
  bool isSpill(LocIdx Idx) const { return LocIdxToLocID[Idx] >= NumRegs; }

  MLocIterator begin() { return MLocIterator(LocIdxToIDNum, 0); }

  MLocIterator end() {
    return MLocIterator(LocIdxToIDNum, LocIdxToIDNum.size());
  }

  /// Return a range over all locations currently tracked.
  iterator_range<MLocIterator> locations() {
    return llvm::make_range(begin(), end());
  }

  std::string LocIdxToName(LocIdx Idx) const;

  std::string IDAsString(const ValueIDNum &Num) const;

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump();

  LLVM_DUMP_METHOD void dump_mloc_map();
#endif

  /// Create a DBG_VALUE based on  machine location \p MLoc. Qualify it with the
  /// information in \pProperties, for variable Var. Don't insert it anywhere,
  /// just return the builder for it.
  MachineInstrBuilder emitLoc(Optional<LocIdx> MLoc, const DebugVariable &Var,
                              const DbgValueProperties &Properties);
};

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

  void defVar(const MachineInstr &MI, const DbgValueProperties &Properties,
              Optional<ValueIDNum> ID) {
    assert(MI.isDebugValue() || MI.isDebugRef());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    DbgValue Rec = (ID) ? DbgValue(*ID, Properties, DbgValue::Def)
                        : DbgValue(Properties, DbgValue::Undef);

    // Attempt insertion; overwrite if it's already mapped.
    auto Result = Vars.insert(std::make_pair(Var, Rec));
    if (!Result.second)
      Result.first->second = Rec;
    Scopes[Var] = MI.getDebugLoc().get();
  }

  void defVar(const MachineInstr &MI, const MachineOperand &MO) {
    // Only DBG_VALUEs can define constant-valued variables.
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

/// Types for recording sets of variable fragments that overlap. For a given
/// local variable, we record all other fragments of that variable that could
/// overlap it, to reduce search time.
using FragmentOfVar =
    std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
using OverlapMap =
    DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

// XXX XXX docs
class InstrRefBasedLDV : public LDVImpl {
public:
  friend class ::InstrRefLDVTest;

  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// Machine location/value transfer function, a mapping of which locations
  /// are assigned which new values.
  using MLocTransferMap = std::map<LocIdx, ValueIDNum>;

  /// Live in/out structure for the variable values: a per-block map of
  /// variables to their values. XXX, better name?
  using LiveIdxT =
      DenseMap<const MachineBasicBlock *, DenseMap<DebugVariable, DbgValue> *>;

  using VarAndLoc = std::pair<DebugVariable, DbgValue>;

  /// Type for a live-in value: the predecessor block, and its value.
  using InValueT = std::pair<MachineBasicBlock *, DbgValue *>;

  /// Vector (per block) of a collection (inner smallvector) of live-ins.
  /// Used as the result type for the variable value dataflow problem.
  using LiveInsT = SmallVector<SmallVector<VarAndLoc, 8>, 8>;

private:
  MachineDominatorTree *DomTree;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  const MachineFrameInfo *MFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;
  TargetPassConfig *TPC;

  // An empty DIExpression. Used default / placeholder DbgValueProperties
  // objects, as we can't have null expressions.
  const DIExpression *EmptyExpr;

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
  DenseMap<const MachineBasicBlock *, unsigned int> BBToOrder;
  DenseMap<unsigned, unsigned> BBNumToRPO;

  /// Pair of MachineInstr, and its 1-based offset into the containing block.
  using InstAndNum = std::pair<const MachineInstr *, unsigned>;
  /// Map from debug instruction number to the MachineInstr labelled with that
  /// number, and its location within the function. Used to transform
  /// instruction numbers in DBG_INSTR_REFs into machine value numbers.
  std::map<uint64_t, InstAndNum> DebugInstrNumToInstr;

  /// Record of where we observed a DBG_PHI instruction.
  class DebugPHIRecord {
  public:
    uint64_t InstrNum;      ///< Instruction number of this DBG_PHI.
    MachineBasicBlock *MBB; ///< Block where DBG_PHI occurred.
    ValueIDNum ValueRead;   ///< The value number read by the DBG_PHI.
    LocIdx ReadLoc;         ///< Register/Stack location the DBG_PHI reads.

    operator unsigned() const { return InstrNum; }
  };

  /// Map from instruction numbers defined by DBG_PHIs to a record of what that
  /// DBG_PHI read and where. Populated and edited during the machine value
  /// location problem -- we use LLVMs SSA Updater to fix changes by
  /// optimizations that destroy PHI instructions.
  SmallVector<DebugPHIRecord, 32> DebugPHINumToValue;

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
  void process(MachineInstr &MI, ValueIDNum **MLiveOuts = nullptr,
               ValueIDNum **MLiveIns = nullptr);

  /// Examines whether \p MI is a DBG_VALUE and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferDebugValue(const MachineInstr &MI);

  /// Examines whether \p MI is a DBG_INSTR_REF and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferDebugInstrRef(MachineInstr &MI, ValueIDNum **MLiveOuts,
                             ValueIDNum **MLiveIns);

  /// Stores value-information about where this PHI occurred, and what
  /// instruction number is associated with it.
  /// \returns true if MI was recognized and processed.
  bool transferDebugPHI(MachineInstr &MI);

  /// Examines whether \p MI is copy instruction, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferRegisterCopy(MachineInstr &MI);

  /// Examines whether \p MI is stack spill or restore  instruction, and
  /// notifies trackers. \returns true if MI was recognized and processed.
  bool transferSpillOrRestoreInst(MachineInstr &MI);

  /// Examines \p MI for any registers that it defines, and notifies trackers.
  void transferRegisterDef(MachineInstr &MI);

  /// Copy one location to the other, accounting for movement of subregisters
  /// too.
  void performCopy(Register Src, Register Dst);

  void accumulateFragmentMap(MachineInstr &MI);

  /// Determine the machine value number referred to by (potentially several)
  /// DBG_PHI instructions. Block duplication and tail folding can duplicate
  /// DBG_PHIs, shifting the position where values in registers merge, and
  /// forming another mini-ssa problem to solve.
  /// \p Here the position of a DBG_INSTR_REF seeking a machine value number
  /// \p InstrNum Debug instruction number defined by DBG_PHI instructions.
  /// \returns The machine value number at position Here, or None.
  Optional<ValueIDNum> resolveDbgPHIs(MachineFunction &MF,
                                      ValueIDNum **MLiveOuts,
                                      ValueIDNum **MLiveIns, MachineInstr &Here,
                                      uint64_t InstrNum);

  /// Step through the function, recording register definitions and movements
  /// in an MLocTracker. Convert the observations into a per-block transfer
  /// function in \p MLocTransfer, suitable for using with the machine value
  /// location dataflow problem.
  void
  produceMLocTransferFunction(MachineFunction &MF,
                              SmallVectorImpl<MLocTransferMap> &MLocTransfer,
                              unsigned MaxNumBlocks);

  /// Solve the machine value location dataflow problem. Takes as input the
  /// transfer functions in \p MLocTransfer. Writes the output live-in and
  /// live-out arrays to the (initialized to zero) multidimensional arrays in
  /// \p MInLocs and \p MOutLocs. The outer dimension is indexed by block
  /// number, the inner by LocIdx.
  void buildMLocValueMap(MachineFunction &MF, ValueIDNum **MInLocs,
                         ValueIDNum **MOutLocs,
                         SmallVectorImpl<MLocTransferMap> &MLocTransfer);

  /// Install PHI values into the live-in array for each block, according to
  /// the IDF of each register.
  void placeMLocPHIs(MachineFunction &MF,
                     SmallPtrSetImpl<MachineBasicBlock *> &AllBlocks,
                     ValueIDNum **MInLocs,
                     SmallVectorImpl<MLocTransferMap> &MLocTransfer);

  /// Calculate the iterated-dominance-frontier for a set of defs, using the
  /// existing LLVM facilities for this. Works for a single "value" or
  /// machine/variable location.
  /// \p AllBlocks Set of blocks where we might consume the value.
  /// \p DefBlocks Set of blocks where the value/location is defined.
  /// \p PHIBlocks Output set of blocks where PHIs must be placed.
  void BlockPHIPlacement(const SmallPtrSetImpl<MachineBasicBlock *> &AllBlocks,
                         const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                         SmallVectorImpl<MachineBasicBlock *> &PHIBlocks);

  /// Perform a control flow join (lattice value meet) of the values in machine
  /// locations at \p MBB. Follows the algorithm described in the file-comment,
  /// reading live-outs of predecessors from \p OutLocs, the current live ins
  /// from \p InLocs, and assigning the newly computed live ins back into
  /// \p InLocs. \returns two bools -- the first indicates whether a change
  /// was made, the second whether a lattice downgrade occurred. If the latter
  /// is true, revisiting this block is necessary.
  bool mlocJoin(MachineBasicBlock &MBB,
                SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
                ValueIDNum **OutLocs, ValueIDNum *InLocs);

  /// Solve the variable value dataflow problem, for a single lexical scope.
  /// Uses the algorithm from the file comment to resolve control flow joins
  /// using PHI placement and value propagation. Reads the locations of machine
  /// values from the \p MInLocs and \p MOutLocs arrays (see buildMLocValueMap)
  /// and reads the variable values transfer function from \p AllTheVlocs.
  /// Live-in and Live-out variable values are stored locally, with the live-ins
  /// permanently stored to \p Output once a fixedpoint is reached.
  /// \p VarsWeCareAbout contains a collection of the variables in \p Scope
  /// that we should be tracking.
  /// \p AssignBlocks contains the set of blocks that aren't in \p DILoc's
  /// scope, but which do contain DBG_VALUEs, which VarLocBasedImpl tracks
  /// locations through.
  void buildVLocValueMap(const DILocation *DILoc,
                    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
                    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
                    LiveInsT &Output, ValueIDNum **MOutLocs,
                    ValueIDNum **MInLocs,
                    SmallVectorImpl<VLocTracker> &AllTheVLocs);

  /// Attempt to eliminate un-necessary PHIs on entry to a block. Examines the
  /// live-in values coming from predecessors live-outs, and replaces any PHIs
  /// already present in this blocks live-ins with a live-through value if the
  /// PHI isn't needed. Live out and live in variable values are stored in
  /// \p VLOCOutLocs and \p VLOCInLocs. The live-ins for \p MBB are computed and
  /// stored into \p VLOCInLocs.
  /// \p InLocsT Output argument, where calculated live-in values are also
  ///          stored.
  /// \returns true if any live-ins change value, either from value propagation
  ///          or PHI elimination.
  bool vlocJoin(MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
                LiveIdxT &VLOCInLocs,
                const SmallSet<DebugVariable, 4> &AllVars,
                SmallPtrSet<const MachineBasicBlock *, 8> &InScopeBlocks,
                SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
                DenseMap<DebugVariable, DbgValue> &InLocsT);

  /// For the given block and live-outs feeding into it, try to find a
  /// machine location where all the variable values join together.
  /// \returns Value ID of a machine PHI if an appropriate one is available.
  Optional<ValueIDNum>
  pickVPHILoc(const MachineBasicBlock &MBB, const DebugVariable &Var,
              const LiveIdxT &LiveOuts, ValueIDNum **MOutLocs,
              const SmallVectorImpl<const MachineBasicBlock *> &BlockOrders);

  /// Given the solutions to the two dataflow problems, machine value locations
  /// in \p MInLocs and live-in variable values in \p SavedLiveIns, runs the
  /// TransferTracker class over the function to produce live-in and transfer
  /// DBG_VALUEs, then inserts them. Groups of DBG_VALUEs are inserted in the
  /// order given by AllVarsNumbering -- this could be any stable order, but
  /// right now "order of appearence in function, when explored in RPO", so
  /// that we can compare explictly against VarLocBasedImpl.
  void emitLocations(MachineFunction &MF, LiveInsT SavedLiveIns,
                     ValueIDNum **MOutLocs, ValueIDNum **MInLocs,
                     DenseMap<DebugVariable, unsigned> &AllVarsNumbering,
                     const TargetPassConfig &TPC);

  /// Boilerplate computation of some initial sets, artifical blocks and
  /// RPOT block ordering.
  void initialSetup(MachineFunction &MF);

  bool ExtendRanges(MachineFunction &MF, MachineDominatorTree *DomTree,
                    TargetPassConfig *TPC, unsigned InputBBLimit,
                    unsigned InputDbgValLimit) override;

public:
  /// Default construct and initialize the pass.
  InstrRefBasedLDV();

  LLVM_DUMP_METHOD
  void dump_mloc_transfer(const MLocTransferMap &mloc_transfer) const;

  bool isCalleeSaved(LocIdx L) const;
};

} // namespace LiveDebugValues

#endif /* LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_INSTRREFBASEDLDV_H */
