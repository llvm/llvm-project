//===-- llvm/CodeGen/LiveInterval.h - Interval representation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveRange and LiveInterval classes.  Given some
// numbering of each the machine instructions an interval [i, j) is said to be a
// live interval for register v if there is no instruction with number j' >= j
// such that v is live at j' and there is no instruction with number i' < i such
// that v is live at i'. In this implementation intervals can have holes,
// i.e. an interval might look like [1,20), [50,65), [1000,1001).  Each
// individual range is represented as an instance of LiveRange, and the whole
// interval is represented as an instance of LiveInterval.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <cassert>
#include <climits>

namespace llvm {
  class CoalescerPair;
  class LiveIntervals;
  class MachineInstr;
  class MachineRegisterInfo;
  class TargetRegisterInfo;
  class raw_ostream;

  /// VNInfo - Value Number Information.
  /// This class holds information about a machine level values, including
  /// definition and use points.
  ///
  class VNInfo {
  public:
    typedef BumpPtrAllocator Allocator;

    /// The ID number of this value.
    unsigned id;

    /// The index of the defining instruction.
    SlotIndex def;

    /// VNInfo constructor.
    VNInfo(unsigned i, SlotIndex d)
      : id(i), def(d)
    { }

    /// VNInfo construtor, copies values from orig, except for the value number.
    VNInfo(unsigned i, const VNInfo &orig)
      : id(i), def(orig.def)
    { }

    /// Copy from the parameter into this VNInfo.
    void copyFrom(VNInfo &src) {
      def = src.def;
    }

    /// Returns true if this value is defined by a PHI instruction (or was,
    /// PHI instrucions may have been eliminated).
    /// PHI-defs begin at a block boundary, all other defs begin at register or
    /// EC slots.
    bool isPHIDef() const { return def.isBlock(); }

    /// Returns true if this value is unused.
    bool isUnused() const { return !def.isValid(); }

    /// Mark this value as unused.
    void markUnused() { def = SlotIndex(); }
  };

  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    SlotIndex start;  // Start point of the interval (inclusive)
    SlotIndex end;    // End point of the interval (exclusive)
    VNInfo *valno;   // identifier for the value contained in this interval.

    LiveRange(SlotIndex S, SlotIndex E, VNInfo *V)
      : start(S), end(E), valno(V) {

      assert(S < E && "Cannot create empty or backwards range");
    }

    /// contains - Return true if the index is covered by this range.
    ///
    bool contains(SlotIndex I) const {
      return start <= I && I < end;
    }

    /// containsRange - Return true if the given range, [S, E), is covered by
    /// this range.
    bool containsRange(SlotIndex S, SlotIndex E) const {
      assert((S < E) && "Backwards interval?");
      return (start <= S && S < end) && (start < E && E <= end);
    }

    bool operator<(const LiveRange &LR) const {
      return start < LR.start || (start == LR.start && end < LR.end);
    }
    bool operator==(const LiveRange &LR) const {
      return start == LR.start && end == LR.end;
    }

    void dump() const;
    void print(raw_ostream &os) const;
  };

  template <> struct isPodLike<LiveRange> { static const bool value = true; };

  raw_ostream& operator<<(raw_ostream& os, const LiveRange &LR);


  inline bool operator<(SlotIndex V, const LiveRange &LR) {
    return V < LR.start;
  }

  inline bool operator<(const LiveRange &LR, SlotIndex V) {
    return LR.start < V;
  }

  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  class LiveInterval {
  public:

    typedef SmallVector<LiveRange,4> Ranges;
    typedef SmallVector<VNInfo*,4> VNInfoList;

    const unsigned reg;  // the register or stack slot of this interval.
    float weight;        // weight of this interval
    Ranges ranges;       // the ranges in which this register is live
    VNInfoList valnos;   // value#'s

    struct InstrSlots {
      enum {
        LOAD  = 0,
        USE   = 1,
        DEF   = 2,
        STORE = 3,
        NUM   = 4
      };

    };

    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), weight(Weight) {}

    typedef Ranges::iterator iterator;
    iterator begin() { return ranges.begin(); }
    iterator end()   { return ranges.end(); }

    typedef Ranges::const_iterator const_iterator;
    const_iterator begin() const { return ranges.begin(); }
    const_iterator end() const  { return ranges.end(); }

    typedef VNInfoList::iterator vni_iterator;
    vni_iterator vni_begin() { return valnos.begin(); }
    vni_iterator vni_end() { return valnos.end(); }

    typedef VNInfoList::const_iterator const_vni_iterator;
    const_vni_iterator vni_begin() const { return valnos.begin(); }
    const_vni_iterator vni_end() const { return valnos.end(); }

    /// advanceTo - Advance the specified iterator to point to the LiveRange
    /// containing the specified position, or end() if the position is past the
    /// end of the interval.  If no LiveRange contains this position, but the
    /// position is in a hole, this method returns an iterator pointing to the
    /// LiveRange immediately after the hole.
    iterator advanceTo(iterator I, SlotIndex Pos) {
      assert(I != end());
      if (Pos >= endIndex())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }

    /// find - Return an iterator pointing to the first range that ends after
    /// Pos, or end(). This is the same as advanceTo(begin(), Pos), but faster
    /// when searching large intervals.
    ///
    /// If Pos is contained in a LiveRange, that range is returned.
    /// If Pos is in a hole, the following LiveRange is returned.
    /// If Pos is beyond endIndex, end() is returned.
    iterator find(SlotIndex Pos);

    const_iterator find(SlotIndex Pos) const {
      return const_cast<LiveInterval*>(this)->find(Pos);
    }

    void clear() {
      valnos.clear();
      ranges.clear();
    }

    bool hasAtLeastOneValue() const { return !valnos.empty(); }

    bool containsOneValue() const { return valnos.size() == 1; }

    unsigned getNumValNums() const { return (unsigned)valnos.size(); }

    /// getValNumInfo - Returns pointer to the specified val#.
    ///
    inline VNInfo *getValNumInfo(unsigned ValNo) {
      return valnos[ValNo];
    }
    inline const VNInfo *getValNumInfo(unsigned ValNo) const {
      return valnos[ValNo];
    }

    /// containsValue - Returns true if VNI belongs to this interval.
    bool containsValue(const VNInfo *VNI) const {
      return VNI && VNI->id < getNumValNums() && VNI == getValNumInfo(VNI->id);
    }

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    VNInfo *getNextValue(SlotIndex def, VNInfo::Allocator &VNInfoAllocator) {
      VNInfo *VNI =
        new (VNInfoAllocator) VNInfo((unsigned)valnos.size(), def);
      valnos.push_back(VNI);
      return VNI;
    }

    /// createDeadDef - Make sure the interval has a value defined at Def.
    /// If one already exists, return it. Otherwise allocate a new value and
    /// add liveness for a dead def.
    VNInfo *createDeadDef(SlotIndex Def, VNInfo::Allocator &VNInfoAllocator);

    /// Create a copy of the given value. The new value will be identical except
    /// for the Value number.
    VNInfo *createValueCopy(const VNInfo *orig,
                            VNInfo::Allocator &VNInfoAllocator) {
      VNInfo *VNI =
        new (VNInfoAllocator) VNInfo((unsigned)valnos.size(), *orig);
      valnos.push_back(VNI);
      return VNI;
    }

    /// RenumberValues - Renumber all values in order of appearance and remove
    /// unused values.
    void RenumberValues(LiveIntervals &lis);

    /// MergeValueNumberInto - This method is called when two value nubmers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// LiveRanges with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    VNInfo* MergeValueNumberInto(VNInfo *V1, VNInfo *V2);

    /// MergeValueInAsValue - Merge all of the live ranges of a specific val#
    /// in RHS into this live interval as the specified value number.
    /// The LiveRanges in RHS are allowed to overlap with LiveRanges in the
    /// current interval, it will replace the value numbers of the overlaped
    /// live ranges with the specified value number.
    void MergeRangesInAsValue(const LiveInterval &RHS, VNInfo *LHSValNo);

    /// MergeValueInAsValue - Merge all of the live ranges of a specific val#
    /// in RHS into this live interval as the specified value number.
    /// The LiveRanges in RHS are allowed to overlap with LiveRanges in the
    /// current interval, but only if the overlapping LiveRanges have the
    /// specified value number.
    void MergeValueInAsValue(const LiveInterval &RHS,
                             const VNInfo *RHSValNo, VNInfo *LHSValNo);

    bool empty() const { return ranges.empty(); }

    /// beginIndex - Return the lowest numbered slot covered by interval.
    SlotIndex beginIndex() const {
      assert(!empty() && "Call to beginIndex() on empty interval.");
      return ranges.front().start;
    }

    /// endNumber - return the maximum point of the interval of the whole,
    /// exclusive.
    SlotIndex endIndex() const {
      assert(!empty() && "Call to endIndex() on empty interval.");
      return ranges.back().end;
    }

    bool expiredAt(SlotIndex index) const {
      return index >= endIndex();
    }

    bool liveAt(SlotIndex index) const {
      const_iterator r = find(index);
      return r != end() && r->start <= index;
    }

    /// killedAt - Return true if a live range ends at index. Note that the kill
    /// point is not contained in the half-open live range. It is usually the
    /// getDefIndex() slot following its last use.
    bool killedAt(SlotIndex index) const {
      const_iterator r = find(index.getRegSlot(true));
      return r != end() && r->end == index;
    }

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    const LiveRange *getLiveRangeContaining(SlotIndex Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    LiveRange *getLiveRangeContaining(SlotIndex Idx) {
      iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// getVNInfoAt - Return the VNInfo that is live at Idx, or NULL.
    VNInfo *getVNInfoAt(SlotIndex Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : I->valno;
    }

    /// getVNInfoBefore - Return the VNInfo that is live up to but not
    /// necessarilly including Idx, or NULL. Use this to find the reaching def
    /// used by an instruction at this SlotIndex position.
    VNInfo *getVNInfoBefore(SlotIndex Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx.getPrevSlot());
      return I == end() ? 0 : I->valno;
    }

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    iterator FindLiveRangeContaining(SlotIndex Idx) {
      iterator I = find(Idx);
      return I != end() && I->start <= Idx ? I : end();
    }

    const_iterator FindLiveRangeContaining(SlotIndex Idx) const {
      const_iterator I = find(Idx);
      return I != end() && I->start <= Idx ? I : end();
    }

    /// overlaps - Return true if the intersection of the two live intervals is
    /// not empty.
    bool overlaps(const LiveInterval& other) const {
      if (other.empty())
        return false;
      return overlapsFrom(other, other.begin());
    }

    /// overlaps - Return true if the two intervals have overlapping segments
    /// that are not coalescable according to CP.
    ///
    /// Overlapping segments where one interval is defined by a coalescable
    /// copy are allowed.
    bool overlaps(const LiveInterval &Other, const CoalescerPair &CP,
                  const SlotIndexes&) const;

    /// overlaps - Return true if the live interval overlaps a range specified
    /// by [Start, End).
    bool overlaps(SlotIndex Start, SlotIndex End) const;

    /// overlapsFrom - Return true if the intersection of the two live intervals
    /// is not empty.  The specified iterator is a hint that we can begin
    /// scanning the Other interval starting at I.
    bool overlapsFrom(const LiveInterval& other, const_iterator I) const;

    /// addRange - Add the specified LiveRange to this interval, merging
    /// intervals as appropriate.  This returns an iterator to the inserted live
    /// range (which may have grown since it was inserted.
    void addRange(LiveRange LR) {
      addRangeFrom(LR, ranges.begin());
    }

    /// extendInBlock - If this interval is live before Kill in the basic block
    /// that starts at StartIdx, extend it to be live up to Kill, and return
    /// the value. If there is no live range before Kill, return NULL.
    VNInfo *extendInBlock(SlotIndex StartIdx, SlotIndex Kill);

    /// join - Join two live intervals (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS intervals as specified.  If
    /// the intervals are not joinable, this aborts.
    void join(LiveInterval &Other,
              const int *ValNoAssignments,
              const int *RHSValNoAssignments,
              SmallVector<VNInfo*, 16> &NewVNInfo,
              MachineRegisterInfo *MRI);

    /// isInOneLiveRange - Return true if the range specified is entirely in the
    /// a single LiveRange of the live interval.
    bool isInOneLiveRange(SlotIndex Start, SlotIndex End) const {
      const_iterator r = find(Start);
      return r != end() && r->containsRange(Start, End);
    }

    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must be a single LiveRange in its entirety.
    void removeRange(SlotIndex Start, SlotIndex End,
                     bool RemoveDeadValNo = false);

    void removeRange(LiveRange LR, bool RemoveDeadValNo = false) {
      removeRange(LR.start, LR.end, RemoveDeadValNo);
    }

    /// removeValNo - Remove all the ranges defined by the specified value#.
    /// Also remove the value# from value# list.
    void removeValNo(VNInfo *ValNo);

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    /// Returns true if the live interval is zero length, i.e. no live ranges
    /// span instructions. It doesn't pay to spill such an interval.
    bool isZeroLength(SlotIndexes *Indexes) const {
      for (const_iterator i = begin(), e = end(); i != e; ++i)
        if (Indexes->getNextNonNullIndex(i->start).getBaseIndex() <
            i->end.getBaseIndex())
          return false;
      return true;
    }

    /// isSpillable - Can this interval be spilled?
    bool isSpillable() const {
      return weight != HUGE_VALF;
    }

    /// markNotSpillable - Mark interval as not spillable
    void markNotSpillable() {
      weight = HUGE_VALF;
    }

    bool operator<(const LiveInterval& other) const {
      const SlotIndex &thisIndex = beginIndex();
      const SlotIndex &otherIndex = other.beginIndex();
      return (thisIndex < otherIndex ||
              (thisIndex == otherIndex && reg < other.reg));
    }

    void print(raw_ostream &OS) const;
    void dump() const;

    /// \brief Walk the interval and assert if any invariants fail to hold.
    ///
    /// Note that this is a no-op when asserts are disabled.
#ifdef NDEBUG
    void verify() const {}
#else
    void verify() const;
#endif

  private:

    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, SlotIndex NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, SlotIndex NewStr);
    void markValNoForDeletion(VNInfo *V);
    void mergeIntervalRanges(const LiveInterval &RHS,
                             VNInfo *LHSValNo = 0,
                             const VNInfo *RHSValNo = 0);

    LiveInterval& operator=(const LiveInterval& rhs) LLVM_DELETED_FUNCTION;

  };

  inline raw_ostream &operator<<(raw_ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }

  /// LiveRangeQuery - Query information about a live range around a given
  /// instruction. This class hides the implementation details of live ranges,
  /// and it should be used as the primary interface for examining live ranges
  /// around instructions.
  ///
  class LiveRangeQuery {
    VNInfo *EarlyVal;
    VNInfo *LateVal;
    SlotIndex EndPoint;
    bool Kill;

  public:
    /// Create a LiveRangeQuery for the given live range and instruction index.
    /// The sub-instruction slot of Idx doesn't matter, only the instruction it
    /// refers to is considered.
    LiveRangeQuery(const LiveInterval &LI, SlotIndex Idx)
      : EarlyVal(0), LateVal(0), Kill(false) {
      // Find the segment that enters the instruction.
      LiveInterval::const_iterator I = LI.find(Idx.getBaseIndex());
      LiveInterval::const_iterator E = LI.end();
      if (I == E)
        return;
      // Is this an instruction live-in segment?
      // If Idx is the start index of a basic block, include live-in segments
      // that start at Idx.getBaseIndex().
      if (I->start <= Idx.getBaseIndex()) {
        EarlyVal = I->valno;
        EndPoint = I->end;
        // Move to the potentially live-out segment.
        if (SlotIndex::isSameInstr(Idx, I->end)) {
          Kill = true;
          if (++I == E)
            return;
        }
        // Special case: A PHIDef value can have its def in the middle of a
        // segment if the value happens to be live out of the layout
        // predecessor.
        // Such a value is not live-in.
        if (EarlyVal->def == Idx.getBaseIndex())
          EarlyVal = 0;
      }
      // I now points to the segment that may be live-through, or defined by
      // this instr. Ignore segments starting after the current instr.
      if (SlotIndex::isEarlierInstr(Idx, I->start))
        return;
      LateVal = I->valno;
      EndPoint = I->end;
    }

    /// Return the value that is live-in to the instruction. This is the value
    /// that will be read by the instruction's use operands. Return NULL if no
    /// value is live-in.
    VNInfo *valueIn() const {
      return EarlyVal;
    }

    /// Return true if the live-in value is killed by this instruction. This
    /// means that either the live range ends at the instruction, or it changes
    /// value.
    bool isKill() const {
      return Kill;
    }

    /// Return true if this instruction has a dead def.
    bool isDeadDef() const {
      return EndPoint.isDead();
    }

    /// Return the value leaving the instruction, if any. This can be a
    /// live-through value, or a live def. A dead def returns NULL.
    VNInfo *valueOut() const {
      return isDeadDef() ? 0 : LateVal;
    }

    /// Return the value defined by this instruction, if any. This includes
    /// dead defs, it is the value created by the instruction's def operands.
    VNInfo *valueDefined() const {
      return EarlyVal == LateVal ? 0 : LateVal;
    }

    /// Return the end point of the last live range segment to interact with
    /// the instruction, if any.
    ///
    /// The end point is an invalid SlotIndex only if the live range doesn't
    /// intersect the instruction at all.
    ///
    /// The end point may be at or past the end of the instruction's basic
    /// block. That means the value was live out of the block.
    SlotIndex endPoint() const {
      return EndPoint;
    }
  };

  /// ConnectedVNInfoEqClasses - Helper class that can divide VNInfos in a
  /// LiveInterval into equivalence clases of connected components. A
  /// LiveInterval that has multiple connected components can be broken into
  /// multiple LiveIntervals.
  ///
  /// Given a LiveInterval that may have multiple connected components, run:
  ///
  ///   unsigned numComps = ConEQ.Classify(LI);
  ///   if (numComps > 1) {
  ///     // allocate numComps-1 new LiveIntervals into LIS[1..]
  ///     ConEQ.Distribute(LIS);
  /// }

  class ConnectedVNInfoEqClasses {
    LiveIntervals &LIS;
    IntEqClasses EqClass;

    // Note that values a and b are connected.
    void Connect(unsigned a, unsigned b);

    unsigned Renumber();

  public:
    explicit ConnectedVNInfoEqClasses(LiveIntervals &lis) : LIS(lis) {}

    /// Classify - Classify the values in LI into connected components.
    /// Return the number of connected components.
    unsigned Classify(const LiveInterval *LI);

    /// getEqClass - Classify creates equivalence classes numbered 0..N. Return
    /// the equivalence class assigned the VNI.
    unsigned getEqClass(const VNInfo *VNI) const { return EqClass[VNI->id]; }

    /// Distribute - Distribute values in LIV[0] into a separate LiveInterval
    /// for each connected component. LIV must have a LiveInterval for each
    /// connected component. The LiveIntervals in Liv[1..] must be empty.
    /// Instructions using LIV[0] are rewritten.
    void Distribute(LiveInterval *LIV[], MachineRegisterInfo &MRI);

  };

}
#endif
