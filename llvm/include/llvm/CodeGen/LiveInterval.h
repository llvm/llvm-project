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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <iosfwd>
#include <cassert>

namespace llvm {
  class MachineInstr;
  class TargetRegisterInfo;
  struct LiveInterval;

  /// VNInfo - If the value number definition is undefined (e.g. phi
  /// merge point), it contains ~0u,x. If the value number is not in use, it
  /// contains ~1u,x to indicate that the value # is not used. 
  ///   def   - Instruction # of the definition.
  ///         - or reg # of the definition if it's a stack slot liveinterval.
  ///   copy  - Copy iff val# is defined by a copy; zero otherwise.
  ///   hasPHIKill - One or more of the kills are PHI nodes.
  ///   redefByEC - Re-defined by early clobber somewhere during the live range.
  ///   kills - Instruction # of the kills.
  struct VNInfo {
    unsigned id;
    unsigned def;
    MachineInstr *copy;
    bool hasPHIKill : 1;
    bool redefByEC : 1;
    SmallVector<unsigned, 4> kills;
    VNInfo()
      : id(~1U), def(~1U), copy(0), hasPHIKill(false), redefByEC(false) {}
    VNInfo(unsigned i, unsigned d, MachineInstr *c)
      : id(i), def(d), copy(c), hasPHIKill(false), redefByEC(false) {}
  };

  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    unsigned start;  // Start point of the interval (inclusive)
    unsigned end;    // End point of the interval (exclusive)
    VNInfo *valno;   // identifier for the value contained in this interval.

    LiveRange(unsigned S, unsigned E, VNInfo *V) : start(S), end(E), valno(V) {
      assert(S < E && "Cannot create empty or backwards range");
    }

    /// contains - Return true if the index is covered by this range.
    ///
    bool contains(unsigned I) const {
      return start <= I && I < end;
    }

    bool operator<(const LiveRange &LR) const {
      return start < LR.start || (start == LR.start && end < LR.end);
    }
    bool operator==(const LiveRange &LR) const {
      return start == LR.start && end == LR.end;
    }

    void dump() const;
    void print(std::ostream &os) const;
    void print(std::ostream *os) const { if (os) print(*os); }

  private:
    LiveRange(); // DO NOT IMPLEMENT
  };

  std::ostream& operator<<(std::ostream& os, const LiveRange &LR);


  inline bool operator<(unsigned V, const LiveRange &LR) {
    return V < LR.start;
  }

  inline bool operator<(const LiveRange &LR, unsigned V) {
    return LR.start < V;
  }

  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  struct LiveInterval {
    typedef SmallVector<LiveRange,4> Ranges;
    typedef SmallVector<VNInfo*,4> VNInfoList;

    unsigned reg;        // the register or stack slot of this interval
                         // if the top bits is set, it represents a stack slot.
    float weight;        // weight of this interval
    unsigned short preference; // preferred register for this interval
    Ranges ranges;       // the ranges in which this register is live
    VNInfoList valnos;   // value#'s

  public:
    LiveInterval(unsigned Reg, float Weight, bool IsSS = false)
      : reg(Reg), weight(Weight), preference(0)  {
      if (IsSS)
        reg = reg | (1U << (sizeof(unsigned)*8-1));
    }

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
    /// position is in a hole, this method returns an iterator pointing the the
    /// LiveRange immediately after the hole.
    iterator advanceTo(iterator I, unsigned Pos) {
      if (Pos >= endNumber())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }
    
    void clear() {
      while (!valnos.empty()) {
        VNInfo *VNI = valnos.back();
        valnos.pop_back();
        VNI->~VNInfo();
      }
      
      ranges.clear();
    }

    /// isStackSlot - Return true if this is a stack slot interval.
    ///
    bool isStackSlot() const {
      return reg & (1U << (sizeof(unsigned)*8-1));
    }

    /// getStackSlotIndex - Return stack slot index if this is a stack slot
    /// interval.
    int getStackSlotIndex() const {
      assert(isStackSlot() && "Interval is not a stack slot interval!");
      return reg & ~(1U << (sizeof(unsigned)*8-1));
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
    
    /// copyValNumInfo - Copy the value number info for one value number to
    /// another.
    void copyValNumInfo(VNInfo *DstValNo, const VNInfo *SrcValNo) {
      DstValNo->def = SrcValNo->def;
      DstValNo->copy = SrcValNo->copy;
      DstValNo->hasPHIKill = SrcValNo->hasPHIKill;
      DstValNo->redefByEC = SrcValNo->redefByEC;
      DstValNo->kills = SrcValNo->kills;
    }

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    VNInfo *getNextValue(unsigned MIIdx, MachineInstr *CopyMI,
                         BumpPtrAllocator &VNInfoAllocator) {
#ifdef __GNUC__
      unsigned Alignment = (unsigned)__alignof__(VNInfo);
#else
      // FIXME: ugly.
      unsigned Alignment = 8;
#endif
      VNInfo *VNI =
        static_cast<VNInfo*>(VNInfoAllocator.Allocate((unsigned)sizeof(VNInfo),
                                                      Alignment));
      new (VNI) VNInfo((unsigned)valnos.size(), MIIdx, CopyMI);
      valnos.push_back(VNI);
      return VNI;
    }

    /// addKill - Add a kill instruction index to the specified value
    /// number.
    static void addKill(VNInfo *VNI, unsigned KillIdx) {
      SmallVector<unsigned, 4> &kills = VNI->kills;
      if (kills.empty()) {
        kills.push_back(KillIdx);
      } else {
        SmallVector<unsigned, 4>::iterator
          I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
        kills.insert(I, KillIdx);
      }
    }

    /// addKills - Add a number of kills into the VNInfo kill vector. If this
    /// interval is live at a kill point, then the kill is not added.
    void addKills(VNInfo *VNI, const SmallVector<unsigned, 4> &kills) {
      for (unsigned i = 0, e = static_cast<unsigned>(kills.size());
           i != e; ++i) {
        unsigned KillIdx = kills[i];
        if (!liveBeforeAndAt(KillIdx)) {
          SmallVector<unsigned, 4>::iterator
            I = std::lower_bound(VNI->kills.begin(), VNI->kills.end(), KillIdx);
          VNI->kills.insert(I, KillIdx);
        }
      }
    }

    /// removeKill - Remove the specified kill from the list of kills of
    /// the specified val#.
    static bool removeKill(VNInfo *VNI, unsigned KillIdx) {
      SmallVector<unsigned, 4> &kills = VNI->kills;
      SmallVector<unsigned, 4>::iterator
        I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
      if (I != kills.end() && *I == KillIdx) {
        kills.erase(I);
        return true;
      }
      return false;
    }

    /// removeKills - Remove all the kills in specified range
    /// [Start, End] of the specified val#.
    static void removeKills(VNInfo *VNI, unsigned Start, unsigned End) {
      SmallVector<unsigned, 4> &kills = VNI->kills;
      SmallVector<unsigned, 4>::iterator
        I = std::lower_bound(kills.begin(), kills.end(), Start);
      SmallVector<unsigned, 4>::iterator
        E = std::upper_bound(kills.begin(), kills.end(), End);
      kills.erase(I, E);
    }

    /// isKill - Return true if the specified index is a kill of the
    /// specified val#.
    static bool isKill(const VNInfo *VNI, unsigned KillIdx) {
      const SmallVector<unsigned, 4> &kills = VNI->kills;
      SmallVector<unsigned, 4>::const_iterator
        I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
      return I != kills.end() && *I == KillIdx;
    }
    
    /// MergeValueNumberInto - This method is called when two value nubmers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// LiveRanges with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    VNInfo* MergeValueNumberInto(VNInfo *V1, VNInfo *V2);

    /// MergeInClobberRanges - For any live ranges that are not defined in the
    /// current interval, but are defined in the Clobbers interval, mark them
    /// used with an unknown definition value. Caller must pass in reference to
    /// VNInfoAllocator since it will create a new val#.
    void MergeInClobberRanges(const LiveInterval &Clobbers,
                              BumpPtrAllocator &VNInfoAllocator);

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

    /// Copy - Copy the specified live interval. This copies all the fields
    /// except for the register of the interval.
    void Copy(const LiveInterval &RHS, BumpPtrAllocator &VNInfoAllocator);
    
    bool empty() const { return ranges.empty(); }

    /// beginNumber - Return the lowest numbered slot covered by interval.
    unsigned beginNumber() const {
      if (empty())
        return 0;
      return ranges.front().start;
    }

    /// endNumber - return the maximum point of the interval of the whole,
    /// exclusive.
    unsigned endNumber() const {
      if (empty())
        return 0;
      return ranges.back().end;
    }

    bool expiredAt(unsigned index) const {
      return index >= endNumber();
    }

    bool liveAt(unsigned index) const;

    // liveBeforeAndAt - Check if the interval is live at the index and the
    // index just before it. If index is liveAt, check if it starts a new live
    // range.If it does, then check if the previous live range ends at index-1.
    bool liveBeforeAndAt(unsigned index) const;

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    const LiveRange *getLiveRangeContaining(unsigned Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    const_iterator FindLiveRangeContaining(unsigned Idx) const;

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    iterator FindLiveRangeContaining(unsigned Idx);

    /// findDefinedVNInfo - Find the VNInfo that's defined at the specified
    /// index (register interval) or defined by the specified register (stack
    /// inteval).
    VNInfo *findDefinedVNInfo(unsigned DefIdxOrReg) const;
    
    /// overlaps - Return true if the intersection of the two live intervals is
    /// not empty.
    bool overlaps(const LiveInterval& other) const {
      return overlapsFrom(other, other.begin());
    }

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

    /// join - Join two live intervals (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS intervals as specified.  If
    /// the intervals are not joinable, this aborts.
    void join(LiveInterval &Other, const int *ValNoAssignments,
              const int *RHSValNoAssignments,
              SmallVector<VNInfo*, 16> &NewVNInfo);

    /// isInOneLiveRange - Return true if the range specified is entirely in the
    /// a single LiveRange of the live interval.
    bool isInOneLiveRange(unsigned Start, unsigned End);

    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must be a single LiveRange in its entirety.
    void removeRange(unsigned Start, unsigned End, bool RemoveDeadValNo = false);

    void removeRange(LiveRange LR, bool RemoveDeadValNo = false) {
      removeRange(LR.start, LR.end, RemoveDeadValNo);
    }

    /// removeValNo - Remove all the ranges defined by the specified value#.
    /// Also remove the value# from value# list.
    void removeValNo(VNInfo *ValNo);

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    bool operator<(const LiveInterval& other) const {
      return beginNumber() < other.beginNumber();
    }

    void print(std::ostream &OS, const TargetRegisterInfo *TRI = 0) const;
    void print(std::ostream *OS, const TargetRegisterInfo *TRI = 0) const {
      if (OS) print(*OS, TRI);
    }
    void dump() const;

  private:
    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, unsigned NewStr);
    LiveInterval& operator=(const LiveInterval& rhs); // DO NOT IMPLEMENT
  };

  inline std::ostream &operator<<(std::ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }
}

#endif
