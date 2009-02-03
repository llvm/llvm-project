//===-- SimpleRegisterCoalescing.h - Register Coalescing --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register copy coalescing phase.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H
#define LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/ADT/BitVector.h"
#include <queue>

namespace llvm {
  class SimpleRegisterCoalescing;
  class LiveVariables;
  class TargetRegisterInfo;
  class TargetInstrInfo;
  class VirtRegMap;
  class MachineLoopInfo;

  /// CopyRec - Representation for copy instructions in coalescer queue.
  ///
  struct CopyRec {
    MachineInstr *MI;
    unsigned LoopDepth;
    bool isBackEdge;
    CopyRec(MachineInstr *mi, unsigned depth, bool be)
      : MI(mi), LoopDepth(depth), isBackEdge(be) {};
  };

  template<class SF> class JoinPriorityQueue;

  /// CopyRecSort - Sorting function for coalescer queue.
  ///
  struct CopyRecSort : public std::binary_function<CopyRec,CopyRec,bool> {
    JoinPriorityQueue<CopyRecSort> *JPQ;
    explicit CopyRecSort(JoinPriorityQueue<CopyRecSort> *jpq) : JPQ(jpq) {}
    CopyRecSort(const CopyRecSort &RHS) : JPQ(RHS.JPQ) {}
    bool operator()(CopyRec left, CopyRec right) const;
  };

  /// JoinQueue - A priority queue of copy instructions the coalescer is
  /// going to process.
  template<class SF>
  class JoinPriorityQueue {
    SimpleRegisterCoalescing *Rc;
    std::priority_queue<CopyRec, std::vector<CopyRec>, SF> Queue;

  public:
    explicit JoinPriorityQueue(SimpleRegisterCoalescing *rc)
      : Rc(rc), Queue(SF(this)) {}

    bool empty() const { return Queue.empty(); }
    void push(CopyRec R) { Queue.push(R); }
    CopyRec pop() {
      if (empty()) return CopyRec(0, 0, false);
      CopyRec R = Queue.top();
      Queue.pop();
      return R;
    }

    // Callbacks to SimpleRegisterCoalescing.
    unsigned getRepIntervalSize(unsigned Reg);
  };

  class SimpleRegisterCoalescing : public MachineFunctionPass,
                                   public RegisterCoalescer {
    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    LiveIntervals *li_;
    const MachineLoopInfo* loopInfo;
    
    BitVector allocatableRegs_;
    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs_;

    /// JoinQueue - A priority queue of copy instructions the coalescer is
    /// going to process.
    JoinPriorityQueue<CopyRecSort> *JoinQueue;

    /// JoinedCopies - Keep track of copies eliminated due to coalescing.
    ///
    SmallPtrSet<MachineInstr*, 32> JoinedCopies;

    /// ReMatCopies - Keep track of copies eliminated due to remat.
    ///
    SmallPtrSet<MachineInstr*, 32> ReMatCopies;

    /// ReMatDefs - Keep track of definition instructions which have
    /// been remat'ed.
    SmallPtrSet<MachineInstr*, 8> ReMatDefs;

  public:
    static char ID; // Pass identifcation, replacement for typeid
    SimpleRegisterCoalescing() : MachineFunctionPass(&ID) {}

    struct InstrSlots {
      enum {
        LOAD  = 0,
        USE   = 1,
        DEF   = 2,
        STORE = 3,
        NUM   = 4
      };
    };
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    bool coalesceFunction(MachineFunction &mf, RegallocQuery &) {
      // This runs as an independent pass, so don't do anything.
      return false;
    };

    /// getRepIntervalSize - Called from join priority queue sorting function.
    /// It returns the size of the interval that represent the given register.
    unsigned getRepIntervalSize(unsigned Reg) {
      if (!li_->hasInterval(Reg))
        return 0;
      return li_->getApproximateInstructionCount(li_->getInterval(Reg)) *
             LiveIntervals::InstrSlots::NUM;
    }

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;
    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }

  private:
    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void CopyCoalesceInMBB(MachineBasicBlock *MBB,
                           std::vector<CopyRec> &TryAgain);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coalesced away. If it is not currently
    /// possible to coalesce this interval, but it may be possible if other
    /// things get coalesced, then it returns true by reference in 'Again'.
    bool JoinCopy(CopyRec &TheCopy, bool &Again);
    
    /// JoinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  Otherwise, if one of the intervals being joined is a
    /// physreg, this method always canonicalizes DestInt to be it.  The output
    /// "SrcInt" will not have been modified, so we can use this information
    /// below to update aliases.
    bool JoinIntervals(LiveInterval &LHS, LiveInterval &RHS, bool &Swapped);
    
    /// SimpleJoin - Attempt to join the specified interval into this one. The
    /// caller of this method must guarantee that the RHS only contains a single
    /// value number and that the RHS is not defined by a copy from this
    /// interval.  This returns false if the intervals are not joinable, or it
    /// joins them and returns true.
    bool SimpleJoin(LiveInterval &LHS, LiveInterval &RHS);
    
    /// Return true if the two specified registers belong to different register
    /// classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;


    /// AdjustCopiesBackFrom - We found a non-trivially-coalescable copy. If
    /// the source value number is defined by a copy from the destination reg
    /// see if we can merge these two destination reg valno# into a single
    /// value number, eliminating a copy.
    bool AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                              MachineInstr *CopyMI);

    /// HasOtherReachingDefs - Return true if there are definitions of IntB
    /// other than BValNo val# that can reach uses of AValno val# of IntA.
    bool HasOtherReachingDefs(LiveInterval &IntA, LiveInterval &IntB,
                              VNInfo *AValNo, VNInfo *BValNo);

    /// RemoveCopyByCommutingDef - We found a non-trivially-coalescable copy.
    /// If the source value number is defined by a commutable instruction and
    /// its other operand is coalesced to the copy dest register, see if we
    /// can transform the copy into a noop by commuting the definition.
    bool RemoveCopyByCommutingDef(LiveInterval &IntA, LiveInterval &IntB,
                                  MachineInstr *CopyMI);

    bool ReMaterializeTrivialDef(LiveInterval &SrcInt, unsigned DstReg,
                                 MachineInstr *CopyMI);

    /// TurnCopyIntoImpDef - If source of the specified copy is an implicit def,
    /// turn the copy into an implicit def.
    bool TurnCopyIntoImpDef(MachineBasicBlock::iterator &I,
                            MachineBasicBlock *MBB,
                            unsigned DstReg, unsigned SrcReg);

    /// CanCoalesceWithImpDef - Returns true if the specified copy instruction
    /// from an implicit def to another register can be coalesced away.
    bool CanCoalesceWithImpDef(MachineInstr *CopyMI,
                               LiveInterval &li, LiveInterval &ImpLi) const;

    /// RemoveCopiesFromValNo - The specified value# is defined by an implicit
    /// def and it is being removed. Turn all copies from this value# into
    /// identity copies so they will be removed.
    void RemoveCopiesFromValNo(LiveInterval &li, VNInfo *VNI);

    /// isWinToJoinCrossClass - Return true if it's profitable to coalesce
    /// two virtual registers from different register classes.
    bool isWinToJoinCrossClass(unsigned LargeReg, unsigned SmallReg,
                               unsigned Threshold);

    /// HasIncompatibleSubRegDefUse - If we are trying to coalesce a virtual
    /// register with a physical register, check if any of the virtual register
    /// operand is a sub-register use or def. If so, make sure it won't result
    /// in an illegal extract_subreg or insert_subreg instruction.
    bool HasIncompatibleSubRegDefUse(MachineInstr *CopyMI,
                                     unsigned VirtReg, unsigned PhysReg);

    /// CanJoinExtractSubRegToPhysReg - Return true if it's possible to coalesce
    /// an extract_subreg where dst is a physical register, e.g.
    /// cl = EXTRACT_SUBREG reg1024, 1
    bool CanJoinExtractSubRegToPhysReg(unsigned DstReg, unsigned SrcReg,
                                       unsigned SubIdx, unsigned &RealDstReg);

    /// CanJoinInsertSubRegToPhysReg - Return true if it's possible to coalesce
    /// an insert_subreg where src is a physical register, e.g.
    /// reg1024 = INSERT_SUBREG reg1024, c1, 0
    bool CanJoinInsertSubRegToPhysReg(unsigned DstReg, unsigned SrcReg,
                                      unsigned SubIdx, unsigned &RealDstReg);

    /// RangeIsDefinedByCopyFromReg - Return true if the specified live range of
    /// the specified live interval is defined by a copy from the specified
    /// register.
    bool RangeIsDefinedByCopyFromReg(LiveInterval &li, LiveRange *LR,
                                     unsigned Reg);

    /// isBackEdgeCopy - Return true if CopyMI is a back edge copy.
    ///
    bool isBackEdgeCopy(MachineInstr *CopyMI, unsigned DstReg) const;

    /// UpdateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
    /// update the subregister number if it is not zero. If DstReg is a
    /// physical register and the existing subregister number of the def / use
    /// being updated is not zero, make sure to set it to the correct physical
    /// subregister.
    void UpdateRegDefsUses(unsigned SrcReg, unsigned DstReg, unsigned SubIdx);

    /// RemoveDeadImpDef - Remove implicit_def instructions which are
    /// "re-defining" registers due to insert_subreg coalescing. e.g.
    void RemoveDeadImpDef(unsigned Reg, LiveInterval &LI);

    /// RemoveUnnecessaryKills - Remove kill markers that are no longer accurate
    /// due to live range lengthening as the result of coalescing.
    void RemoveUnnecessaryKills(unsigned Reg, LiveInterval &LI);

    /// ShortenDeadCopyLiveRange - Shorten a live range defined by a dead copy.
    /// Return true if live interval is removed.
    bool ShortenDeadCopyLiveRange(LiveInterval &li, MachineInstr *CopyMI);

    /// ShortenDeadCopyLiveRange - Shorten a live range as it's artificially
    /// extended by a dead copy. Mark the last use (if any) of the val# as kill
    /// as ends the live range there. If there isn't another use, then this
    /// live range is dead. Return true if live interval is removed.
    bool ShortenDeadCopySrcLiveRange(LiveInterval &li, MachineInstr *CopyMI);

    /// RemoveDeadDef - If a def of a live interval is now determined dead,
    /// remove the val# it defines. If the live interval becomes empty, remove
    /// it as well.
    bool RemoveDeadDef(LiveInterval &li, MachineInstr *DefMI);

    /// lastRegisterUse - Returns the last use of the specific register between
    /// cycles Start and End or NULL if there are no uses.
    MachineOperand *lastRegisterUse(unsigned Start, unsigned End, unsigned Reg,
                                    unsigned &LastUseIdx) const;

    void printRegName(unsigned reg) const;
  };

} // End llvm namespace

#endif
