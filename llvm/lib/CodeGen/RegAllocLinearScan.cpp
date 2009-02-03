//===-- RegAllocLinearScan.cpp - Linear Scan register allocator -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a linear scan register allocator.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "PhysRegTracker.h"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <set>
#include <queue>
#include <memory>
#include <cmath>
using namespace llvm;

STATISTIC(NumIters     , "Number of iterations performed");
STATISTIC(NumBacktracks, "Number of times we had to backtrack");
STATISTIC(NumCoalesce,   "Number of copies coalesced");

static cl::opt<bool>
NewHeuristic("new-spilling-heuristic",
             cl::desc("Use new spilling heuristic"),
             cl::init(false), cl::Hidden);

static cl::opt<bool>
PreSplitIntervals("pre-alloc-split",
                  cl::desc("Pre-register allocation live interval splitting"),
                  cl::init(false), cl::Hidden);

static RegisterRegAlloc
linearscanRegAlloc("linearscan", "linear scan register allocator",
                   createLinearScanRegisterAllocator);

namespace {
  struct VISIBILITY_HIDDEN RALinScan : public MachineFunctionPass {
    static char ID;
    RALinScan() : MachineFunctionPass(&ID) {}

    typedef std::pair<LiveInterval*, LiveInterval::iterator> IntervalPtr;
    typedef SmallVector<IntervalPtr, 32> IntervalPtrs;
  private:
    /// RelatedRegClasses - This structure is built the first time a function is
    /// compiled, and keeps track of which register classes have registers that
    /// belong to multiple classes or have aliases that are in other classes.
    EquivalenceClasses<const TargetRegisterClass*> RelatedRegClasses;
    DenseMap<unsigned, const TargetRegisterClass*> OneClassForEachPhysReg;

    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    BitVector allocatableRegs_;
    LiveIntervals* li_;
    LiveStacks* ls_;
    const MachineLoopInfo *loopInfo;

    /// handled_ - Intervals are added to the handled_ set in the order of their
    /// start value.  This is uses for backtracking.
    std::vector<LiveInterval*> handled_;

    /// fixed_ - Intervals that correspond to machine registers.
    ///
    IntervalPtrs fixed_;

    /// active_ - Intervals that are currently being processed, and which have a
    /// live range active for the current point.
    IntervalPtrs active_;

    /// inactive_ - Intervals that are currently being processed, but which have
    /// a hold at the current point.
    IntervalPtrs inactive_;

    typedef std::priority_queue<LiveInterval*,
                                SmallVector<LiveInterval*, 64>,
                                greater_ptr<LiveInterval> > IntervalHeap;
    IntervalHeap unhandled_;
    std::auto_ptr<PhysRegTracker> prt_;
    std::auto_ptr<VirtRegMap> vrm_;
    std::auto_ptr<Spiller> spiller_;

  public:
    virtual const char* getPassName() const {
      return "Linear Scan Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LiveIntervals>();
      if (StrongPHIElim)
        AU.addRequiredID(StrongPHIEliminationID);
      // Make sure PassManager knows which analyses to make available
      // to coalescing and which analyses coalescing invalidates.
      AU.addRequiredTransitive<RegisterCoalescer>();
      if (PreSplitIntervals)
        AU.addRequiredID(PreAllocSplittingID);
      AU.addRequired<LiveStacks>();
      AU.addPreserved<LiveStacks>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - register allocate the whole function
    bool runOnMachineFunction(MachineFunction&);

  private:
    /// linearScan - the linear scan algorithm
    void linearScan();

    /// initIntervalSets - initialize the interval sets.
    ///
    void initIntervalSets();

    /// processActiveIntervals - expire old intervals and move non-overlapping
    /// ones to the inactive list.
    void processActiveIntervals(unsigned CurPoint);

    /// processInactiveIntervals - expire old intervals and move overlapping
    /// ones to the active list.
    void processInactiveIntervals(unsigned CurPoint);

    /// assignRegOrStackSlotAtInterval - assign a register if one
    /// is available, or spill.
    void assignRegOrStackSlotAtInterval(LiveInterval* cur);

    /// findIntervalsToSpill - Determine the intervals to spill for the
    /// specified interval. It's passed the physical registers whose spill
    /// weight is the lowest among all the registers whose live intervals
    /// conflict with the interval.
    void findIntervalsToSpill(LiveInterval *cur,
                            std::vector<std::pair<unsigned,float> > &Candidates,
                            unsigned NumCands,
                            SmallVector<LiveInterval*, 8> &SpillIntervals);

    /// attemptTrivialCoalescing - If a simple interval is defined by a copy,
    /// try allocate the definition the same register as the source register
    /// if the register is not defined during live time of the interval. This
    /// eliminate a copy. This is used to coalesce copies which were not
    /// coalesced away before allocation either due to dest and src being in
    /// different register classes or because the coalescer was overly
    /// conservative.
    unsigned attemptTrivialCoalescing(LiveInterval &cur, unsigned Reg);

    ///
    /// register handling helpers
    ///

    /// getFreePhysReg - return a free physical register for this virtual
    /// register interval if we have one, otherwise return 0.
    unsigned getFreePhysReg(LiveInterval* cur);

    /// assignVirt2StackSlot - assigns this virtual register to a
    /// stack slot. returns the stack slot
    int assignVirt2StackSlot(unsigned virtReg);

    void ComputeRelatedRegClasses();

    template <typename ItTy>
    void printIntervals(const char* const str, ItTy i, ItTy e) const {
      if (str) DOUT << str << " intervals:\n";
      for (; i != e; ++i) {
        DOUT << "\t" << *i->first << " -> ";
        unsigned reg = i->first->reg;
        if (TargetRegisterInfo::isVirtualRegister(reg)) {
          reg = vrm_->getPhys(reg);
        }
        DOUT << tri_->getName(reg) << '\n';
      }
    }
  };
  char RALinScan::ID = 0;
}

static RegisterPass<RALinScan>
X("linearscan-regalloc", "Linear Scan Register Allocator");

void RALinScan::ComputeRelatedRegClasses() {
  const TargetRegisterInfo &TRI = *tri_;
  
  // First pass, add all reg classes to the union, and determine at least one
  // reg class that each register is in.
  bool HasAliases = false;
  for (TargetRegisterInfo::regclass_iterator RCI = TRI.regclass_begin(),
       E = TRI.regclass_end(); RCI != E; ++RCI) {
    RelatedRegClasses.insert(*RCI);
    for (TargetRegisterClass::iterator I = (*RCI)->begin(), E = (*RCI)->end();
         I != E; ++I) {
      HasAliases = HasAliases || *TRI.getAliasSet(*I) != 0;
      
      const TargetRegisterClass *&PRC = OneClassForEachPhysReg[*I];
      if (PRC) {
        // Already processed this register.  Just make sure we know that
        // multiple register classes share a register.
        RelatedRegClasses.unionSets(PRC, *RCI);
      } else {
        PRC = *RCI;
      }
    }
  }
  
  // Second pass, now that we know conservatively what register classes each reg
  // belongs to, add info about aliases.  We don't need to do this for targets
  // without register aliases.
  if (HasAliases)
    for (DenseMap<unsigned, const TargetRegisterClass*>::iterator
         I = OneClassForEachPhysReg.begin(), E = OneClassForEachPhysReg.end();
         I != E; ++I)
      for (const unsigned *AS = TRI.getAliasSet(I->first); *AS; ++AS)
        RelatedRegClasses.unionSets(I->second, OneClassForEachPhysReg[*AS]);
}

/// attemptTrivialCoalescing - If a simple interval is defined by a copy,
/// try allocate the definition the same register as the source register
/// if the register is not defined during live time of the interval. This
/// eliminate a copy. This is used to coalesce copies which were not
/// coalesced away before allocation either due to dest and src being in
/// different register classes or because the coalescer was overly
/// conservative.
unsigned RALinScan::attemptTrivialCoalescing(LiveInterval &cur, unsigned Reg) {
  if ((cur.preference && cur.preference == Reg) || !cur.containsOneValue())
    return Reg;

  VNInfo *vni = cur.begin()->valno;
  if (!vni->def || vni->def == ~1U || vni->def == ~0U)
    return Reg;
  MachineInstr *CopyMI = li_->getInstructionFromIndex(vni->def);
  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (!CopyMI ||
      !tii_->isMoveInstr(*CopyMI, SrcReg, DstReg, SrcSubReg, DstSubReg))
    return Reg;
  if (TargetRegisterInfo::isVirtualRegister(SrcReg)) {
    if (!vrm_->isAssignedReg(SrcReg))
      return Reg;
    else
      SrcReg = vrm_->getPhys(SrcReg);
  }
  if (Reg == SrcReg)
    return Reg;

  const TargetRegisterClass *RC = mri_->getRegClass(cur.reg);
  if (!RC->contains(SrcReg))
    return Reg;

  // Try to coalesce.
  if (!li_->conflictsWithPhysRegDef(cur, *vrm_, SrcReg)) {
    DOUT << "Coalescing: " << cur << " -> " << tri_->getName(SrcReg)
         << '\n';
    vrm_->clearVirt(cur.reg);
    vrm_->assignVirt2Phys(cur.reg, SrcReg);
    ++NumCoalesce;
    return SrcReg;
  }

  return Reg;
}

bool RALinScan::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &fn.getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  allocatableRegs_ = tri_->getAllocatableSet(fn);
  li_ = &getAnalysis<LiveIntervals>();
  ls_ = &getAnalysis<LiveStacks>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  // We don't run the coalescer here because we have no reason to
  // interact with it.  If the coalescer requires interaction, it
  // won't do anything.  If it doesn't require interaction, we assume
  // it was run as a separate pass.

  // If this is the first function compiled, compute the related reg classes.
  if (RelatedRegClasses.empty())
    ComputeRelatedRegClasses();
  
  if (!prt_.get()) prt_.reset(new PhysRegTracker(*tri_));
  vrm_.reset(new VirtRegMap(*mf_));
  if (!spiller_.get()) spiller_.reset(createSpiller());

  initIntervalSets();

  linearScan();

  // Rewrite spill code and update the PhysRegsUsed set.
  spiller_->runOnMachineFunction(*mf_, *vrm_);
  vrm_.reset();  // Free the VirtRegMap

  assert(unhandled_.empty() && "Unhandled live intervals remain!");
  fixed_.clear();
  active_.clear();
  inactive_.clear();
  handled_.clear();

  return true;
}

/// initIntervalSets - initialize the interval sets.
///
void RALinScan::initIntervalSets()
{
  assert(unhandled_.empty() && fixed_.empty() &&
         active_.empty() && inactive_.empty() &&
         "interval sets should be empty on initialization");

  handled_.reserve(li_->getNumIntervals());

  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i) {
    if (TargetRegisterInfo::isPhysicalRegister(i->second->reg)) {
      mri_->setPhysRegUsed(i->second->reg);
      fixed_.push_back(std::make_pair(i->second, i->second->begin()));
    } else
      unhandled_.push(i->second);
  }
}

void RALinScan::linearScan()
{
  // linear scan algorithm
  DOUT << "********** LINEAR SCAN **********\n";
  DOUT << "********** Function: " << mf_->getFunction()->getName() << '\n';

  DEBUG(printIntervals("fixed", fixed_.begin(), fixed_.end()));

  while (!unhandled_.empty()) {
    // pick the interval with the earliest start point
    LiveInterval* cur = unhandled_.top();
    unhandled_.pop();
    ++NumIters;
    DOUT << "\n*** CURRENT ***: " << *cur << '\n';

    if (!cur->empty()) {
      processActiveIntervals(cur->beginNumber());
      processInactiveIntervals(cur->beginNumber());

      assert(TargetRegisterInfo::isVirtualRegister(cur->reg) &&
             "Can only allocate virtual registers!");
    }

    // Allocating a virtual register. try to find a free
    // physical register or spill an interval (possibly this one) in order to
    // assign it one.
    assignRegOrStackSlotAtInterval(cur);

    DEBUG(printIntervals("active", active_.begin(), active_.end()));
    DEBUG(printIntervals("inactive", inactive_.begin(), inactive_.end()));
  }

  // expire any remaining active intervals
  while (!active_.empty()) {
    IntervalPtr &IP = active_.back();
    unsigned reg = IP.first->reg;
    DOUT << "\tinterval " << *IP.first << " expired\n";
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    prt_->delRegUse(reg);
    active_.pop_back();
  }

  // expire any remaining inactive intervals
  DEBUG(for (IntervalPtrs::reverse_iterator
               i = inactive_.rbegin(); i != inactive_.rend(); ++i)
        DOUT << "\tinterval " << *i->first << " expired\n");
  inactive_.clear();

  // Add live-ins to every BB except for entry. Also perform trivial coalescing.
  MachineFunction::iterator EntryMBB = mf_->begin();
  SmallVector<MachineBasicBlock*, 8> LiveInMBBs;
  for (LiveIntervals::iterator i = li_->begin(), e = li_->end(); i != e; ++i) {
    LiveInterval &cur = *i->second;
    unsigned Reg = 0;
    bool isPhys = TargetRegisterInfo::isPhysicalRegister(cur.reg);
    if (isPhys)
      Reg = cur.reg;
    else if (vrm_->isAssignedReg(cur.reg))
      Reg = attemptTrivialCoalescing(cur, vrm_->getPhys(cur.reg));
    if (!Reg)
      continue;
    // Ignore splited live intervals.
    if (!isPhys && vrm_->getPreSplitReg(cur.reg))
      continue;
    for (LiveInterval::Ranges::const_iterator I = cur.begin(), E = cur.end();
         I != E; ++I) {
      const LiveRange &LR = *I;
      if (li_->findLiveInMBBs(LR.start, LR.end, LiveInMBBs)) {
        for (unsigned i = 0, e = LiveInMBBs.size(); i != e; ++i)
          if (LiveInMBBs[i] != EntryMBB)
            LiveInMBBs[i]->addLiveIn(Reg);
        LiveInMBBs.clear();
      }
    }
  }

  DOUT << *vrm_;
}

/// processActiveIntervals - expire old intervals and move non-overlapping ones
/// to the inactive list.
void RALinScan::processActiveIntervals(unsigned CurPoint)
{
  DOUT << "\tprocessing active intervals:\n";

  for (unsigned i = 0, e = active_.size(); i != e; ++i) {
    LiveInterval *Interval = active_[i].first;
    LiveInterval::iterator IntervalPos = active_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {     // Remove expired intervals.
      DOUT << "\t\tinterval " << *Interval << " expired\n";
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);

      // Pop off the end of the list.
      active_[i] = active_.back();
      active_.pop_back();
      --i; --e;

    } else if (IntervalPos->start > CurPoint) {
      // Move inactive intervals to inactive list.
      DOUT << "\t\tinterval " << *Interval << " inactive\n";
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->delRegUse(reg);
      // add to inactive.
      inactive_.push_back(std::make_pair(Interval, IntervalPos));

      // Pop off the end of the list.
      active_[i] = active_.back();
      active_.pop_back();
      --i; --e;
    } else {
      // Otherwise, just update the iterator position.
      active_[i].second = IntervalPos;
    }
  }
}

/// processInactiveIntervals - expire old intervals and move overlapping
/// ones to the active list.
void RALinScan::processInactiveIntervals(unsigned CurPoint)
{
  DOUT << "\tprocessing inactive intervals:\n";

  for (unsigned i = 0, e = inactive_.size(); i != e; ++i) {
    LiveInterval *Interval = inactive_[i].first;
    LiveInterval::iterator IntervalPos = inactive_[i].second;
    unsigned reg = Interval->reg;

    IntervalPos = Interval->advanceTo(IntervalPos, CurPoint);

    if (IntervalPos == Interval->end()) {       // remove expired intervals.
      DOUT << "\t\tinterval " << *Interval << " expired\n";

      // Pop off the end of the list.
      inactive_[i] = inactive_.back();
      inactive_.pop_back();
      --i; --e;
    } else if (IntervalPos->start <= CurPoint) {
      // move re-activated intervals in active list
      DOUT << "\t\tinterval " << *Interval << " active\n";
      assert(TargetRegisterInfo::isVirtualRegister(reg) &&
             "Can only allocate virtual registers!");
      reg = vrm_->getPhys(reg);
      prt_->addRegUse(reg);
      // add to active
      active_.push_back(std::make_pair(Interval, IntervalPos));

      // Pop off the end of the list.
      inactive_[i] = inactive_.back();
      inactive_.pop_back();
      --i; --e;
    } else {
      // Otherwise, just update the iterator position.
      inactive_[i].second = IntervalPos;
    }
  }
}

/// updateSpillWeights - updates the spill weights of the specifed physical
/// register and its weight.
static void updateSpillWeights(std::vector<float> &Weights,
                               unsigned reg, float weight,
                               const TargetRegisterInfo *TRI) {
  Weights[reg] += weight;
  for (const unsigned* as = TRI->getAliasSet(reg); *as; ++as)
    Weights[*as] += weight;
}

static
RALinScan::IntervalPtrs::iterator
FindIntervalInVector(RALinScan::IntervalPtrs &IP, LiveInterval *LI) {
  for (RALinScan::IntervalPtrs::iterator I = IP.begin(), E = IP.end();
       I != E; ++I)
    if (I->first == LI) return I;
  return IP.end();
}

static void RevertVectorIteratorsTo(RALinScan::IntervalPtrs &V, unsigned Point){
  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    RALinScan::IntervalPtr &IP = V[i];
    LiveInterval::iterator I = std::upper_bound(IP.first->begin(),
                                                IP.second, Point);
    if (I != IP.first->begin()) --I;
    IP.second = I;
  }
}

/// addStackInterval - Create a LiveInterval for stack if the specified live
/// interval has been spilled.
static void addStackInterval(LiveInterval *cur, LiveStacks *ls_,
                             LiveIntervals *li_, float &Weight,
                             VirtRegMap &vrm_) {
  int SS = vrm_.getStackSlot(cur->reg);
  if (SS == VirtRegMap::NO_STACK_SLOT)
    return;
  LiveInterval &SI = ls_->getOrCreateInterval(SS);
  SI.weight += Weight;

  VNInfo *VNI;
  if (SI.hasAtLeastOneValue())
    VNI = SI.getValNumInfo(0);
  else
    VNI = SI.getNextValue(~0U, 0, ls_->getVNInfoAllocator());

  LiveInterval &RI = li_->getInterval(cur->reg);
  // FIXME: This may be overly conservative.
  SI.MergeRangesInAsValue(RI, VNI);
}

/// getConflictWeight - Return the number of conflicts between cur
/// live interval and defs and uses of Reg weighted by loop depthes.
static float getConflictWeight(LiveInterval *cur, unsigned Reg,
                                  LiveIntervals *li_,
                                  MachineRegisterInfo *mri_,
                                  const MachineLoopInfo *loopInfo) {
  float Conflicts = 0;
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(Reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineInstr *MI = &*I;
    if (cur->liveAt(li_->getInstructionIndex(MI))) {
      unsigned loopDepth = loopInfo->getLoopDepth(MI->getParent());
      Conflicts += powf(10.0f, (float)loopDepth);
    }
  }
  return Conflicts;
}

/// findIntervalsToSpill - Determine the intervals to spill for the
/// specified interval. It's passed the physical registers whose spill
/// weight is the lowest among all the registers whose live intervals
/// conflict with the interval.
void RALinScan::findIntervalsToSpill(LiveInterval *cur,
                            std::vector<std::pair<unsigned,float> > &Candidates,
                            unsigned NumCands,
                            SmallVector<LiveInterval*, 8> &SpillIntervals) {
  // We have figured out the *best* register to spill. But there are other
  // registers that are pretty good as well (spill weight within 3%). Spill
  // the one that has fewest defs and uses that conflict with cur.
  float Conflicts[3] = { 0.0f, 0.0f, 0.0f };
  SmallVector<LiveInterval*, 8> SLIs[3];

  DOUT << "\tConsidering " << NumCands << " candidates: ";
  DEBUG(for (unsigned i = 0; i != NumCands; ++i)
          DOUT << tri_->getName(Candidates[i].first) << " ";
        DOUT << "\n";);
  
  // Calculate the number of conflicts of each candidate.
  for (IntervalPtrs::iterator i = active_.begin(); i != active_.end(); ++i) {
    unsigned Reg = i->first->reg;
    unsigned PhysReg = vrm_->getPhys(Reg);
    if (!cur->overlapsFrom(*i->first, i->second))
      continue;
    for (unsigned j = 0; j < NumCands; ++j) {
      unsigned Candidate = Candidates[j].first;
      if (tri_->regsOverlap(PhysReg, Candidate)) {
        if (NumCands > 1)
          Conflicts[j] += getConflictWeight(cur, Reg, li_, mri_, loopInfo);
        SLIs[j].push_back(i->first);
      }
    }
  }

  for (IntervalPtrs::iterator i = inactive_.begin(); i != inactive_.end(); ++i){
    unsigned Reg = i->first->reg;
    unsigned PhysReg = vrm_->getPhys(Reg);
    if (!cur->overlapsFrom(*i->first, i->second-1))
      continue;
    for (unsigned j = 0; j < NumCands; ++j) {
      unsigned Candidate = Candidates[j].first;
      if (tri_->regsOverlap(PhysReg, Candidate)) {
        if (NumCands > 1)
          Conflicts[j] += getConflictWeight(cur, Reg, li_, mri_, loopInfo);
        SLIs[j].push_back(i->first);
      }
    }
  }

  // Which is the best candidate?
  unsigned BestCandidate = 0;
  float MinConflicts = Conflicts[0];
  for (unsigned i = 1; i != NumCands; ++i) {
    if (Conflicts[i] < MinConflicts) {
      BestCandidate = i;
      MinConflicts = Conflicts[i];
    }
  }

  std::copy(SLIs[BestCandidate].begin(), SLIs[BestCandidate].end(),
            std::back_inserter(SpillIntervals));
}

namespace {
  struct WeightCompare {
    typedef std::pair<unsigned, float> RegWeightPair;
    bool operator()(const RegWeightPair &LHS, const RegWeightPair &RHS) const {
      return LHS.second < RHS.second;
    }
  };
}

static bool weightsAreClose(float w1, float w2) {
  if (!NewHeuristic)
    return false;

  float diff = w1 - w2;
  if (diff <= 0.02f)  // Within 0.02f
    return true;
  return (diff / w2) <= 0.05f;  // Within 5%.
}

/// assignRegOrStackSlotAtInterval - assign a register if one is available, or
/// spill.
void RALinScan::assignRegOrStackSlotAtInterval(LiveInterval* cur)
{
  DOUT << "\tallocating current interval: ";

  // This is an implicitly defined live interval, just assign any register.
  const TargetRegisterClass *RC = mri_->getRegClass(cur->reg);
  if (cur->empty()) {
    unsigned physReg = cur->preference;
    if (!physReg)
      physReg = *RC->allocation_order_begin(*mf_);
    DOUT <<  tri_->getName(physReg) << '\n';
    // Note the register is not really in use.
    vrm_->assignVirt2Phys(cur->reg, physReg);
    return;
  }

  PhysRegTracker backupPrt = *prt_;

  std::vector<std::pair<unsigned, float> > SpillWeightsToAdd;
  unsigned StartPosition = cur->beginNumber();
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);

  // If start of this live interval is defined by a move instruction and its
  // source is assigned a physical register that is compatible with the target
  // register class, then we should try to assign it the same register.
  // This can happen when the move is from a larger register class to a smaller
  // one, e.g. X86::mov32to32_. These move instructions are not coalescable.
  if (!cur->preference && cur->hasAtLeastOneValue()) {
    VNInfo *vni = cur->begin()->valno;
    if (vni->def && vni->def != ~1U && vni->def != ~0U) {
      MachineInstr *CopyMI = li_->getInstructionFromIndex(vni->def);
      unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
      if (CopyMI &&
          tii_->isMoveInstr(*CopyMI, SrcReg, DstReg, SrcSubReg, DstSubReg)) {
        unsigned Reg = 0;
        if (TargetRegisterInfo::isPhysicalRegister(SrcReg))
          Reg = SrcReg;
        else if (vrm_->isAssignedReg(SrcReg))
          Reg = vrm_->getPhys(SrcReg);
        if (Reg && allocatableRegs_[Reg] && RC->contains(Reg))
          cur->preference = Reg;
      }
    }
  }

  // for every interval in inactive we overlap with, mark the
  // register as not free and update spill weights.
  for (IntervalPtrs::const_iterator i = inactive_.begin(),
         e = inactive_.end(); i != e; ++i) {
    unsigned Reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
           "Can only allocate virtual registers!");
    const TargetRegisterClass *RegRC = mri_->getRegClass(Reg);
    // If this is not in a related reg class to the register we're allocating, 
    // don't check it.
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&
        cur->overlapsFrom(*i->first, i->second-1)) {
      Reg = vrm_->getPhys(Reg);
      prt_->addRegUse(Reg);
      SpillWeightsToAdd.push_back(std::make_pair(Reg, i->first->weight));
    }
  }
  
  // Speculatively check to see if we can get a register right now.  If not,
  // we know we won't be able to by adding more constraints.  If so, we can
  // check to see if it is valid.  Doing an exhaustive search of the fixed_ list
  // is very bad (it contains all callee clobbered registers for any functions
  // with a call), so we want to avoid doing that if possible.
  unsigned physReg = getFreePhysReg(cur);
  unsigned BestPhysReg = physReg;
  if (physReg) {
    // We got a register.  However, if it's in the fixed_ list, we might
    // conflict with it.  Check to see if we conflict with it or any of its
    // aliases.
    SmallSet<unsigned, 8> RegAliases;
    for (const unsigned *AS = tri_->getAliasSet(physReg); *AS; ++AS)
      RegAliases.insert(*AS);
    
    bool ConflictsWithFixed = false;
    for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
      IntervalPtr &IP = fixed_[i];
      if (physReg == IP.first->reg || RegAliases.count(IP.first->reg)) {
        // Okay, this reg is on the fixed list.  Check to see if we actually
        // conflict.
        LiveInterval *I = IP.first;
        if (I->endNumber() > StartPosition) {
          LiveInterval::iterator II = I->advanceTo(IP.second, StartPosition);
          IP.second = II;
          if (II != I->begin() && II->start > StartPosition)
            --II;
          if (cur->overlapsFrom(*I, II)) {
            ConflictsWithFixed = true;
            break;
          }
        }
      }
    }
    
    // Okay, the register picked by our speculative getFreePhysReg call turned
    // out to be in use.  Actually add all of the conflicting fixed registers to
    // prt so we can do an accurate query.
    if (ConflictsWithFixed) {
      // For every interval in fixed we overlap with, mark the register as not
      // free and update spill weights.
      for (unsigned i = 0, e = fixed_.size(); i != e; ++i) {
        IntervalPtr &IP = fixed_[i];
        LiveInterval *I = IP.first;

        const TargetRegisterClass *RegRC = OneClassForEachPhysReg[I->reg];
        if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader &&       
            I->endNumber() > StartPosition) {
          LiveInterval::iterator II = I->advanceTo(IP.second, StartPosition);
          IP.second = II;
          if (II != I->begin() && II->start > StartPosition)
            --II;
          if (cur->overlapsFrom(*I, II)) {
            unsigned reg = I->reg;
            prt_->addRegUse(reg);
            SpillWeightsToAdd.push_back(std::make_pair(reg, I->weight));
          }
        }
      }

      // Using the newly updated prt_ object, which includes conflicts in the
      // future, see if there are any registers available.
      physReg = getFreePhysReg(cur);
    }
  }
    
  // Restore the physical register tracker, removing information about the
  // future.
  *prt_ = backupPrt;
  
  // if we find a free register, we are done: assign this virtual to
  // the free physical register and add this interval to the active
  // list.
  if (physReg) {
    DOUT <<  tri_->getName(physReg) << '\n';
    vrm_->assignVirt2Phys(cur->reg, physReg);
    prt_->addRegUse(physReg);
    active_.push_back(std::make_pair(cur, cur->begin()));
    handled_.push_back(cur);
    return;
  }
  DOUT << "no free registers\n";

  // Compile the spill weights into an array that is better for scanning.
  std::vector<float> SpillWeights(tri_->getNumRegs(), 0.0f);
  for (std::vector<std::pair<unsigned, float> >::iterator
       I = SpillWeightsToAdd.begin(), E = SpillWeightsToAdd.end(); I != E; ++I)
    updateSpillWeights(SpillWeights, I->first, I->second, tri_);
  
  // for each interval in active, update spill weights.
  for (IntervalPtrs::const_iterator i = active_.begin(), e = active_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");
    reg = vrm_->getPhys(reg);
    updateSpillWeights(SpillWeights, reg, i->first->weight, tri_);
  }
 
  DOUT << "\tassigning stack slot at interval "<< *cur << ":\n";

  // Find a register to spill.
  float minWeight = HUGE_VALF;
  unsigned minReg = 0; /*cur->preference*/;  // Try the preferred register first.

  bool Found = false;
  std::vector<std::pair<unsigned,float> > RegsWeights;
  if (!minReg || SpillWeights[minReg] == HUGE_VALF)
    for (TargetRegisterClass::iterator i = RC->allocation_order_begin(*mf_),
           e = RC->allocation_order_end(*mf_); i != e; ++i) {
      unsigned reg = *i;
      float regWeight = SpillWeights[reg];
      if (minWeight > regWeight)
        Found = true;
      RegsWeights.push_back(std::make_pair(reg, regWeight));
    }
  
  // If we didn't find a register that is spillable, try aliases?
  if (!Found) {
    for (TargetRegisterClass::iterator i = RC->allocation_order_begin(*mf_),
           e = RC->allocation_order_end(*mf_); i != e; ++i) {
      unsigned reg = *i;
      // No need to worry about if the alias register size < regsize of RC.
      // We are going to spill all registers that alias it anyway.
      for (const unsigned* as = tri_->getAliasSet(reg); *as; ++as)
        RegsWeights.push_back(std::make_pair(*as, SpillWeights[*as]));
    }
  }

  // Sort all potential spill candidates by weight.
  std::sort(RegsWeights.begin(), RegsWeights.end(), WeightCompare());
  minReg = RegsWeights[0].first;
  minWeight = RegsWeights[0].second;
  if (minWeight == HUGE_VALF) {
    // All registers must have inf weight. Just grab one!
    minReg = BestPhysReg ? BestPhysReg : *RC->allocation_order_begin(*mf_);
    if (cur->weight == HUGE_VALF ||
        li_->getApproximateInstructionCount(*cur) == 0) {
      // Spill a physical register around defs and uses.
      li_->spillPhysRegAroundRegDefsUses(*cur, minReg, *vrm_);
      assignRegOrStackSlotAtInterval(cur);
      return;
    }
  }

  // Find up to 3 registers to consider as spill candidates.
  unsigned LastCandidate = RegsWeights.size() >= 3 ? 3 : 1;
  while (LastCandidate > 1) {
    if (weightsAreClose(RegsWeights[LastCandidate-1].second, minWeight))
      break;
    --LastCandidate;
  }

  DOUT << "\t\tregister(s) with min weight(s): ";
  DEBUG(for (unsigned i = 0; i != LastCandidate; ++i)
          DOUT << tri_->getName(RegsWeights[i].first)
               << " (" << RegsWeights[i].second << ")\n");

  // if the current has the minimum weight, we need to spill it and
  // add any added intervals back to unhandled, and restart
  // linearscan.
  if (cur->weight != HUGE_VALF && cur->weight <= minWeight) {
    DOUT << "\t\t\tspilling(c): " << *cur << '\n';
    float SSWeight;
    SmallVector<LiveInterval*, 8> spillIs;
    std::vector<LiveInterval*> added =
      li_->addIntervalsForSpills(*cur, spillIs, loopInfo, *vrm_, SSWeight);
    addStackInterval(cur, ls_, li_, SSWeight, *vrm_);
    if (added.empty())
      return;  // Early exit if all spills were folded.

    // Merge added with unhandled.  Note that we know that
    // addIntervalsForSpills returns intervals sorted by their starting
    // point.
    for (unsigned i = 0, e = added.size(); i != e; ++i)
      unhandled_.push(added[i]);
    return;
  }

  ++NumBacktracks;

  // push the current interval back to unhandled since we are going
  // to re-run at least this iteration. Since we didn't modify it it
  // should go back right in the front of the list
  unhandled_.push(cur);

  assert(TargetRegisterInfo::isPhysicalRegister(minReg) &&
         "did not choose a register to spill?");

  // We spill all intervals aliasing the register with
  // minimum weight, rollback to the interval with the earliest
  // start point and let the linear scan algorithm run again
  SmallVector<LiveInterval*, 8> spillIs;

  // Determine which intervals have to be spilled.
  findIntervalsToSpill(cur, RegsWeights, LastCandidate, spillIs);

  // Set of spilled vregs (used later to rollback properly)
  SmallSet<unsigned, 8> spilled;

  // The earliest start of a Spilled interval indicates up to where
  // in handled we need to roll back
  unsigned earliestStart = cur->beginNumber();

  // Spill live intervals of virtual regs mapped to the physical register we
  // want to clear (and its aliases).  We only spill those that overlap with the
  // current interval as the rest do not affect its allocation. we also keep
  // track of the earliest start of all spilled live intervals since this will
  // mark our rollback point.
  std::vector<LiveInterval*> added;
  while (!spillIs.empty()) {
    LiveInterval *sli = spillIs.back();
    spillIs.pop_back();
    DOUT << "\t\t\tspilling(a): " << *sli << '\n';
    earliestStart = std::min(earliestStart, sli->beginNumber());
    float SSWeight;
    std::vector<LiveInterval*> newIs =
      li_->addIntervalsForSpills(*sli, spillIs, loopInfo, *vrm_, SSWeight);
    addStackInterval(sli, ls_, li_, SSWeight, *vrm_);
    std::copy(newIs.begin(), newIs.end(), std::back_inserter(added));
    spilled.insert(sli->reg);
  }

  DOUT << "\t\trolling back to: " << earliestStart << '\n';

  // Scan handled in reverse order up to the earliest start of a
  // spilled live interval and undo each one, restoring the state of
  // unhandled.
  while (!handled_.empty()) {
    LiveInterval* i = handled_.back();
    // If this interval starts before t we are done.
    if (i->beginNumber() < earliestStart)
      break;
    DOUT << "\t\t\tundo changes for: " << *i << '\n';
    handled_.pop_back();

    // When undoing a live interval allocation we must know if it is active or
    // inactive to properly update the PhysRegTracker and the VirtRegMap.
    IntervalPtrs::iterator it;
    if ((it = FindIntervalInVector(active_, i)) != active_.end()) {
      active_.erase(it);
      assert(!TargetRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      prt_->delRegUse(vrm_->getPhys(i->reg));
      vrm_->clearVirt(i->reg);
    } else if ((it = FindIntervalInVector(inactive_, i)) != inactive_.end()) {
      inactive_.erase(it);
      assert(!TargetRegisterInfo::isPhysicalRegister(i->reg));
      if (!spilled.count(i->reg))
        unhandled_.push(i);
      vrm_->clearVirt(i->reg);
    } else {
      assert(TargetRegisterInfo::isVirtualRegister(i->reg) &&
             "Can only allocate virtual registers!");
      vrm_->clearVirt(i->reg);
      unhandled_.push(i);
    }

    // It interval has a preference, it must be defined by a copy. Clear the
    // preference now since the source interval allocation may have been undone
    // as well.
    i->preference = 0;
  }

  // Rewind the iterators in the active, inactive, and fixed lists back to the
  // point we reverted to.
  RevertVectorIteratorsTo(active_, earliestStart);
  RevertVectorIteratorsTo(inactive_, earliestStart);
  RevertVectorIteratorsTo(fixed_, earliestStart);

  // scan the rest and undo each interval that expired after t and
  // insert it in active (the next iteration of the algorithm will
  // put it in inactive if required)
  for (unsigned i = 0, e = handled_.size(); i != e; ++i) {
    LiveInterval *HI = handled_[i];
    if (!HI->expiredAt(earliestStart) &&
        HI->expiredAt(cur->beginNumber())) {
      DOUT << "\t\t\tundo changes for: " << *HI << '\n';
      active_.push_back(std::make_pair(HI, HI->begin()));
      assert(!TargetRegisterInfo::isPhysicalRegister(HI->reg));
      prt_->addRegUse(vrm_->getPhys(HI->reg));
    }
  }

  // merge added with unhandled
  for (unsigned i = 0, e = added.size(); i != e; ++i)
    unhandled_.push(added[i]);
}

/// getFreePhysReg - return a free physical register for this virtual register
/// interval if we have one, otherwise return 0.
unsigned RALinScan::getFreePhysReg(LiveInterval *cur) {
  SmallVector<unsigned, 256> inactiveCounts;
  unsigned MaxInactiveCount = 0;
  
  const TargetRegisterClass *RC = mri_->getRegClass(cur->reg);
  const TargetRegisterClass *RCLeader = RelatedRegClasses.getLeaderValue(RC);
 
  for (IntervalPtrs::iterator i = inactive_.begin(), e = inactive_.end();
       i != e; ++i) {
    unsigned reg = i->first->reg;
    assert(TargetRegisterInfo::isVirtualRegister(reg) &&
           "Can only allocate virtual registers!");

    // If this is not in a related reg class to the register we're allocating, 
    // don't check it.
    const TargetRegisterClass *RegRC = mri_->getRegClass(reg);
    if (RelatedRegClasses.getLeaderValue(RegRC) == RCLeader) {
      reg = vrm_->getPhys(reg);
      if (inactiveCounts.size() <= reg)
        inactiveCounts.resize(reg+1);
      ++inactiveCounts[reg];
      MaxInactiveCount = std::max(MaxInactiveCount, inactiveCounts[reg]);
    }
  }

  unsigned FreeReg = 0;
  unsigned FreeRegInactiveCount = 0;

  // If copy coalescer has assigned a "preferred" register, check if it's
  // available first.
  if (cur->preference) {
    if (prt_->isRegAvail(cur->preference) && 
        RC->contains(cur->preference)) {
      DOUT << "\t\tassigned the preferred register: "
           << tri_->getName(cur->preference) << "\n";
      return cur->preference;
    } else
      DOUT << "\t\tunable to assign the preferred register: "
           << tri_->getName(cur->preference) << "\n";
  }

  // Scan for the first available register.
  TargetRegisterClass::iterator I = RC->allocation_order_begin(*mf_);
  TargetRegisterClass::iterator E = RC->allocation_order_end(*mf_);
  assert(I != E && "No allocatable register in this register class!");
  for (; I != E; ++I)
    if (prt_->isRegAvail(*I)) {
      FreeReg = *I;
      if (FreeReg < inactiveCounts.size())
        FreeRegInactiveCount = inactiveCounts[FreeReg];
      else
        FreeRegInactiveCount = 0;
      break;
    }

  // If there are no free regs, or if this reg has the max inactive count,
  // return this register.
  if (FreeReg == 0 || FreeRegInactiveCount == MaxInactiveCount) return FreeReg;
  
  // Continue scanning the registers, looking for the one with the highest
  // inactive count.  Alkis found that this reduced register pressure very
  // slightly on X86 (in rev 1.94 of this file), though this should probably be
  // reevaluated now.
  for (; I != E; ++I) {
    unsigned Reg = *I;
    if (prt_->isRegAvail(Reg) && Reg < inactiveCounts.size() &&
        FreeRegInactiveCount < inactiveCounts[Reg]) {
      FreeReg = Reg;
      FreeRegInactiveCount = inactiveCounts[Reg];
      if (FreeRegInactiveCount == MaxInactiveCount)
        break;    // We found the one with the max inactive count.
    }
  }
  
  return FreeReg;
}

FunctionPass* llvm::createLinearScanRegisterAllocator() {
  return new RALinScan();
}
