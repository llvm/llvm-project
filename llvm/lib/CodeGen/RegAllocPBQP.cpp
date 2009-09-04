//===------ RegAllocPBQP.cpp ---- PBQP Register Allocator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a Partitioned Boolean Quadratic Programming (PBQP) based
// register allocator for LLVM. This allocator works by constructing a PBQP
// problem representing the register allocation problem under consideration,
// solving this using a PBQP solver, and mapping the solution back to a
// register assignment. If any variables are selected for spilling then spill
// code is inserted and the process repeated.
//
// The PBQP solver (pbqp.c) provided for this allocator uses a heuristic tuned
// for register allocation. For more information on PBQP for register
// allocation, see the following papers:
//
//   (1) Hames, L. and Scholz, B. 2006. Nearly optimal register allocation with
//   PBQP. In Proceedings of the 7th Joint Modular Languages Conference
//   (JMLC'06). LNCS, vol. 4228. Springer, New York, NY, USA. 346-361.
//
//   (2) Scholz, B., Eckstein, E. 2002. Register allocation for irregular
//   architectures. In Proceedings of the Joint Conference on Languages,
//   Compilers and Tools for Embedded Systems (LCTES'02), ACM Press, New York,
//   NY, USA, 139-148.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"

#include "PBQP/HeuristicSolver.h"
#include "PBQP/SimpleGraph.h"
#include "PBQP/Heuristics/Briggs.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace llvm;

static RegisterRegAlloc
registerPBQPRepAlloc("pbqp", "PBQP register allocator.",
                      llvm::createPBQPRegisterAllocator);

static cl::opt<bool>
pbqpCoalescing("pbqp-coalescing",
               cl::desc("Attempt coalescing during PBQP register allocation."),
               cl::init(false), cl::Hidden);

namespace {

  ///
  /// PBQP based allocators solve the register allocation problem by mapping
  /// register allocation problems to Partitioned Boolean Quadratic
  /// Programming problems.
  class VISIBILITY_HIDDEN PBQPRegAlloc : public MachineFunctionPass {
  public:

    static char ID;
    
    /// Construct a PBQP register allocator.
    PBQPRegAlloc() : MachineFunctionPass(&ID) {}

    /// Return the pass name.
    virtual const char* getPassName() const {
      return "PBQP Register Allocator";
    }

    /// PBQP analysis usage.
    virtual void getAnalysisUsage(AnalysisUsage &au) const {
      au.addRequired<LiveIntervals>();
      //au.addRequiredID(SplitCriticalEdgesID);
      au.addRequired<RegisterCoalescer>();
      au.addRequired<LiveStacks>();
      au.addPreserved<LiveStacks>();
      au.addRequired<MachineLoopInfo>();
      au.addPreserved<MachineLoopInfo>();
      au.addRequired<VirtRegMap>();
      MachineFunctionPass::getAnalysisUsage(au);
    }

    /// Perform register allocation
    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    typedef std::map<const LiveInterval*, unsigned> LI2NodeMap;
    typedef std::vector<const LiveInterval*> Node2LIMap;
    typedef std::vector<unsigned> AllowedSet;
    typedef std::vector<AllowedSet> AllowedSetMap;
    typedef std::set<unsigned> RegSet;
    typedef std::pair<unsigned, unsigned> RegPair;
    typedef std::map<RegPair, PBQP::PBQPNum> CoalesceMap;

    typedef std::set<LiveInterval*> LiveIntervalSet;

    MachineFunction *mf;
    const TargetMachine *tm;
    const TargetRegisterInfo *tri;
    const TargetInstrInfo *tii;
    const MachineLoopInfo *loopInfo;
    MachineRegisterInfo *mri;

    LiveIntervals *lis;
    LiveStacks *lss;
    VirtRegMap *vrm;

    LI2NodeMap li2Node;
    Node2LIMap node2LI;
    AllowedSetMap allowedSets;
    LiveIntervalSet vregIntervalsToAlloc,
                    emptyVRegIntervals;


    /// Builds a PBQP cost vector.
    template <typename RegContainer>
    PBQP::Vector buildCostVector(unsigned vReg,
                                 const RegContainer &allowed,
                                 const CoalesceMap &cealesces,
                                 PBQP::PBQPNum spillCost) const;

    /// \brief Builds a PBQP interference matrix.
    ///
    /// @return Either a pointer to a non-zero PBQP matrix representing the
    ///         allocation option costs, or a null pointer for a zero matrix.
    ///
    /// Expects allowed sets for two interfering LiveIntervals. These allowed
    /// sets should contain only allocable registers from the LiveInterval's
    /// register class, with any interfering pre-colored registers removed.
    template <typename RegContainer>
    PBQP::Matrix* buildInterferenceMatrix(const RegContainer &allowed1,
                                          const RegContainer &allowed2) const;

    ///
    /// Expects allowed sets for two potentially coalescable LiveIntervals,
    /// and an estimated benefit due to coalescing. The allowed sets should
    /// contain only allocable registers from the LiveInterval's register
    /// classes, with any interfering pre-colored registers removed.
    template <typename RegContainer>
    PBQP::Matrix* buildCoalescingMatrix(const RegContainer &allowed1,
                                        const RegContainer &allowed2,
                                        PBQP::PBQPNum cBenefit) const;

    /// \brief Finds coalescing opportunities and returns them as a map.
    ///
    /// Any entries in the map are guaranteed coalescable, even if their
    /// corresponding live intervals overlap.
    CoalesceMap findCoalesces();

    /// \brief Finds the initial set of vreg intervals to allocate.
    void findVRegIntervalsToAlloc();

    /// \brief Constructs a PBQP problem representation of the register
    /// allocation problem for this function.
    ///
    /// @return a PBQP solver object for the register allocation problem.
    PBQP::SimpleGraph constructPBQPProblem();

    /// \brief Adds a stack interval if the given live interval has been
    /// spilled. Used to support stack slot coloring.
    void addStackInterval(const LiveInterval *spilled,MachineRegisterInfo* mri);

    /// \brief Given a solved PBQP problem maps this solution back to a register
    /// assignment.
    bool mapPBQPToRegAlloc(const PBQP::Solution &solution);

    /// \brief Postprocessing before final spilling. Sets basic block "live in"
    /// variables.
    void finalizeAlloc() const;

  };

  char PBQPRegAlloc::ID = 0;
}


template <typename RegContainer>
PBQP::Vector PBQPRegAlloc::buildCostVector(unsigned vReg,
                                           const RegContainer &allowed,
                                           const CoalesceMap &coalesces,
                                           PBQP::PBQPNum spillCost) const {

  typedef typename RegContainer::const_iterator AllowedItr;

  // Allocate vector. Additional element (0th) used for spill option
  PBQP::Vector v(allowed.size() + 1, 0);

  v[0] = spillCost;

  // Iterate over the allowed registers inserting coalesce benefits if there
  // are any.
  unsigned ai = 0;
  for (AllowedItr itr = allowed.begin(), end = allowed.end();
       itr != end; ++itr, ++ai) {

    unsigned pReg = *itr;

    CoalesceMap::const_iterator cmItr =
      coalesces.find(RegPair(vReg, pReg));

    // No coalesce - on to the next preg.
    if (cmItr == coalesces.end())
      continue;

    // We have a coalesce - insert the benefit.
    v[ai + 1] = -cmItr->second;
  }

  return v;
}

template <typename RegContainer>
PBQP::Matrix* PBQPRegAlloc::buildInterferenceMatrix(
      const RegContainer &allowed1, const RegContainer &allowed2) const {

  typedef typename RegContainer::const_iterator RegContainerIterator;

  // Construct a PBQP matrix representing the cost of allocation options. The
  // rows and columns correspond to the allocation options for the two live
  // intervals.  Elements will be infinite where corresponding registers alias,
  // since we cannot allocate aliasing registers to interfering live intervals.
  // All other elements (non-aliasing combinations) will have zero cost. Note
  // that the spill option (element 0,0) has zero cost, since we can allocate
  // both intervals to memory safely (the cost for each individual allocation
  // to memory is accounted for by the cost vectors for each live interval).
  PBQP::Matrix *m =
    new PBQP::Matrix(allowed1.size() + 1, allowed2.size() + 1, 0);

  // Assume this is a zero matrix until proven otherwise.  Zero matrices occur
  // between interfering live ranges with non-overlapping register sets (e.g.
  // non-overlapping reg classes, or disjoint sets of allowed regs within the
  // same class). The term "overlapping" is used advisedly: sets which do not
  // intersect, but contain registers which alias, will have non-zero matrices.
  // We optimize zero matrices away to improve solver speed.
  bool isZeroMatrix = true;


  // Row index. Starts at 1, since the 0th row is for the spill option, which
  // is always zero.
  unsigned ri = 1;

  // Iterate over allowed sets, insert infinities where required.
  for (RegContainerIterator a1Itr = allowed1.begin(), a1End = allowed1.end();
       a1Itr != a1End; ++a1Itr) {

    // Column index, starts at 1 as for row index.
    unsigned ci = 1;
    unsigned reg1 = *a1Itr;

    for (RegContainerIterator a2Itr = allowed2.begin(), a2End = allowed2.end();
         a2Itr != a2End; ++a2Itr) {

      unsigned reg2 = *a2Itr;

      // If the row/column regs are identical or alias insert an infinity.
      if (tri->regsOverlap(reg1, reg2)) {
        (*m)[ri][ci] = std::numeric_limits<PBQP::PBQPNum>::infinity();
        isZeroMatrix = false;
      }

      ++ci;
    }

    ++ri;
  }

  // If this turns out to be a zero matrix...
  if (isZeroMatrix) {
    // free it and return null.
    delete m;
    return 0;
  }

  // ...otherwise return the cost matrix.
  return m;
}

template <typename RegContainer>
PBQP::Matrix* PBQPRegAlloc::buildCoalescingMatrix(
      const RegContainer &allowed1, const RegContainer &allowed2,
      PBQP::PBQPNum cBenefit) const {

  typedef typename RegContainer::const_iterator RegContainerIterator;

  // Construct a PBQP Matrix representing the benefits of coalescing. As with
  // interference matrices the rows and columns represent allowed registers
  // for the LiveIntervals which are (potentially) to be coalesced. The amount
  // -cBenefit will be placed in any element representing the same register
  // for both intervals.
  PBQP::Matrix *m =
    new PBQP::Matrix(allowed1.size() + 1, allowed2.size() + 1, 0);

  // Reset costs to zero.
  m->reset(0);

  // Assume the matrix is zero till proven otherwise. Zero matrices will be
  // optimized away as in the interference case.
  bool isZeroMatrix = true;

  // Row index. Starts at 1, since the 0th row is for the spill option, which
  // is always zero.
  unsigned ri = 1;

  // Iterate over the allowed sets, insert coalescing benefits where
  // appropriate.
  for (RegContainerIterator a1Itr = allowed1.begin(), a1End = allowed1.end();
       a1Itr != a1End; ++a1Itr) {

    // Column index, starts at 1 as for row index.
    unsigned ci = 1;
    unsigned reg1 = *a1Itr;

    for (RegContainerIterator a2Itr = allowed2.begin(), a2End = allowed2.end();
         a2Itr != a2End; ++a2Itr) {

      // If the row and column represent the same register insert a beneficial
      // cost to preference this allocation - it would allow us to eliminate a
      // move instruction.
      if (reg1 == *a2Itr) {
        (*m)[ri][ci] = -cBenefit;
        isZeroMatrix = false;
      }

      ++ci;
    }

    ++ri;
  }

  // If this turns out to be a zero matrix...
  if (isZeroMatrix) {
    // ...free it and return null.
    delete m;
    return 0;
  }

  return m;
}

PBQPRegAlloc::CoalesceMap PBQPRegAlloc::findCoalesces() {

  typedef MachineFunction::const_iterator MFIterator;
  typedef MachineBasicBlock::const_iterator MBBIterator;
  typedef LiveInterval::const_vni_iterator VNIIterator;

  CoalesceMap coalescesFound;

  // To find coalesces we need to iterate over the function looking for
  // copy instructions.
  for (MFIterator bbItr = mf->begin(), bbEnd = mf->end();
       bbItr != bbEnd; ++bbItr) {

    const MachineBasicBlock *mbb = &*bbItr;

    for (MBBIterator iItr = mbb->begin(), iEnd = mbb->end();
         iItr != iEnd; ++iItr) {

      const MachineInstr *instr = &*iItr;
      unsigned srcReg, dstReg, srcSubReg, dstSubReg;

      // If this isn't a copy then continue to the next instruction.
      if (!tii->isMoveInstr(*instr, srcReg, dstReg, srcSubReg, dstSubReg))
        continue;

      // If the registers are already the same our job is nice and easy.
      if (dstReg == srcReg)
        continue;

      bool srcRegIsPhysical = TargetRegisterInfo::isPhysicalRegister(srcReg),
           dstRegIsPhysical = TargetRegisterInfo::isPhysicalRegister(dstReg);

      // If both registers are physical then we can't coalesce.
      if (srcRegIsPhysical && dstRegIsPhysical)
        continue;

      // If it's a copy that includes a virtual register but the source and
      // destination classes differ then we can't coalesce, so continue with
      // the next instruction.
      const TargetRegisterClass *srcRegClass = srcRegIsPhysical ?
          tri->getPhysicalRegisterRegClass(srcReg) : mri->getRegClass(srcReg);

      const TargetRegisterClass *dstRegClass = dstRegIsPhysical ?
          tri->getPhysicalRegisterRegClass(dstReg) : mri->getRegClass(dstReg);

      if (srcRegClass != dstRegClass)
        continue;

      // We also need any physical regs to be allocable, coalescing with
      // a non-allocable register is invalid.
      if (srcRegIsPhysical) {
        if (std::find(srcRegClass->allocation_order_begin(*mf),
                      srcRegClass->allocation_order_end(*mf), srcReg) ==
            srcRegClass->allocation_order_end(*mf))
          continue;
      }

      if (dstRegIsPhysical) {
        if (std::find(dstRegClass->allocation_order_begin(*mf),
                      dstRegClass->allocation_order_end(*mf), dstReg) ==
            dstRegClass->allocation_order_end(*mf))
          continue;
      }

      // If we've made it here we have a copy with compatible register classes.
      // We can probably coalesce, but we need to consider overlap.
      const LiveInterval *srcLI = &lis->getInterval(srcReg),
                         *dstLI = &lis->getInterval(dstReg);

      if (srcLI->overlaps(*dstLI)) {
        // Even in the case of an overlap we might still be able to coalesce,
        // but we need to make sure that no definition of either range occurs
        // while the other range is live.

        // Otherwise start by assuming we're ok.
        bool badDef = false;

        // Test all defs of the source range.
        for (VNIIterator
               vniItr = srcLI->vni_begin(), vniEnd = srcLI->vni_end();
               vniItr != vniEnd; ++vniItr) {

          // If we find a def that kills the coalescing opportunity then
          // record it and break from the loop.
          if (dstLI->liveAt((*vniItr)->def)) {
            badDef = true;
            break;
          }
        }

        // If we have a bad def give up, continue to the next instruction.
        if (badDef)
          continue;

        // Otherwise test definitions of the destination range.
        for (VNIIterator
               vniItr = dstLI->vni_begin(), vniEnd = dstLI->vni_end();
               vniItr != vniEnd; ++vniItr) {

          // We want to make sure we skip the copy instruction itself.
          if ((*vniItr)->getCopy() == instr)
            continue;

          if (srcLI->liveAt((*vniItr)->def)) {
            badDef = true;
            break;
          }
        }

        // As before a bad def we give up and continue to the next instr.
        if (badDef)
          continue;
      }

      // If we make it to here then either the ranges didn't overlap, or they
      // did, but none of their definitions would prevent us from coalescing.
      // We're good to go with the coalesce.

      float cBenefit = powf(10.0f, loopInfo->getLoopDepth(mbb)) / 5.0;

      coalescesFound[RegPair(srcReg, dstReg)] = cBenefit;
      coalescesFound[RegPair(dstReg, srcReg)] = cBenefit;
    }

  }

  return coalescesFound;
}

void PBQPRegAlloc::findVRegIntervalsToAlloc() {

  // Iterate over all live ranges.
  for (LiveIntervals::iterator itr = lis->begin(), end = lis->end();
       itr != end; ++itr) {

    // Ignore physical ones.
    if (TargetRegisterInfo::isPhysicalRegister(itr->first))
      continue;

    LiveInterval *li = itr->second;

    // If this live interval is non-empty we will use pbqp to allocate it.
    // Empty intervals we allocate in a simple post-processing stage in
    // finalizeAlloc.
    if (!li->empty()) {
      vregIntervalsToAlloc.insert(li);
    }
    else {
      emptyVRegIntervals.insert(li);
    }
  }
}

PBQP::SimpleGraph PBQPRegAlloc::constructPBQPProblem() {

  typedef std::vector<const LiveInterval*> LIVector;
  typedef std::vector<unsigned> RegVector;
  typedef std::vector<PBQP::SimpleGraph::NodeIterator> NodeVector;

  // This will store the physical intervals for easy reference.
  LIVector physIntervals;

  // Start by clearing the old node <-> live interval mappings & allowed sets
  li2Node.clear();
  node2LI.clear();
  allowedSets.clear();

  // Populate physIntervals, update preg use:
  for (LiveIntervals::iterator itr = lis->begin(), end = lis->end();
       itr != end; ++itr) {

    if (TargetRegisterInfo::isPhysicalRegister(itr->first)) {
      physIntervals.push_back(itr->second);
      mri->setPhysRegUsed(itr->second->reg);
    }
  }

  // Iterate over vreg intervals, construct live interval <-> node number
  //  mappings.
  for (LiveIntervalSet::const_iterator
       itr = vregIntervalsToAlloc.begin(), end = vregIntervalsToAlloc.end();
       itr != end; ++itr) {
    const LiveInterval *li = *itr;

    li2Node[li] = node2LI.size();
    node2LI.push_back(li);
  }

  // Get the set of potential coalesces.
  CoalesceMap coalesces;

  if (pbqpCoalescing) {
    coalesces = findCoalesces();
  }

  // Construct a PBQP solver for this problem
  PBQP::SimpleGraph problem;
  NodeVector problemNodes(vregIntervalsToAlloc.size());

  // Resize allowedSets container appropriately.
  allowedSets.resize(vregIntervalsToAlloc.size());

  // Iterate over virtual register intervals to compute allowed sets...
  for (unsigned node = 0; node < node2LI.size(); ++node) {

    // Grab pointers to the interval and its register class.
    const LiveInterval *li = node2LI[node];
    const TargetRegisterClass *liRC = mri->getRegClass(li->reg);

    // Start by assuming all allocable registers in the class are allowed...
    RegVector liAllowed(liRC->allocation_order_begin(*mf),
                        liRC->allocation_order_end(*mf));

    // Eliminate the physical registers which overlap with this range, along
    // with all their aliases.
    for (LIVector::iterator pItr = physIntervals.begin(),
       pEnd = physIntervals.end(); pItr != pEnd; ++pItr) {

      if (!li->overlaps(**pItr))
        continue;

      unsigned pReg = (*pItr)->reg;

      // If we get here then the live intervals overlap, but we're still ok
      // if they're coalescable.
      if (coalesces.find(RegPair(li->reg, pReg)) != coalesces.end())
        continue;

      // If we get here then we have a genuine exclusion.

      // Remove the overlapping reg...
      RegVector::iterator eraseItr =
        std::find(liAllowed.begin(), liAllowed.end(), pReg);

      if (eraseItr != liAllowed.end())
        liAllowed.erase(eraseItr);

      const unsigned *aliasItr = tri->getAliasSet(pReg);

      if (aliasItr != 0) {
        // ...and its aliases.
        for (; *aliasItr != 0; ++aliasItr) {
          RegVector::iterator eraseItr =
            std::find(liAllowed.begin(), liAllowed.end(), *aliasItr);

          if (eraseItr != liAllowed.end()) {
            liAllowed.erase(eraseItr);
          }
        }
      }
    }

    // Copy the allowed set into a member vector for use when constructing cost
    // vectors & matrices, and mapping PBQP solutions back to assignments.
    allowedSets[node] = AllowedSet(liAllowed.begin(), liAllowed.end());

    // Set the spill cost to the interval weight, or epsilon if the
    // interval weight is zero
    PBQP::PBQPNum spillCost = (li->weight != 0.0) ?
        li->weight : std::numeric_limits<PBQP::PBQPNum>::min();

    // Build a cost vector for this interval.
    problemNodes[node] =
      problem.addNode(
        buildCostVector(li->reg, allowedSets[node], coalesces, spillCost));

  }


  // Now add the cost matrices...
  for (unsigned node1 = 0; node1 < node2LI.size(); ++node1) {
    const LiveInterval *li = node2LI[node1];

    // Test for live range overlaps and insert interference matrices.
    for (unsigned node2 = node1 + 1; node2 < node2LI.size(); ++node2) {
      const LiveInterval *li2 = node2LI[node2];

      CoalesceMap::const_iterator cmItr =
        coalesces.find(RegPair(li->reg, li2->reg));

      PBQP::Matrix *m = 0;

      if (cmItr != coalesces.end()) {
        m = buildCoalescingMatrix(allowedSets[node1], allowedSets[node2],
                                  cmItr->second);
      }
      else if (li->overlaps(*li2)) {
        m = buildInterferenceMatrix(allowedSets[node1], allowedSets[node2]);
      }

      if (m != 0) {
        problem.addEdge(problemNodes[node1],
                        problemNodes[node2],
                        *m);

        delete m;
      }
    }
  }

  problem.assignNodeIDs();

  assert(problem.getNumNodes() == allowedSets.size());
  for (unsigned i = 0; i < allowedSets.size(); ++i) {
    assert(problem.getNodeItr(i) == problemNodes[i]);
  }
/*
  std::cerr << "Allocating for " << problem.getNumNodes() << " nodes, "
            << problem.getNumEdges() << " edges.\n";

  problem.printDot(std::cerr);
*/
  // We're done, PBQP problem constructed - return it.
  return problem;
}

void PBQPRegAlloc::addStackInterval(const LiveInterval *spilled,
                                    MachineRegisterInfo* mri) {
  int stackSlot = vrm->getStackSlot(spilled->reg);

  if (stackSlot == VirtRegMap::NO_STACK_SLOT)
    return;

  const TargetRegisterClass *RC = mri->getRegClass(spilled->reg);
  LiveInterval &stackInterval = lss->getOrCreateInterval(stackSlot, RC);

  VNInfo *vni;
  if (stackInterval.getNumValNums() != 0)
    vni = stackInterval.getValNumInfo(0);
  else
    vni = stackInterval.getNextValue(0, 0, false, lss->getVNInfoAllocator());

  LiveInterval &rhsInterval = lis->getInterval(spilled->reg);
  stackInterval.MergeRangesInAsValue(rhsInterval, vni);
}

bool PBQPRegAlloc::mapPBQPToRegAlloc(const PBQP::Solution &solution) {

  static unsigned round = 0;
  (void) round;

  // Set to true if we have any spills
  bool anotherRoundNeeded = false;

  // Clear the existing allocation.
  vrm->clearAllVirt();
  
  // Iterate over the nodes mapping the PBQP solution to a register assignment.
  for (unsigned node = 0; node < node2LI.size(); ++node) {
    unsigned virtReg = node2LI[node]->reg,
             allocSelection = solution.getSelection(node);


    // If the PBQP solution is non-zero it's a physical register...
    if (allocSelection != 0) {
      // Get the physical reg, subtracting 1 to account for the spill option.
      unsigned physReg = allowedSets[node][allocSelection - 1];

      DEBUG(errs() << "VREG " << virtReg << " -> "
                   << tri->getName(physReg) << "\n");

      assert(physReg != 0);

      // Add to the virt reg map and update the used phys regs.
      vrm->assignVirt2Phys(virtReg, physReg);
    }
    // ...Otherwise it's a spill.
    else {

      // Make sure we ignore this virtual reg on the next round
      // of allocation
      vregIntervalsToAlloc.erase(&lis->getInterval(virtReg));

      // Insert spill ranges for this live range
      const LiveInterval *spillInterval = node2LI[node];
      double oldSpillWeight = spillInterval->weight;
      SmallVector<LiveInterval*, 8> spillIs;
      std::vector<LiveInterval*> newSpills =
        lis->addIntervalsForSpills(*spillInterval, spillIs, loopInfo, *vrm);
      addStackInterval(spillInterval, mri);

      (void) oldSpillWeight;
      DEBUG(errs() << "VREG " << virtReg << " -> SPILLED (Cost: "
                   << oldSpillWeight << ", New vregs: ");

      // Copy any newly inserted live intervals into the list of regs to
      // allocate.
      for (std::vector<LiveInterval*>::const_iterator
           itr = newSpills.begin(), end = newSpills.end();
           itr != end; ++itr) {

        assert(!(*itr)->empty() && "Empty spill range.");

        DEBUG(errs() << (*itr)->reg << " ");

        vregIntervalsToAlloc.insert(*itr);
      }

      DEBUG(errs() << ")\n");

      // We need another round if spill intervals were added.
      anotherRoundNeeded |= !newSpills.empty();
    }
  }

  return !anotherRoundNeeded;
}

void PBQPRegAlloc::finalizeAlloc() const {
  typedef LiveIntervals::iterator LIIterator;
  typedef LiveInterval::Ranges::const_iterator LRIterator;

  // First allocate registers for the empty intervals.
  for (LiveIntervalSet::const_iterator
	 itr = emptyVRegIntervals.begin(), end = emptyVRegIntervals.end();
         itr != end; ++itr) {
    LiveInterval *li = *itr;

    unsigned physReg = vrm->getRegAllocPref(li->reg);

    if (physReg == 0) {
      const TargetRegisterClass *liRC = mri->getRegClass(li->reg);
      physReg = *liRC->allocation_order_begin(*mf);
    }

    vrm->assignVirt2Phys(li->reg, physReg);
  }

  // Finally iterate over the basic blocks to compute and set the live-in sets.
  SmallVector<MachineBasicBlock*, 8> liveInMBBs;
  MachineBasicBlock *entryMBB = &*mf->begin();

  for (LIIterator liItr = lis->begin(), liEnd = lis->end();
       liItr != liEnd; ++liItr) {

    const LiveInterval *li = liItr->second;
    unsigned reg = 0;

    // Get the physical register for this interval
    if (TargetRegisterInfo::isPhysicalRegister(li->reg)) {
      reg = li->reg;
    }
    else if (vrm->isAssignedReg(li->reg)) {
      reg = vrm->getPhys(li->reg);
    }
    else {
      // Ranges which are assigned a stack slot only are ignored.
      continue;
    }

    if (reg == 0) {
      // Filter out zero regs - they're for intervals that were spilled.
      continue;
    }

    // Iterate over the ranges of the current interval...
    for (LRIterator lrItr = li->begin(), lrEnd = li->end();
         lrItr != lrEnd; ++lrItr) {

      // Find the set of basic blocks which this range is live into...
      if (lis->findLiveInMBBs(lrItr->start, lrItr->end,  liveInMBBs)) {
        // And add the physreg for this interval to their live-in sets.
        for (unsigned i = 0; i < liveInMBBs.size(); ++i) {
          if (liveInMBBs[i] != entryMBB) {
            if (!liveInMBBs[i]->isLiveIn(reg)) {
              liveInMBBs[i]->addLiveIn(reg);
            }
          }
        }
        liveInMBBs.clear();
      }
    }
  }

}

bool PBQPRegAlloc::runOnMachineFunction(MachineFunction &MF) {

  mf = &MF;
  tm = &mf->getTarget();
  tri = tm->getRegisterInfo();
  tii = tm->getInstrInfo();
  mri = &mf->getRegInfo();

  lis = &getAnalysis<LiveIntervals>();
  lss = &getAnalysis<LiveStacks>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  vrm = &getAnalysis<VirtRegMap>();

  DEBUG(errs() << "PBQP2 Register Allocating for " << mf->getFunction()->getName() << "\n");

  // Allocator main loop:
  //
  // * Map current regalloc problem to a PBQP problem
  // * Solve the PBQP problem
  // * Map the solution back to a register allocation
  // * Spill if necessary
  //
  // This process is continued till no more spills are generated.

  // Find the vreg intervals in need of allocation.
  findVRegIntervalsToAlloc();

  // If there aren't any then we're done here.
  if (vregIntervalsToAlloc.empty() && emptyVRegIntervals.empty())
    return true;

  // If there are non-empty intervals allocate them using pbqp.
  if (!vregIntervalsToAlloc.empty()) {

    bool pbqpAllocComplete = false;
    unsigned round = 0;

    while (!pbqpAllocComplete) {
      DEBUG(errs() << "  PBQP Regalloc round " << round << ":\n");

      PBQP::SimpleGraph problem = constructPBQPProblem();
      PBQP::HeuristicSolver<PBQP::Heuristics::Briggs> solver;
      problem.assignNodeIDs();
      PBQP::Solution solution = solver.solve(problem);

      pbqpAllocComplete = mapPBQPToRegAlloc(solution);

      ++round;
    }
  }

  // Finalise allocation, allocate empty ranges.
  finalizeAlloc();

  vregIntervalsToAlloc.clear();
  emptyVRegIntervals.clear();
  li2Node.clear();
  node2LI.clear();
  allowedSets.clear();

  DEBUG(errs() << "Post alloc VirtRegMap:\n" << *vrm << "\n");

  // Run rewriter
  std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());

  rewriter->runOnMachineFunction(*mf, *vrm, lis);

  return true;
}

FunctionPass* llvm::createPBQPRegisterAllocator() {
  return new PBQPRegAlloc();
}


#undef DEBUG_TYPE
