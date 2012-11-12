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

#include "Spiller.h"
#include "VirtRegMap.h"
#include "RegisterCoalescer.h"
#include "llvm/Module.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/RegAllocPBQP.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PBQP/HeuristicSolver.h"
#include "llvm/CodeGen/PBQP/Graph.h"
#include "llvm/CodeGen/PBQP/Heuristics/Briggs.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

using namespace llvm;

static RegisterRegAlloc
registerPBQPRepAlloc("pbqp", "PBQP register allocator",
                       createDefaultPBQPRegisterAllocator);

static cl::opt<bool>
pbqpCoalescing("pbqp-coalescing",
                cl::desc("Attempt coalescing during PBQP register allocation."),
                cl::init(false), cl::Hidden);

#ifndef NDEBUG
static cl::opt<bool>
pbqpDumpGraphs("pbqp-dump-graphs",
               cl::desc("Dump graphs for each function/round in the compilation unit."),
               cl::init(false), cl::Hidden);
#endif

namespace {

///
/// PBQP based allocators solve the register allocation problem by mapping
/// register allocation problems to Partitioned Boolean Quadratic
/// Programming problems.
class RegAllocPBQP : public MachineFunctionPass {
public:

  static char ID;

  /// Construct a PBQP register allocator.
  RegAllocPBQP(std::auto_ptr<PBQPBuilder> b, char *cPassID=0)
      : MachineFunctionPass(ID), builder(b), customPassID(cPassID) {
    initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
    initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
    initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
    initializeLiveStacksPass(*PassRegistry::getPassRegistry());
    initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
    initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
  }

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "PBQP Register Allocator";
  }

  /// PBQP analysis usage.
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

  /// Perform register allocation
  virtual bool runOnMachineFunction(MachineFunction &MF);

private:

  typedef std::map<const LiveInterval*, unsigned> LI2NodeMap;
  typedef std::vector<const LiveInterval*> Node2LIMap;
  typedef std::vector<unsigned> AllowedSet;
  typedef std::vector<AllowedSet> AllowedSetMap;
  typedef std::pair<unsigned, unsigned> RegPair;
  typedef std::map<RegPair, PBQP::PBQPNum> CoalesceMap;
  typedef std::set<unsigned> RegSet;


  std::auto_ptr<PBQPBuilder> builder;

  char *customPassID;

  MachineFunction *mf;
  const TargetMachine *tm;
  const TargetRegisterInfo *tri;
  const TargetInstrInfo *tii;
  const MachineLoopInfo *loopInfo;
  MachineRegisterInfo *mri;

  std::auto_ptr<Spiller> spiller;
  LiveIntervals *lis;
  LiveStacks *lss;
  VirtRegMap *vrm;

  RegSet vregsToAlloc, emptyIntervalVRegs;

  /// \brief Finds the initial set of vreg intervals to allocate.
  void findVRegIntervalsToAlloc();

  /// \brief Given a solved PBQP problem maps this solution back to a register
  /// assignment.
  bool mapPBQPToRegAlloc(const PBQPRAProblem &problem,
                         const PBQP::Solution &solution);

  /// \brief Postprocessing before final spilling. Sets basic block "live in"
  /// variables.
  void finalizeAlloc() const;

};

char RegAllocPBQP::ID = 0;

} // End anonymous namespace.

unsigned PBQPRAProblem::getVRegForNode(PBQP::Graph::ConstNodeItr node) const {
  Node2VReg::const_iterator vregItr = node2VReg.find(node);
  assert(vregItr != node2VReg.end() && "No vreg for node.");
  return vregItr->second;
}

PBQP::Graph::NodeItr PBQPRAProblem::getNodeForVReg(unsigned vreg) const {
  VReg2Node::const_iterator nodeItr = vreg2Node.find(vreg);
  assert(nodeItr != vreg2Node.end() && "No node for vreg.");
  return nodeItr->second;

}

const PBQPRAProblem::AllowedSet&
  PBQPRAProblem::getAllowedSet(unsigned vreg) const {
  AllowedSetMap::const_iterator allowedSetItr = allowedSets.find(vreg);
  assert(allowedSetItr != allowedSets.end() && "No pregs for vreg.");
  const AllowedSet &allowedSet = allowedSetItr->second;
  return allowedSet;
}

unsigned PBQPRAProblem::getPRegForOption(unsigned vreg, unsigned option) const {
  assert(isPRegOption(vreg, option) && "Not a preg option.");

  const AllowedSet& allowedSet = getAllowedSet(vreg);
  assert(option <= allowedSet.size() && "Option outside allowed set.");
  return allowedSet[option - 1];
}

std::auto_ptr<PBQPRAProblem> PBQPBuilder::build(MachineFunction *mf,
                                                const LiveIntervals *lis,
                                                const MachineLoopInfo *loopInfo,
                                                const RegSet &vregs) {

  LiveIntervals *LIS = const_cast<LiveIntervals*>(lis);
  MachineRegisterInfo *mri = &mf->getRegInfo();
  const TargetRegisterInfo *tri = mf->getTarget().getRegisterInfo();

  std::auto_ptr<PBQPRAProblem> p(new PBQPRAProblem());
  PBQP::Graph &g = p->getGraph();
  RegSet pregs;

  // Collect the set of preg intervals, record that they're used in the MF.
  for (unsigned Reg = 1, e = tri->getNumRegs(); Reg != e; ++Reg) {
    if (mri->def_empty(Reg))
      continue;
    pregs.insert(Reg);
    mri->setPhysRegUsed(Reg);
  }

  // Iterate over vregs.
  for (RegSet::const_iterator vregItr = vregs.begin(), vregEnd = vregs.end();
       vregItr != vregEnd; ++vregItr) {
    unsigned vreg = *vregItr;
    const TargetRegisterClass *trc = mri->getRegClass(vreg);
    LiveInterval *vregLI = &LIS->getInterval(vreg);

    // Record any overlaps with regmask operands.
    BitVector regMaskOverlaps;
    LIS->checkRegMaskInterference(*vregLI, regMaskOverlaps);

    // Compute an initial allowed set for the current vreg.
    typedef std::vector<unsigned> VRAllowed;
    VRAllowed vrAllowed;
    ArrayRef<uint16_t> rawOrder = trc->getRawAllocationOrder(*mf);
    for (unsigned i = 0; i != rawOrder.size(); ++i) {
      unsigned preg = rawOrder[i];
      if (mri->isReserved(preg))
        continue;

      // vregLI crosses a regmask operand that clobbers preg.
      if (!regMaskOverlaps.empty() && !regMaskOverlaps.test(preg))
        continue;

      // vregLI overlaps fixed regunit interference.
      bool Interference = false;
      for (MCRegUnitIterator Units(preg, tri); Units.isValid(); ++Units) {
        if (vregLI->overlaps(LIS->getRegUnit(*Units))) {
          Interference = true;
          break;
        }
      }
      if (Interference)
        continue;

      // preg is usable for this virtual register.
      vrAllowed.push_back(preg);
    }

    // Construct the node.
    PBQP::Graph::NodeItr node =
      g.addNode(PBQP::Vector(vrAllowed.size() + 1, 0));

    // Record the mapping and allowed set in the problem.
    p->recordVReg(vreg, node, vrAllowed.begin(), vrAllowed.end());

    PBQP::PBQPNum spillCost = (vregLI->weight != 0.0) ?
        vregLI->weight : std::numeric_limits<PBQP::PBQPNum>::min();

    addSpillCosts(g.getNodeCosts(node), spillCost);
  }

  for (RegSet::const_iterator vr1Itr = vregs.begin(), vrEnd = vregs.end();
         vr1Itr != vrEnd; ++vr1Itr) {
    unsigned vr1 = *vr1Itr;
    const LiveInterval &l1 = lis->getInterval(vr1);
    const PBQPRAProblem::AllowedSet &vr1Allowed = p->getAllowedSet(vr1);

    for (RegSet::const_iterator vr2Itr = llvm::next(vr1Itr);
         vr2Itr != vrEnd; ++vr2Itr) {
      unsigned vr2 = *vr2Itr;
      const LiveInterval &l2 = lis->getInterval(vr2);
      const PBQPRAProblem::AllowedSet &vr2Allowed = p->getAllowedSet(vr2);

      assert(!l2.empty() && "Empty interval in vreg set?");
      if (l1.overlaps(l2)) {
        PBQP::Graph::EdgeItr edge =
          g.addEdge(p->getNodeForVReg(vr1), p->getNodeForVReg(vr2),
                    PBQP::Matrix(vr1Allowed.size()+1, vr2Allowed.size()+1, 0));

        addInterferenceCosts(g.getEdgeCosts(edge), vr1Allowed, vr2Allowed, tri);
      }
    }
  }

  return p;
}

void PBQPBuilder::addSpillCosts(PBQP::Vector &costVec,
                                PBQP::PBQPNum spillCost) {
  costVec[0] = spillCost;
}

void PBQPBuilder::addInterferenceCosts(
                                    PBQP::Matrix &costMat,
                                    const PBQPRAProblem::AllowedSet &vr1Allowed,
                                    const PBQPRAProblem::AllowedSet &vr2Allowed,
                                    const TargetRegisterInfo *tri) {
  assert(costMat.getRows() == vr1Allowed.size() + 1 && "Matrix height mismatch.");
  assert(costMat.getCols() == vr2Allowed.size() + 1 && "Matrix width mismatch.");

  for (unsigned i = 0; i != vr1Allowed.size(); ++i) {
    unsigned preg1 = vr1Allowed[i];

    for (unsigned j = 0; j != vr2Allowed.size(); ++j) {
      unsigned preg2 = vr2Allowed[j];

      if (tri->regsOverlap(preg1, preg2)) {
        costMat[i + 1][j + 1] = std::numeric_limits<PBQP::PBQPNum>::infinity();
      }
    }
  }
}

std::auto_ptr<PBQPRAProblem> PBQPBuilderWithCoalescing::build(
                                                MachineFunction *mf,
                                                const LiveIntervals *lis,
                                                const MachineLoopInfo *loopInfo,
                                                const RegSet &vregs) {

  std::auto_ptr<PBQPRAProblem> p = PBQPBuilder::build(mf, lis, loopInfo, vregs);
  PBQP::Graph &g = p->getGraph();

  const TargetMachine &tm = mf->getTarget();
  CoalescerPair cp(*tm.getRegisterInfo());

  // Scan the machine function and add a coalescing cost whenever CoalescerPair
  // gives the Ok.
  for (MachineFunction::const_iterator mbbItr = mf->begin(),
                                       mbbEnd = mf->end();
       mbbItr != mbbEnd; ++mbbItr) {
    const MachineBasicBlock *mbb = &*mbbItr;

    for (MachineBasicBlock::const_iterator miItr = mbb->begin(),
                                           miEnd = mbb->end();
         miItr != miEnd; ++miItr) {
      const MachineInstr *mi = &*miItr;

      if (!cp.setRegisters(mi)) {
        continue; // Not coalescable.
      }

      if (cp.getSrcReg() == cp.getDstReg()) {
        continue; // Already coalesced.
      }

      unsigned dst = cp.getDstReg(),
               src = cp.getSrcReg();

      const float copyFactor = 0.5; // Cost of copy relative to load. Current
      // value plucked randomly out of the air.

      PBQP::PBQPNum cBenefit =
        copyFactor * LiveIntervals::getSpillWeight(false, true,
                                                   loopInfo->getLoopDepth(mbb));

      if (cp.isPhys()) {
        if (!mf->getRegInfo().isAllocatable(dst)) {
          continue;
        }

        const PBQPRAProblem::AllowedSet &allowed = p->getAllowedSet(src);
        unsigned pregOpt = 0;
        while (pregOpt < allowed.size() && allowed[pregOpt] != dst) {
          ++pregOpt;
        }
        if (pregOpt < allowed.size()) {
          ++pregOpt; // +1 to account for spill option.
          PBQP::Graph::NodeItr node = p->getNodeForVReg(src);
          addPhysRegCoalesce(g.getNodeCosts(node), pregOpt, cBenefit);
        }
      } else {
        const PBQPRAProblem::AllowedSet *allowed1 = &p->getAllowedSet(dst);
        const PBQPRAProblem::AllowedSet *allowed2 = &p->getAllowedSet(src);
        PBQP::Graph::NodeItr node1 = p->getNodeForVReg(dst);
        PBQP::Graph::NodeItr node2 = p->getNodeForVReg(src);
        PBQP::Graph::EdgeItr edge = g.findEdge(node1, node2);
        if (edge == g.edgesEnd()) {
          edge = g.addEdge(node1, node2, PBQP::Matrix(allowed1->size() + 1,
                                                      allowed2->size() + 1,
                                                      0));
        } else {
          if (g.getEdgeNode1(edge) == node2) {
            std::swap(node1, node2);
            std::swap(allowed1, allowed2);
          }
        }

        addVirtRegCoalesce(g.getEdgeCosts(edge), *allowed1, *allowed2,
                           cBenefit);
      }
    }
  }

  return p;
}

void PBQPBuilderWithCoalescing::addPhysRegCoalesce(PBQP::Vector &costVec,
                                                   unsigned pregOption,
                                                   PBQP::PBQPNum benefit) {
  costVec[pregOption] += -benefit;
}

void PBQPBuilderWithCoalescing::addVirtRegCoalesce(
                                    PBQP::Matrix &costMat,
                                    const PBQPRAProblem::AllowedSet &vr1Allowed,
                                    const PBQPRAProblem::AllowedSet &vr2Allowed,
                                    PBQP::PBQPNum benefit) {

  assert(costMat.getRows() == vr1Allowed.size() + 1 && "Size mismatch.");
  assert(costMat.getCols() == vr2Allowed.size() + 1 && "Size mismatch.");

  for (unsigned i = 0; i != vr1Allowed.size(); ++i) {
    unsigned preg1 = vr1Allowed[i];
    for (unsigned j = 0; j != vr2Allowed.size(); ++j) {
      unsigned preg2 = vr2Allowed[j];

      if (preg1 == preg2) {
        costMat[i + 1][j + 1] += -benefit;
      }
    }
  }
}


void RegAllocPBQP::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesCFG();
  au.addRequired<AliasAnalysis>();
  au.addPreserved<AliasAnalysis>();
  au.addRequired<SlotIndexes>();
  au.addPreserved<SlotIndexes>();
  au.addRequired<LiveIntervals>();
  au.addPreserved<LiveIntervals>();
  //au.addRequiredID(SplitCriticalEdgesID);
  if (customPassID)
    au.addRequiredID(*customPassID);
  au.addRequired<CalculateSpillWeights>();
  au.addRequired<LiveStacks>();
  au.addPreserved<LiveStacks>();
  au.addRequired<MachineDominatorTree>();
  au.addPreserved<MachineDominatorTree>();
  au.addRequired<MachineLoopInfo>();
  au.addPreserved<MachineLoopInfo>();
  au.addRequired<VirtRegMap>();
  au.addPreserved<VirtRegMap>();
  MachineFunctionPass::getAnalysisUsage(au);
}

void RegAllocPBQP::findVRegIntervalsToAlloc() {

  // Iterate over all live ranges.
  for (unsigned i = 0, e = mri->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (mri->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *li = &lis->getInterval(Reg);

    // If this live interval is non-empty we will use pbqp to allocate it.
    // Empty intervals we allocate in a simple post-processing stage in
    // finalizeAlloc.
    if (!li->empty()) {
      vregsToAlloc.insert(li->reg);
    } else {
      emptyIntervalVRegs.insert(li->reg);
    }
  }
}

bool RegAllocPBQP::mapPBQPToRegAlloc(const PBQPRAProblem &problem,
                                     const PBQP::Solution &solution) {
  // Set to true if we have any spills
  bool anotherRoundNeeded = false;

  // Clear the existing allocation.
  vrm->clearAllVirt();

  const PBQP::Graph &g = problem.getGraph();
  // Iterate over the nodes mapping the PBQP solution to a register
  // assignment.
  for (PBQP::Graph::ConstNodeItr node = g.nodesBegin(),
                                 nodeEnd = g.nodesEnd();
       node != nodeEnd; ++node) {
    unsigned vreg = problem.getVRegForNode(node);
    unsigned alloc = solution.getSelection(node);

    if (problem.isPRegOption(vreg, alloc)) {
      unsigned preg = problem.getPRegForOption(vreg, alloc);
      DEBUG(dbgs() << "VREG " << PrintReg(vreg, tri) << " -> "
            << tri->getName(preg) << "\n");
      assert(preg != 0 && "Invalid preg selected.");
      vrm->assignVirt2Phys(vreg, preg);
    } else if (problem.isSpillOption(vreg, alloc)) {
      vregsToAlloc.erase(vreg);
      SmallVector<LiveInterval*, 8> newSpills;
      LiveRangeEdit LRE(&lis->getInterval(vreg), newSpills, *mf, *lis, vrm);
      spiller->spill(LRE);

      DEBUG(dbgs() << "VREG " << PrintReg(vreg, tri) << " -> SPILLED (Cost: "
                   << LRE.getParent().weight << ", New vregs: ");

      // Copy any newly inserted live intervals into the list of regs to
      // allocate.
      for (LiveRangeEdit::iterator itr = LRE.begin(), end = LRE.end();
           itr != end; ++itr) {
        assert(!(*itr)->empty() && "Empty spill range.");
        DEBUG(dbgs() << PrintReg((*itr)->reg, tri) << " ");
        vregsToAlloc.insert((*itr)->reg);
      }

      DEBUG(dbgs() << ")\n");

      // We need another round if spill intervals were added.
      anotherRoundNeeded |= !LRE.empty();
    } else {
      llvm_unreachable("Unknown allocation option.");
    }
  }

  return !anotherRoundNeeded;
}


void RegAllocPBQP::finalizeAlloc() const {
  // First allocate registers for the empty intervals.
  for (RegSet::const_iterator
         itr = emptyIntervalVRegs.begin(), end = emptyIntervalVRegs.end();
         itr != end; ++itr) {
    LiveInterval *li = &lis->getInterval(*itr);

    unsigned physReg = vrm->getRegAllocPref(li->reg);

    if (physReg == 0) {
      const TargetRegisterClass *liRC = mri->getRegClass(li->reg);
      physReg = liRC->getRawAllocationOrder(*mf).front();
    }

    vrm->assignVirt2Phys(li->reg, physReg);
  }
}

bool RegAllocPBQP::runOnMachineFunction(MachineFunction &MF) {

  mf = &MF;
  tm = &mf->getTarget();
  tri = tm->getRegisterInfo();
  tii = tm->getInstrInfo();
  mri = &mf->getRegInfo();

  lis = &getAnalysis<LiveIntervals>();
  lss = &getAnalysis<LiveStacks>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  vrm = &getAnalysis<VirtRegMap>();
  spiller.reset(createInlineSpiller(*this, MF, *vrm));

  mri->freezeReservedRegs(MF);

  DEBUG(dbgs() << "PBQP Register Allocating for " << mf->getName() << "\n");

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

#ifndef NDEBUG
  const Function* func = mf->getFunction();
  std::string fqn =
    func->getParent()->getModuleIdentifier() + "." +
    func->getName().str();
#endif

  // If there are non-empty intervals allocate them using pbqp.
  if (!vregsToAlloc.empty()) {

    bool pbqpAllocComplete = false;
    unsigned round = 0;

    while (!pbqpAllocComplete) {
      DEBUG(dbgs() << "  PBQP Regalloc round " << round << ":\n");

      std::auto_ptr<PBQPRAProblem> problem =
        builder->build(mf, lis, loopInfo, vregsToAlloc);

#ifndef NDEBUG
      if (pbqpDumpGraphs) {
        std::ostringstream rs;
        rs << round;
        std::string graphFileName(fqn + "." + rs.str() + ".pbqpgraph");
        std::string tmp;
        raw_fd_ostream os(graphFileName.c_str(), tmp);
        DEBUG(dbgs() << "Dumping graph for round " << round << " to \""
              << graphFileName << "\"\n");
        problem->getGraph().dump(os);
      }
#endif

      PBQP::Solution solution =
        PBQP::HeuristicSolver<PBQP::Heuristics::Briggs>::solve(
          problem->getGraph());

      pbqpAllocComplete = mapPBQPToRegAlloc(*problem, solution);

      ++round;
    }
  }

  // Finalise allocation, allocate empty ranges.
  finalizeAlloc();
  vregsToAlloc.clear();
  emptyIntervalVRegs.clear();

  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *vrm << "\n");

  return true;
}

FunctionPass* llvm::createPBQPRegisterAllocator(
                                           std::auto_ptr<PBQPBuilder> builder,
                                           char *customPassID) {
  return new RegAllocPBQP(builder, customPassID);
}

FunctionPass* llvm::createDefaultPBQPRegisterAllocator() {
  if (pbqpCoalescing) {
    return createPBQPRegisterAllocator(
             std::auto_ptr<PBQPBuilder>(new PBQPBuilderWithCoalescing()));
  } // else
  return createPBQPRegisterAllocator(
           std::auto_ptr<PBQPBuilder>(new PBQPBuilder()));
}

#undef DEBUG_TYPE
