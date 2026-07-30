
//===----- HexagonGlobalScheduler.cpp - Global Scheduler ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic infrastructure for the global scheduling + Hexagon pull-up pass.
// Currently run at the very end of code generation for Hexagon, cleans
// up lost scheduling opportunities. Currently breaks liveness, so no passes
// that rely on liveness info should run afterwards. Will be fixed in future
// versions.
//
//===----------------------------------------------------------------------===//
#include "Hexagon.h"
#include "HexagonGlobalRegion.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonVLIWPacketizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <list>
#include <map>

#define DEBUG_TYPE "global_sched"

using namespace llvm;

STATISTIC(HexagonNumPullUps, "Number of instructions pull-ups");
STATISTIC(HexagonNumDualJumps, "Number of dual jumps formed");

static cl::opt<bool> DisablePullUp("disable-pull-up", cl::Hidden,
                                   cl::desc("Disable Hexagon pull-up pass"));

static cl::opt<bool> EnableSpeculativePullUp(
    "enable-speculative-pull-up", cl::Hidden,
    cl::desc("Enable speculation during Hexagon pull-up pass"));

static cl::opt<bool> EnableLocalPullUp(
    "enable-local-pull-up", cl::Hidden, cl::init(true),
    cl::desc("Enable same BB pull during Hexagon pull-up pass"));

static cl::opt<bool> AllowSpeculateLoads(
    "speculate-loads-on-pull-up", cl::Hidden, cl::init(true),
    cl::desc("Allow speculative loads during Hexagon pull-up pass"));

static cl::opt<bool> AllowCmpBranchLoads(
    "cmp-branch-loads-pull-up", cl::Hidden, cl::init(true),
    cl::desc("Allow compare-branch loads during Hexagon pull-up pass"));

static cl::opt<bool> AllowUnlikelyPath("unlikely-path-pull-up", cl::Hidden,
                                       cl::init(true),
                                       cl::desc("Allow unlikely path pull up"));

static cl::opt<bool>
    PerformDualJumps("dual-jump-in-pull-up", cl::Hidden, cl::init(true),
                     cl::desc("Perform dual jump formation during pull up"));

static cl::opt<bool> AllowDependentPullUp(
    "enable-dependent-pull-up", cl::Hidden, cl::init(true),
    cl::desc("Perform dual jump formation during pull up"));

static cl::opt<bool>
    AllowBBPeelPullUp("enable-bb-peel-pull-up", cl::Hidden, cl::init(true),
                      cl::desc("Peel a reg copy out of a BBloop"));

static cl::opt<bool> PreventCompoundSeparation(
    "prevent-compound-separation", cl::Hidden,
    cl::desc("Do not destroy existing compounds during pull up"));

static cl::opt<bool> PreventDuplexSeparation(
    "prevent-duplex-separation", cl::Hidden, cl::init(true),
    cl::desc("Do not destroy existing duplexes during pull up"));

static cl::opt<unsigned> MainCandidateQueueSize("pull-up-main-queue-size",
                                                cl::Hidden, cl::init(8));

static cl::opt<unsigned> SecondaryCandidateQueueSize("pull-up-sec-queue-size",
                                                     cl::Hidden, cl::init(2));

static cl::opt<bool> PostPullUpOpt(
    "post-pull-up-opt", cl::Hidden, cl::Optional, cl::init(true),
    cl::desc("Enable opt. exposed by pull-up e.g., remove redundant jumps"));

static cl::opt<bool> SpeculateNonPredInsn(
    "speculate-non-pred-insn", cl::Hidden, cl::Optional, cl::init(true),
    cl::desc("Speculate non-predicable instructions in parent BB"));

static cl::opt<bool>
    DisableCheckBundles("disable-hexagon-check-bundles", cl::Hidden,
                        cl::init(true),
                        cl::desc("Disable Hexagon check bundles pass"));

static cl::opt<bool>
    WarnOnBundleSize("warn-on-bundle-size", cl::Hidden,
                     cl::desc("Hexagon check bundles and warn on size"));

static cl::opt<bool>
    ForceNoopHazards("force-noop-hazards", cl::Hidden, cl::init(false),
                     cl::desc("Force noop hazards in scheduler"));
static cl::opt<bool> OneFloatPerPacket(
    "single-float-packet", cl::Hidden,
    cl::desc("Allow only one single floating point instruction in a packet"));
static cl::opt<bool> OneComplexPerPacket(
    "single-complex-packet", cl::Hidden,
    cl::desc("Allow only one complex instruction in a packet"));

namespace llvm {
FunctionPass *createHexagonGlobalScheduler();
void initializeHexagonGlobalSchedulerPass(PassRegistry &);
} // namespace llvm

namespace {
class HexagonGlobalSchedulerImpl;

class HexagonGlobalScheduler : public MachineFunctionPass {
public:
  static char ID;
  HexagonGlobalScheduler() : MachineFunctionPass(ID) {
    initializeHexagonGlobalSchedulerPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredID(MachineDominatorsID);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Hexagon Global Scheduler"; }

  bool runOnMachineFunction(MachineFunction &Fn) override;
};
char HexagonGlobalScheduler::ID = 0;

// Describes a single pull-up candidate.
class PullUpCandidate {
  MachineBasicBlock::instr_iterator CandidateLocation;
  MachineBasicBlock::iterator HomeBundle;
  bool DependentOp;
  signed BenefitCost;
  std::vector<MachineInstr *> Backtrack;

public:
  PullUpCandidate(MachineBasicBlock::instr_iterator MII) {
    CandidateLocation = MII;
    BenefitCost = 0;
  }

  PullUpCandidate(MachineBasicBlock::instr_iterator MII,
                  MachineBasicBlock::iterator HomeBundle,
                  std::vector<MachineInstr *> &backtrack, bool DependentOp,
                  signed Cost)
      : CandidateLocation(MII), HomeBundle(HomeBundle),
        DependentOp(DependentOp), BenefitCost(Cost) {
    // Copy of the backtrack.
    Backtrack = backtrack;
  }

  void populate(MachineBasicBlock::instr_iterator &MII,
                MachineBasicBlock::iterator &WorkPoint,
                std::vector<MachineInstr *> &backtrack, bool &dependentOp) {
    MII = CandidateLocation;
    WorkPoint = HomeBundle;
    backtrack = Backtrack;
    dependentOp = DependentOp;
  }

  signed getCost() { return BenefitCost; }

  MachineInstr *getCandidate() { return &*CandidateLocation; }

  void dump() {
    dbgs() << "Cost(" << BenefitCost;
    dbgs() << ") Dependent(" << DependentOp;
    dbgs() << ") backtrack size(" << Backtrack.size() << ")\t";
    CandidateLocation->dump();
  }
};

/// PullUpCandidateSorter - A Sort utility for pull-up candidates.
struct PullUpCandidateSorter {
  PullUpCandidateSorter() {}
  bool operator()(PullUpCandidate *LHS, PullUpCandidate *RHS) {
    return LHS->getCost() > RHS->getCost();
  }
};

// Describes a single pull-up opportunity: location to which
// pull-up is possible with additional information about it.
// Also contains a list of pull-up candidates for this location.
class PullUpState {
  friend class HexagonGlobalSchedulerImpl;
  // Available opportunity for pull-up.
  // FAIAP a bundle with an empty slot.
  MachineBasicBlock::iterator HomeLocation;
  // Home bundle copy. This is here for speed of iteration.
  SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE> HomeBundle;
  // Multiple candidates for the Home location.
  SmallVector<PullUpCandidate *, 8> PullUpCandidates;

  const HexagonInstrInfo *QII;

public:
  PullUpState(const HexagonInstrInfo *QII) : HomeLocation(NULL), QII(QII) {}

  ~PullUpState() { reset(); }

  void addPullUpCandidate(MachineBasicBlock::instr_iterator MII,
                          MachineBasicBlock::iterator HomeBundle,
                          std::vector<MachineInstr *> &backtrack,
                          bool DependentOp, signed Cost) {
    LLVM_DEBUG(dbgs() << "\t[addPullUpCandidate]: "; (*MII).dump());
    PullUpCandidate *PUI =
        new PullUpCandidate(MII, HomeBundle, backtrack, DependentOp, Cost);
    PullUpCandidates.push_back(PUI);
  }

  void dump() {
    unsigned element = 0;
    for (unsigned i = 0; i < HomeBundle.size(); i++) {
      dbgs() << "[" << element++;
      dbgs() << "] Home Duplex("
             << QII->getDuplexCandidateGroup(*HomeBundle[i]);
      dbgs() << ") Compound (" << QII->getCompoundCandidateGroup(*HomeBundle[i])
             << ") ";
      HomeBundle[i]->dump();
    }
    dbgs() << "\n";
    element = 0;
    for (SmallVector<PullUpCandidate *, 4>::iterator
             I = PullUpCandidates.begin(),
             E = PullUpCandidates.end();
         I != E; ++I) {
      dbgs() << "[" << element++ << "] Cand: Compound(";
      dbgs() << QII->getCompoundCandidateGroup(*(*I)->getCandidate()) << ") ";
      (*I)->dump();
    }
  }

  void reset() {
    HomeLocation = NULL;
    for (SmallVector<PullUpCandidate *, 4>::iterator
             I = PullUpCandidates.begin(),
             E = PullUpCandidates.end();
         I != E; ++I)
      delete *I;
    PullUpCandidates.clear();
    HomeBundle.clear();
  }

  void addHomeLocation(MachineBasicBlock::iterator WorkPoint) {
    reset();
    HomeLocation = WorkPoint;
  }

  unsigned haveCandidates() { return PullUpCandidates.size(); }
};

class HexagonGlobalSchedulerImpl : public HexagonPacketizerList {
  // List of PullUp regions for this function.
  std::vector<BasicBlockRegion *> PullUpRegions;
  // Map of approximate distance for each BB from the
  // function base.
  DenseMap<MachineBasicBlock *, unsigned> BlockToInstOffset;
  // Keep track of multiple pull-up candidates.
  PullUpState CurrentState;
  // Empty basic blocks as a result of pull-up.
  std::vector<MachineBasicBlock *> EmptyBBs;
  // Save all the Speculated MachineInstr that were moved
  // FROM MachineBasicBlock because we don't want to have
  // more than one speculated instructions pulled into one packet.
  // TODO: This can be removed once we have a use-def dependency chain
  // for all the instructions in a function.
  std::map<MachineInstr *, MachineBasicBlock *> SpeculatedIns;
  // All the regs and their aliases used by an instruction.
  std::map<MachineInstr *, std::vector<unsigned>> MIUseSet;
  // All the regs and their aliases defined by an instruction.
  std::map<MachineInstr *, std::vector<unsigned>> MIDefSet;

  AliasAnalysis *AA;
  const MachineBranchProbabilityInfo *MBPI;
  const MachineBlockFrequencyInfo *MBFI;
  const MachineRegisterInfo *MRI;
  const MachineFrameInfo &MFI;
  const HexagonRegisterInfo *QRI;
  const HexagonInstrInfo *QII;
  MachineLoopInfo &MLI;
  MachineDominatorTree &MDT;
  MachineInstrBuilder Ext;
  MachineInstrBuilder Nop;
  const unsigned PacketSize;
  TargetSchedModel TSchedModel;

public:
  // Ctor.
  HexagonGlobalSchedulerImpl(MachineFunction &MF, MachineLoopInfo &MLI,
                             MachineDominatorTree &MDT, AliasAnalysis *AA,
                             const MachineBranchProbabilityInfo *MBPI,
                             const MachineBlockFrequencyInfo *MBFI,
                             const MachineRegisterInfo *MRI,
                             const MachineFrameInfo &MFI,
                             const HexagonRegisterInfo *QRI);
  HexagonGlobalSchedulerImpl(const HexagonGlobalSchedulerImpl &) = delete;
  HexagonGlobalSchedulerImpl &
  operator=(const HexagonGlobalSchedulerImpl &) = delete;

  ~HexagonGlobalSchedulerImpl() {
    // Free regions.
    for (std::vector<BasicBlockRegion *>::iterator I = PullUpRegions.begin(),
                                                   E = PullUpRegions.end();
         I != E; ++I)
      delete *I;
    MF.deleteMachineInstr(Ext);
    MF.deleteMachineInstr(Nop);
  }

  // initPacketizerState - initialize some internal flags.
  void initPacketizerState() override;

  // ignorePseudoInstruction - Ignore bundling of pseudo instructions.
  bool ignoreInstruction(MachineInstr *MI);

  // isSoloInstruction - return true if instruction MI can not be packetized
  // with any other instruction, which means that MI itself is a packet.
  bool isSoloInstruction(const MachineInstr &MI) override;

  // Add MI to packetizer state. Returns false if it cannot fit in the packet.
  bool incrementalAddToPacket(MachineInstr &MI);

  // formPullUpRegions - Top level call to form regions.
  bool formPullUpRegions(MachineFunction &Fn);

  // performPullUp - Top level call for pull-up.
  bool performPullUp();

  // performPullUpCFG - Top level call for pull-up CFG.
  bool performPullUpCFG(MachineFunction &Fn);

  // performExposedOptimizations -
  // Look for optimization opportunities after pullup.
  bool performExposedOptimizations(MachineFunction &Fn);

  // optimizeBranching -
  // 1. A conditional-jump transfers control to a BB with
  // jump as the only instruction.
  // if(p0) jump t1
  // // ...
  // t1: jump t2
  // 2. When a BB with a single conditional jump, jumps to succ-of-succ and
  // falls-through BB with only jump instruction.
  // { if(p0) jump t1 }
  // { jump t2 }
  // t1: { ... }
  MachineBasicBlock *optimizeBranches(MachineBasicBlock *MBB,
                                      MachineBasicBlock *TBB,
                                      MachineInstr *FirstTerm,
                                      MachineBasicBlock *FBB);

  // removeRedundantBranches -
  // 1. Remove jump to the layout successor.
  // 2. Remove multiple (dual) jump to the same target.
  bool removeRedundantBranches(MachineBasicBlock *MBB, MachineBasicBlock *TBB,
                               MachineInstr *FirstTerm, MachineBasicBlock *FBB,
                               MachineInstr *SecondTerm);

  // optimizeDualJumps - optimize dual jumps in a packet
  // For now: Replace dual jump by single jump in case of a fall through.
  bool optimizeDualJumps(MachineBasicBlock *MBB, MachineBasicBlock *TBB,
                         MachineInstr *FirstTerm, MachineBasicBlock *FBB,
                         MachineInstr *SecondTerm);

  void GenUseDefChain(MachineFunction &Fn);

  // Return region pointer or null if none found.
  BasicBlockRegion *getRegionForMBB(std::vector<BasicBlockRegion *> &Regions,
                                    MachineBasicBlock *MBB);

  // Saves all the used-regs and their aliases in Uses.
  // Saves all the defined-regs and their aliases in Defs.
  void MIUseDefSet(MachineInstr *MI, std::vector<unsigned> &Defs,
                   std::vector<unsigned> &Uses);

  // This is a very useful debug utility.
  unsigned countCompounds(MachineFunction &Fn);

  // Check bundle counts
  void checkBundleCounts(MachineFunction &Fn);

private:
  // Get next BB to be included into the region.
  MachineBasicBlock *getNextPURBB(MachineBasicBlock *MBB, bool SecondBest);

  void setUsedRegs(BitVector &Set, unsigned Reg);
  bool AliasingRegs(unsigned RegA, unsigned RegB);

  // Test is true if the two MIs cannot be safely reordered.
  bool ReorderDependencyTest(MachineInstr *MIa, MachineInstr *MIb);

  bool canAddMIToThisPacket(
      MachineInstr *MI,
      SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE> &Bundle);

  bool CanPromoteToDotNew(MachineInstr *MI, unsigned Reg);

  bool pullUpPeelBBLoop(MachineBasicBlock *PredBB, MachineBasicBlock *LoopBB);

  MachineInstr *findBundleAndBranch(MachineBasicBlock *BB,
                                    MachineBasicBlock::iterator &Bundle);

  // Does this bundle have any slots left?
  bool ResourcesAvailableInBundle(BasicBlockRegion *CurrentRegion,
                                  MachineBasicBlock::iterator &TargetPacket);

  // Perform the actual move.
  MachineInstr *MoveAndUpdateLiveness(
      BasicBlockRegion *CurrentRegion, MachineBasicBlock *HomeBB,
      MachineInstr *InstrToMove, bool NeedToNewify, unsigned DepReg,
      bool MovingDependentOp, MachineBasicBlock *OriginBB,
      MachineInstr *OriginalInstruction, SmallVector<MachineOperand, 4> &Cond,
      MachineBasicBlock::iterator &SourceLocation,
      MachineBasicBlock::iterator &TargetPacket,
      MachineBasicBlock::iterator &NextMI,
      std::vector<MachineInstr *> &backtrack);

  // Updates incremental kill patterns along the backtrack.
  void updateKillAlongThePath(MachineBasicBlock *HomeBB,
                              MachineBasicBlock *OriginBB,
                              MachineBasicBlock::instr_iterator &Head,
                              MachineBasicBlock::instr_iterator &Tail,
                              MachineBasicBlock::iterator &SourcePacket,
                              MachineBasicBlock::iterator &TargetPacket,
                              std::vector<MachineInstr *> &backtrack);

  // Gather list of pull-up candidates.
  bool findPullUpCandidates(MachineBasicBlock::iterator &WorkPoint,
                            MachineBasicBlock::iterator &FromHere,
                            std::vector<MachineInstr *> &backtrack,
                            unsigned MaxCandidates);

  // See if the instruction could be pulled up.
  bool tryMultipleInstructions(
      MachineBasicBlock::iterator &RetVal, /* output parameter */
      std::vector<BasicBlockRegion *>::iterator &CurrentRegion,
      MachineBasicBlock::iterator &NextMI,
      MachineBasicBlock::iterator &ToThisBBEnd,
      MachineBasicBlock::iterator &FromThisBBEnd, bool PathInRegion = true);

  // Try to move MI into existing bundle.
  bool MoveMItoBundle(BasicBlockRegion *CurrentRegion,
                      MachineBasicBlock::instr_iterator &InstrToMove,
                      MachineBasicBlock::iterator &NextMI,
                      MachineBasicBlock::iterator &TargetPacket,
                      MachineBasicBlock::iterator &SourceLocation,
                      std::vector<MachineInstr *> &backtrack,
                      bool MovingDependentOp, bool PathInRegion);

  // Insert temporary MI copy into MBB.
  MachineBasicBlock::instr_iterator
  insertTempCopy(MachineBasicBlock *MBB,
                 MachineBasicBlock::iterator &TargetPacket, MachineInstr *MI,
                 bool DeleteOldCopy);

  MachineBasicBlock::instr_iterator
  findInsertPositionInBundle(MachineBasicBlock::iterator &Bundle,
                             MachineInstr *MI, bool &LastInBundle);

  bool NeedToNewify(MachineBasicBlock::instr_iterator NewMI, unsigned *DepReg,
                    MachineInstr *TargetPacket);

  bool CanNewifiedBeUsedInBundle(MachineBasicBlock::instr_iterator NewMI,
                                 unsigned DepReg, MachineInstr *TargetPacket);

  void addInstructionToExistingBundle(MachineBasicBlock *HomeBB,
                                      MachineBasicBlock::instr_iterator &Head,
                                      MachineBasicBlock::instr_iterator &Tail,
                                      MachineBasicBlock::instr_iterator &NewMI,
                                      MachineBasicBlock::iterator &TargetPacket,
                                      MachineBasicBlock::iterator &NextMI,
                                      std::vector<MachineInstr *> &backtrack);

  void removeInstructionFromExistingBundle(
      MachineBasicBlock *HomeBB, MachineBasicBlock::instr_iterator &Head,
      MachineBasicBlock::instr_iterator &Tail,
      MachineBasicBlock::iterator &SourceLocation,
      MachineBasicBlock::iterator &NextMI, bool MovingDependentOp,
      std::vector<MachineInstr *> &backtrack);

  // Check for conditional register operaton.
  bool MIsCondAssign(MachineInstr *BMI, MachineInstr *MI,
                     SmallVector<unsigned, 4> &Defs);

  // Test all the conditions required for instruction to be
  // speculative. These are just required conditions, cost
  // or benefit should be computed elsewhere.
  bool canMIBeSpeculated(MachineInstr *MI, MachineBasicBlock *ToBB,
                         MachineBasicBlock *FromBB,
                         std::vector<MachineInstr *> &backtrack);

  // See if this branch target belongs to the current region.
  bool isBranchWithinRegion(BasicBlockRegion *CurrentRegion, MachineInstr *MI);

  // A collection of low level utilities.
  bool MIsAreDependent(MachineInstr *MIa, MachineInstr *MIb);
  bool MIsHaveTrueDependency(MachineInstr *MIa, MachineInstr *MIb);
  bool canReorderMIs(MachineInstr *MIa, MachineInstr *MIb);
  bool canCauseStall(MachineInstr *MI, MachineInstr *MJ);
  bool canThisMIBeMoved(MachineInstr *MI,
                        MachineBasicBlock::iterator &WorkPoint,
                        bool &MovingDependentOp, int &Cost);
  bool MIisDualJumpCandidate(MachineInstr *MI,
                             MachineBasicBlock::iterator &WorkPoint);
  bool DemoteToDotOld(MachineInstr *MI);
  bool isNewifiable(MachineBasicBlock::instr_iterator MII, unsigned DepReg,
                    MachineInstr *TargetPacket);
  bool IsNewifyStore(MachineInstr *MI);
  bool isJumpOutOfRange(MachineInstr *MI);
  bool IsDualJumpFirstCandidate(MachineInstr *MI);
  bool IsDualJumpFirstCandidate(MachineBasicBlock *MBB);
  bool IsDualJumpFirstCandidate(MachineBasicBlock::iterator &TargetPacket);
  bool IsNotDualJumpFirstCandidate(MachineInstr *MI);
  bool isJumpOutOfRange(MachineInstr *UnCond, MachineInstr *Cond);
  bool IsDualJumpSecondCandidate(MachineInstr *MI);
  bool tryAllocateResourcesForConstExt(MachineInstr *MI, bool UpdateState);
  bool isCompoundPair(MachineInstr *MIa, MachineInstr *MIb);
  bool doesMIDefinesPredicate(MachineInstr *MI, SmallVector<unsigned, 4> &Defs);
  bool AnalyzeBBBranches(MachineBasicBlock *MBB, MachineBasicBlock *&TBB,
                         MachineInstr *&FirstTerm, MachineBasicBlock *&FBB,
                         MachineInstr *&SecondTerm);
  inline bool multipleBranchesFromToBB(MachineBasicBlock *BB) const;
};
} // namespace

INITIALIZE_PASS_BEGIN(HexagonGlobalScheduler, "global-sched",
                      "Hexagon Global Scheduler", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(HexagonGlobalScheduler, "global-sched",
                    "Hexagon Global Scheduler", false, false)

/// HexagonGlobalSchedulerImpl Ctor.
HexagonGlobalSchedulerImpl::HexagonGlobalSchedulerImpl(
    MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree &MDT,
    AliasAnalysis *AA, const MachineBranchProbabilityInfo *MBPI,
    const MachineBlockFrequencyInfo *MBFI, const MachineRegisterInfo *MRI,
    const MachineFrameInfo &MFI, const HexagonRegisterInfo *QRI)
    : HexagonPacketizerList(MF, MLI, AA, nullptr, false), PullUpRegions(0),
      CurrentState((const HexagonInstrInfo *)TII), AA(AA), MBPI(MBPI),
      MBFI(MBFI), MRI(MRI), MFI(MFI), QRI(QRI), MLI(MLI), MDT(MDT),
      PacketSize(MF.getSubtarget().getSchedModel().IssueWidth) {
  QII = (const HexagonInstrInfo *)TII;
  Ext = BuildMI(MF, DebugLoc(), QII->get(Hexagon::A4_ext));
  Nop = BuildMI(MF, DebugLoc(), QII->get(Hexagon::A2_nop));
  TSchedModel.init(&MF.getSubtarget());
}

// Return bundle size without debug instructions.
static unsigned nonDbgBundleSize(MachineBasicBlock::iterator &TargetPacket) {
  MachineBasicBlock::instr_iterator MII = TargetPacket.getInstrIterator();
  MachineBasicBlock::instr_iterator End = MII->getParent()->instr_end();
  unsigned count = 0;
  for (++MII; MII != End && MII->isInsideBundle(); ++MII) {
    if (MII->isDebugInstr())
      continue;
    count++;
  }
  return count;
}

/// The pass main entry point.
bool HexagonGlobalScheduler::runOnMachineFunction(MachineFunction &Fn) {
  auto &HST = Fn.getSubtarget<HexagonSubtarget>();
  if (DisablePullUp || !HST.usePackets() || skipFunction(Fn.getFunction()))
    return false;

  const MachineRegisterInfo *MRI = &Fn.getRegInfo();
  const MachineFrameInfo &MFI = Fn.getFrameInfo();
  const HexagonRegisterInfo *QRI = HST.getRegisterInfo();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  const MachineBranchProbabilityInfo *MBPI =
      &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  const MachineBlockFrequencyInfo *MBFI =
      &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  // Preserve comounds if Opt Size.
  const Function &F = Fn.getFunction();
  if (F.hasOptSize() && PreventCompoundSeparation.getNumOccurrences() == 0)
    PreventCompoundSeparation = true;

  // Instantiate the Scheduler.
  HexagonGlobalSchedulerImpl GlobalSchedulerState(Fn, MLI, MDT, AA, MBPI, MBFI,
                                                  MRI, MFI, QRI);

  // DFA state table should not be empty.
  assert(GlobalSchedulerState.getResourceTracker() && "Empty DFA table!");

  // Loop over all of the basic blocks.
  // PullUp regions are basically traces with no side entrances.
  // Might want to traverse BB by frequency.
  GlobalSchedulerState.checkBundleCounts(Fn);

  // Pullup does not handle hazards yet.
  if (!DisablePullUp.getPosition() && ForceNoopHazards)
    return true;

  LLVM_DEBUG(GlobalSchedulerState.countCompounds(Fn));
  GlobalSchedulerState.GenUseDefChain(Fn);
  GlobalSchedulerState.formPullUpRegions(Fn);
  GlobalSchedulerState.performPullUp();
  GlobalSchedulerState.performPullUpCFG(Fn);
  if (PostPullUpOpt) {
    GlobalSchedulerState.formPullUpRegions(Fn);
    GlobalSchedulerState.performExposedOptimizations(Fn);
  }
  LLVM_DEBUG(GlobalSchedulerState.countCompounds(Fn));

  return true;
}

/// Allocate resources (i.e. 4 bytes) for constant extender. If succeess, return
/// true, otherwise, return false.
bool HexagonGlobalSchedulerImpl::tryAllocateResourcesForConstExt(
    MachineInstr *MI, bool UpdateState = true) {
  if (ResourceTracker->canReserveResources(*Ext)) {
    // We do not always want to change the state of ResourceTracker.
    // When we do not want to change it, we need to test for additional
    // corner cases.
    if (UpdateState)
      ResourceTracker->reserveResources(*Ext);
    else if (CurrentPacketMIs.size() >= PacketSize - 1)
      return false;
    return true;
  }

  return false;
}

static bool IsSchedBarrier(const MachineInstr *MI) {
  return MI->getOpcode() == Hexagon::Y2_barrier;
}

static bool IsIndirectCall(const MachineInstr *MI) {
  return MI->getOpcode() == Hexagon::J2_callr;
}

#ifndef NDEBUG
static void DumpLinked(MachineInstr *MI) {
  if (MI->isBundledWithPred())
    dbgs() << "^";
  else
    dbgs() << " ";
  if (MI->isBundledWithSucc())
    dbgs() << "v";
  else
    dbgs() << " ";
  MI->dump();
}

static void DumpPacket(MachineBasicBlock::instr_iterator MII) {
  if (MII == MachineBasicBlock::instr_iterator()) {
    dbgs() << "\tNULL\n";
    return;
  }
  MachineInstr *MI = &*MII;
  MachineBasicBlock *MBB = MI->getParent();
  // Uninserted instruction.
  if (!MBB) {
    dbgs() << "\tUnattached: ";
    DumpLinked(MI);
    return;
  }
  dbgs() << "\t";
  DumpLinked(MI);
  if (MI->isBundle()) {
    MachineBasicBlock::instr_iterator MIE = MI->getParent()->instr_end();
    for (++MII; MII != MIE && MII->isInsideBundle() && !MII->isBundle();
         ++MII) {
      dbgs() << "\t\t*";
      DumpLinked(&*MII);
    }
  }
}

static void DumpPacket(MachineBasicBlock::instr_iterator MII,
                       MachineBasicBlock::instr_iterator BBEnd) {
  if (MII == BBEnd) {
    dbgs() << "\tBBEnd\n";
    return;
  }

  DumpPacket(MII);
}
#endif

static bool isBranch(MachineInstr *MI) {
  if (MI->isBundle()) {
    MachineBasicBlock::instr_iterator MII = MI->getIterator();
    MachineBasicBlock::instr_iterator MIE = MI->getParent()->instr_end();
    for (++MII; MII != MIE && MII->isInsideBundle() && !MII->isBundle();
         ++MII) {
      if (MII->isBranch())
        return true;
    }
  } else
    return MI->isBranch();
  return false;
}

/// Any of those must not be first dual jump. Everything else is OK.
bool HexagonGlobalSchedulerImpl::IsNotDualJumpFirstCandidate(MachineInstr *MI) {
  if (MI->isCall() || (MI->isBranch() && !QII->isPredicated(*MI)) ||
      MI->isReturn() || QII->isEndLoopN(MI->getOpcode()))
    return true;
  return false;
}

/// These four functions clearly belong in HexagonInstrInfo.cpp.
/// Is this MI could be first dual jump instruction?
bool HexagonGlobalSchedulerImpl::IsDualJumpFirstCandidate(MachineInstr *MI) {
  if (!PerformDualJumps)
    return false;
  if (MI->isBranch() && QII->isPredicated(*MI) && !QII->isNewValueJump(*MI) &&
      !MI->isIndirectBranch() && !QII->isEndLoopN(MI->getOpcode()))
    return true;
  // Missing loopN here, but not sure if there will be any benefit from it.
  return false;
}

/// This version covers the whole packet.
bool HexagonGlobalSchedulerImpl::IsDualJumpFirstCandidate(
    MachineBasicBlock::iterator &TargetPacket) {
  if (!PerformDualJumps)
    return false;
  MachineInstr *MI = &*TargetPacket;

  if (MI->isBundle()) {
    // If this is a bundle, it must be the last bundle in BB.
    if (&(*MI->getParent()->rbegin()) != MI)
      return false;

    MachineBasicBlock::instr_iterator MII = MI->getIterator();
    MachineBasicBlock::instr_iterator BBEnd = MI->getParent()->instr_end();
    // If there is a control flow op in this packet, this is the case
    // we look for, even if they are dependent on other members.
    for (++MII; MII != BBEnd && MII->isInsideBundle() && !MII->isBundle();
         ++MII)
      if (IsNotDualJumpFirstCandidate(&*MII))
        return false;
  } else
    return IsDualJumpFirstCandidate(MI);

  return true;
}

/// This version cover whole BB. There could be a BB
/// with no control flow in it. In this case we can still pull-up a jump
/// into it. Negative proof.
bool HexagonGlobalSchedulerImpl::IsDualJumpFirstCandidate(
    MachineBasicBlock *MBB) {
  if (!PerformDualJumps)
    return false;

  for (MachineBasicBlock::instr_iterator MII = MBB->instr_begin(),
                                         MBBEnd = MBB->instr_end();
       MII != MBBEnd; ++MII) {
    MachineInstr *MI = &*MII;
    if (MI->isDebugInstr())
      continue;
    if (!MI->isBundle() && IsNotDualJumpFirstCandidate(MI))
      return false;
  }
  return true;
}

/// Is this MI could be second dual jump instruction?
bool HexagonGlobalSchedulerImpl::IsDualJumpSecondCandidate(MachineInstr *MI) {
  if (!PerformDualJumps)
    return false;
  if ((MI->isBranch() && !QII->isNewValueJump(*MI) && !MI->isIndirectBranch() &&
       !QII->isEndLoopN(MI->getOpcode())) ||
      (MI->isCall() && !IsIndirectCall(MI)))
    return true;
  return false;
}

// Since we have no exact knowledge of code layout,
// allow some safety buffer for jump target.
// This is measured in bytes.
static const unsigned SafetyBuffer = 200;

static MachineBasicBlock::instr_iterator
getHexagonFirstInstrTerminator(MachineBasicBlock *MBB) {
  MachineBasicBlock::instr_iterator MIB = MBB->instr_begin();
  MachineBasicBlock::instr_iterator MIE = MBB->instr_end();
  MachineBasicBlock::instr_iterator MII = MIB;
  while (MII != MIE) {
    if (!MII->isBundle() && MII->isTerminator())
      return MII;
    ++MII;
  }
  return MIE;
}

/// Check if a given instruction is:
/// - a jump to a distant target
/// - that exceeds its immediate range
/// If both conditions are true, it requires constant extension.
bool HexagonGlobalSchedulerImpl::isJumpOutOfRange(MachineInstr *MI) {
  if (!MI || !MI->isBranch())
    return false;
  MachineBasicBlock *MBB = MI->getParent();
  auto FirstTerm = getHexagonFirstInstrTerminator(MBB);
  if (FirstTerm == MBB->instr_end())
    return false;

  unsigned InstOffset = BlockToInstOffset[MBB];
  unsigned Distance = 0;
  MachineBasicBlock::instr_iterator FTMII = FirstTerm;

  // To save time, estimate exact position of a branch instruction
  // as one at the end of the MBB.
  // Number of instructions times typical instruction size.
  InstOffset += (QII->nonDbgBBSize(MBB) * HEXAGON_INSTR_SIZE);

  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  SmallVector<MachineOperand, 4> Cond;

  // Try to analyze this branch.
  if (QII->analyzeBranch(*MBB, TBB, FBB, Cond, false)) {
    // Could not analyze it. See if this is something we can recognize.
    // If it is a NVJ, it should always have its target in
    // a fixed location.
    if (QII->isNewValueJump(*FirstTerm))
      TBB = FirstTerm->getOperand(QII->getCExtOpNum(*FirstTerm)).getMBB();
  }
  if (TBB && (MI == &*FirstTerm)) {
    Distance =
        (unsigned)std::abs((long long)InstOffset - BlockToInstOffset[TBB]) +
        SafetyBuffer;
    LLVM_DEBUG(dbgs() << "\tFirst term offset(" << Distance << "): ";
               FirstTerm->dump());
    return !QII->isJumpWithinBranchRange(*FirstTerm, Distance);
  }
  if (FBB) {
    // Look for second terminator.
    FTMII++;
    MachineInstr *SecondTerm = &*FTMII;
    assert(FTMII != MBB->instr_end() &&
           (SecondTerm->isBranch() || SecondTerm->isCall()) &&
           "Bad second terminator");
    if (MI != SecondTerm)
      return false;
    // Analyze the second branch in the BB.
    Distance =
        (unsigned)std::abs((long long)InstOffset - BlockToInstOffset[FBB]) +
        SafetyBuffer;
    LLVM_DEBUG(dbgs() << "\tSecond term offset(" << Distance << "): ";
               FirstTerm->dump());
    return !QII->isJumpWithinBranchRange(*SecondTerm, Distance);
  }
  return false;
}

/// Returns true if an instruction can be promoted to .new predicate
/// or new-value store.
/// Performs implicit version checking.
bool HexagonGlobalSchedulerImpl::isNewifiable(
    MachineBasicBlock::instr_iterator MII, unsigned DepReg,
    MachineInstr *TargetPacket) {
  MachineInstr *MI = &*MII;
  if (QII->isDotNewInst(*MI) ||
      !CanNewifiedBeUsedInBundle(MII, DepReg, TargetPacket))
    return false;
  return (QII->isPredicated(*MI) && QII->getDotNewPredOp(*MI, nullptr) > 0) ||
         QII->mayBeNewStore(*MI);
}

bool HexagonGlobalSchedulerImpl::DemoteToDotOld(MachineInstr *MI) {
  int NewOpcode = QII->getDotOldOp(*MI);
  MI->setDesc(QII->get(NewOpcode));
  return true;
}

// initPacketizerState - Initialize packetizer flags
void HexagonGlobalSchedulerImpl::initPacketizerState(void) {
  CurrentPacketMIs.clear();
  return;
}

// ignorePseudoInstruction - Ignore bundling of pseudo instructions.
bool HexagonGlobalSchedulerImpl::ignoreInstruction(MachineInstr *MI) {
  if (MI->isDebugInstr())
    return true;

  // We must print out inline assembly
  if (MI->isInlineAsm())
    return false;

  // We check if MI has any functional units mapped to it.
  // If it doesn't, we ignore the instruction.
  const MCInstrDesc &TID = MI->getDesc();
  unsigned SchedClass = TID.getSchedClass();
  const InstrStage *IS =
      ResourceTracker->getInstrItins()->beginStage(SchedClass);
  unsigned FuncUnits = IS->getUnits();
  return !FuncUnits;
}

// isSoloInstruction: - Returns true for instructions that must be
// scheduled in their own packet.
bool HexagonGlobalSchedulerImpl::isSoloInstruction(const MachineInstr &MI) {
  if (MI.isInlineAsm())
    return true;

  if (MI.isEHLabel())
    return true;

  // From Hexagon V4 Programmer's Reference Manual 3.4.4 Grouping constraints:
  // trap, pause, barrier, icinva, isync, and syncht are solo instructions.
  // They must not be grouped with other instructions in a packet.
  if (IsSchedBarrier(&MI))
    return true;

  if (MI.getOpcode() == Hexagon::A2_nop)
    return true;

  return false;
}

/// Return region ptr or null if non found.
BasicBlockRegion *HexagonGlobalSchedulerImpl::getRegionForMBB(
    std::vector<BasicBlockRegion *> &Regions, MachineBasicBlock *MBB) {
  for (std::vector<BasicBlockRegion *>::iterator I = Regions.begin(),
                                                 E = Regions.end();
       I != E; ++I) {
    if ((*I)->findMBB(MBB))
      return *I;
  }
  return NULL;
}

/// Select best candidate to form regions.
static inline bool selectBestBB(BlockFrequency &BBaFreq, unsigned BBaSize,
                                BlockFrequency &BBbFreq, unsigned BBbSize) {
  if (BBaFreq.getFrequency() > BBbFreq.getFrequency())
    return true;
  // TODO: This needs fine tuning.
  // if (BBaSize < BBbSize)
  //  return true;
  if (BBaFreq.getFrequency() == BBbFreq.getFrequency())
    return true;
  return false;
}

/// Returns BB pointer if one of MBB successors should be added to the
/// current PullUp Region, NULL otherwise.
/// If SecondBest is defined, get next one after Best match.
/// Most of the time, since we practically always have only two successors,
/// this is "the other" BB successor which still matches original
/// selection criterion.
MachineBasicBlock *
HexagonGlobalSchedulerImpl::getNextPURBB(MachineBasicBlock *MBB,
                                         bool SecondBest = false) {
  if (!MBB)
    return NULL;

  BlockFrequency BestBlockFreq = BlockFrequency(0);
  unsigned BestBlockSize = 0;
  MachineBasicBlock *BestBB = NULL;
  MachineBasicBlock *SecondBestBB = NULL;

  // Catch single BB loops.
  for (MachineBasicBlock *Succ : MBB->successors())
    if (Succ == MBB)
      return NULL;

  // Iterate through successors to MBB.
  for (MachineBasicBlock *Succ : MBB->successors()) {
    BlockFrequency BlockFreq = MBFI->getBlockFreq(Succ);

    LLVM_DEBUG(dbgs() << "\tsucc BB(" << Succ->getNumber() << ") freq("
                      << BlockFreq.getFrequency() << ")");

    if (!SecondBest && getRegionForMBB(PullUpRegions, Succ))
      continue;

    // If there is more then one predecessor to this block, do not include it.
    // It means there is a side entrance to it.
    if (Succ->pred_size() > 1)
      continue;

    // If this block is a target of an indirect branch, it should
    // also not be included.
    if (Succ->isEHPad() || Succ->hasAddressTaken())
      continue;

    // Get BB edge frequency.
    BlockFrequency EdgeFreq = BlockFreq * MBPI->getEdgeProbability(MBB, Succ);
    LLVM_DEBUG(dbgs() << "\tedge with freq(" << EdgeFreq.getFrequency()
                      << ")\n");

    if (selectBestBB(EdgeFreq, QII->nonDbgBBSize(Succ), BestBlockFreq,
                     BestBlockSize)) {
      BestBlockFreq = EdgeFreq;
      BestBlockSize = QII->nonDbgBBSize(Succ);
      SecondBestBB = BestBB;
      BestBB = Succ;
    } else if (!SecondBestBB) {
      SecondBestBB = Succ;
    }
  }
  if (SecondBest)
    return SecondBestBB;
  else
    return BestBB;
}

/// Form region to perform pull-up.
bool HexagonGlobalSchedulerImpl::formPullUpRegions(MachineFunction &Fn) {
  const Function &F = Fn.getFunction();
  // Check for single-block functions and skip them.
  if (std::next(F.begin()) == F.end())
    return false;

  // Compute map for BB distances.
  // Offset of the current instruction from the start.
  unsigned InstOffset = 0;

  LLVM_DEBUG(dbgs() << "****** Form PullUpRegions **************\n");
  // Loop over all basic blocks.
  // PullUp regions are basically traces with no side entrances.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end(); MBB != MBBe;
       ++MBB) {
    if (MBB->getAlignment() > llvm::Align(1)) {
      // Although we don't know the exact layout of the final code, we need
      // to account for alignment padding somehow. This heuristic pads each
      // aligned basic block according to the alignment value.
      int ByteAlign = MBB->getAlignment().value() - 1;
      InstOffset = (InstOffset + ByteAlign) & ~(ByteAlign);
    }
    // Remember BB layout offset.
    BlockToInstOffset[&*MBB] = InstOffset;
    for (MachineBasicBlock::instr_iterator MII = MBB->instr_begin(),
                                           MIE = MBB->instr_end();
         MII != MIE; ++MII)
      if (!MII->isBundle())
        InstOffset += QII->getSize(*MII);

    // If this BB is already in a region, move on.
    if (getRegionForMBB(PullUpRegions, &*MBB))
      continue;

    LLVM_DEBUG(dbgs() << "\nRoot BB(" << MBB->getNumber() << ") name("
                      << MBB->getName() << ") size(" << QII->nonDbgBBSize(&*MBB)
                      << ") freq(" << printBlockFreq(*MBFI, *MBB)
                      << ") pred_size(" << MBB->pred_size() << ") in_func("
                      << MBB->getParent()->getFunction().getName() << ")\n");

    BasicBlockRegion *PUR = new BasicBlockRegion(TII, QRI, &*MBB);
    PullUpRegions.push_back(PUR);

    for (MachineBasicBlock *MBBR = getNextPURBB(&*MBB); MBBR;
         MBBR = getNextPURBB(MBBR)) {
      LLVM_DEBUG(dbgs() << "Add BB(" << MBBR->getNumber() << ") name("
                        << MBBR->getName() << ") size("
                        << QII->nonDbgBBSize(MBBR) << ") freq("
                        << printBlockFreq(*MBFI, *MBBR) << ") in_func("
                        << MBBR->getParent()->getFunction().getName() << ")\n");
      PUR->addBBtoRegion(MBBR);
    }
  }
  return true;
}

/// Return true if MI is an instruction we are unable to reason about
/// (like something with unmodeled memory side effects).
static inline bool isGlobalMemoryObject(MachineInstr *MI) {
  if (MI->hasUnmodeledSideEffects() || MI->hasOrderedMemoryRef() ||
      MI->isCall() ||
      (MI->getOpcode() == Hexagon::J2_jump && !MI->getOperand(0).isMBB()))
    return true;
  return false;
}

// This MI might have either incomplete info, or known to be unsafe
// to deal with (i.e. volatile object).
static inline bool isUnsafeMemoryObject(MachineInstr *MI) {
  if (!MI || MI->memoperands_empty())
    return true;

  // We purposefully do no check for hasOneMemOperand() here
  // in hope to trigger an assert downstream in order to
  // finish implementation.
  if ((*MI->memoperands_begin())->isVolatile() || MI->hasUnmodeledSideEffects())
    return true;

  if (!(*MI->memoperands_begin())->getValue())
    return true;

  return false;
}

/// This returns true if the two MIs could be memory dependent.
static bool MIsNeedChainEdge(AliasAnalysis *AA, const TargetInstrInfo *TII,
                             MachineInstr *MIa, MachineInstr *MIb) {
  // Cover a trivial case - no edge is need to itself.
  if (MIa == MIb)
    return false;

  if (TII->areMemAccessesTriviallyDisjoint(*MIa, *MIb))
    return false;

  if (isUnsafeMemoryObject(MIa) || isUnsafeMemoryObject(MIb))
    return true;

  // If we are dealing with two "normal" loads, we do not need an edge
  // between them - they could be reordered.
  if (!MIa->mayStore() && !MIb->mayStore())
    return false;

  // To this point analysis is generic. From here on we do need AA.
  if (!AA)
    return true;

  MachineMemOperand *MMOa = *MIa->memoperands_begin();
  MachineMemOperand *MMOb = *MIb->memoperands_begin();

  // TODO: Need to handle multiple memory operands.
  // if either instruction has more than one memory operand, punt.
  if (!(MIa->hasOneMemOperand() && MIb->hasOneMemOperand()))
    return true;

  if (!MMOa->getSize().hasValue() || !MMOb->getSize().hasValue())
    return true;

  assert((MMOa->getOffset() >= 0) && "Negative MachineMemOperand offset");
  assert((MMOb->getOffset() >= 0) && "Negative MachineMemOperand offset");
  assert((MMOa->getSize().hasValue() && MMOb->getSize().hasValue()) &&
         "Size 0 memory access");

  // If the base address of the two memoperands is the same. For instance,
  // x and x+4, then we can easily reason about them using the offset and size
  // of access.
  if (MMOa->getValue() == MMOb->getValue()) {
    if (MMOa->getOffset() > MMOb->getOffset()) {
      uint64_t offDiff = MMOa->getOffset() - MMOb->getOffset();
      return !(MMOb->getSize().getValue() <= offDiff);
    } else if (MMOa->getOffset() < MMOb->getOffset()) {
      uint64_t offDiff = MMOb->getOffset() - MMOa->getOffset();
      return !(MMOa->getSize().getValue() <= offDiff);
    }
    // MMOa->getOffset() == MMOb->getOffset()
    return true;
  }

  int64_t MinOffset = std::min(MMOa->getOffset(), MMOb->getOffset());
  int64_t Overlapa = MMOa->getSize().getValue() + MMOa->getOffset() - MinOffset;
  int64_t Overlapb = MMOb->getSize().getValue() + MMOb->getOffset() - MinOffset;

  AliasResult AAResult =
      AA->alias(MemoryLocation(MMOa->getValue(), Overlapa, MMOa->getAAInfo()),
                MemoryLocation(MMOb->getValue(), Overlapb, MMOb->getAAInfo()));

  return (AAResult != AliasResult::NoAlias);
}

/// Gather register def/uses from MI.
/// This treats possible (predicated) defs
/// as actually happening ones (conservatively).
static inline void parseOperands(MachineInstr *MI,
                                 SmallVector<unsigned, 4> &Defs,
                                 SmallVector<unsigned, 8> &Uses) {
  Defs.clear();
  Uses.clear();

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    if (MO.isReg()) {
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(Register::isPhysicalRegister(Reg));
      if (MO.isUse())
        Uses.push_back(MO.getReg());
      if (MO.isDef())
        Defs.push_back(MO.getReg());
    } else if (MO.isRegMask()) {
      for (unsigned R = 1, NR = Hexagon::NUM_TARGET_REGS; R != NR; ++R)
        if (MO.clobbersPhysReg(R))
          Defs.push_back(R);
    }
  }
}

void HexagonGlobalSchedulerImpl::MIUseDefSet(MachineInstr *MI,
                                             std::vector<unsigned> &Defs,
                                             std::vector<unsigned> &Uses) {
  Defs.clear();
  Uses.clear();
  assert(!MI->isBundle() && "Cannot parse regs of a bundle.");
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    if (MO.isReg()) {
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(Register::isPhysicalRegister(Reg));
      std::vector<unsigned> &Refs = MO.isUse() ? Uses : Defs;
      for (MCRegAliasIterator AI(MO.getReg(), QRI, true); AI.isValid(); ++AI)
        Refs.push_back(*AI);
    } else if (MO.isRegMask()) {
      for (unsigned R = 1, NR = Hexagon::NUM_TARGET_REGS; R != NR; ++R)
        if (MO.clobbersPhysReg(R))
          Defs.push_back(R);
    }
  }
}

/// Some apparent dependencies are not actually restricting us since there
/// is a delay between assignment and actual usage, like in case of a call.
/// There could be more cases here, but this one seems the most obvious.
static bool isDelayedUseException(MachineInstr *MIa, MachineInstr *MIb) {
  if (MIa->isCall() && !MIb->isCall())
    return true;
  if (!MIa->isCall() && MIb->isCall())
    return true;
  return false;
}

/// This is a check for resources availability and dependency
/// for an MI being tried for an existing bundle.
/// This is needed because we can:
///   - save time by filtering out trivial cases
///   - we want to reuse infrastructure that does not really knows
///     how to deal with parallel semantics of a bundle that already
///     exists. For instance, the following case:
///     SI %R6<def> = L2_ploadrif_io %P0<kill>, %R7, 4;
///     SJ %R6<def> = A2_tfr %R0;
///     will be happily allowed by isLegalToPacketizeTogether since in serial
///     semantics it never happens, and even if it does, it is legal. Not so
///     for when we __speculatively__ trying and MI for a bundle.
///
/// Note:  This is not equivalent to MIsAreDependent().
/// MIsAreDependent only understands serial semantics.
/// These are OK to packetize together:
///  %R0<def> = L2_loadri_io %R18, 76; mem:LD4[%sunkaddr226](tbaa=!"int")
///  %R2<def> = ASL %R0<kill>, 3; flags:  Inside bundle
///
bool HexagonGlobalSchedulerImpl::canAddMIToThisPacket(
    MachineInstr *MI,
    SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE> &Bundle) {
  if (!MI)
    return false;
  LLVM_DEBUG(dbgs() << "\n\t[canAddMIToThisPacket]: "; MI->dump());

  // Const extenders need custom resource checking...
  // Should be OK if we can update the check everywhere.
  if ((QII->isConstExtended(*MI) || QII->isExtended(*MI) ||
       isJumpOutOfRange(MI)) &&
      !tryAllocateResourcesForConstExt(MI, false))
    return false;

  // Ask DFA if machine resource is available for MI.
  if (!ResourceTracker->canReserveResources(*MI) || !shouldAddToPacket(*MI)) {
    LLVM_DEBUG(dbgs() << "\tNo DFA resources.\n");
    return false;
  }

  SmallVector<unsigned, 4> BundleDefs;
  SmallVector<unsigned, 8> BundleUses;
  SmallVector<unsigned, 4> Defs;
  SmallVector<unsigned, 8> Uses;
  MachineInstr *FirstCompound = NULL, *SecondCompound = NULL;
  MachineInstr *FirstDuplex = NULL, *SecondDuplex = NULL;

  parseOperands(MI, Defs, Uses);
  for (SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE>::iterator
           BI = Bundle.begin(),
           BE = Bundle.end();
       BI != BE; ++BI) {
    BundleDefs.clear();
    BundleUses.clear();
    parseOperands(*BI, BundleDefs, BundleUses);

    MachineInstr *Inst1 = *BI;
    MachineInstr *Inst2 = MI;

    if (Inst1->getParent() && OneFloatPerPacket && QII->isFloat(*Inst1) &&
        QII->isFloat(*Inst2))
      return false;

    if (Inst1->getParent() && OneComplexPerPacket && QII->isComplex(*Inst1) &&
        QII->isComplex(*Inst2))
      return false;

    if (PreventCompoundSeparation)
      if (QII->getCompoundCandidateGroup(**BI)) {
        if (!FirstCompound)
          FirstCompound = *BI;
        else {
          SecondCompound = *BI;
          if (isCompoundPair(FirstCompound, SecondCompound)) {
            if (MI->mayLoad() || MI->mayStore()) {
              LLVM_DEBUG(dbgs() << "\tPrevent compound destruction.\n");
              return false;
            }
          }
        }
      }
    if (PreventDuplexSeparation)
      if (QII->getDuplexCandidateGroup(**BI)) {
        if (!FirstDuplex)
          FirstDuplex = *BI;
        else {
          SecondDuplex = *BI;
          if (QII->isDuplexPair(*FirstDuplex, *SecondDuplex)) {
            if (MI->mayLoad() || MI->mayStore()) {
              LLVM_DEBUG(dbgs() << "\tPrevent duplex destruction.\n");
              return false;
            }
          }
        }
      }

    for (unsigned i = 0; i < Defs.size(); i++) {
      // Check for multiple definitions in the same packet.
      for (unsigned j = 0; j < BundleDefs.size(); j++)
        // Multiple defs in the same packet.
        // Calls are OK here.
        // Also if we have multiple defs of PC, this simply means we are
        // dealing with dual jumps.
        if (AliasingRegs(Defs[i], BundleDefs[j]) &&
            !isDelayedUseException(MI, *BI) &&
            !(IsDualJumpFirstCandidate(*BI) && IsDualJumpSecondCandidate(MI))) {
          LLVM_DEBUG(dbgs() << "\tMultiple defs.\n\t"; MI->dump();
                     dbgs() << "\t"; (*BI)->dump());
          return false;
        }

      // See if we are creating a swap case as we go, and disallow
      // it for now.
      // Also, this is not OK:
      //  if (!p0) r7 = r5
      //  if (!p0) r5 = #0
      // But this is fine:
      //  if (!p0) r7 = r5
      //  if (p0) r5 = #0
      // Aslo - this is not a swap, but an opportunity to newify:
      // %P1<def> = C2_cmpeqi %R0, 0; flags:
      // %R0<def> = L2_ploadrif_io %P1<kill>, %R29, 8;
      // TODO: Handle this.
      for (unsigned j = 0; j < BundleUses.size(); j++)
        if (AliasingRegs(Defs[i], BundleUses[j])) {
          for (unsigned k = 0; k < BundleDefs.size(); k++)
            for (unsigned l = 0; l < Uses.size(); l++) {
              if (AliasingRegs(BundleDefs[k], Uses[l]) &&
                  !isDelayedUseException(MI, *BI)) {
                LLVM_DEBUG(dbgs() << "\tSwap detected:\n\t"; MI->dump();
                           dbgs() << "\t"; (*BI)->dump());
                return false;
              }
            }
        }
    }

    for (unsigned i = 0; i < Uses.size(); i++) {
      // Check for true data dependency.
      for (unsigned j = 0; j < BundleDefs.size(); j++)
        if (AliasingRegs(Uses[i], BundleDefs[j]) &&
            !isDelayedUseException(MI, *BI)) {
          LLVM_DEBUG(dbgs() << "\tImmediate Use detected on reg("
                            << printReg(Uses[i], QRI) << ")\n\t";
                     MI->dump(); dbgs() << "\t"; (*BI)->dump());
          // TODO: This could be an opportunity for newifying:
          //  %P0<def> = C2_cmpeqi %R26, 0
          //  %R26<def> = A2_tfr %R0<kill>
          // if (CanPromoteToDotNew(MI, Uses[i]))
          //  LLVM_DEBUG(dbgs() << "\tCan promoto to .new form.\n");
          // else
          return false;
        }
    }

    // For calls we also check callee save regs.
    if ((*BI)->isCall()) {
      for (const uint16_t *I = QRI->getCalleeSavedRegs(&MF); *I; ++I) {
        for (unsigned i = 0; i < Defs.size(); i++) {
          if (AliasingRegs(Defs[i], *I)) {
            LLVM_DEBUG(dbgs() << "\tAlias with call.\n");
            return false;
          }
        }
      }
    }

    // If this is return, we are probably speculating (otherwise
    // we could not pull in there) and will not win from pulling
    // into this location anyhow.
    // Example: a side exit.
    // if (!p0) dealloc_return
    // TODO: Can check that we do not overwrite return value
    // and proceed.
    if ((*BI)->isBarrier()) {
      LLVM_DEBUG(dbgs() << "\tBarrier interference.\n");
      return false;
    }

    // \ref-manual (7.3.4) A loop setup packet in loopN or spNloop0 cannot
    // contain a speculative indirect jump,
    // a new-value compare jump or a dealloc_return.
    // Speculative indirect jumps (predicate + .new + indirect):
    // if ([!]Ps.new) jumpr:t Rs
    // if ([!]Ps.new) jumpr:nt Rs
    // @note: We don't want to pull across a call to be on the safe side.
    if (QII->isLoopN(*MI) &&
        ((QII->isPredicated(**BI) && QII->isPredicatedNew(**BI) &&
          QII->isJumpR(**BI)) ||
         QII->isNewValueJump(**BI) || QII->isDeallocRet(**BI) ||
         (*BI)->isCall())) {
      LLVM_DEBUG(dbgs() << "\tLoopN pull interference.\n");
      return false;
    }

    // The opposite is also true.
    if (QII->isLoopN(**BI) &&
        ((QII->isPredicated(*MI) && QII->isPredicatedNew(*MI) &&
          QII->isJumpR(*MI)) ||
         QII->isNewValueJump(*MI) || QII->isDeallocRet(*MI) || MI->isCall())) {
      LLVM_DEBUG(dbgs() << "\tResident LoopN.\n");
      return false;
    }

    // @todo \ref-manual 7.6.1
    // Presence of NVJ adds more restrictions.
    if (QII->isNewValueJump(**BI) &&
        (MI->mayStore() || MI->getOpcode() == Hexagon::S2_allocframe ||
         MI->isCall())) {
      LLVM_DEBUG(dbgs() << "\tNew val Jump.\n");
      return false;
    }

    // For memory operations, check aliasing.
    // First, be conservative on these objects. Might be overly constraining,
    // so recheck.
    if (isGlobalMemoryObject(*BI) || isGlobalMemoryObject(MI))
      // Currently it catches things like this:
      // S2_storerinew_io %R29, 32, %R16
      // S2_storeri_io %R29, 68, %R0
      // which we can reason about.
      // TODO: revisit.
      return false;

    // If packet has a new-value store, MI can't be a store instruction.
    if (QII->isNewValueStore(**BI) && MI->mayStore()) {
      LLVM_DEBUG(dbgs() << "\tNew Value Store to store.\n");
      return false;
    }

    if ((QII->isMemOp(**BI) && MI->mayStore()) ||
        (QII->isMemOp(*MI) && (*BI)->mayStore())) {
      LLVM_DEBUG(
          dbgs() << "\tSlot 0 not available for store because of memop.\n");
      return false;
    }

    // If any of these is true, check aliasing.
    if ((MI->mayLoad() && (*BI)->mayStore()) ||
        (MI->mayStore() && (*BI)->mayLoad()) ||
        (MI->mayStore() && (*BI)->mayStore())) {
      if (MIsNeedChainEdge(AA, TII, MI, *BI)) {
        LLVM_DEBUG(dbgs() << "\tAliasing detected:\n\t"; MI->dump();
                   dbgs() << "\t"; (*BI)->dump());
        return false;
      }
    }
    // Do not move an instruction to this packet if this packet
    // already contains a speculated instruction.
    std::map<MachineInstr *, MachineBasicBlock *>::iterator MIMoved;
    MIMoved = SpeculatedIns.find(*BI);
    if ((MIMoved != SpeculatedIns.end()) &&
        (MIMoved->second != (*BI)->getParent())) {
      LLVM_DEBUG(
          dbgs() << "This packet already contains a speculated instruction";
          (*BI)->dump(););
      return false;
    }
  }

  // Do not pull-up vector instructions because these instructions have
  // multi-cycle latencies, and the pull-up pass doesn't correctly account
  // for instructions that stall for more than one cycle.
  if (QII->isHVXVec(*MI))
    return false;

  return true;
}

/// Test is true if the two MIs cannot be safely reordered.
bool HexagonGlobalSchedulerImpl::ReorderDependencyTest(MachineInstr *MIa,
                                                       MachineInstr *MIb) {
  SmallVector<unsigned, 4> DefsA;
  SmallVector<unsigned, 4> DefsB;
  SmallVector<unsigned, 8> UsesA;
  SmallVector<unsigned, 8> UsesB;

  parseOperands(MIa, DefsA, UsesA);
  parseOperands(MIb, DefsB, UsesB);

  for (SmallVector<unsigned, 4>::iterator IDA = DefsA.begin(),
                                          IDAE = DefsA.end();
       IDA != IDAE; ++IDA) {
    for (SmallVector<unsigned, 8>::iterator IUB = UsesB.begin(),
                                            IUBE = UsesB.end();
         IUB != IUBE; ++IUB)
      // True data dependency.
      if (AliasingRegs(*IDA, *IUB))
        return true;

    for (SmallVector<unsigned, 4>::iterator IDB = DefsB.begin(),
                                            IDBE = DefsB.end();
         IDB != IDBE; ++IDB)
      // Output dependency.
      if (AliasingRegs(*IDA, *IDB))
        return true;
  }

  for (SmallVector<unsigned, 4>::iterator IDB = DefsB.begin(),
                                          IDBE = DefsB.end();
       IDB != IDBE; ++IDB) {
    for (SmallVector<unsigned, 8>::iterator IUA = UsesA.begin(),
                                            IUAE = UsesA.end();
         IUA != IUAE; ++IUA)
      // True data dependency.
      if (AliasingRegs(*IDB, *IUA))
        return true;
  }

  // Do not reorder two calls...
  if (MIa->isCall() && MIb->isCall())
    return true;

  // For calls we also check callee save regs.
  if (MIa->isCall())
    for (const uint16_t *I = QRI->getCalleeSavedRegs(&MF); *I; ++I) {
      for (unsigned i = 0; i < DefsB.size(); i++) {
        if (AliasingRegs(DefsB[i], *I))
          return true;
      }
    }

  if (MIb->isCall())
    for (const uint16_t *I = QRI->getCalleeSavedRegs(&MF); *I; ++I) {
      for (unsigned i = 0; i < DefsA.size(); i++) {
        if (AliasingRegs(DefsA[i], *I))
          return true;
      }
    }

  // For memory operations, check aliasing.
  // First, be conservative on these objects.
  // Might be overly constraining, so recheck.
  if ((isGlobalMemoryObject(MIa)) || (isGlobalMemoryObject(MIb)))
    return true;

  // If any of these is true, check aliasing.
  if (((MIa->mayLoad() && MIb->mayStore()) ||
       (MIa->mayStore() && MIb->mayLoad()) ||
       (MIa->mayStore() && MIb->mayStore())) &&
      MIsNeedChainEdge(AA, TII, MIa, MIb))
    return true;

  return false;
}

/// Serial semantics.
bool HexagonGlobalSchedulerImpl::MIsAreDependent(MachineInstr *MIa,
                                                 MachineInstr *MIb) {
  if (MIa == MIb)
    return false;

  if (ReorderDependencyTest(MIa, MIb)) {
    LLVM_DEBUG(dbgs() << "\t\t[MIsAreDependent]:\n\t\t"; MIa->dump();
               dbgs() << "\t\t"; MIb->dump());
    return true;
  }
  return false;
}

/// Serial semantics.
bool HexagonGlobalSchedulerImpl::MIsHaveTrueDependency(MachineInstr *MIa,
                                                       MachineInstr *MIb) {
  if (MIa == MIb)
    return false;

  SmallVector<unsigned, 4> DefsA;
  SmallVector<unsigned, 4> DefsB;
  SmallVector<unsigned, 8> UsesA;
  SmallVector<unsigned, 8> UsesB;

  parseOperands(MIa, DefsA, UsesA);
  parseOperands(MIb, DefsB, UsesB);

  for (SmallVector<unsigned, 4>::iterator IDA = DefsA.begin(),
                                          IDAE = DefsA.end();
       IDA != IDAE; ++IDA) {
    for (SmallVector<unsigned, 8>::iterator IUB = UsesB.begin(),
                                            IUBE = UsesB.end();
         IUB != IUBE; ++IUB)
      // True data dependency.
      if (AliasingRegs(*IDA, *IUB))
        return true;
  }
  return false;
}

/// Sequential semantics. Can these two MIs be reordered?
/// Moving MIa from "behind" to "in front" of MIb.
bool HexagonGlobalSchedulerImpl::canReorderMIs(MachineInstr *MIa,
                                               MachineInstr *MIb) {
  if (!MIa || !MIb)
    return false;

  // Within bundle semantics are parallel.
  if (MIa->isBundle()) {
    MachineBasicBlock::instr_iterator MII = MIa->getIterator();
    MachineBasicBlock::instr_iterator MIIE = MIa->getParent()->instr_end();
    for (++MII; MII != MIIE && MII->isInsideBundle(); ++MII) {
      if (MII->isDebugInstr())
        continue;
      if (MIsAreDependent(&*MII, MIb))
        return false;
    }
    return true;
  }
  return !MIsAreDependent(MIa, MIb);
}

static inline bool MIMustNotBePulledUp(MachineInstr *MI) {
  if (MI->isInlineAsm() || MI->isEHLabel() || IsSchedBarrier(MI))
    return true;
  return false;
}

static inline bool MIShouldNotBePulledUp(MachineInstr *MI) {
  if (MI->isBranch() || MI->isReturn() || MI->isCall() || MI->isBarrier() ||
      MI->isTerminator() || MIMustNotBePulledUp(MI))
    return true;
  return false;
}

// Only approve dual jump candidate:
// It is a branch, and we move it to last packet of the target location.
bool HexagonGlobalSchedulerImpl::MIisDualJumpCandidate(
    MachineInstr *MI, MachineBasicBlock::iterator &WorkPoint) {
  if (!PerformDualJumps || !IsDualJumpSecondCandidate(MI) ||
      MIMustNotBePulledUp(MI) || ignoreInstruction(MI))
    return false;

  MachineBasicBlock *FromThisBB = MI->getParent();
  MachineBasicBlock *ToThisBB = WorkPoint->getParent();

  LLVM_DEBUG(dbgs() << "\t\t[MIisDualJumpCandidate] To BB("
                    << ToThisBB->getNumber() << ") From BB("
                    << FromThisBB->getNumber() << ")\n");
  // If the question is about the same BB, we do not want to get
  // dual jump involved - it is a different case.
  if (FromThisBB == ToThisBB)
    return false;

  // Dual jump could only be done on neigboring BBs.
  // The FromThisBB must only have one predecessor - the basic
  // block we are trying to merge.
  if ((*(FromThisBB->pred_begin()) != ToThisBB) ||
      (std::next(FromThisBB->pred_begin()) != FromThisBB->pred_end()))
    return false;

  // If this block is a target of an indirect branch, it should
  // also not be included.
  if (FromThisBB->isEHPad() || FromThisBB->hasAddressTaken())
    return false;

  // Now we must preserve original fall through paths. In fact we
  // might be dealing with 3way branching.
  MachineBasicBlock *ToTBB = NULL, *ToFBB = NULL;

  if (ToThisBB->succ_size() == 2) {
    // Check the branch from target block.
    // If we have two successors, we must understand the branch.
    SmallVector<MachineOperand, 4> ToCond;
    if (!QII->analyzeBranch(*ToThisBB, ToTBB, ToFBB, ToCond, false)) {
      // Have the branch. Check the topology.
      LLVM_DEBUG(dbgs() << "\t\tToThisBB has two successors: TBB("
                        << ToTBB->getNumber() << ") and FBB(";
                 if (ToFBB) dbgs() << ToFBB->getNumber() << ").\n";
                 else dbgs() << "None"
                             << ").\n";);
      if (ToTBB == FromThisBB) {
        // If the from BB is not the fall through, we can only handle case
        // when second branch is unconditional jump.
        return false;
      } else if (ToFBB == FromThisBB || !ToFBB) {
        // If the fall through path of ToBB is our FromBB, we have more freedom
        // of operation.
        LLVM_DEBUG(dbgs() << "\t\tFall through jump target.\n");
      }
    } else {
      LLVM_DEBUG(dbgs() << "\t\tUnable to analyze first branch.\n");
      return false;
    }
  } else if (ToThisBB->succ_size() == 1) {
    ToFBB = *ToThisBB->succ_begin();
    assert(ToFBB == FromThisBB && "Bad CFG layout");
  } else
    return false;

  // First unbundled control flow instruction in the BB.
  if (!MI->isBundled() && MI == &*FromThisBB->getFirstNonDebugInstr())
    return IsDualJumpFirstCandidate(WorkPoint);

  return false;
}

// Check whether moving MI to MJ's packet would cause a stall from a previous
// packet.
bool HexagonGlobalSchedulerImpl::canCauseStall(MachineInstr *MI,
                                               MachineInstr *MJ) {
  SmallVector<unsigned, 4> DefsMJI;
  SmallVector<unsigned, 8> UsesMJI;
  SmallVector<unsigned, 4> DefsMI;
  SmallVector<unsigned, 8> UsesMI;
  parseOperands(MI, DefsMI, UsesMI);

  for (auto Use : UsesMI) {
    int UseIdx = MI->findRegisterUseOperandIdx(Use, /*TRI=*/nullptr);
    if (UseIdx == -1)
      continue;
    bool ShouldBreak = false;
    int BundleCount = 0;
    for (MachineBasicBlock::instr_iterator
             Begin = MJ->getParent()->instr_begin(),
             MJI = MJ->getIterator();
         MJI != Begin; --MJI) {
      if (MJI->isBundle()) {
        ++BundleCount;
        continue;
      }
      parseOperands(&*MJI, DefsMJI, UsesMJI);
      for (auto Def : DefsMJI) {
        if (Def == Use || AliasingRegs(Def, Use)) {
          int DefIdx = MJI->findRegisterDefOperandIdx(Def, /*TRI=*/nullptr);
          if (DefIdx >= 0) {
            int Latency =
                TSchedModel.computeOperandLatency(&*MJI, DefIdx, MI, UseIdx);
            if (Latency > BundleCount)
              // There will be a stall if MI is moved to MJ's packet.
              return true;
            // We found the def for the use and it does not cause a stall.
            // Continue checking the next use for a potential stall.
            ShouldBreak = true;
            break;
          }
        }
      }
      if (ShouldBreak)
        break;
      if (!MJI->isBundled() && !MJI->isDebugInstr())
        ++BundleCount;
    }
  }
  return false;
}

/// Analyze this instruction. If this is an unbundled instruction, see
/// if it in theory could be packetized.
/// If it is already part of a packet, see if it has internal
/// dependencies to this packet.
bool HexagonGlobalSchedulerImpl::canThisMIBeMoved(
    MachineInstr *MI, MachineBasicBlock::iterator &WorkPoint,
    bool &MovingDependentOp, int &Cost) {
  if (!MI)
    return false;
  // By default, it is a normal move.
  MovingDependentOp = false;
  Cost = 0;
  // If MI is a 'formed' compound not potential compound, bail out.
  if (QII->isCompoundBranchInstr(*MI))
    return false;
  // See if we can potentially break potential compound candidates,
  // and do not do it.
  if (PreventCompoundSeparation && MI->isBundled()) {
    enum HexagonII::CompoundGroup MICG = QII->getCompoundCandidateGroup(*MI);
    if (MICG != HexagonII::HCG_None) {
      // Check internal dependencies in the bundle.
      // First, find the bundle header.
      MachineBasicBlock::instr_iterator MII = MI->getIterator();
      for (--MII; MII->isBundled(); --MII)
        if (MII->isBundle())
          break;

      MachineBasicBlock::instr_iterator BBEnd = MI->getParent()->instr_end();
      for (++MII; MII != BBEnd && MII->isInsideBundle() && !MII->isBundle();
           ++MII) {
        if (&(*MII) == MI)
          continue;
        if (isCompoundPair(&*MII, MI)) {
          LLVM_DEBUG(dbgs() << "\tPrevent Compound separation.\n");
          return false;
        }
      }
    }
  }
  // Same thing for duplex candidates.
  if (PreventDuplexSeparation && MI->isBundled()) {
    if (QII->getDuplexCandidateGroup(*MI) != HexagonII::HSIG_None) {
      // Check internal dependencies in the bundle.
      // First, find the bundle header.
      MachineBasicBlock::instr_iterator MII = MI->getIterator();
      for (--MII; MII->isBundled(); --MII)
        if (MII->isBundle())
          break;

      MachineBasicBlock::instr_iterator BBEnd = MI->getParent()->instr_end();
      for (++MII; MII != BBEnd && MII->isInsideBundle() && !MII->isBundle();
           ++MII) {
        if ((&(*MII) != MI) && QII->isDuplexPair(*MII, *MI)) {
          LLVM_DEBUG(dbgs() << "\tPrevent Duplex separation.\n");
          return false;
        }
      }
    }
  }

  // If we perform dual jump formation during the pull-up,
  // then we want to consider several additional situations.
  // a) Allow moving of dependent instruction from a packet
  // b) Allow moving some control flow instructions if they meet
  //    dual jump criteria.
  if (MIisDualJumpCandidate(MI, WorkPoint)) {
    LLVM_DEBUG(dbgs() << "\t\tDual jump candidate:\t"; MI->dump());
    // Here we are breaking our general assumption about not moving dependent
    // instructions. To save us two more expensive checks down the line,
    // propagate the information directly.
    MovingDependentOp = true;
    return true;
  }

  // Any of these should not even be tried.
  if (MIShouldNotBePulledUp(MI) || ignoreInstruction(MI))
    return false;
  // Pulling up these instructions could put them
  // out of jump range/offset size.
  if (QII->isLoopN(*MI)) {
    unsigned dist_looplabel =
        BlockToInstOffset.find(MI->getOperand(0).getMBB())->second;
    unsigned dist_newloop0 =
        BlockToInstOffset.find(WorkPoint->getParent())->second;
    // Check if the jump in the last instruction is within range.
    unsigned Distance =
        (unsigned)std::abs((long long)dist_looplabel - dist_newloop0) +
        QII->nonDbgBBSize(WorkPoint->getParent()) * 4 + SafetyBuffer;
    const HexagonInstrInfo *HII = (const HexagonInstrInfo *)TII;
    if (!HII->isJumpWithinBranchRange(*MI, Distance)) {
      LLVM_DEBUG(dbgs() << "\nloopN cannot be moved since Distance: "
                        << Distance << " outside branch range.";);
      return false;
    }
    LLVM_DEBUG(dbgs() << "\nloopN can be moved since Distance: " << Distance
                      << " within branch range.";);
  }
  // If the def-set of an MI is one of the live-ins then MI should
  // kill that reg and no instruction before MI should use it.
  // For simplicity, allow only if MI is the first instruction in the MBB.
  std::map<MachineInstr *, std::vector<unsigned>>::const_iterator DefIter =
      MIDefSet.find(MI);
  MachineBasicBlock *MBB = MI->getParent();
  for (unsigned i = 0; DefIter != MIDefSet.end() && i < DefIter->second.size();
       ++i) {
    if (MBB->isLiveIn(DefIter->second[i]) &&
        &*MBB->getFirstNonDebugInstr() != MI)
      return false;
  }
  // If it is part of a bundle, analyze it.
  if (MI->isBundled()) {
    // Cannot move bundle header itself. This function is about
    // individual MI move.
    if (MI->isBundle())
      return false;

    // Check internal dependencies in the bundle.
    // First, find the bundle header.
    MachineBasicBlock::instr_iterator MII = MI->getIterator();
    for (--MII; MII->isBundled(); --MII)
      if (MII->isBundle())
        break;

    MachineBasicBlock::instr_iterator BBEnd = MI->getParent()->instr_end();
    for (++MII; MII != BBEnd && MII->isInsideBundle() && !MII->isBundle();
         ++MII) {
      if (MII->isDebugInstr())
        continue;
      if (MIsAreDependent(&*MII, MI)) {
        if (!AllowDependentPullUp) {
          LLVM_DEBUG(dbgs() << "\t\tDependent.\n");
          return false;
        } else {
          // There are a few cases that we can safely move a dependent
          // instruction away from this packet.
          // One example is an instruction setting a call operands.
          if ((MII->isCall() && !IsIndirectCall(&*MII)) ||
              IsDualJumpSecondCandidate(&*MII) || MI->isBranch()) {
            LLVM_DEBUG(dbgs() << "\t\tDependent, but allow to move.\n");
            MovingDependentOp = true;
            Cost -= 10;
            continue;
          } else {
            LLVM_DEBUG(dbgs() << "\t\tDependent, and do not allow for now.\n");
            return false;
          }
        }
      }
    }
  }
  return true;
}

/// Return true if MI defines a predicate and parse all defs.
bool HexagonGlobalSchedulerImpl::doesMIDefinesPredicate(
    MachineInstr *MI, SmallVector<unsigned, 4> &Defs) {
  bool defsPredicate = false;
  Defs.clear();

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    // Regmasks are considered "implicit".
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();

    if (!Reg || QRI->isFakeReg(Reg))
      continue;

    assert(Register::isPhysicalRegister(Reg));

    if (MO.isDef() && !MO.isImplicit()) {
      const TargetRegisterClass *RC = QRI->getMinimalPhysRegClass(Reg);
      if (RC == &Hexagon::PredRegsRegClass) {
        defsPredicate = true;
        Defs.push_back(MO.getReg());
      }
    }
  }
  return defsPredicate;
}

/// We have just tentatively added a predicated MI to an existing packet.
/// Now we need to determine if it needs to be changed to .new form.
/// It only handles compare/predicate right now.
/// TODO - clean this logic up.
/// TODO - generalize to handle any .new
bool HexagonGlobalSchedulerImpl::NeedToNewify(
    MachineBasicBlock::instr_iterator NewMI, unsigned *DepReg,
    MachineInstr *TargetPacket = NULL) {
  MachineBasicBlock::instr_iterator MII = NewMI;
  SmallVector<unsigned, 4> DefsA;
  SmallVector<unsigned, 4> DefsB;
  SmallVector<unsigned, 8> UsesB;

  // If this is not a normal bundle, we are probably
  // trying to size two lonesome instructions together,
  // and trying to say if one of them will need to be
  // newified. In this is the case we have something like this:
  //   BB#5:
  //    %P0<def> = CMPGEri %R4, 2
  //    S2_pstorerif_io %P0<kill>, %R29, 16, %R21<kill>
  //    BUNDLE %R7<imp-def>, %R4<imp-def>, %R7<imp-use>
  parseOperands(&*NewMI, DefsB, UsesB);
  if (TargetPacket && !TargetPacket->isBundled()) {
    if (doesMIDefinesPredicate(TargetPacket, DefsA)) {
      for (SmallVector<unsigned, 4>::iterator IA = DefsA.begin(),
                                              IAE = DefsA.end();
           IA != IAE; ++IA)
        for (SmallVector<unsigned, 8>::iterator IB = UsesB.begin(),
                                                IBE = UsesB.end();
             IB != IBE; ++IB)
          if (*IA == *IB) {
            *DepReg = *IA;
            return true;
          }
    }
    return false;
  }

  // Find bundle header.
  for (--MII; MII->isBundled(); --MII)
    if (MII->isBundle())
      break;

  // Iterate down, if there is data dependent cmp found, need to .newify.
  // Also, we can have the following:
  //   {
  //  p0 = r7
  //  if (!p0.new) jump:t .LBB4_18
  //  if (p0.new) r8 = zxth(r12)
  //   }
  MachineBasicBlock::instr_iterator BBEnd = MII->getParent()->instr_end();
  for (++MII; MII != BBEnd && MII->isBundled() && !MII->isBundle(); ++MII) {
    if (MII == NewMI)
      continue;
    if (doesMIDefinesPredicate(&*MII, DefsA)) {
      for (SmallVector<unsigned, 4>::iterator IA = DefsA.begin(),
                                              IAE = DefsA.end();
           IA != IAE; ++IA)
        for (SmallVector<unsigned, 8>::iterator IB = UsesB.begin(),
                                                IBE = UsesB.end();
             IB != IBE; ++IB)
          // We do not have multiple predicate regs defined in any instruction,
          // if we ever will, this needs to be generalized.
          if (*IA == *IB) {
            *DepReg = *IA;
            return true;
          }
      DefsA.clear();
    }
  }
  LLVM_DEBUG(dbgs() << "\nNo need to newify:"; NewMI->dump());
  return false;
}

/// We know this instruction needs to be newified to be added to the packet,
/// but not all combinations are legal.
/// It is a complimentary check to NeedToNewify().
/// The packet actually contains the new instruction during the check.
bool HexagonGlobalSchedulerImpl::CanNewifiedBeUsedInBundle(
    MachineBasicBlock::instr_iterator NewMI, unsigned DepReg,
    MachineInstr *TargetPacket) {
  MachineBasicBlock::instr_iterator MII = NewMI;
  if (!TargetPacket || !TargetPacket->isBundled())
    return true;

  // Find the bundle header.
  for (--MII; MII->isBundled(); --MII)
    if (MII->isBundle())
      break;

  MachineBasicBlock::instr_iterator BBEnd = MII->getParent()->instr_end();
  for (++MII; MII != BBEnd && MII->isBundled() && !MII->isBundle(); ++MII) {
    // Effectively we look for the case of late predicates.
    // No additional checks at the time.
    if (MII == NewMI || !QII->isPredicateLate(MII->getOpcode()))
      continue;
    SmallVector<unsigned, 4> DefsA;
    if (!doesMIDefinesPredicate(&*MII, DefsA))
      continue;
    for (auto &IA : DefsA)
      if (IA == DepReg)
        return false;
  }
  return true;
}

/// setUsed - Set the register and its sub-registers as being used.
/// Similar to RegScavenger::setUsed().
void HexagonGlobalSchedulerImpl::setUsedRegs(BitVector &Set, unsigned Reg) {
  Set.reset(Reg);
  for (MCSubRegIterator SubRegs(Reg, QRI); SubRegs.isValid(); ++SubRegs)
    Set.reset(*SubRegs);
}

/// Are these two registers overlaping?
bool HexagonGlobalSchedulerImpl::AliasingRegs(unsigned RegA, unsigned RegB) {
  if (RegA == RegB)
    return true;

  for (MCSubRegIterator SubRegs(RegA, QRI); SubRegs.isValid(); ++SubRegs)
    if (RegB == *SubRegs)
      return true;

  for (MCSubRegIterator SubRegs(RegB, QRI); SubRegs.isValid(); ++SubRegs)
    if (RegA == *SubRegs)
      return true;

  return false;
}

/// Find use with this reg, and unmark the kill flag.
static inline void unmarkKillReg(MachineInstr *MI, unsigned Reg) {
  if (MI->isDebugInstr())
    return;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);

    if (!MO.isReg())
      continue;

    if (MO.isKill() && (MO.getReg() == Reg))
      MO.setIsKill(false);
  }
}

/// Find use with this reg, and unmark the kill flag.
static inline void markKillReg(MachineInstr *MI, unsigned Reg) {
  if (MI->isDebugInstr())
    return;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);

    if (!MO.isReg())
      continue;

    if (MO.isUse() && (MO.getReg() == Reg))
      MO.setIsKill(true);
  }
}

/// We have just moved an instruction that could have changed kill patterns
/// along the path it was moved. We need to update it.
void HexagonGlobalSchedulerImpl::updateKillAlongThePath(
    MachineBasicBlock *HomeBB, MachineBasicBlock *OriginBB,
    MachineBasicBlock::instr_iterator &Head,
    MachineBasicBlock::instr_iterator &Tail,
    MachineBasicBlock::iterator &SourcePacket,
    MachineBasicBlock::iterator &TargetPacket,
    std::vector<MachineInstr *> &backtrack) {
  // This is the instruction being moved.
  MachineInstr *MI = &*Head;
  MachineBasicBlock *CurrentBB = OriginBB;
  SmallSet<unsigned, 8> KilledUseSet;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;

    if (MO.isKill())
      KilledUseSet.insert(Reg);
  }

  // If there are no kills here, we are done.
  if (KilledUseSet.empty())
    return;

  LLVM_DEBUG(dbgs() << "\n[updateKillAlongThePath]\n");
  LLVM_DEBUG(dbgs() << "\t\tInstrToMove   :\t"; MI->dump());
  LLVM_DEBUG(dbgs() << "\t\tSourceLocation:\n";
             DumpPacket(SourcePacket.getInstrIterator()));
  LLVM_DEBUG(dbgs() << "\t\tTargetPacket  :\n";
             DumpPacket(TargetPacket.getInstrIterator()));
  LLVM_DEBUG(dbgs() << "\tUpdate Kills. Need to update (" << KilledUseSet.size()
                    << ")kills. From BB (" << OriginBB->getNumber() << ")\n");
  LLVM_DEBUG(dbgs() << "\tMove path:\n");
  assert(!backtrack.empty() && "Empty back track");

  // We have pulled up an instruction, with one of its uses marked as kill.
  // If there is any other use of the same register along the move path,
  // and there are no side exits with killed register live-in along them,
  // we need to mark last use of that reg as kill.
  for (signed i = backtrack.size() - 1; i >= 0; --i) {
    LLVM_DEBUG(dbgs() << "\t\t[" << i << "]BB("
                      << backtrack[i]->getParent()->getNumber() << ")\t";
               backtrack[i]->dump());
    if (CurrentBB != backtrack[i]->getParent()) {
      LLVM_DEBUG(dbgs() << "\t\tChange BB from (" << CurrentBB->getNumber()
                        << ") to(" << backtrack[i]->getParent()->getNumber()
                        << ")\n");
      for (MachineBasicBlock::const_succ_iterator
               SI = backtrack[i]->getParent()->succ_begin(),
               SE = backtrack[i]->getParent()->succ_end();
           SI != SE; ++SI) {
        if (*SI == CurrentBB)
          continue;

        LLVM_DEBUG(dbgs() << "\t\tSide Exit:\n\t"; (*SI)->dump());
        // If any reg kill is live along this side exit, it is not
        // a kill any more.
        for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
                                                E = (*SI)->livein_end();
             I != E; ++I) {
          if (KilledUseSet.count((*I).PhysReg)) {
            LLVM_DEBUG(dbgs() << "\t\tReg (" << printReg((*I).PhysReg, QRI)
                              << ") is LiveIn along side exit.\n");
            KilledUseSet.erase((*I).PhysReg);
            unmarkKillReg(MI, (*I).PhysReg);
          }
          if (KilledUseSet.empty())
            return;
        }
      }
      CurrentBB = backtrack[i]->getParent();
    }

    // Done with the whole path.
    if (backtrack[i] == &*TargetPacket)
      return;

    // Starting the tracking. Do not update source bundle.
    // If TargetPacket == SourcePacket we have returned
    // in the previous check.
    if (backtrack[i] == &*SourcePacket)
      continue;

    // Ignore DBG_VALUE.
    if (backtrack[i]->isDebugInstr())
      continue;

    // Encountered an intermediary bundle. Process it.
    // Beware, sometimes check for backtrack[i] == TargetPacket
    // does not work, so this instruction could be one from the target bundle.
    SmallVector<unsigned, 4> Defs;
    SmallVector<unsigned, 8> Uses;
    MachineInstr *MIU = backtrack[i];
    parseOperands(MIU, Defs, Uses);

    for (SmallVector<unsigned, 8>::iterator IA = Uses.begin(), IAE = Uses.end();
         IA != IAE; ++IA) {
      if (KilledUseSet.count(*IA)) {
        // Now this is new kill point for this Reg.
        // Update the bundle, and any local uses.
        markKillReg(MIU, *IA);

        // Unmark the current MI.
        unmarkKillReg(MI, *IA);

        if (MIU->isBundle()) {
          // TODO: Can do this cleaner and faster.
          MachineBasicBlock::instr_iterator MII = MIU->getIterator();
          MachineBasicBlock::instr_iterator End = CurrentBB->instr_end();
          for (++MII; MII != End && MII->isInsideBundle(); ++MII)
            markKillReg(&*MII, *IA);
        }

        // We have updated this kill reg, if there are more, keep on going.
        KilledUseSet.erase(*IA);

        // If the set is exhausted, just leave.
        if (KilledUseSet.empty())
          return;
      }
    }
  }
}

/// This is houskeeping for bundle with instruction just added to it.
void HexagonGlobalSchedulerImpl::addInstructionToExistingBundle(
    MachineBasicBlock *HomeBB, MachineBasicBlock::instr_iterator &Head,
    MachineBasicBlock::instr_iterator &Tail,
    MachineBasicBlock::instr_iterator &NewMI,
    MachineBasicBlock::iterator &TargetPacket,
    MachineBasicBlock::iterator &NextMI,
    std::vector<MachineInstr *> &backtrack) {
  Tail = getBundleEnd(Head);
  LLVM_DEBUG(dbgs() << "\t\t\t[Add] Head home: "; DumpPacket(Head));

  // Old header to be deleted shortly.
  MachineBasicBlock::instr_iterator Outcast = Head;
  // Unbundle old header.
  if (Outcast->isBundle() && Outcast->isBundledWithSucc())
    Outcast->unbundleFromSucc();

  bool memShufDisabled = QII->getBundleNoShuf(*Outcast);

  // Create new bundle header and update MI flags.
  finalizeBundle(*HomeBB, ++Head, Tail);
  MachineBasicBlock::instr_iterator BundleMII = std::prev(Head);
  if (memShufDisabled)
    QII->setBundleNoShuf(BundleMII);
  --Head;

  LLVM_DEBUG(dbgs() << "\t\t\t[Add] New Head : "; DumpPacket(Head));

  // The old header could be listed in the back tracking,
  // so if it is, we need to update it.
  for (unsigned i = 0; i < backtrack.size(); ++i)
    if (backtrack[i] == &*Outcast)
      backtrack[i] = &*Head;

  // Same for top MI iterator.
  if (NextMI == Outcast)
    NextMI = Head;

  TargetPacket = Head;
  HomeBB->erase(Outcast);
}

/// This handles houskeeping for bundle with instruction just deleted from it.
/// We do not see the original moved instruction in here.
void HexagonGlobalSchedulerImpl::removeInstructionFromExistingBundle(
    MachineBasicBlock *HomeBB, MachineBasicBlock::instr_iterator &Head,
    MachineBasicBlock::instr_iterator &Tail,
    MachineBasicBlock::iterator &SourceLocation,
    MachineBasicBlock::iterator &NextMI, bool MovingDependentOp,
    std::vector<MachineInstr *> &backtrack) {
  // Empty BBs will be deleted shortly.
  if (HomeBB->empty()) {
    Head = MachineBasicBlock::instr_iterator();
    Tail = MachineBasicBlock::instr_iterator();
    return;
  }

  if (!SourceLocation->isBundle()) {
    LLVM_DEBUG(dbgs() << "\t\t\tOriginal instruction was not bundled.\n\t\t\t";
               SourceLocation->dump());
    // If original instruction was not bundled, and we have moved it
    // and it is in the back track, we probably want to remove it from there.
    LLVM_DEBUG(dbgs() << "\t\t\t[Rem] New head: "; backtrack.back()->dump());

    for (unsigned i = 0; i < backtrack.size(); ++i) {
      if (backtrack[i] == &*SourceLocation) {
        // By definition, this should be the last instruction in the backtrack.
        assert((backtrack[i] == backtrack.back()) && "Lost back track");
        backtrack.pop_back();
      }
      // Point the main iterator to the next instruction.
      if (NextMI == SourceLocation)
        NextMI++;
    }
    SourceLocation = MachineBasicBlock::iterator();
    Head = MachineBasicBlock::instr_iterator();
    Tail = MachineBasicBlock::instr_iterator();
    return;
  }

  // The old header, soon to be deleted.
  MachineBasicBlock::instr_iterator Outcast = SourceLocation.getInstrIterator();
  LLVM_DEBUG(dbgs() << "\t\t\t[Rem] SourceLocation after bundle update: ";
             DumpPacket(Outcast));

  // If bundle has been already destroyed. BB->splat seems to do it some times
  // but not the other.
  // We already know that SourceLocation is bundle header.
  if (!SourceLocation->isBundledWithSucc()) {
    assert(!Head->isBundledWithSucc() && !Head->isBundledWithPred() &&
           "Bad bundle");
  } else {
    Head = SourceLocation.getInstrIterator();
    Tail = getBundleEnd(Head);
    unsigned Size = 0;
    unsigned BBSizeWithDbg = 0;
    MachineBasicBlock::const_instr_iterator I(Head);
    MachineBasicBlock::const_instr_iterator E = Head->getParent()->instr_end();

    for (++I; I != E && I->isBundledWithPred(); ++I) {
      ++BBSizeWithDbg;
      if (!I->isDebugInstr())
        ++Size;
    }

    LLVM_DEBUG(dbgs() << "\t\t\t[Rem] Size(" << Size << ") Head orig: ";
               DumpPacket(Head));
    // The old header, soon to be deleted.
    Outcast = Head;

    // The old Header is still counted here.
    if (Size > 1) {
      if (Outcast->isBundle() && Outcast->isBundledWithSucc())
        Outcast->unbundleFromSucc();

      bool memShufDisabled = QII->getBundleNoShuf(*Outcast);
      // The finalizeBundle() assumes that "original" sequence
      // it is finalizing is sequentially correct. That basically
      // means that swap case might not be handled properly.
      // I find insert point for the pull-up instruction myself,
      // and I should try to catch that swap case there, and refuse
      // to insert if I cannot guarantee correct serial semantics.
      // In the future, I need my own incremental "inserToBundle"
      // function.
      finalizeBundle(*HomeBB, ++Head, Tail);
      MachineBasicBlock::instr_iterator BundleMII = std::prev(Head);
      if (memShufDisabled)
        QII->setBundleNoShuf(BundleMII);

      --Head;
    } else if (Size == 1) {
      // There is only one non-debug instruction in the bundle.
      if (BBSizeWithDbg > 1) {
        // There are some debug instructions that should be unbundled too.
        MachineBasicBlock::instr_iterator I(Head);
        MachineBasicBlock::instr_iterator E = Head->getParent()->instr_end();
        for (++I; I != E && I->isBundledWithPred(); ++I) {
          I->unbundleFromPred();
          // Set Head to the non-debug instruction.
          if (!I->isDebugInstr())
            Head = I;
        }
      } else {
        // This means that only one original instruction is
        // left in the bundle. We need to "unbundle" it because the
        // rest of API will not like it.
        ++Head;
        if (Head->isBundledWithPred())
          Head->unbundleFromPred();
        if (Head->isBundledWithSucc())
          Head->unbundleFromSucc();
      }
    } else
      llvm_unreachable("Corrupt bundle");
  }

  LLVM_DEBUG(dbgs() << "\t\t\t[Rem] New Head : "; DumpPacket(Head));
  SourceLocation = Head;

  // The old header could be listed in the back tracking,
  // so if it is, we need to update it.
  for (unsigned i = 0; i < backtrack.size(); ++i)
    if (backtrack[i] == &*Outcast)
      backtrack[i] = &*Head;

  // Same for top MI iterator.
  if (NextMI == Outcast)
    NextMI = Head;

  HomeBB->erase(Outcast);
}

#ifndef NDEBUG
static void debugLivenessForBB(const MachineBasicBlock *MBB,
                               const TargetRegisterInfo *TRI) {
  LLVM_DEBUG(dbgs() << "\tLiveness for BB:\n"; MBB->dump());
  for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
                                              SE = MBB->succ_end();
       SI != SE; ++SI) {
    LLVM_DEBUG(dbgs() << "\tSuccessor BB (" << (*SI)->getNumber() << "):");
    for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
                                            E = (*SI)->livein_end();
         I != E; ++I)
      LLVM_DEBUG(dbgs() << "\t" << printReg((*I).PhysReg, TRI));
    LLVM_DEBUG(dbgs() << "\n");
  }
}
#endif

// Blocks should be considered empty if they contain only debug info;
// else the debug info would affect codegen.
static bool IsEmptyBlock(MachineBasicBlock *MBB) {
  if (MBB->empty())
    return true;
  for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
       MBBI != MBBE; ++MBBI) {
    if (!MBBI->isDebugInstr())
      return false;
  }
  return true;
}

/// Treat given instruction as a branch, go through its operands
/// and see if any of them is a BB address. If so, return it.
/// Return NULL otherwise.
static inline MachineBasicBlock *getBranchDestination(MachineInstr *MI) {
  if (!MI || !MI->isBranch() || MI->isBundle())
    return NULL;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isMBB())
      return MO.getMBB();
  }
  return NULL;
}

/// Similar to HexagonInstrInfo::analyzeBranch but handles
/// serveral more general cases including parsing empty BBs when possible.
bool HexagonGlobalSchedulerImpl::AnalyzeBBBranches(MachineBasicBlock *MBB,
                                                   MachineBasicBlock *&TBB,
                                                   MachineInstr *&FirstTerm,
                                                   MachineBasicBlock *&FBB,
                                                   MachineInstr *&SecondTerm) {
  // Hexagon allowes up to two jumps in MBB.
  FirstTerm = NULL;
  SecondTerm = NULL;

  LLVM_DEBUG(dbgs() << "\n\t\tAnalyze Branches in BB(" << MBB->getNumber()
                    << ")\n");
  if (MBB->succ_size() == 0) {
    LLVM_DEBUG(dbgs() << "\n\t\tBlock has no successors.\n");
    return true;
  }
  // Find both jumps.
  // We largely rely on implied assumption that BB branching always
  // looks like this:
  //  J2_jumpf %P0, <BB#60>, %PC<imp-def>;
  //  J2_jump <BB#49>
  // Branches also could be in different packets.
  MachineBasicBlock::instr_iterator MIB = MBB->instr_begin();
  MachineBasicBlock::instr_iterator MIE = MBB->instr_end();
  MachineBasicBlock::instr_iterator MII = MIB;

  if (QII->nonDbgBBSize(MBB) == 1) {
    MII = MBB->getFirstNonDebugInstr().getInstrIterator();
    if (MII->isBranch())
      FirstTerm = &*MII;
  } else {
    // We have already eliminated the case when MIB == MIE.
    while (MII != MIE) {
      if (!MII->isBundle() && MII->isBranch()) {
        if (!FirstTerm)
          FirstTerm = &*MII;
        else
          SecondTerm = &*MII;
      }
      ++MII;
    }
  }
  if ((FirstTerm && FirstTerm->isIndirectBranch()) ||
      (SecondTerm && SecondTerm->isIndirectBranch())) {
    LLVM_DEBUG(dbgs() << "\n\t\tCannot analyze BB with indirect branch.");
    return true;
  }
  if ((FirstTerm && FirstTerm->getOpcode() == Hexagon::J2_jump &&
       !FirstTerm->getOperand(0).isMBB()) ||
      (SecondTerm && SecondTerm->getOpcode() == Hexagon::J2_jump &&
       !SecondTerm->getOperand(0).isMBB())) {
    LLVM_DEBUG(
        dbgs() << "\n\t\tCannot analyze BB with a branch out of function.");
    return true;
  }

  // Now try to analyze this branch.
  SmallVector<MachineOperand, 4> Cond;
  if (QII->analyzeBranch(*MBB, TBB, FBB, Cond, false)) {
    LLVM_DEBUG(dbgs() << "\t\tFail to analyze with analyzeBranch.\n");
    LLVM_DEBUG(dbgs() << "\t\tFirst term: "; if (FirstTerm) FirstTerm->dump();
               else dbgs() << "None\n";);
    // Could not analyze it. See if this is something we can recognize.
    TBB = getBranchDestination(FirstTerm);
  }
  // There are several cases not handled by HexagonInstrInfo::analyzeBranch.
  if (!TBB) {
    LLVM_DEBUG(dbgs() << "\t\tMissing TBB.\n");
    // There is a branch, but TBB is not found.
    // The BB could also be empty at this point. See if it is a trivial
    // layout case.
    if (MBB->succ_size() == 1) {
      TBB = *MBB->succ_begin();
      LLVM_DEBUG(dbgs() << "\t\tFall through TBB(" << TBB->getNumber()
                        << ").\n");
      return false;
    } else if (MBB->succ_size() == 2) {
      // This should cover majority of remaining cases.
      if (FirstTerm && SecondTerm &&
          (QII->isPredicated(*FirstTerm) || QII->isNewValueJump(*FirstTerm)) &&
          !QII->isPredicated(*SecondTerm)) {
        TBB = getBranchDestination(FirstTerm);
        FBB = getBranchDestination(SecondTerm);
        LLVM_DEBUG(dbgs() << "\t\tCanonical dual jump layout: TBB("
                          << TBB->getNumber() << ") FBB(" << FBB->getNumber()
                          << ").\n");
        return false;
      } else if (SecondTerm && SecondTerm->getOpcode() == Hexagon::J2_jump &&
                 SecondTerm->getOperand(0).isMBB()) {
        // Look at the second term if I know it, to find out what is the fall
        // through for this BB.
        FBB = SecondTerm->getOperand(0).getMBB();
        assert(MBB->succ_size() == 2 && "Expected exactly 2 successors");
        MachineBasicBlock *Succ0 = *MBB->succ_begin();
        MachineBasicBlock *Succ1 = *std::next(MBB->succ_begin());
        if (FBB == Succ0)
          TBB = Succ1;
        else
          TBB = Succ0;
        LLVM_DEBUG(dbgs() << "\t\tSecond br is J2_jump TBB(" << TBB->getNumber()
                          << ") FBB(" << FBB->getNumber() << ").\n");
        return false;
      } else {
        // This might be an empty BB but still with two
        // successors set. Try to use CFG layout to sort it out.
        // This could happen when last jump was pulled up from a BB, and
        // CFG is being updated. At that point this method is called and
        // returns best guess possible for TBB/FBB. Fortunately order of those
        // is irrelevant, and rather used a worklist for CFG update.
        MachineFunction::iterator MBBIter = MBB->getIterator();
        MachineFunction &MF = *MBB->getParent();
        (void)MF; // supress compiler warning
        // If there are no other clues, assume next sequential BB
        // in CFG as FBB.
        ++MBBIter;
        assert(MBBIter != MF.end() && "I give up.");
        FBB = &(*MBBIter);
        assert(MBB->succ_size() == 2 && "Expected exactly 2 successors");
        MachineBasicBlock *S0 = *MBB->succ_begin();
        MachineBasicBlock *S1 = *std::next(MBB->succ_begin());
        if (FBB == S0)
          TBB = S1;
        else if (FBB == S1) {
          TBB = S0;
        } else {
          // This case can arise when the layout successor basic block (++IMBB)
          // got empty during pull-up.
          // As a result, ++IMBB is not one of MBB's successors.
          MBBIter = MF.begin();
          while (!MBB->isSuccessor(&*MBBIter) && (MBBIter != MF.end()))
            ++MBBIter;
          assert(MBBIter != MF.end() && "Malformed BB with invalid successors");
          FBB = &*MBBIter;
          if (FBB == S0)
            TBB = S1;
          else
            TBB = S0;
        }
        LLVM_DEBUG(dbgs() << "\t\tUse layout TBB(" << TBB->getNumber()
                          << ") FBB(" << FBB->getNumber() << ").\n");
        return false;
      }
    }
    assert(!FirstTerm && "Bad BB");
    return true;
  }
  // Ok, we have TBB, but maybe missing FBB.
  if (!FBB && SecondTerm) {
    LLVM_DEBUG(dbgs() << "\t\tMissing FBB.\n");
    // analyzeBranch could lie to us, ignore it in this case.
    // For the canonical case simply take known branch targets.
    if ((QII->isPredicated(*FirstTerm) || QII->isNewValueJump(*FirstTerm)) &&
        !QII->isPredicated(*SecondTerm)) {
      FBB = getBranchDestination(SecondTerm);
    } else {
      // Second term is also predicated.
      // Use CFG layout. Assign layout successor as FBB.
      for (MachineBasicBlock *Succ : MBB->successors()) {
        if (MBB->isLayoutSuccessor(Succ))
          FBB = Succ;
      }
      if (FBB == NULL) {
        LLVM_DEBUG(dbgs() << "\nNo layout successor found.");
        LLVM_DEBUG(dbgs() << "Possibly the layout successor is an empty BB");
        return true;
      }
      if (TBB == FBB)
        LLVM_DEBUG(dbgs() << "Malformed branch with useless branch condition";);
    }
    LLVM_DEBUG(dbgs() << "\t\tSecond term: "; SecondTerm->dump());
  } else if (TBB && !FBB) {
    // If BB ends in endloop, and it is a single BB hw loop,
    // we will have a single terminator, but we can figure FBB
    // easily from CFG.
    if (MBB->succ_size() == 2) {
      MachineBasicBlock *S0 = *MBB->succ_begin();
      MachineBasicBlock *S1 = *std::next(MBB->succ_begin());
      if (TBB == S0)
        FBB = S1;
      else
        FBB = S0;
    }
  }

  LLVM_DEBUG(dbgs() << "\t\tFinal TBB(" << TBB->getNumber() << ").\n";
             if (FBB) dbgs() << "\t\tFinal FBB(" << FBB->getNumber() << ").\n";
             else dbgs() << "\t\tFinal FBB(None)\n";);
  return false;
}

/// updateBranches - Updates all branches to \p From in the basic block \p
/// InBlock to branches to \p To.
static void updateBranches(MachineBasicBlock &InBlock, MachineBasicBlock *From,
                           MachineBasicBlock *To) {
  for (MachineBasicBlock::instr_iterator BI = InBlock.instr_begin(),
                                         E = InBlock.instr_end();
       BI != E; ++BI) {
    MachineInstr *Inst = &*BI;
    // Ignore anything that is not a branch.
    if (!Inst->isBranch())
      continue;
    for (MachineInstr::mop_iterator OI = Inst->operands_begin(),
                                    OE = Inst->operands_end();
         OI != OE; ++OI) {
      MachineOperand &Opd = *OI;
      // Look for basic block "From".
      if (!Opd.isMBB() || Opd.getMBB() != From)
        continue;
      // Update it.
      Opd.setMBB(To);
    }
  }
}

/// Rewrite all predecessors of the old block to go to the fallthrough
/// instead.
/// NB: Collect predecessors into a snapshot vector before iterating to
/// avoid iterator invalidation on MBB's predecessor list. Each call to
/// ReplaceUsesOfBlockWith modifies both the successor list of Pred and
/// the predecessor list of MBB, which invalidates debug-mode iterators
/// (detected by _GLIBCXX_DEBUG).
static void updatePredecessors(MachineBasicBlock &MBB,
                               MachineBasicBlock *MFBB) {
  MachineFunction &MF = *MBB.getParent();

  if (MFBB->getIterator() == MF.end())
    return;

  // Snapshot the predecessor list to avoid iterator invalidation.
  SmallVector<MachineBasicBlock *, 4> Preds(MBB.pred_begin(), MBB.pred_end());
  for (MachineBasicBlock *Pred : Preds) {
    if (!Pred->isSuccessor(&MBB))
      continue;
    Pred->ReplaceUsesOfBlockWith(&MBB, MFBB);
    updateBranches(*Pred, &MBB, MFBB);
  }
}

static void UpdateCFG(MachineBasicBlock *HomeBB, MachineBasicBlock *OriginBB,
                      MachineInstr *MII, MachineBasicBlock *HomeTBB,
                      MachineBasicBlock *HomeFBB, MachineInstr *FTA,
                      MachineInstr *STA,
                      const MachineBranchProbabilityInfo *MBPI) {
  MachineBasicBlock *S2Add = NULL, *S2Remove = NULL;
  bool RemoveLSIfPresent = false;
  if ((&*MII == FTA) && MII->isConditionalBranch()) {
    LLVM_DEBUG(dbgs() << "\nNew firstterm conditional jump added to HomeBB";);
    S2Add = HomeTBB;
    S2Remove = HomeTBB;
  } else if ((&*MII == STA) && MII->isConditionalBranch()) {
    LLVM_DEBUG(dbgs() << "\nNew secondterm conditional jump added to HomeBB";);
    // AnalyzeBBBranches might not give correct information in this case.
    // The branch destination may be a symbol, not necessarily a block.
    if (MachineBasicBlock *Dest = getBranchDestination(MII)) {
      LLVM_DEBUG(dbgs() << "\nBranch destination for pulled instruction is BB#"
                        << Dest->getNumber(););
      S2Add = Dest;
      S2Remove = Dest;
    }
  } else if ((&*MII == FTA) && MII->isUnconditionalBranch()) {
    LLVM_DEBUG(dbgs() << "\nNew firstterm unconditional jump added to HomeBB";);
    S2Add = HomeTBB;
    S2Remove = HomeTBB;
    RemoveLSIfPresent = true;
  } else if ((&*MII == STA) && MII->isUnconditionalBranch()) {
    LLVM_DEBUG(
        dbgs() << "\nNew secondterm unconditional jump added to HomeBB";);
    S2Add = HomeFBB;
    S2Remove = HomeFBB;
    RemoveLSIfPresent = true;
  }
  if (S2Add && !HomeBB->isSuccessor(S2Add)) {
    HomeBB->addSuccessor(S2Add, MBPI->getEdgeProbability(OriginBB, S2Add));
  }
  if (S2Remove)
    OriginBB->removeSuccessor(S2Remove);
  if (RemoveLSIfPresent) {
    MachineFunction::iterator HomeBBLS = HomeBB->getIterator();
    ++HomeBBLS;
    if (HomeBBLS != HomeBB->getParent()->end() &&
        HomeBB->isLayoutSuccessor(&*HomeBBLS)) {
      LLVM_DEBUG(dbgs() << "\nRemoving LayoutSucc BB#" << HomeBBLS->getNumber()
                        << "from list of successors";);
      HomeBB->removeSuccessor(&*HomeBBLS);
    }
  }
}

/// Move instruction from/to BB, Update liveness info,
/// return pointer to the newly inserted and modified
/// instruction.
MachineInstr *HexagonGlobalSchedulerImpl::MoveAndUpdateLiveness(
    BasicBlockRegion *CurrentRegion, MachineBasicBlock *HomeBB,
    MachineInstr *InstrToMove, bool NeedToNewify, unsigned DepReg,
    bool MovingDependentOp, MachineBasicBlock *OriginBB,
    MachineInstr *OriginalInstruction, SmallVector<MachineOperand, 4> &Cond,
    MachineBasicBlock::iterator &SourceLocation,
    MachineBasicBlock::iterator &TargetPacket,
    MachineBasicBlock::iterator &NextMI,
    std::vector<MachineInstr *> &backtrack) {
  LLVM_DEBUG(
      dbgs() << "\n...............[MoveAndUpdateLiveness]..............\n");
  LLVM_DEBUG(dbgs() << "\t\tInstrToMove        :\t"; InstrToMove->dump());
  LLVM_DEBUG(dbgs() << "\t\tOriginalInstruction:\t";
             OriginalInstruction->dump());
  LLVM_DEBUG(dbgs() << "\t\tSourceLocation     :\t";
             DumpPacket(SourceLocation.getInstrIterator()));
  LLVM_DEBUG(dbgs() << "\t\tTargetPacket       :\t";
             DumpPacket(TargetPacket.getInstrIterator()));

  MachineBasicBlock::instr_iterator OriginalHead =
      SourceLocation.getInstrIterator();
  MachineBasicBlock::instr_iterator OriginalTail = getBundleEnd(OriginalHead);
  MachineBasicBlock::instr_iterator OutcastFrom =
      OriginalInstruction->getIterator();

  // Remove our temporary instruction.
  MachineBasicBlock::instr_iterator kill_it(InstrToMove);
  HomeBB->erase(kill_it);

  MachineBasicBlock::instr_iterator TargetHead(TargetPacket.getInstrIterator());
  MachineBasicBlock::instr_iterator TargetTail = getBundleEnd(TargetHead);

  LLVM_DEBUG(dbgs() << "\n\tTo BB before:\n"; debugLivenessForBB(HomeBB, QRI));
  LLVM_DEBUG(dbgs() << "\n\tFrom BB before:\n";
             debugLivenessForBB(OriginBB, QRI));

  // Before we perform the move, we need to collect the worklist
  // of BBs for liveness updated.
  std::list<MachineBasicBlock *> WorkList;

  // Insert into the work list all BBs along the backtrace.
  for (std::vector<MachineInstr *>::iterator RI = backtrack.begin(),
                                             RIE = backtrack.end();
       RI != RIE; RI++)
    WorkList.push_back((*RI)->getParent());

  // Only keep unique entries.
  // TODO: Use a different container here.
  WorkList.unique();

  // Move the original instruction.
  // If this instruction is inside a bundle, update the bundle.
  MachineBasicBlock::instr_iterator BBEnd =
      TargetHead->getParent()->instr_end();
  bool LastInstructionInBundle = false;
  MachineBasicBlock::instr_iterator MII = findInsertPositionInBundle(
      TargetPacket, &*OutcastFrom, LastInstructionInBundle);

  (void)BBEnd;
  LLVM_DEBUG(dbgs() << "\n\t\t\tHead target      : "; DumpPacket(TargetHead));
  LLVM_DEBUG(dbgs() << "\t\t\tTail target        : ";
             DumpPacket(TargetTail, BBEnd));
  LLVM_DEBUG(dbgs() << "\t\t\tInsert right before: "; DumpPacket(MII, BBEnd));

  MIBundleBuilder Bundle(&*TargetHead);

  // Actual move. One day liveness might be updated here.
  if (OriginalInstruction->isBundled()) {
    Bundle.insert(MII, OriginalInstruction->removeFromBundle());
    --MII;
  } else {
    // This is one case currently unhandled by Bundle.insert
    // and needs to be fixed upstream. Meanwhile use old way to handle
    // this odd case.
    if (OriginalInstruction->getIterator() == TargetTail) {
      LLVM_DEBUG(dbgs() << "\t\t\tSpecial case move.\n");
      MachineBasicBlock::instr_iterator MIIToPred = MII;
      --MIIToPred;
      LLVM_DEBUG(dbgs() << "\t\t\tInser after        : ";
                 DumpPacket(MIIToPred, BBEnd));
      // Unbundle it in its current location.
      if (OutcastFrom->isBundledWithSucc()) {
        OutcastFrom->clearFlag(MachineInstr::BundledSucc);
        OutcastFrom->clearFlag(MachineInstr::BundledPred);
      } else if (OutcastFrom->isBundledWithPred()) {
        OutcastFrom->unbundleFromPred();
      }
      HomeBB->splice(MII, OriginBB, OutcastFrom);
      if (!MII->isBundledWithPred())
        MII->bundleWithPred();
      if (!LastInstructionInBundle && !MII->isBundledWithSucc())
        MII->bundleWithSucc();
      // This is the instruction after which we have inserted.
      if (!MIIToPred->isBundledWithSucc())
        MIIToPred->bundleWithSucc();
    } else {
      Bundle.insert(MII, OriginalInstruction->removeFromParent());
      --MII;
    }
  }
  // Source location bundle is updated later in the
  // removeInstructionFromExistingBundle().

  LLVM_DEBUG(dbgs() << "\t\t\tNew packet head: "; DumpPacket(TargetHead));
  LLVM_DEBUG(dbgs() << "\t\t\tInserted op    : "; MII->dump());
  LLVM_DEBUG(dbgs() << "\n\tTo BB after move:\n";
             debugLivenessForBB(HomeBB, QRI));
  LLVM_DEBUG(dbgs() << "\n\tFrom BB after:\n";
             debugLivenessForBB(OriginBB, QRI));

  // Update kill patterns. Do it before we have predicated the moved
  // instruction.
  updateKillAlongThePath(HomeBB, OriginBB, MII, TargetTail, SourceLocation,
                         TargetPacket, backtrack);
  // I need to know:
  // - true/false predication
  // - do I need to .new it?
  // - do I need to .old it?
  // If the original instruction used new value operands,
  // it might need to be changed to the generic form
  // before further processing.
  if (QII->isDotNewInst(*MII)) {
    DemoteToDotOld(&*MII);
    LLVM_DEBUG(dbgs() << "\t\t\tDemoted to .old\t:"; MII->dump());
  }

  // We have previously checked whether this instruction could
  // be placed in this packet, including all possible transformations
  // it might need, so if any request will fail now, something is wrong.
  //
  // Need for predication and the exact condition is determined by
  // the path between original and current instruction location.
  if (!Cond.empty()) { // To be predicated
    LLVM_DEBUG(dbgs() << "\t\t\tPredicating:"; MII->dump());
    assert(TII->isPredicable(*MII) && "MII is not predicable");
    TII->PredicateInstruction(*MII, Cond);
    if (NeedToNewify) {
      assert((DepReg < std::numeric_limits<unsigned>::max()) &&
             "Invalid pred reg value");
      LLVM_DEBUG(dbgs() << "\t\t\tNeeds to NEWify on Reg("
                        << printReg(DepReg, QRI) << ").\n");
      int NewOpcode = QII->getDotNewPredOp(*MII, MBPI);
      MII->setDesc(QII->get(NewOpcode));

      // Now we need to mark newly created predicate operand as
      // internal read.
      // TODO: Better look for predicate operand.
      for (unsigned i = 0, e = MII->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MII->getOperand(i);
        if (!MO.isReg())
          continue;
        if (MO.isDef())
          continue;
        if (DepReg == MO.getReg())
          MO.setIsInternalRead();
      }
    }
    LLVM_DEBUG(dbgs() << "\t\t\tNew predicated form:\t"; MII->dump());
    // If the predicate has changed kill pattern, now we need to propagate
    // that again. This is important for liveness computation.
    updateKillAlongThePath(HomeBB, OriginBB, MII, TargetTail, SourceLocation,
                           TargetPacket, backtrack);
  }

  // Create new bundle header, remove the old one.
  addInstructionToExistingBundle(HomeBB, TargetHead, TargetTail, MII,
                                 TargetPacket, NextMI, backtrack);

  // If moved instruction was inside a bundle, update that bundle.
  removeInstructionFromExistingBundle(OriginBB, ++OriginalHead, OriginalTail,
                                      SourceLocation, NextMI, MovingDependentOp,
                                      backtrack);

  // If removed instruction could have been dependent on any
  // of the remaining ops, we need to oldify possible affected ones.
  LLVM_DEBUG(dbgs() << "\t\tTargetHead:\t"; DumpPacket(TargetHead, BBEnd));
  LLVM_DEBUG(dbgs() << "\t\tOriginalHead:\t"; DumpPacket(OriginalHead, BBEnd));
  LLVM_DEBUG(dbgs() << "\t\tOriginalInstruction:\t";
             DumpPacket(OriginalInstruction->getIterator(), BBEnd));
  LLVM_DEBUG(dbgs() << "\t\tOutcastFrom:\t"; DumpPacket(OutcastFrom, BBEnd));

  // Clean up the original source bundle on a global scope.
  if (OriginalHead != MachineBasicBlock::instr_iterator() &&
      QII->isEndLoopN(OriginalHead->getOpcode())) {
    // Single endloop left. Since it is not a real instruction,
    // we can simply add it to a non empty previous bundle, if one exist,
    // or let assembler to produce a fake bundle for it.
    LLVM_DEBUG(dbgs() << "\t\tOnly endloop in packet.\n");
    MachineBasicBlock::instr_iterator I(OriginalHead);
    if (OriginBB->begin() != I) {
      --I;
      if (I->isBundled()) {
        if (!I->isBundledWithSucc())
          I->bundleWithSucc();
        if (!OriginalHead->isBundledWithPred())
          OriginalHead->bundleWithPred();
      }
      // else we probably need to create a new bundle here.
      // SourceLocation = NULL;
    }
  } else if (MovingDependentOp &&
             OriginalHead != MachineBasicBlock::instr_iterator()) {
    if (OriginalHead->isBundled()) {
      for (MachineBasicBlock::instr_iterator J = ++OriginalHead;
           J != OriginalTail && J->isInsideBundle() && !J->isBundle(); ++J) {
        // Need to oldify it.
        if (MIsHaveTrueDependency(OriginalInstruction, &*J) &&
            QII->isDotNewInst(*J)) {
          LLVM_DEBUG(dbgs() << "\t\tDemoting to .old:\t"; J->dump());
          DemoteToDotOld(&*J);
        }
      }
    } else {
      // Single instruction left.
      if (MIsHaveTrueDependency(OriginalInstruction, &*OriginalHead) &&
          QII->isDotNewInst(*OriginalHead)) {
        LLVM_DEBUG(dbgs() << "\t\tDemoting to .old op:\t";
                   OriginalHead->dump());
        DemoteToDotOld(&*OriginalHead);
      }
    }
  }

  // Now we need to update liveness to all BBs involved
  // including those we might have "passed" through on the way here.
  LLVM_DEBUG(dbgs() << "\n\tTo BB after bundle update:\n"; HomeBB->dump());
  LLVM_DEBUG(dbgs() << "\n\n\tFrom BB after bundle update:\n";
             OriginBB->dump());

  // Update global liveness.
  LLVM_DEBUG(dbgs() << "\n\tWorkList:\t");
  for (std::list<MachineBasicBlock *>::iterator BBI = WorkList.begin(),
                                                BBIE = WorkList.end();
       BBI != BBIE; BBI++) {
    LLVM_DEBUG(dbgs() << "BB#" << (*BBI)->getNumber() << " ");
  }
  LLVM_DEBUG(dbgs() << "\n");

  do {
    MachineBasicBlock *BB = WorkList.back();
    WorkList.pop_back();
    CurrentRegion->getLivenessInfoForBB(BB)->UpdateLiveness(BB);
  } while (!WorkList.empty());

  // No need to analyze for empty BB or update CFG for same BB pullup.
  if (OriginBB == HomeBB)
    return &*TargetHead;
  // If the instruction moved was a branch we need to update the
  // successor/predecessor of OriginBB and HomeBB accordingly.
  MachineBasicBlock *HomeTBB, *HomeFBB;
  MachineInstr *FTA = NULL, *STA = NULL;
  bool HomeBBAnalyzed = !AnalyzeBBBranches(HomeBB, HomeTBB, FTA, HomeFBB, STA);
  if (MII->isBranch()) {
    if (HomeBBAnalyzed) {
      UpdateCFG(HomeBB, OriginBB, &*MII, HomeTBB, HomeFBB, FTA, STA, MBPI);
    } else {
      llvm_unreachable("Underimplememted AnalyzeBBBranches");
    }
  }
  // If we have exhausted the OriginBB clean it up.
  // Beware that we could have created dual conditional jumps, which
  // ultimately means we can have three way jumps.
  if (IsEmptyBlock(OriginBB) && !OriginBB->isEHPad() &&
      !OriginBB->hasAddressTaken() && !OriginBB->succ_empty()) {
    // Dead block? Unlikely, but check.
    LLVM_DEBUG(dbgs() << "Empty BB(" << OriginBB->getNumber() << ").\n");
    // Update region map.
    CurrentRegion->RemoveBBFromRegion(OriginBB);
    // Keep the list of empty basic blocks to be freed later.
    EmptyBBs.push_back(OriginBB);
    if (OriginBB->pred_empty() || OriginBB->succ_empty())
      return &*TargetHead;

    if (OriginBB->succ_size() == 1) {
      // Find empty block's successor.
      MachineBasicBlock *CommonFBB = *OriginBB->succ_begin();
      updatePredecessors(*OriginBB, CommonFBB);
      // Remove the only successor entry for empty BB.
      OriginBB->removeSuccessor(CommonFBB);
    } else {
      // Three way branching is not yet fully supported.
      assert((OriginBB->succ_size() == 2) && "Underimplemented 3way branch.");
      MachineBasicBlock *OriginTBB, *OriginFBB;
      MachineInstr *FTB = NULL, *STB = NULL;

      LLVM_DEBUG(dbgs() << "\tComplex case.\n");
      if (HomeBBAnalyzed &&
          !AnalyzeBBBranches(OriginBB, OriginTBB, FTB, OriginFBB, STB)) {
        assert(OriginFBB && "Missing Origin FBB");
        if (HomeFBB == OriginBB) {
          // OriginBB is FBB for HomeBB.
          if (HomeTBB == OriginTBB) {
            // Shared TBB target, common FBB.
            updatePredecessors(*OriginBB, OriginFBB);
          } else if (HomeTBB == OriginFBB) {
            // Shared TBB target, common FBB.
            updatePredecessors(*OriginBB, OriginTBB);
          } else {
            // Three way branch. Add new successor to HomeBB.
            updatePredecessors(*OriginBB, OriginFBB);
            // TODO: Update the weight as well.
            // Adding the successor to make updatePredecessor happy.
            HomeBB->addSuccessor(OriginBB);
            updatePredecessors(*OriginBB, OriginTBB);
          }
        } else if (HomeTBB == OriginBB) {
          // OriginBB is TBB for HomeBB.
          if (HomeFBB == OriginTBB) {
            // Shared TBB target, common FBB.
            updatePredecessors(*OriginBB, OriginFBB);
          } else if (HomeFBB == OriginFBB) {
            // Shared TBB target, common FBB.
            updatePredecessors(*OriginBB, OriginTBB);
          } else {
            // Three way branch. Add new successor to HomeBB.
            updatePredecessors(*OriginBB, OriginFBB);
            // TODO: Update the weight as well.
            // Adding the successor to make updatePredecessor happy.
            HomeBB->addSuccessor(OriginBB);
            updatePredecessors(*OriginBB, OriginTBB);
          }
        } else
          llvm_unreachable("CFG update failed");
        // The empty BB can now be relieved of its successors.
        OriginBB->removeSuccessor(OriginFBB);
        OriginBB->removeSuccessor(OriginTBB);
      } else
        llvm_unreachable("Underimplemented analyzeBranch");
    }
    LLVM_DEBUG(dbgs() << "Updated BB(" << HomeBB->getNumber() << ").\n";
               HomeBB->dump());
  }
  return &*TargetHead;
}

// Find where inside a given bundle current instruction should be inserted.
// Instruction will be inserted _before_ this position.
MachineBasicBlock::instr_iterator
HexagonGlobalSchedulerImpl::findInsertPositionInBundle(
    MachineBasicBlock::iterator &Bundle, MachineInstr *MI, bool &LastInBundle) {
  MachineBasicBlock::instr_iterator MII = Bundle.getInstrIterator();
  MachineBasicBlock *MBB = MII->getParent();
  MachineBasicBlock::instr_iterator BBEnd = MBB->instr_end();
  MachineBasicBlock::instr_iterator FirstBranch = BBEnd;
  MachineBasicBlock::instr_iterator LastBundledInstruction = BBEnd;
  MachineBasicBlock::instr_iterator DualJumpFirstCandidate = BBEnd;

  assert(MII->isBundle() && "Missing insert location");
  bool isDualJumpSecondCandidate = IsDualJumpSecondCandidate(MI);
  LastInBundle = false;

  for (++MII; MII != BBEnd && MII->isInsideBundle() && !MII->isBundle();
       ++MII) {
    if (MII->isBranch() && (FirstBranch == BBEnd))
      FirstBranch = MII;
    // If what we insert is a dual jump, we need to find
    // first jump, and insert new instruction after it.
    if (isDualJumpSecondCandidate && IsDualJumpFirstCandidate(&*MII))
      DualJumpFirstCandidate = MII;
    LastBundledInstruction = MII;
  }

  if (DualJumpFirstCandidate != BBEnd) {
    // First respect dual jumps.
    ++DualJumpFirstCandidate;
    if (DualJumpFirstCandidate == BBEnd ||
        DualJumpFirstCandidate == LastBundledInstruction)
      LastInBundle = true;
    return DualJumpFirstCandidate;
  } else if (FirstBranch != BBEnd) {
    // If we have no dual jumps, but do have a single
    // branch in the bundle, add our new instruction
    // right before it.
    return FirstBranch;
  } else if (LastBundledInstruction != BBEnd) {
    LastInBundle = true;
    return ++LastBundledInstruction;
  } else
    llvm_unreachable("Lost in bundle");
  return MBB->instr_begin();
}

/// This function for now needs to try to insert new instruction
/// in correct serial semantics fashion - i.e. find "correct" insert
/// point for instruction as if inserting in serial sequence.
MachineBasicBlock::instr_iterator HexagonGlobalSchedulerImpl::insertTempCopy(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator &TargetPacket,
    MachineInstr *MI, bool DeleteOldCopy) {
  MachineBasicBlock::instr_iterator MII;
  MachineBasicBlock *CurrentBB = MI->getParent();

  assert(CurrentBB && "Corrupt instruction");
  // Create a temporary copy of the instruction we are considering.
  // LLVM refuses to deal with an instruction which was not inserted
  // to any BB. We can visit multiple BBs on the way "up", so we
  // create a temp copy of the original instruction and delete it later.
  // It is way cheaper than using splice and then
  // needing to undo it most of the time.
  MachineInstr *NewMI = MI->getParent()->getParent()->CloneMachineInstr(MI);
  // Make sure all bundling flags are cleared.
  if (NewMI->isBundledWithPred())
    NewMI->unbundleFromPred();
  if (NewMI->isBundledWithSucc())
    NewMI->unbundleFromSucc();

  if (DeleteOldCopy) {
    // Remove our temporary instruction.
    // MachineBasicBlock::erase method calls unbundleSingleMI()
    // prior to deletion, so we do not have to do it here.
    MachineBasicBlock::instr_iterator kill_it(MI);
    CurrentBB->erase(kill_it);
  }

  // If the original instruction used new value operands,
  // it might need to be changed to generic form
  // before further processing.
  if (QII->isDotNewInst(*NewMI))
    DemoteToDotOld(NewMI);

  // Insert new temporary instruction.
  // If this is the destination packet, insert the tmp after
  // its header. Otherwise, as second instr in BB.
  if (TargetPacket->getParent() == MBB) {
    MII = TargetPacket.getInstrIterator();

    if (MII->isBundled()) {
      bool LastInBundle = false;
      MachineBasicBlock::instr_iterator InsertBefore =
          findInsertPositionInBundle(TargetPacket, NewMI, LastInBundle);
      MIBundleBuilder Bundle(&*TargetPacket);
      Bundle.insert(InsertBefore, NewMI);
    } else
      MBB->insertAfter(MII, NewMI);
  } else {
    MII = MBB->instr_begin();

    // Skip debug instructions.
    while (MII->isDebugInstr())
      MII++;

    if (MII->isBundled()) {
      MIBundleBuilder Bundle(&*MII);
      Bundle.insert(++MII, NewMI);
    } else
      MBB->insertAfter(MII, NewMI);
  }
  return NewMI->getIterator();
}

// Check for a conditionally assigned register within the block.
bool HexagonGlobalSchedulerImpl::MIsCondAssign(MachineInstr *BMI,
                                               MachineInstr *MI,
                                               SmallVector<unsigned, 4> &Defs) {
  if (!QII->isPredicated(*BMI))
    return false;
  // Its a conditional instruction, now is it the same registers as MI?
  SmallVector<unsigned, 4> CondDefs;
  SmallVector<unsigned, 8> CondUses;
  parseOperands(BMI, CondDefs, CondUses);

  for (SmallVector<unsigned, 4>::iterator ID = Defs.begin(), IDE = Defs.end();
       ID != IDE; ++ID) {
    for (SmallVector<unsigned, 4>::iterator CID = CondDefs.begin(),
                                            CIDE = CondDefs.end();
         CID != CIDE; ++CID) {
      if (AliasingRegs(*CID, *ID)) {
        LLVM_DEBUG(dbgs() << "\tFound conditional def, can't move\n";
                   BMI->dump());
        return true;
      }
    }
  }
  return false;
}

// Returns the Union of all the elements in Set1 and
// Union of all the elements in Set2 separately.
// Constraints:
// Set1 and Set2 should contain an entry for each element in Range.
template <typename ElemType, typename IndexType>
void Unify(std::vector<ElemType> Range,
           std::map<ElemType, std::vector<IndexType>> &Set1,
           std::map<ElemType, std::vector<IndexType>> &Set2,
           std::pair<std::vector<IndexType>, std::vector<IndexType>> &UnionSet,
           unsigned union_size = 100) {
  typedef
      typename std::map<ElemType, std::vector<IndexType>>::iterator PosIter_t;
  typedef typename std::vector<IndexType>::iterator IndexIter_t;
  std::vector<IndexType> &Union1 = UnionSet.first;
  std::vector<IndexType> &Union2 = UnionSet.second;
  Union1.resize(union_size, 0);
  Union2.resize(union_size, 0);
  LLVM_DEBUG(dbgs() << "\n\t\tElements in the range:\n";);
  typename std::vector<ElemType>::iterator iter = Range.begin();
  while (iter != Range.end()) {
    if ((*iter)->isDebugInstr()) {
      ++iter;
      continue;
    }
    LLVM_DEBUG((*iter)->dump());
    PosIter_t set1_pos = Set1.find(*iter);
    assert(set1_pos != Set1.end() &&
           "Set1 should contain an entry for each element in Range.");
    IndexIter_t set1idx = set1_pos->second.begin();
    while (set1idx != set1_pos->second.end()) {
      Union1[*set1idx] = 1;
      ++set1idx;
    }
    PosIter_t set2_pos = Set2.find(*iter);
    assert(set2_pos != Set2.end() &&
           "Set2 should contain an entry for each element in Range.");
    IndexIter_t set2idx = set2_pos->second.begin();
    while (set2idx != set2_pos->second.end()) {
      Union2[*set2idx] = 1;
      ++set2idx;
    }
    ++iter;
  }
}

static void UpdateBundle(MachineInstr *BundleHead) {
  assert(BundleHead->isBundle() && "Not a bundle header");
  if (!BundleHead)
    return;
  unsigned Size = BundleHead->getBundleSize();
  if (Size >= 2)
    return;
  if (Size == 1) {
    MachineBasicBlock::instr_iterator MIter = BundleHead->getIterator();
    MachineInstr *MI = &*(++MIter);
    MI->unbundleFromPred();
  }
  BundleHead->eraseFromParent();
}

/// Gatekeeper for instruction speculation.
/// If all MI defs are dead (not live-in) to any other
/// BB but the one we are moving into, and it could not cause
/// exception by early execution, allow it to be pulled up.
bool HexagonGlobalSchedulerImpl::canMIBeSpeculated(
    MachineInstr *MI, MachineBasicBlock *ToBB, MachineBasicBlock *FromBB,
    std::vector<MachineInstr *> &backtrack) {
  // For now disallow memory accesses from speculation.
  // Generally we can check if they potentially may trap/cause an exception.
  if (!EnableSpeculativePullUp || !MI || MI->mayStore())
    return false;

  LLVM_DEBUG(dbgs() << "\t[canMIBeSpeculated] From BB(" << FromBB->getNumber()
                    << "):\t";
             MI->dump());
  LLVM_DEBUG(dbgs() << "\tTo this BB:\n"; ToBB->dump());

  if (!ToBB->isSuccessor(FromBB))
    return false;

  // This is a very tricky topic. Speculating arithmetic instructions with
  // results dead out of a loop more times then required by number of
  // iterations is safe, while speculating loads can cause an exception.
  // Simplest of checks is to not cross loop exit edge, or in our case
  // do not pull-in to a loop exit BB, but there are implications for
  // non-natural loops (not recognized by LLVM as loops) and multi-threaded
  // code.
  if (AllowSpeculateLoads && MI->mayLoad()) {
    // Invariant loads should always be safe.
    if (!MI->isDereferenceableInvariantLoad())
      return false;
    LLVM_DEBUG(dbgs() << "\tSpeculating a Load.\n");
  }

  SmallVector<unsigned, 4> Defs;
  SmallVector<unsigned, 8> Uses;
  parseOperands(MI, Defs, Uses);

  // Do not speculate instructions that modify reserved global registers.
  for (unsigned R : Defs)
    if (MRI->isReserved(R) && QRI->isGlobalReg(R))
      return false;

  for (MachineBasicBlock::const_succ_iterator SI = ToBB->succ_begin(),
                                              SE = ToBB->succ_end();
       SI != SE; ++SI) {
    // TODO: Allow an instruction (I) which 'defines' the live-in reg (R)
    // along the path when I is the first instruction to use the R.
    // i.e., I kills R before any other instruction in the BB uses it.
    // TODO: We have already parsed live sets - reuse them.
    if (*SI == FromBB)
      continue;
    LLVM_DEBUG(dbgs() << "\tTarget succesor BB to check:\n"; (*SI)->dump());
    LLVM_DEBUG(
        for (MachineBasicBlock::const_succ_iterator SII = (*SI)->succ_begin(),
             SIE = (*SI)->succ_end();
             SII != SIE; ++SII)(*SII)
            ->dump());
    for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
                                            E = (*SI)->livein_end();
         I != E; ++I)
      for (SmallVector<unsigned, 4>::iterator ID = Defs.begin(),
                                              IDE = Defs.end();
           ID != IDE; ++ID) {
        if (AliasingRegs((*I).PhysReg, *ID))
          return false;
      }

    // Check the successor blocks for conditional define.
    // TODO: We should really test the whole path here.
    for (MachineBasicBlock::instr_iterator BI = (*SI)->instr_begin(),
                                           E = (*SI)->instr_end();
         BI != E; ++BI) {
      if (BI->isBundle() || BI->isDebugInstr())
        continue;
      LLVM_DEBUG(dbgs() << "\t\tcheck against:\t"; BI->dump());
      if (MIsCondAssign(&*BI, MI, Defs))
        return false;
    }
  }
  // Taking a very conservative approach during speculation.
  // Traverse the path (FromBB, ToBB] and make sure
  // that the def-use set of the instruction to be moved
  // are not modified.
  std::vector<MachineBasicBlock *> PathBB;
  for (unsigned i = 0; i < backtrack.size(); ++i) {
    // Insert unique BB along the path but skip FromBB
    MachineBasicBlock *MBB = backtrack[i]->getParent();
    if ((MBB != FromBB) &&
        (std::find(PathBB.begin(), PathBB.end(), MBB) == PathBB.end()))
      PathBB.push_back(MBB);
  }
  bool WaitingForTargetPacket = true;
  MachineBasicBlock::instr_iterator MII;
  std::vector<MachineInstr *> TraversalRange;
  LLVM_DEBUG(dbgs() << "\n\tElements in the range:");
  // TODO: Use just the backtrack to get TraversalRange because it
  // contains the path (only when speculated from a path in region).
  // Note: We check the dependency of instruction-to-move with
  // all the instructions (starting from backtrack[0]) in the parent BBs
  // because a BB might have a branching from in between due to packetization
  // and just checking packets in the backtrack won't be comprehensive.
  for (unsigned i = 0; i < PathBB.size(); ++i) {
    for (MII = PathBB[i]->instr_begin(); MII != PathBB[i]->instr_end(); ++MII) {
      // Skip instructions until the target packet is found.
      // although target packet is already checked for correctness,
      // it is good to check here to validate intermediate pullups.
      if (backtrack[0] == &*MII)
        WaitingForTargetPacket = false;
      if (WaitingForTargetPacket)
        continue;
      if (MII->isBundle())
        continue;
      // TODO: Ideally we should check that there is a `linear' control flow
      // in the TraversalRange in all possible manner. For e.g.,
      // BB0 { packet1: if(p0) indirect_jump BB1;
      //      packet2: jump BB2 }
      // BB1 { i1 }. In this case we should not pull `i1' into packet2.
      if (MII->isCall() || MII->isReturn() ||
          (MII->getOpcode() == Hexagon::J2_jump && !MII->getOperand(0).isMBB()))
        return false;
      if (MI != &*MII) {
        TraversalRange.push_back(&*MII);
        LLVM_DEBUG(MII->dump(););
      }
    }
  }
  // Get the union of def/use set of all the instructions along TraversalRange.
  std::pair<std::vector<unsigned>, std::vector<unsigned>> RangeDefUse;
  Unify(TraversalRange, MIDefSet, MIUseSet, RangeDefUse, QRI->getNumRegs());
  // No instruction (along TraversalRange) should 'define' the use set of MI
  for (unsigned j = 0; j < Uses.size(); ++j)
    if (RangeDefUse.first[Uses[j]]) {
      LLVM_DEBUG(dbgs() << "\n\t\tUnresolved dependency along path to HOME for "
                        << printReg(Uses[j], QRI););
      return false;
    }
  // No instruction (along TraversalRange) should 'define' or 'use'
  // the def set of MI
  for (unsigned j = 0; j < Defs.size(); ++j)
    if (RangeDefUse.first[Defs[j]] || RangeDefUse.second[Defs[j]]) {
      LLVM_DEBUG(dbgs() << "\n\t\tUnresolved dependency along path to HOME for "
                        << printReg(Defs[j], QRI););
      return false;
    }
  return true;
}

/// Try to move InstrToMove to TargetPacket using path stored in backtrack.
/// SourceLocation is current iterator point. It must be updated to the new
/// iteration location after all updates.
/// Alogrithm:
/// To move an instruction (I) from OriginBB through HomeBB via backtrack.
/// for each packet (i) in backtrack, analyzeBranch
/// case 1 (success)
///   case Pulling from conditional branch:
///     if I is predicable
///         Try to predicate on the branch condition
///     else
///         Try to speculate I to backtrack[i].
///   case Pulling from unconditional branch:
///         Just pullup. (TODO: Speculate here as well)
/// case 2 (fails)
///     Try to speculate I backtrack[i].
bool HexagonGlobalSchedulerImpl::MoveMItoBundle(
    BasicBlockRegion *CurrentRegion,
    MachineBasicBlock::instr_iterator &InstrToMove,
    MachineBasicBlock::iterator &NextMI,
    MachineBasicBlock::iterator &TargetPacket,
    MachineBasicBlock::iterator &SourceLocation,
    std::vector<MachineInstr *> &backtrack, bool MovingDependentOp,
    bool PathInRegion) {
  MachineBasicBlock *HomeBB = TargetPacket->getParent();
  MachineBasicBlock *OriginBB = InstrToMove->getParent();
  MachineBasicBlock *CurrentBB = OriginBB;
  MachineBasicBlock *CleanupBB = OriginBB;
  MachineBasicBlock *PreviousBB = OriginBB;
  MachineInstr *OriginalInstructionToMove = &*InstrToMove;

  assert(HomeBB && "Missing HomeBB");
  assert(OriginBB && "Missing OriginBB");

  LLVM_DEBUG(dbgs() << "\n.........[MoveMItoBundle]..............\n");
  LLVM_DEBUG(dbgs() << "\t\tInstrToMove   :\t"; InstrToMove->dump());
  LLVM_DEBUG(dbgs() << "\t\tTargetPacket  :\t";
             DumpPacket(TargetPacket.getInstrIterator()));
  LLVM_DEBUG(dbgs() << "\t\tSourceLocation:\t";
             DumpPacket(SourceLocation.getInstrIterator()));

  // We do not allow to move instructions in the same BB.
  if (HomeBB == OriginBB) {
    LLVM_DEBUG(dbgs() << "\t\tSame BB pull-up.\n");
    if (!EnableLocalPullUp)
      return false;
  }

  if (OneFloatPerPacket && QII->isFloat(*TargetPacket) &&
      QII->isFloat(*InstrToMove))
    return false;

  if (OneComplexPerPacket && QII->isComplex(*TargetPacket) &&
      QII->isComplex(*InstrToMove))
    return false;

  LLVM_DEBUG(dbgs() << "\t\tWay home:\n");
  // Test integrity of the back track.
  for (unsigned i = 0; i < backtrack.size(); ++i) {
    assert(backtrack[i]->getParent() && "Messed back track.");
    LLVM_DEBUG(dbgs() << "\t\t[" << i << "] BB("
                      << backtrack[i]->getParent()->getNumber() << ")\t";
               backtrack[i]->dump());
  }
  LLVM_DEBUG(dbgs() << "\n");

  bool NeedCleanup = false;
  bool NeedToPredicate = false;
  bool MINeedToNewify = false;
  unsigned DepReg = std::numeric_limits<unsigned>::max();
  bool isDualJump = false;
  SmallVector<MachineOperand, 4> Cond;
  SmallVector<MachineOperand, 4> PredCond;
  std::vector<MachineInstr *> PullUpPath;
  if (PathInRegion)
    PullUpPath = backtrack;
  else {
    PullUpPath.push_back(&*TargetPacket);
    PullUpPath.push_back(&*InstrToMove);
  }

  // Now start iterating over all instructions
  // preceeding the one we are trying to move,
  // and see if they could be reodered/bypassed.
  for (std::vector<MachineInstr *>::reverse_iterator RI = backtrack.rbegin(),
                                                     RIE = backtrack.rend();
       RI < RIE; ++RI) {
    // Once most of debug will be gone, this will be a real assert.
    // assert((backtrack.front() == ToThisBundle) && "Lost my way home.");
    MachineInstr *MIWH = *RI;
    if (QII->isDotNewInst(*InstrToMove)) {
      LLVM_DEBUG(dbgs() << "Cannot move a dot new instruction:";
                 InstrToMove->dump());
      if (NeedCleanup)
        CleanupBB->erase(InstrToMove);
      return false;
    }
    if (canCauseStall(&*InstrToMove, MIWH)) {
      if (NeedCleanup)
        CleanupBB->erase(InstrToMove);
      return false;
    }
    LLVM_DEBUG(dbgs() << "\t> Step home BB(" << MIWH->getParent()->getNumber()
                      << "):\t";
               DumpPacket(MIWH->getIterator()));

    // See if we cross a jump, and possibly change the form of instruction.
    // Passing through BBs with dual jumps in different packets
    // takes extra care.
    bool isBranchMIWH = isBranch(MIWH);
    if (((&*SourceLocation != MIWH) && isBranchMIWH) ||
        (CurrentBB != MIWH->getParent())) {
      LLVM_DEBUG(dbgs() << "\tChange BB from(" << CurrentBB->getNumber()
                        << ") to (" << MIWH->getParent()->getNumber() << ")\n");
      PreviousBB = CurrentBB;
      CurrentBB = MIWH->getParent();

      // See what kind of branch we are dealing with.
      MachineBasicBlock *PredTBB = NULL;
      MachineBasicBlock *PredFBB = NULL;

      if (QII->analyzeBranch(*CurrentBB, PredTBB, PredFBB, Cond, false)) {
        // We currently do not handle NV jumps of this kind:
        // if (cmp.eq(r0.new, #0)) jump:t .LBB12_69
        // TODO: Need to handle them.
        LLVM_DEBUG(dbgs() << "\tCould not analyze branch.\n");

        // This is the main point of lost performance.
        // We could try to speculate here, but for that we need accurate
        // liveness info, and it is not ready yet.
        if (!canMIBeSpeculated(&*InstrToMove, CurrentBB, PreviousBB,
                               PullUpPath)) {
          if (NeedCleanup)
            CleanupBB->erase(InstrToMove);
          return false;
        } else {
          // Save speculated instruction moved.
          SpeculatedIns.insert(
              std::make_pair(OriginalInstructionToMove, OriginBB));
          LLVM_DEBUG(dbgs() << "\nSpeculatedInsToMove"; InstrToMove->dump());
        }

        LLVM_DEBUG(dbgs() << "\tSpeculating.\n");
        // If we are speculating, we can come through a predication
        // into an unconditional branch...
        // For now simply bail out.
        // TODO: See if this ever happens.
        if (NeedToPredicate) {
          LLVM_DEBUG(dbgs()
                     << "\tUnderimplemented pred for speculative move.\n");
          if (NeedCleanup)
            CleanupBB->erase(InstrToMove);
          return false;
        }
        InstrToMove =
            insertTempCopy(CurrentBB, TargetPacket, &*InstrToMove, NeedCleanup);
        NeedCleanup = true;
        NeedToPredicate = false;
        assert(!NeedToPredicate && "Need to handle predication for this case");
        CleanupBB = CurrentBB;
        // No need to recheck for resources - instruction did not change.
        LLVM_DEBUG(dbgs() << "\tUpdated BB:\n"; CurrentBB->dump());
      } else {
        bool LocalNeedPredication = true;
        // We were able to analyze the branch.
        if (!isBranchMIWH && !PredTBB) {
          LLVM_DEBUG(dbgs() << "\tDo not need predicate for this case.\n");
          LocalNeedPredication = false;
        }
        // First see if this is a potential dual jump situation.
        if (IsDualJumpSecondCandidate(&*InstrToMove) &&
            IsDualJumpFirstCandidate(TargetPacket)) {
          LLVM_DEBUG(dbgs() << "\tPerforming unrestricted dual jump.\n");
          isDualJump = true;
        } else if (LocalNeedPredication && (PredFBB != PreviousBB)) {
          // Predicate instruction based on condition feeding it.
          // This is generally a statefull pull-up path.
          // Can this insn be predicated? If so, try to do it.
          if (TII->isPredicable(*InstrToMove)) {
            if (PredTBB) {
              if (PreviousBB != PredTBB) {
                // If we "came" not from TBB, we need to invert condition.
                if (TII->reverseBranchCondition(Cond)) {
                  LLVM_DEBUG(dbgs() << "\tUnable to invert condition.\n");
                  if (NeedCleanup)
                    CleanupBB->erase(InstrToMove);
                  return false;
                }
              }
              LLVM_DEBUG(dbgs() << "\tTBB(" << PredTBB->getNumber()
                                << ")InvertCondition("
                                << (PreviousBB != PredTBB) << ")\n");
            }
            // Create a new copy of the instruction we are trying to move.
            // It changes enough (new BB, predicated form) and untill we
            // reach home, we do not even know if it is going to work.
            InstrToMove = insertTempCopy(CurrentBB, TargetPacket, &*InstrToMove,
                                         NeedCleanup);
            NeedCleanup = true;
            NeedToPredicate = true;
            CleanupBB = CurrentBB;

            if (PredCond.empty() && // If not already predicated.
                TII->PredicateInstruction(*InstrToMove, Cond)) {
              LLVM_DEBUG(dbgs() << "\tNew predicated insn:\t";
                         InstrToMove->dump());
              // After predication some instruction could become const extended:
              // L2_loadrigp == "$dst=memw(#$global)"
              // L4_ploadrit_abs == "if ($src1) $dst=memw(##$global)"
              // Resource checking for those is different.
              if ((QII->isExtended(*InstrToMove) ||
                   QII->isConstExtended(*InstrToMove) ||
                   isJumpOutOfRange(&*InstrToMove)) &&
                  !tryAllocateResourcesForConstExt(&*InstrToMove, false)) {
                // If we cannot, do not modify the state.
                LLVM_DEBUG(dbgs()
                           << "\tEI Could not be added to the packet.\n");
                CleanupBB->erase(InstrToMove);
                return false;
              }

              if (!ResourceTracker->canReserveResources(*InstrToMove) ||
                  !shouldAddToPacket(*InstrToMove)) {
                // It will not fit in its new form...
                LLVM_DEBUG(dbgs() << "\tCould not be added in its new form.\n");
                CurrentBB->erase(InstrToMove);
                return false;
              }

              // Need also verify that we can newify it if we want to.
              if (NeedToNewify(InstrToMove, &DepReg, &*TargetPacket)) {
                if (isNewifiable(InstrToMove, DepReg, &*TargetPacket)) {
                  MINeedToNewify = true;
                  LLVM_DEBUG(dbgs() << "\t\t\tNeeds to NEWify on Reg("
                                    << printReg(DepReg, QRI) << ").\n");
                } else {
                  LLVM_DEBUG(dbgs() << "\tNon newifiable in this bundle: ";
                             InstrToMove->dump());
                  CleanupBB->erase(InstrToMove);
                  return false;
                }
              }

              LLVM_DEBUG(dbgs() << "\tUpdated BB:\n"; CurrentBB->dump());
              PredCond = Cond;
              // Now the instruction uses the pred-reg as well.
              if (!Cond.empty() && (Cond.size() == 2)) {
                MIUseSet[OriginalInstructionToMove].push_back(Cond[1].getReg());
              }
              assert(((Cond.size() <= 2) &&
                      !(QII->isNewValueJump(Cond[0].getImm()))) &&
                     "Update MIUseSet for new-value compare jumps");
            } else {
              LLVM_DEBUG(dbgs() << "\tCould not predicate it\n");
              LLVM_DEBUG(dbgs() << "\tTrying to speculate!\t";
                         InstrToMove->dump());
              bool DistantSpeculation = false;
              std::vector<MachineInstr *> NonPredPullUpPath;
              unsigned btidx = 0;
              // Generate a backtrack path for instruction to be speculated.
              // Original backtrack may start from a different (ancestor)
              // target packet.
              while (btidx < backtrack.size()) {
                const MachineBasicBlock *btBB = backtrack[btidx]->getParent();
                if ((btBB == PreviousBB) || (btBB == CurrentBB))
                  NonPredPullUpPath.push_back(backtrack[btidx]);
                ++btidx;
              }
              // Speculate only to immediate predecessor.
              if (PreviousBB != CurrentBB) {
                if (*(PreviousBB->pred_begin()) != CurrentBB) {
                  // In a region there are no side entries.
                  DistantSpeculation = true;
                  LLVM_DEBUG(dbgs()
                                 << "\n\tMI not in immediate successor of BB#"
                                 << CurrentBB->getNumber() << ", MI is in BB#"
                                 << PreviousBB->getNumber(););
                }
                assert((PreviousBB->pred_size() < 2) &&
                       "Region with a side entry");
              }
              // TODO: Speculate ins. when pulled from unlikely path.
              if (DistantSpeculation || /*!PathInRegion ||*/
                  InstrToMove->mayLoad() || InstrToMove->mayStore() ||
                  InstrToMove->hasUnmodeledSideEffects() ||
                  !canMIBeSpeculated(&*InstrToMove, CurrentBB, PreviousBB,
                                     NonPredPullUpPath)) {
                CleanupBB->erase(InstrToMove);
                return false;
              } else {
                // Save speculated instruction moved.
                NeedToPredicate = false;
                SpeculatedIns.insert(
                    std::make_pair(OriginalInstructionToMove, OriginBB));
                LLVM_DEBUG(dbgs() << "\nPredicable+SpeculatedInsToMove";
                           InstrToMove->dump());
              }
            }
          } else {
            // This is a non-predicable instruction. We still can try to
            // speculate it here.
            LLVM_DEBUG(dbgs() << "\tNon predicable insn!\t";
                       InstrToMove->dump());
            // TODO: Speculate ins. when pulled from unlikely path.
            if (!SpeculateNonPredInsn || !PathInRegion ||
                InstrToMove->mayLoad() || InstrToMove->mayStore() ||
                InstrToMove->hasUnmodeledSideEffects() ||
                !canMIBeSpeculated(&*InstrToMove, CurrentBB, PreviousBB,
                                   PullUpPath)) {
              if (NeedCleanup)
                CleanupBB->erase(InstrToMove);
              return false;
            } else {
              // Save speculated instruction moved.
              SpeculatedIns.insert(
                  std::make_pair(OriginalInstructionToMove, OriginBB));
              LLVM_DEBUG(dbgs() << "\nNonPredicable+SpeculatedInsToMove";
                         InstrToMove->dump());
            }

            InstrToMove = insertTempCopy(CurrentBB, TargetPacket, &*InstrToMove,
                                         NeedCleanup);
            NeedCleanup = true;
            CleanupBB = CurrentBB;
          }
        } else {
          // No branch. Fall through.
          LLVM_DEBUG(dbgs() << "\tFall through BB.\n"
                            << "\tCurrentBB:" << CurrentBB->getNumber()
                            << "\tPreviousBB:" << PreviousBB->getNumber();
                     if (PredFBB) dbgs()
                     << "\tPredFBB:" << PredFBB->getNumber(););
          // Even though this is a fall though case, we still can
          // have a dual jump situation here with a CALL involved.
          // For now simply avoid it.
          if (IsDualJumpSecondCandidate(&*InstrToMove)) {
            llvm_unreachable("Dual jumps with known?");
            LLVM_DEBUG(dbgs() << "\tUnderimplemented dual jump formation.\n");
            if (NeedCleanup)
              CleanupBB->erase(InstrToMove);
            return false;
          }

          if (!CurrentBB->isSuccessor(PreviousBB)) {
            LLVM_DEBUG(dbgs() << "\tNon-successor fall through.\n");
            if (NeedCleanup)
              CleanupBB->erase(InstrToMove);
            return false;
          }
          SpeculatedIns.insert(
              std::make_pair(OriginalInstructionToMove, OriginBB));
          LLVM_DEBUG(dbgs() << "\nSpeculatedInsToMove+FallThroughBB";
                     InstrToMove->dump());
          // Create a temp copy.
          InstrToMove = insertTempCopy(CurrentBB, TargetPacket, &*InstrToMove,
                                       NeedCleanup);
          NeedCleanup = true;
          NeedToPredicate = false;
          CleanupBB = CurrentBB;
          LLVM_DEBUG(dbgs() << "\tUpdated BB:\n"; CurrentBB->dump());
        }
      }
    }
    // If we have reached Home, great.
    // Original check should have verified that instruction could be added
    // to the target packet, so here we do nothing for deps.
    if (MIWH == backtrack.front()) {
      LLVM_DEBUG(dbgs() << "\tHOME!\n");
      break;
    }

    // Test if we can reorder the two MIs.
    // The exception is when we are forming dual jumps - we can pull up
    // dependent instruction to the last bundle of an immediate predecesor
    // of the current BB if control flow permits it.
    // In this special case we also need to update the bundle we are moving
    // from.
    if (!(MovingDependentOp && (MIWH == &*SourceLocation)) &&
        !canReorderMIs(MIWH, &*InstrToMove)) {
      if (NeedCleanup)
        CleanupBB->erase(InstrToMove);
      return false;
    }
  }
  // We have previously tested this instruction, but has not updated the state
  // for it. Do it now.
  if (QII->isExtended(*InstrToMove) || QII->isConstExtended(*InstrToMove) ||
      isJumpOutOfRange(&*InstrToMove)) {
    if (!tryAllocateResourcesForConstExt(&*InstrToMove))
      llvm_unreachable("Missed dependency test");
  }

  // Ok. We can safely move this instruction all the way up.
  // We also potentially have a slot for it.
  // During move original instruction could have changed (becoming predicated).
  // Now try to place the final instance of it into the current packet.
  LLVM_DEBUG(dbgs() << "\nWant to move ";
             if (MovingDependentOp) dbgs() << "dependent op"; dbgs() << ": ";
             InstrToMove->dump(); dbgs() << "To BB:\n"; HomeBB->dump();
             dbgs() << "From BB:\n"; OriginBB->dump());

  // Keep these two statistics separately.
  if (!isDualJump)
    HexagonNumPullUps++;
  else
    HexagonNumDualJumps++;

  // This means we have not yet inserted the temp copy of InstrToMove
  // in the target bundle. We are probably inside the same BB.
  if (!NeedCleanup) {
    InstrToMove =
        insertTempCopy(HomeBB, TargetPacket, &*InstrToMove, NeedCleanup);
    NeedCleanup = true;
  }

  // No problems detected. Add it.
  // If we were adding InstrToMove to a single, not yet packetized
  // instruction, we need to create bundle header for it before proceeding.
  // Be carefull since endPacket also resets the DFA state.
  if (!TargetPacket->isBundle()) {
    LLVM_DEBUG(dbgs() << "\tForm a new bundle.\n");
    finalizeBundle(*HomeBB, TargetPacket.getInstrIterator(),
                   std::next(InstrToMove));
    LLVM_DEBUG(HomeBB->dump());
    // Now we need to adjust pointer to the newly created packet header.
    MachineBasicBlock::instr_iterator MII = TargetPacket.getInstrIterator();
    MII--;

    // Is it also on the way home?
    for (unsigned i = 0; i < backtrack.size(); ++i)
      if (backtrack[i] == &*TargetPacket)
        backtrack[i] = &*MII;

    // Is it where our next MI is pointing?
    if (NextMI == TargetPacket)
      NextMI = MII;
    TargetPacket = MII;
  }

  // Move and Update Liveness info.
  MoveAndUpdateLiveness(CurrentRegion, HomeBB, &*InstrToMove, MINeedToNewify,
                        DepReg, MovingDependentOp, OriginBB,
                        OriginalInstructionToMove, PredCond, SourceLocation,
                        TargetPacket, NextMI, backtrack);

  LLVM_DEBUG(dbgs() << "\n______Updated______\n"; HomeBB->dump();
             OriginBB->dump());

  return true;
}

/// Verify that we respect CFG layout during pull-up.
bool HexagonGlobalSchedulerImpl::isBranchWithinRegion(
    BasicBlockRegion *CurrentRegion, MachineInstr *MI) {
  assert(MI && MI->isBranch() && "Missing call info");

  MachineBasicBlock *MBB = MI->getParent();
  LLVM_DEBUG(dbgs() << "\t[isBranchWithinRegion] BB(" << MBB->getNumber()
                    << ") Branch instr:\t";
             MI->dump());
  // If there is only one successor, it is safe to pull.
  if (MBB->succ_size() <= 1)
    return true;
  // If there are multiple successors (jump table), we should
  // not allow pull up over this instruction.
  if (MBB->succ_size() > 2)
    return false;

  MachineBasicBlock *NextRegionBB;
  MachineBasicBlock *TBB, *FBB;
  MachineInstr *FirstTerm = NULL;
  MachineInstr *SecondTerm = NULL;

  if (AnalyzeBBBranches(MBB, TBB, FirstTerm, FBB, SecondTerm)) {
    LLVM_DEBUG(dbgs() << "\t\tAnalyzeBBBranches failed!\n");
    return false;
  }

  // If there is no jump in this BB, it simply falls through.
  if (!FirstTerm) {
    LLVM_DEBUG(dbgs() << "\t\tNo FirstTerm\n");
    return true;
  } else if (QII->isEndLoopN(FirstTerm->getOpcode())) {
    // We can easily analyze where endloop would take us
    // but here it would be pointless either way since
    // the region will not cross it.
    LLVM_DEBUG(dbgs() << "\t\tEndloop terminator\n");
    return false;
  }
  // On some occasions we see code like this:
  // BB#142: derived from LLVM BB %init, Align 4 (16 bytes)
  //  Live Ins: %R17 %R18
  //  Predecessors according to CFG: BB#2
  //  EH_LABEL <MCSym=.Ltmp35>
  //  J2_jump <BB#3>, %PC<imp-def>
  //  Successors according to CFG: BB#3(1048575) BB#138(1)
  // It breaks most assumptions about CFG layout, so untill we know
  // the source of it, let's have a safeguard.
  if (MBB->succ_size() > 1 && !TII->isPredicated(*FirstTerm) &&
      !QII->isNewValueJump(*FirstTerm)) {
    LLVM_DEBUG(dbgs() << "\t\tBadly formed BB.\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "\t\tFirstTerm:  "; FirstTerm->dump());
  LLVM_DEBUG(dbgs() << "\t\tSecondTerm: "; if (SecondTerm) SecondTerm->dump();
             else dbgs() << "None\n";);

  // All cases where there is only one branch in BB are OK to proceed.
  if (!SecondTerm)
    return true;

  assert(!QII->isEndLoopN(SecondTerm->getOpcode()) && "Found endloop.");

  // Find next BB in this region - if there is none, we will likely
  // stop pulling in the next check outside of this function.
  // This largely is don't care.
  NextRegionBB = CurrentRegion->findNextMBB(MBB);
  if (!NextRegionBB) {
    LLVM_DEBUG(dbgs() << "\t\tNo next BB in the region...\n");
    return true;
  }
  LLVM_DEBUG(dbgs() << "\t\tNextRegionBB(" << NextRegionBB->getNumber()
                    << ")\n");
  assert(TBB && "Corrupt BB layout");
  // This means we are trying to pull into a packet _before_ the first
  // branch in the MBB.
  if (MI == FirstTerm) {
    LLVM_DEBUG(dbgs() << "\t\tTBB(" << TBB->getNumber()
                      << ") NextBB in the region(" << NextRegionBB->getNumber()
                      << ")\n");
    return (TBB == NextRegionBB);
  }
  assert(FBB && "Corrupt BB layout");
  // This means we are trying to pull into the packet _after_ first branch,
  // and it is OK if we pull from the second branch target.
  // This pull is always speculative.
  if ((MI != SecondTerm)) {
    LLVM_DEBUG(dbgs() << "\t\tDual terminator not matching SecondTerm.\n");
    return false;
  }
  // Analyze the second branch in the BB.
  LLVM_DEBUG(dbgs() << "\t\tFBB(" << FBB->getNumber()
                    << ") NextBB in the region(" << NextRegionBB->getNumber()
                    << ")\n");
  return (FBB == NextRegionBB);
}

/// Check if a given instruction is:
/// - a jump to a distant target
/// - that exceeds its immediate range
/// If both conditions are true, it requires constant extension.
bool HexagonGlobalSchedulerImpl::isJumpOutOfRange(MachineInstr *UnCond,
                                                  MachineInstr *Cond) {
  if (!UnCond || !UnCond->isBranch())
    return false;

  MachineBasicBlock *UnCondBB = UnCond->getParent();
  MachineBasicBlock *CondBB = Cond->getParent();
  MachineInstr *FirstTerm = &*(CondBB->getFirstInstrTerminator());
  // This might be worth an assert.
  if (FirstTerm == &*CondBB->instr_end())
    return false;

  unsigned InstOffset = BlockToInstOffset[UnCondBB];
  unsigned Distance = 0;

  // To save time, estimate exact position of a branch instruction
  // as one at the end of the UnCondBB.
  // Number of instructions times typical instruction size.
  InstOffset += (QII->nonDbgBBSize(UnCondBB) * HEXAGON_INSTR_SIZE);

  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  SmallVector<MachineOperand, 4> CondList;

  // Find the target of the unconditional branch in UnCondBB, which is returned
  // in TBB. Then use the CondBB to extract the FirsTerm. We desire to replace
  // the branch target in FirstTerm with the branch location from the UnCondBB,
  // provided it is within the distance of the opcode in FirstTerm.
  if (QII->analyzeBranch(*UnCondBB, TBB, FBB, CondList, false))
    // Could not analyze it. give up.
    return false;

  if (TBB && (Cond == FirstTerm)) {
    Distance =
        (unsigned)std::abs((long long)InstOffset - BlockToInstOffset[TBB]) +
        SafetyBuffer;
    return !QII->isJumpWithinBranchRange(*FirstTerm, Distance);
  }
  return false;
}

// findBundleAndBranch returns the branch instruction and the
// bundle which contains it. Null is returned if not found.
MachineInstr *HexagonGlobalSchedulerImpl::findBundleAndBranch(
    MachineBasicBlock *BB, MachineBasicBlock::iterator &Bundle) {
  // Find the conditional branch out of BB.
  if (!BB)
    return NULL;
  MachineInstr *CondBranch = NULL;
  Bundle = BB->end();
  for (MachineBasicBlock::instr_iterator MII = BB->getFirstInstrTerminator(),
                                         MBBEnd = BB->instr_end();
       MII != MBBEnd; ++MII) {
    MachineInstr *MI = &*MII;
    if (MII->isConditionalBranch()) {
      CondBranch = MI;
    }
  }
  if (!CondBranch)
    return NULL;
  MachineBasicBlock::instr_iterator MII = CondBranch->getIterator();
  if (!MII->isBundled())
    return NULL;
  // Find bundle header.
  for (--MII; MII->isBundled(); --MII)
    if (MII->isBundle()) {
      Bundle = MII;
      break;
    }
  return CondBranch;
}

// pullUpPeelBBLoop
// A single BB loop with a register copy at the beginning in its
// own bundle, benefits from eliminating the extra bundle. We do
// this by predicating the register copy in the predecessor BB, and
// again in the last bundle of the loop.
bool HexagonGlobalSchedulerImpl::pullUpPeelBBLoop(MachineBasicBlock *PredBB,
                                                  MachineBasicBlock *LoopBB) {
  if (!AllowBBPeelPullUp)
    return false;
  if (!LoopBB || !PredBB)
    return false;

  // We consider single BB loops only. Check for it here.
  if (LoopBB->isEHPad() || LoopBB->hasAddressTaken())
    return false;
  if (LoopBB->succ_size() != 2)
    return false;
  if (LoopBB->pred_size() != 2)
    return false;
  // Make sure one of the successors and one of the predecssors is to self.
  if (!(LoopBB->isSuccessor(LoopBB) && LoopBB->isPredecessor(LoopBB)))
    return false;

  // Find the none self successor block. We know we only have 2 successors.
  MachineBasicBlock *SuccBB = NULL;
  for (MachineBasicBlock::succ_iterator SI = LoopBB->succ_begin(),
                                        SE = LoopBB->succ_end();
       SI != SE; ++SI)
    if (*SI != LoopBB) {
      SuccBB = *SI;
      break;
    }
  if (!SuccBB)
    return false;

  // Find the conditional branch and its bundle inside PredBB.
  MachineBasicBlock::iterator PredBundle;
  MachineInstr *PredCondBranch = NULL;
  PredCondBranch = findBundleAndBranch(PredBB, PredBundle);
  if (!PredCondBranch)
    return false;
  if (PredBundle == PredBB->end())
    return false;
  LLVM_DEBUG(dbgs() << "PredBB's Branch: ");
  LLVM_DEBUG(dbgs() << *PredCondBranch);

  // Look for leading reg copy as single bundle and make sure its live in.
  MachineBasicBlock::instr_iterator FMI = LoopBB->instr_begin();
  // Skip debug instructions.
  while (FMI->isDebugInstr())
    FMI++;

  MachineInstr *RegMI = &*FMI;
  if (RegMI->isBundle())
    return false;
  int TfrOpcode = RegMI->getOpcode();
  if (TfrOpcode != Hexagon::A2_tfr && TfrOpcode != Hexagon::A2_tfr)
    return false;
  if (!(RegMI->getOperand(0).isReg() && RegMI->getOperand(1).isReg()))
    return false;
  unsigned InLoopReg = RegMI->getOperand(1).getReg();
  if (!LoopBB->isLiveIn(InLoopReg))
    return false;

  // Create a region to pass to ResourcesAvailableInBundle.
  BasicBlockRegion PUR(BasicBlockRegion(TII, QRI, PredBB));
  PUR.addBBtoRegion(LoopBB);
  PUR.addBBtoRegion(SuccBB);

  // Make sure we have space in PredBB's last bundle.
  if (!ResourcesAvailableInBundle(&PUR, PredBundle))
    return false;
  SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE> PredBundlePkt(
      CurrentState.HomeBundle);

  // Find condition to use for predicating the reg copy into PredBB.
  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  SmallVector<MachineOperand, 4> Cond;
  if (QII->analyzeBranch(*PredBB, TBB, FBB, Cond, false))
    return false;
  if (Cond.empty())
    return false;

  // Find condition to use for predicating the reg copy at the end of LoopBB.
  MachineBasicBlock *LTBB = NULL, *LFBB = NULL;
  SmallVector<MachineOperand, 4> LCond;
  if (QII->analyzeBranch(*LoopBB, LTBB, LFBB, LCond, false))
    return false;
  if (LCond.empty())
    return false;

  // Move predicated reg copy to previous BB's last bundle.
  if (!TII->isPredicable(*RegMI))
    return false;
  MachineInstr *InstrToMove =
      &*insertTempCopy(PredBB, PredBundle, RegMI, false);
  if (!canAddMIToThisPacket(InstrToMove, PredBundlePkt)) {
    PredBB->erase_instr(InstrToMove);
    return false;
  }

  if (!TII->PredicateInstruction(*InstrToMove, Cond)) {
    // Failed to predicate the copy reg.
    PredBB->erase_instr(InstrToMove);
    return false;
  }

  // Can we newify this instruction?
  unsigned DepReg = 0;
  if (NeedToNewify(InstrToMove->getIterator(), &DepReg, &*PredBundle) &&
      !isNewifiable(InstrToMove->getIterator(), DepReg, &*PredBundle)) {
    PredBB->erase_instr(InstrToMove);
    return false;
  }
  // Newify it, and then undo it if we determine we are using a .old.
  int NewOpcode = QII->getDotNewPredOp(*InstrToMove, MBPI);
  // Undo newify if we have a non .new predicated jump we are matching.
  if (!QII->isDotNewInst(*PredCondBranch))
    NewOpcode = QII->getDotOldOp(*InstrToMove);
  NewOpcode = QII->getInvertedPredicatedOpcode(NewOpcode);
  // Properly set the opcode on the new hoisted reg copy instruction.
  InstrToMove->setDesc(QII->get(NewOpcode));
  if (!incrementalAddToPacket(*InstrToMove)) {
    PredBB->erase_instr(InstrToMove);
    return false;
  }

  // Find the conditional branch and its bundle for LoopBB.
  MachineBasicBlock::iterator LoopBundle;
  MachineInstr *LoopCondBranch = NULL;
  LoopCondBranch = findBundleAndBranch(LoopBB, LoopBundle);
  if (!LoopCondBranch)
    return false;
  if (LoopBundle == LoopBB->end())
    return false;
  LLVM_DEBUG(dbgs() << "LoopBB's Branch: ");
  LLVM_DEBUG(dbgs() << *LoopCondBranch);

  // Make sure we have space in LoopBB's last bundle.
  if (!ResourcesAvailableInBundle(&PUR, LoopBundle))
    return false;
  SmallVector<MachineInstr *, HEXAGON_PACKET_SIZE> LoopBundlePkt(
      CurrentState.HomeBundle);

  // Move predicated reg copy to last bundle of LoopBB.
  MachineInstr *InstrToSink =
      &*insertTempCopy(LoopBB, LoopBundle, RegMI, false);
  if (!canAddMIToThisPacket(InstrToSink, LoopBundlePkt)) {
    // Get rid of previous instruction as well.
    PredBB->erase_instr(InstrToMove);
    LoopBB->erase_instr(InstrToSink);
    return false;
  }

  if (!TII->PredicateInstruction(*InstrToSink, LCond)) {
    // Get rid of previous instruction as well.
    PredBB->erase_instr(InstrToMove);
    LoopBB->erase_instr(InstrToSink);
    return false;
  }
  // Can we newify this instruction?
  if (NeedToNewify(InstrToSink->getIterator(), &DepReg, &*LoopBundle) &&
      !isNewifiable(InstrToSink->getIterator(), DepReg, &*LoopBundle)) {
    // Get rid of previous instruction as well.
    PredBB->erase_instr(InstrToMove);
    PredBB->erase_instr(InstrToSink);
    return false;
  }
  NewOpcode = QII->getDotNewPredOp(*InstrToSink, MBPI);
  // Undo newify if we have a non .new predicated jump we are matching.
  if (!QII->isDotNewInst(*LoopCondBranch))
    NewOpcode = QII->getDotOldOp(*InstrToSink);
  InstrToSink->setDesc(QII->get(NewOpcode));
  if (!incrementalAddToPacket(*InstrToSink)) {
    // Get rid of previous instruction as well.
    PredBB->erase_instr(InstrToMove);
    LoopBB->erase_instr(InstrToSink);
    return false;
  }

  // Remove old instruction.
  LoopBB->erase_instr(RegMI);
  // Set loop alignment to 32.
  LoopBB->setAlignment(llvm::Align(32));

  LLVM_DEBUG(dbgs() << "Peeled Single BBLoop copy\n");
  LLVM_DEBUG(dbgs() << *InstrToMove);
  LLVM_DEBUG(dbgs() << *InstrToSink);
  LLVM_DEBUG(dbgs() << *PredBB);
  LLVM_DEBUG(dbgs() << *LoopBB);
  LLVM_DEBUG(dbgs() << *SuccBB);
  LLVM_DEBUG(dbgs() << "--- BBLoop ---\n\n");
  return true;
}

bool HexagonGlobalSchedulerImpl::performPullUpCFG(MachineFunction &Fn) {
  const Function &F = Fn.getFunction();
  // Check for single-block functions and skip them.
  if (std::next(F.begin()) == F.end())
    return false;
  bool Changed = false;
  LLVM_DEBUG(dbgs() << "****** PullUpCFG **************\n");

  // Loop over all basic blocks, asking if 3 consecutive blocks are
  // the jump opportunity.
  MachineBasicBlock *PrevBlock = NULL;
  MachineBasicBlock *JumpBlock = NULL;
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end(); MBB != MBBe;
       ++MBB) {
    MachineBasicBlock *FallBlock = &*MBB;
    if (PrevBlock && JumpBlock) {
      Changed |= pullUpPeelBBLoop(PrevBlock, JumpBlock);
    }
    PrevBlock = JumpBlock;
    JumpBlock = FallBlock;
  }
  return Changed;
}

void HexagonGlobalSchedulerImpl::GenUseDefChain(MachineFunction &Fn) {
  std::vector<unsigned> Defs;
  std::vector<unsigned> Uses;
  for (MachineFunction::iterator MBBIter = Fn.begin(); MBBIter != Fn.end();
       ++MBBIter) {
    for (MachineBasicBlock::instr_iterator MIter = MBBIter->instr_begin();
         MIter != MBBIter->instr_end(); ++MIter) {
      if (MIter->isBundle() || MIter->isDebugInstr())
        continue;
      LLVM_DEBUG(dbgs() << "\n\nInserted Ins:"; MIter->dump());
      MIUseDefSet(&*MIter, Defs, Uses);
      LLVM_DEBUG(dbgs() << "\n\tDefs:";
                 for (unsigned i = 0; i < Defs.size(); ++i) dbgs()
                 << printReg(Defs[i], QRI) << ",");
      LLVM_DEBUG(dbgs() << "\n\tUses:";
                 for (unsigned i = 0; i < Uses.size(); ++i) dbgs()
                 << printReg(Uses[i], QRI) << ",");
      MIDefSet[&*MIter] = Defs;
      MIUseSet[&*MIter] = Uses;
    }
  }
}

// optimizeBranching -
// 1. A conditional-jump transfers control to a BB with
// jump as the only instruction.
// if(p0) jump t1
// // ...
// t1: jump t2
// 2. When a BB with a single conditional jump, jumps to succ-of-succ and
// falls-through BB with only jump instruction.
// { if(p0) jump t1 }
// { jump t2 }
// t1: { ... }
MachineBasicBlock *HexagonGlobalSchedulerImpl::optimizeBranches(
    MachineBasicBlock *MBB, MachineBasicBlock *TBB, MachineInstr *FirstTerm,
    MachineBasicBlock *FBB) {
  LLVM_DEBUG(dbgs() << "\n\t\t[optimizeBranching]\n");
  if ((TBB == MBB) || (FBB == MBB))
    LLVM_DEBUG(dbgs() << "Cannot deal with loops in BB#" << MBB->getNumber(););

  // LLVM_DEBUG(dbgs() << "\n\t\tTBBMIb:"; MII->dump(););
  // { if(p) jump t1; }
  // t1: { jump t2; }
  // --> { if(p) jump t2
  // remove t1: { jump t2; }, if it's address is not taken/not a landing pad.
  if (QII->nonDbgBBSize(TBB) == 1) {
    MachineInstr *TBBMIb = &*TBB->getFirstNonDebugInstr();
    if (TBBMIb->getOpcode() == Hexagon::J2_jump &&
        TBBMIb->getOperand(0).isMBB()) {
      MachineBasicBlock *NewTarget = TBBMIb->getOperand(0).getMBB();
      if (TBB == NewTarget) // Infinite loop.
        return NULL;

      LLVM_DEBUG(dbgs() << "\nSuboptimal branching in TBB");
      // Check if the jump in the last instruction is within range.
      int64_t InstOffset =
          BlockToInstOffset.find(MBB)->second + QII->nonDbgBBSize(MBB) * 4;
      unsigned Distance = (unsigned)std::abs(
          InstOffset - BlockToInstOffset.find(NewTarget)->second);
      if (!QII->isJumpWithinBranchRange(*FirstTerm, Distance)) {
        LLVM_DEBUG(dbgs() << "\nUnconditional jump target:" << Distance
                          << " out of range.");
        return NULL;
      }
      // We need to make sure that the TBB is _not_ also a target for another
      // branch. This is suboptimal since theoretically we can update both
      // branches.
      if (!TBB->hasAddressTaken() && !TBB->isEHPad() && TBB->pred_size() == 1) {
        updatePredecessors(*TBB, NewTarget);
        // TBB has only one successor since only one J2_jump instr.
        TBB->removeSuccessor(TBB->succ_begin());
        TBBMIb->removeFromParent();
        if (!TBB->empty()) {
          // There are only debug instructions in TBB now. Move them to
          // the beginning of NewTarget.
          NewTarget->splice(NewTarget->getFirstNonPHI(), TBB, TBB->begin(),
                            TBB->end());
        }
        return TBB;
      } else {
        MBB->ReplaceUsesOfBlockWith(TBB, NewTarget);
        return NULL;
      }
    }
  }
  // { if(p) jump t1; } may contain more instructions
  // { jump t2; } --only one instruction
  // t1: {...}
  // TBB is layout successor of FBB, then we can change the branch target
  // for conditional jump and invert the predicate to remove jump t2.
  // { if(!p) jump t2; }
  // t1: {...}
  if (QII->nonDbgBBSize(FBB) == 1) {
    MachineInstr *FBBMIb = &*FBB->getFirstNonDebugInstr();
    if (FBBMIb->getOpcode() == Hexagon::J2_jump &&
        FBBMIb->getOperand(0).isMBB()) {
      MachineBasicBlock *NewTarget = FBBMIb->getOperand(0).getMBB();
      if (FBB->hasAddressTaken() || FBB->isEHPad() ||
          !FBB->isLayoutSuccessor(TBB) || (FBB == NewTarget /*Infinite loop*/))
        return NULL;

      LLVM_DEBUG(dbgs() << "\nSuboptimal branching in FBB");
      // Check if the jump in the last instruction is within range.
      int64_t InstOffset =
          BlockToInstOffset.find(MBB)->second + QII->nonDbgBBSize(MBB) * 4;
      unsigned Distance = (unsigned)std::abs(
          InstOffset - BlockToInstOffset.find(NewTarget)->second);
      if (!QII->isJumpWithinBranchRange(*FirstTerm, Distance)) {
        LLVM_DEBUG(dbgs() << "\nUnconditional jump target:" << Distance
                          << " out of range.");
        return NULL;
      }
      if (!QII->invertAndChangeJumpTarget(*FirstTerm, NewTarget))
        return NULL;
      LLVM_DEBUG(dbgs() << "\nNew instruction:"; FirstTerm->dump(););
      updatePredecessors(*FBB, NewTarget);
      // Only one successor remains for FBB
      FBB->removeSuccessor(FBB->succ_begin());
      FBBMIb->removeFromParent();
      return FBB;
    }
  }
  return NULL;
}

// performExposedOptimizations -
// look for optimization opportunities after pullup.
// e.g. jump to adjacent targets
bool HexagonGlobalSchedulerImpl::performExposedOptimizations(
    MachineFunction &Fn) {
  // Check for single-block functions and skip them.
  if (std::next(Fn.getFunction().begin()) == Fn.getFunction().end())
    return true;
  LLVM_DEBUG(dbgs() << "\n\t\t[performExposedOptimizations]\n");
  // Erasing the empty basic blocks formed during pullup.
  std::vector<MachineBasicBlock *>::iterator ebb = EmptyBBs.begin();
  while (ebb != EmptyBBs.end()) {
    assert(IsEmptyBlock(*ebb) && "Pullup inserted packets into an empty BB");
    LLVM_DEBUG(dbgs() << "Removing BB(" << (*ebb)->getNumber()
                      << ") from parent.\n");
    (*ebb)->eraseFromParent();
    ++ebb;
  }
  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  MachineInstr *FirstTerm = NULL, *SecondTerm = NULL;

  SmallVector<MachineBasicBlock *, 4> Erase;

  for (MachineBasicBlock &MBB : Fn) {
    if (MBB.succ_size() > 2 ||
        AnalyzeBBBranches(&MBB, TBB, FirstTerm, FBB, SecondTerm)) {
      LLVM_DEBUG(dbgs() << "\nAnalyzeBBBranches failed in BB#"
                        << MBB.getNumber() << "\n";);
      continue;
    }
    if (FirstTerm && QII->isCompoundBranchInstr(*FirstTerm))
      continue;
    if (TBB && FirstTerm &&
        removeRedundantBranches(&MBB, TBB, FirstTerm, FBB, SecondTerm)) {
      LLVM_DEBUG(dbgs() << "\nRemoved redundant branches in BB#"
                        << MBB.getNumber(););
      continue;
    }
    if (FirstTerm && SecondTerm &&
        optimizeDualJumps(&MBB, TBB, FirstTerm, FBB, SecondTerm)) {
      LLVM_DEBUG(dbgs() << "\nRemoved dual jumps in in BB#"
                        << MBB.getNumber(););
      continue;
    }
    if (TBB && FBB && FirstTerm && !SecondTerm) {
      MachineBasicBlock *MBBToErase =
          optimizeBranches(&MBB, TBB, FirstTerm, FBB);
      if (MBBToErase) {
        assert(IsEmptyBlock(MBBToErase) && "Erasing non-empty BB");
        Erase.push_back(MBBToErase);
        LLVM_DEBUG(dbgs() << "\nOptimized jump from BB#" << MBB.getNumber());
      }
    }
  }
  for (MachineBasicBlock *MBB : Erase)
    MBB->eraseFromParent();

  return false;
}

// 1. Remove jump to the layout successor.
// 2. Remove multiple (dual) jump to the same target.
bool HexagonGlobalSchedulerImpl::removeRedundantBranches(
    MachineBasicBlock *MBB, MachineBasicBlock *TBB, MachineInstr *FirstTerm,
    MachineBasicBlock *FBB, MachineInstr *SecondTerm) {
  bool Analyzed = false;
  LLVM_DEBUG(dbgs() << "\n\t\t[removeRedundantBranches]\n");
  MachineInstr *Head = NULL, *ToErase = NULL;
  if (!FBB && (FirstTerm->getOpcode() == Hexagon::J2_jump) &&
      MBB->isLayoutSuccessor(TBB)) {
    // Jmp layout_succ_basic_block <-- Remove
    LLVM_DEBUG(
        dbgs() << "\nRemoving Uncond. jump to the layout successor in BB#"
               << MBB->getNumber());
    ToErase = FirstTerm;
  } else if (SecondTerm && (TBB == FBB) &&
             (SecondTerm->getOpcode() == Hexagon::J2_jump)) {
    // If both branching instructions in same packet or are consecutive.
    // Jmp_c t1 <-- Remove
    // Jmp t1
    // @Note: If they are in different packets or if they are separated
    // by packet(s), this opt. cannot be done.
    MachineBasicBlock::instr_iterator FirstTermIter = FirstTerm->getIterator();
    MachineBasicBlock::instr_iterator SecondTermIter =
        SecondTerm->getIterator();
    if (++FirstTermIter == SecondTermIter) {
      LLVM_DEBUG(dbgs() << "\nRemoving multiple branching to same target in BB#"
                        << MBB->getNumber());
      // TODO: This might make the `p' register assignment instruction dead.
      // and can be removed.
      ToErase = FirstTerm;
    }
  } else if (SecondTerm && (SecondTerm->getOpcode() == Hexagon::J2_jump) &&
             FBB && MBB->isLayoutSuccessor(FBB)) {
    // Jmp_c t1
    // Jmp layout_succ_basic_block <-- Remove
    LLVM_DEBUG(dbgs() << "\nRemoving fall through branch in BB#"
                      << MBB->getNumber());
    ToErase = SecondTerm;
  } else if (SecondTerm && QII->PredOpcodeHasJMP_c(SecondTerm->getOpcode()) &&
             MBB->isLayoutSuccessor(getBranchDestination(SecondTerm))) {
    // Jmp_c t1
    // Jmp_c layout_succ_basic_block <-- Remove
    // In this case AnalyzeBBBranches might assign FBB to some other BB.
    // So using the jump target of SecondTerm to check.
    LLVM_DEBUG(dbgs() << "\nRemoving Cond. jump to the layout successor in BB#"
                      << MBB->getNumber());
    ToErase = SecondTerm;
  }
  // Remove the instruction from the BB
  if (ToErase) {
    if (ToErase->isBundled()) {
      Head = &*getBundleStart(ToErase->getIterator());
      ToErase->eraseFromBundle();
      UpdateBundle(Head);
    } else
      ToErase->eraseFromParent();
    Analyzed = true;
  }
  return Analyzed;
}

// ----- convert
// p = <expr>
// if(p) jump layout_succ_basic_block
// jump t
// ----- to
// p = <expr>
// if(!p) jump t
// for now only looking at the dual jump
bool HexagonGlobalSchedulerImpl::optimizeDualJumps(MachineBasicBlock *MBB,
                                                   MachineBasicBlock *TBB,
                                                   MachineInstr *FirstTerm,
                                                   MachineBasicBlock *FBB,
                                                   MachineInstr *SecondTerm) {
  LLVM_DEBUG(dbgs() << "\n******* optimizeDualJumps *******");

  bool Analyzed = false;

  if (QII->PredOpcodeHasJMP_c(FirstTerm->getOpcode()) &&
      (SecondTerm->getOpcode() == Hexagon::J2_jump)) {

    if (TBB == FBB) {
      LLVM_DEBUG(dbgs() << "\nBoth successors are the same.");
      return Analyzed;
    }

    // Do not optimize for dual jumps if this MBB
    // contains a speculatively pulled-up instruction.
    // A speculated instruction is more likely to be at the end of MBB.
    MachineBasicBlock::reverse_instr_iterator SII = MBB->instr_rbegin();
    while (SII != MBB->instr_rend()) {
      MachineInstr *SI = &*SII;
      std::map<MachineInstr *, MachineBasicBlock *>::iterator MIMoved;
      MIMoved = SpeculatedIns.find(SI);
      if ((MIMoved != SpeculatedIns.end()) &&
          (MIMoved->second != SI->getParent())) {
        return Analyzed;
      }
      ++SII;
    }

    LLVM_DEBUG(dbgs() << "\nCandidate for jump optimization in BB("
                      << MBB->getNumber() << ").\n";);

    // Predicated jump to layout successor followed by an unconditional jump.
    if (MBB->isLayoutSuccessor(TBB)) {

      // Check if the jump in the last instruction is within range.
      int64_t InstOffset =
          BlockToInstOffset.find(&*MBB)->second + QII->nonDbgBBSize(MBB) * 4;
      unsigned Distance =
          (unsigned)std::abs(InstOffset - BlockToInstOffset.find(FBB)->second) +
          SafetyBuffer;
      if (!QII->isJumpWithinBranchRange(*FirstTerm, Distance)) {
        LLVM_DEBUG(dbgs() << "\nUnconditional jump target:" << Distance
                          << " out of range.");
        return Analyzed;
      }

      // modify the second last -predicated- instruction (sense and target)
      LLVM_DEBUG(dbgs() << "\nFirst Instr:" << *FirstTerm;);
      LLVM_DEBUG(dbgs() << "\nSecond Instr:" << *SecondTerm;);
      LLVM_DEBUG(dbgs() << "\nOld Succ BB(" << TBB->getNumber() << ").";);

      QII->invertAndChangeJumpTarget(*FirstTerm, FBB);

      LLVM_DEBUG(dbgs() << "\nNew First Instruction:" << *FirstTerm;);

      // unbundle if there is only one instruction left
      MachineInstr *SecondHead, *FirstHead;
      FirstHead = FirstTerm->isBundled()
                      ? &*getBundleStart(FirstTerm->getIterator())
                      : nullptr;
      SecondHead = SecondTerm->isBundled()
                       ? &*getBundleStart(SecondTerm->getIterator())
                       : nullptr;

      // 1. Both unbundled, 2. FirstTerm inside bundle, second outside.
      if (!SecondHead)
        SecondTerm->eraseFromParent();
      else if (!FirstHead) {
        // 3. FirstHead outside, SecondHead inside.
        SecondTerm->eraseFromBundle();
        UpdateBundle(SecondHead);
      } else if (FirstHead == SecondHead) {
        // 4. Both are in the same bundle
        assert((FirstHead && SecondHead) && "Unbundled Instruction");
        SecondTerm->eraseFromBundle();
        if (SecondHead->getBundleSize() < 2)
          UpdateBundle(SecondHead);
      } else {
        // 5. Both are in different bundles
        SecondTerm->eraseFromBundle();
        UpdateBundle(SecondHead);
      }
      Analyzed = true;
    }
  }
  return Analyzed;
}

/// Are there any resources left in this bundle?
bool HexagonGlobalSchedulerImpl::ResourcesAvailableInBundle(
    BasicBlockRegion *CurrentRegion,
    MachineBasicBlock::iterator &TargetPacket) {
  MachineBasicBlock::instr_iterator MII = TargetPacket.getInstrIterator();

  // If this is a single instruction, form new packet around it.
  if (!TargetPacket->isBundle()) {
    if (ignoreInstruction(&*MII) || isSoloInstruction(*MII))
      return false;

    // Before we begin, we need to make sure that we do not
    // look at an unconditional jump outside the current region.
    if (MII->isBranch() && !isBranchWithinRegion(CurrentRegion, &*MII))
      return false;

    // Build up state for this new packet.
    // Note, we cannot create a bundle header for it,
    // so this "bundle" only exist in DFA state, and not in code.
    initPacketizerState();
    ResourceTracker->clearResources();
    CurrentState.addHomeLocation(MII);
    return incrementalAddToPacket(*MII);
  }

  MachineBasicBlock::instr_iterator End = MII->getParent()->instr_end();

  // Build up state for this packet.
  initPacketizerState();
  ResourceTracker->clearResources();
  CurrentState.addHomeLocation(MII);

  for (++MII; MII != End && MII->isInsideBundle(); ++MII) {
    if (MII->getOpcode() == TargetOpcode::DBG_VALUE ||
        MII->getOpcode() == TargetOpcode::IMPLICIT_DEF ||
        MII->getOpcode() == TargetOpcode::CFI_INSTRUCTION || MII->isEHLabel())
      continue;

    // Before we begin, we need to make sure that we do not
    // look at an unconditional jump outside the current region.
    // TODO: See if we can profit from handling this kind of cases:
    // B#15: derived from LLVM BB %if.then22
    // Predecessors according to CFG: BB#13
    // BUNDLE %PC<imp-def>, %P2<imp-use,kill>
    //   * J2_jumpf %P2<kill,internal>, <BB#17>, %PC<imp-def>; flags:
    //   * J2_jump <BB#18>, %PC<imp-def>; flags:
    // Successors according to CFG: BB#18(62) BB#17(62)
    // Curently we do not allow them.
    if (MII->isBranch() && !isBranchWithinRegion(CurrentRegion, &*MII))
      return false;

    if (!incrementalAddToPacket(*MII))
      return false;
  }
  return ResourceTracker->canReserveResources(*Nop);
}

/// Symmetrical. See if these two instructions are fit for compound pair.
bool HexagonGlobalSchedulerImpl::isCompoundPair(MachineInstr *MIa,
                                                MachineInstr *MIb) {
  enum HexagonII::CompoundGroup MIaG = QII->getCompoundCandidateGroup(*MIa),
                                MIbG = QII->getCompoundCandidateGroup(*MIb);
  // We have two candidates - check that this is the same register
  // we are talking about.
  unsigned Opcb = MIb->getOpcode();
  if (MIaG == HexagonII::HCG_C && MIbG == HexagonII::HCG_A &&
      (Opcb == Hexagon::A2_tfr || Opcb == Hexagon::A2_tfrsi))
    return true;
  unsigned Opca = MIa->getOpcode();
  if (MIbG == HexagonII::HCG_C && MIaG == HexagonII::HCG_A &&
      (Opca == Hexagon::A2_tfr || Opca == Hexagon::A2_tfrsi))
    return true;
  return (((MIaG == HexagonII::HCG_A && MIbG == HexagonII::HCG_B) ||
           (MIbG == HexagonII::HCG_A && MIaG == HexagonII::HCG_B)) &&
          (MIa->getOperand(0).getReg() == MIb->getOperand(0).getReg()));
}

// This is a weird situation when BB conditionally branches + falls through
// to layout successor. \ref bug17792
inline bool HexagonGlobalSchedulerImpl::multipleBranchesFromToBB(
    MachineBasicBlock *BB) const {
  if (BB->succ_size() != 1)
    return false;
  SmallVector<MachineInstr *, 2> Jumpers = QII->getBranchingInstrs(*BB);
  return ((Jumpers.size() == 1) && !Jumpers[0]->isUnconditionalBranch());
}

/// Gather a worklist of MaxCandidates pull-up candidates.
/// Compute relative cost.
bool HexagonGlobalSchedulerImpl::findPullUpCandidates(
    MachineBasicBlock::iterator &WorkPoint,
    MachineBasicBlock::iterator &FromHere,
    std::vector<MachineInstr *> &backtrack, unsigned MaxCandidates = 1) {

  const HexagonInstrInfo *QII = (const HexagonInstrInfo *)TII;
  MachineBasicBlock *FromThisBB = FromHere->getParent();
  bool MovingDependentOp = false;
  signed CostBenefit = 0;

  // Do not collect more than that many candidates.
  if (CurrentState.haveCandidates() >= MaxCandidates)
    return false;

  LLVM_DEBUG(dbgs() << "\n\tTry from BB(" << FromThisBB->getNumber() << "):\n";
             DumpPacket(FromHere.getInstrIterator()));

  if (FromHere->isBundle()) {
    MachineBasicBlock::instr_iterator MII = FromHere.getInstrIterator();
    for (++MII; MII != FromThisBB->instr_end() && MII->isInsideBundle();
         ++MII) {
      if (MII->isDebugInstr())
        continue;
      LLVM_DEBUG(dbgs() << "\tCandidate from BB("
                        << MII->getParent()->getNumber() << "): ";
                 MII->dump());

      // See if this instruction could be moved.
      if (!canThisMIBeMoved(&*MII, WorkPoint, MovingDependentOp, CostBenefit))
        continue;

      MachineBasicBlock::instr_iterator InstrToMove = MII;
      if (canAddMIToThisPacket(&*InstrToMove, CurrentState.HomeBundle)) {
        CostBenefit -= (backtrack.size() * 4);
        // Prefer instructions in empty packets.
        CostBenefit += (PacketSize - nonDbgBundleSize(FromHere)) * 2;
        // Prefer Compares.
        if (MII->isCompare())
          CostBenefit += 10;
        // Check duplex conditions;
        for (unsigned i = 0; i < CurrentState.HomeBundle.size(); i++) {
          if (QII->isDuplexPair(*CurrentState.HomeBundle[i], *MII)) {
            LLVM_DEBUG(dbgs() << "\tGot real Duplex (bundle).\n");
            CostBenefit += 20;
          }
          if (isCompoundPair(CurrentState.HomeBundle[i], &*MII)) {
            LLVM_DEBUG(dbgs() << "\tGot compound (bundle).\n");
            CostBenefit += 40;
          }
        }
        // Create a record for this location.
        CurrentState.addPullUpCandidate(InstrToMove, WorkPoint, backtrack,
                                        MovingDependentOp, CostBenefit);
      } else
        LLVM_DEBUG(dbgs() << "\tNo resources in the target packet.\n");
    }
  }
  // This is a standalone instruction.
  // First see if this MI can even be moved. Cost model for a single instruction
  // should be rather different from moving something out of a bundle.
  else if (canThisMIBeMoved(&*FromHere, WorkPoint, MovingDependentOp,
                            CostBenefit)) {
    MachineBasicBlock::instr_iterator InstrToMove = FromHere.getInstrIterator();
    if (canAddMIToThisPacket(&*InstrToMove, CurrentState.HomeBundle)) {
      CostBenefit -= (backtrack.size() * 4);
      // Prefer Compares.
      if (InstrToMove->isCompare())
        CostBenefit += 10;
      // It is better to pull a single instruction in to a bundle - save
      // a cycle immediately.
      CostBenefit += 10;
      // Search for duplex match.
      for (unsigned i = 0; i < CurrentState.HomeBundle.size(); i++) {
        if (QII->isDuplexPair(*CurrentState.HomeBundle[i], *InstrToMove)) {
          LLVM_DEBUG(dbgs() << "\tGot real Duplex (single).\n");
          CostBenefit += 30;
        }
        if (isCompoundPair(CurrentState.HomeBundle[i], &*InstrToMove)) {
          LLVM_DEBUG(dbgs() << "\tGot compound (single).\n");
          CostBenefit += 50;
        }
      }
      // Create a record for this location.
      CurrentState.addPullUpCandidate(InstrToMove, WorkPoint, backtrack,
                                      MovingDependentOp, CostBenefit);
    } else
      LLVM_DEBUG(dbgs() << "\tNo resources for single in the target packet.\n");
  }
  return true;
}

/// Try to move a candidate MI.
/// The move can destroy all iterator system, so we have to drag them
/// around to keep them up to date.
bool HexagonGlobalSchedulerImpl::tryMultipleInstructions(
    MachineBasicBlock::iterator &RetVal, /* output parameter */
    std::vector<BasicBlockRegion *>::iterator &CurrentRegion,
    MachineBasicBlock::iterator &NextMI,
    MachineBasicBlock::iterator &ToThisBBEnd,
    MachineBasicBlock::iterator &FromThisBBEnd, bool PathInRegion) {

  MachineBasicBlock::instr_iterator MII;
  MachineBasicBlock::iterator WorkPoint;
  bool MovingDependentOp = false;
  std::vector<MachineInstr *> backtrack;

  LLVM_DEBUG(dbgs() << "\n\tTry Multiple candidates: \n");

  std::sort(CurrentState.PullUpCandidates.begin(),
            CurrentState.PullUpCandidates.end(), PullUpCandidateSorter());
  LLVM_DEBUG(CurrentState.dump());
  // Iterate through candidates in sorted order.
  for (SmallVector<PullUpCandidate *, 4>::iterator
           I = CurrentState.PullUpCandidates.begin(),
           E = CurrentState.PullUpCandidates.end();
       I != E; ++I) {
    (*I)->populate(MII, WorkPoint, backtrack, MovingDependentOp);

    MachineBasicBlock *FromThisBB = MII->getParent();
    MachineBasicBlock *ToThisBB = WorkPoint->getParent();

    LLVM_DEBUG(dbgs() << "\n\tCandidate: "; MII->dump());
    LLVM_DEBUG(dbgs() << "\tDependent(" << MovingDependentOp << ") FromBB("
                      << FromThisBB->getNumber() << ") ToBB("
                      << ToThisBB->getNumber() << ") to this packet:\n";
               DumpPacket(WorkPoint.getInstrIterator()));

    MachineBasicBlock::instr_iterator FromHereII = MII;
    if (MII->isInsideBundle()) {
      while (!FromHereII->isBundle())
        --FromHereII;
      LLVM_DEBUG(dbgs() << "\tFrom here:\n"; DumpPacket(FromHereII));

      MachineBasicBlock::iterator FromHere(FromHereII);
      // We have instruction that could be moved from its current position.
      if (MoveMItoBundle(*CurrentRegion, MII, NextMI, WorkPoint, FromHere,
                         backtrack, MovingDependentOp, PathInRegion)) {
        // If BB from which we pull is now empty, move on.
        if (IsEmptyBlock(FromThisBB)) {
          LLVM_DEBUG(dbgs() << "\n\tExhosted BB (bundle).\n");
          return false;
        }
        FromThisBBEnd = FromThisBB->end();
        ToThisBBEnd = ToThisBB->end();

        LLVM_DEBUG(dbgs() << "\n\tAfter updates(bundle to bundle):\n");
        LLVM_DEBUG(dbgs() << "\t\tWorkPoint: ";
                   DumpPacket(WorkPoint.getInstrIterator()));

        // We should not increment current position,
        // but rather try one more time to pull from the same bundle.
        RetVal = WorkPoint;
        return true;
      } else
        LLVM_DEBUG(dbgs() << "\tCould not move packetized instr.\n");
    } else {
      MachineBasicBlock::iterator FromHere(FromHereII);
      if (MoveMItoBundle(*CurrentRegion, MII, NextMI, WorkPoint, FromHere,
                         backtrack, MovingDependentOp, PathInRegion)) {

        // If BB from which we pull is now empty, move on.
        if (IsEmptyBlock(FromThisBB)) {
          LLVM_DEBUG(dbgs() << "\n\tExhosted BB (single).\n");
          return false;
        }
        FromThisBBEnd = FromThisBB->end();
        ToThisBBEnd = ToThisBB->end();

        LLVM_DEBUG(dbgs() << "\tAfter updates (single to bundle):\n");
        LLVM_DEBUG(dbgs() << "\t\tWorkPoint: ";
                   DumpPacket(WorkPoint.getInstrIterator()));
        // We should not increment current position,
        // but rather try one more time to pull from the same bundle.
        RetVal = WorkPoint;
        return true;
      } else
        LLVM_DEBUG(dbgs() << "\tCould not move single.\n");
    }
  }
  LLVM_DEBUG(dbgs() << "\tNot a single candidate fit.\n");
  return false;
}

/// Main function. Iterate all current regions one at a time,
/// and look for pull-up opportunities.
/// Pseudo sequence:
/// - for all bundles and single instructions in region:
/// - see if resources are available (in the same cycle) - this is HOME.
/// - Starting from next BB in region, find an instruction that could be:
///   - removed from its current location
///   - added to underutilized bundle (including bundles with only one op)
/// - If so, trace path back to HOME and check that candidate could be
///   reordered with all the intermediate instructions.
bool HexagonGlobalSchedulerImpl::performPullUp() {
  std::vector<MachineInstr *> backtrack;
  MachineBasicBlock::iterator FromHere;
  MachineBasicBlock::iterator FromThisBBEnd;

  LLVM_DEBUG(dbgs() << "****** PullUpRegions ***********\n");
  // For all regions...
  for (std::vector<BasicBlockRegion *>::iterator
           CurrentRegion = PullUpRegions.begin(),
           E = PullUpRegions.end();
       CurrentRegion != E; ++CurrentRegion) {

    LLVM_DEBUG(dbgs() << "\n\nRegion with(" << (*CurrentRegion)->size()
                      << ")BBs\n");

    if (!EnableLocalPullUp && (*CurrentRegion)->size() < 2)
      continue;

    // For all MBB in the region... except the last one.
    // ...except when we want to allow local pull-up.
    for (auto ToThisBB = (*CurrentRegion)->getRootMBB(),
              LastBBInRegion = (*CurrentRegion)->getLastMBB();
         ToThisBB != LastBBInRegion; ++ToThisBB) {
      // If we do not want to allow same BB pull-up, take an early exit.
      if (!EnableLocalPullUp && (std::next(ToThisBB) == LastBBInRegion))
        break;
      if (multipleBranchesFromToBB(*ToThisBB))
        break;

      auto FromThisBB = ToThisBB;
      MachineBasicBlock::iterator ToThisBBEnd = (*ToThisBB)->end();
      MachineBasicBlock::iterator MI = (*ToThisBB)->begin();

      LLVM_DEBUG(dbgs() << "\n\tHome iterator moved to new BB("
                        << (*ToThisBB)->getNumber() << ")\n";
                 (*ToThisBB)->dump());

      // For all instructions in the BB.
      while (MI != ToThisBBEnd) {
        MachineBasicBlock::iterator WorkPoint = MI;
        ++MI;

        // Trivial check that there are unused resources
        // in the current location (cycle).
        while (ResourcesAvailableInBundle(*CurrentRegion, WorkPoint)) {
          LLVM_DEBUG(dbgs() << "\nxxxx Next Home in BB("
                            << (*ToThisBB)->getNumber() << "):\n";
                     DumpPacket(WorkPoint.getInstrIterator()));
          // Keep the path to the candidate.
          // It is the traveled path between home and work point.
          // Reset it for the new iteration.
          backtrack.clear();

          // The point of pull-up source (WorkPoint) could begin from the
          // current BB, but only if we allow pull-up in the same BB.
          // At the moment we do not.
          // We also do not process last block in the region,
          // so it is safe to always begin with the next BB in the region.
          // Start from "next" BB in the region.
          if (EnableLocalPullUp) {
            FromThisBB = ToThisBB;
            FromHere = WorkPoint;
            ++FromHere;
            FromThisBBEnd = (*FromThisBB)->end();

            // Initialize backtrack.
            // These are instructions between Home location
            // and the WorkPoint.
            for (MachineBasicBlock::iterator I = WorkPoint, IE = FromHere;
                 I != IE; ++I)
              backtrack.push_back(&*I);
          } else {
            FromThisBB = ToThisBB;
            ++FromThisBB;
            FromHere = (*FromThisBB)->begin();
            FromThisBBEnd = (*FromThisBB)->end();

            // Initialize backtrack.
            // These are instructions between Home location
            // and the end of the home BB.
            for (MachineBasicBlock::iterator I = WorkPoint, IE = ToThisBBEnd;
                 I != IE; ++I)
              backtrack.push_back(&*I);
          }

          // Search for pull-up candidate.
          while (true) {
            // If this BB is over, move onto the next one
            // in this region.
            if (FromHere == FromThisBBEnd) {
              ++FromThisBB;
              // Refresh LastBBInRegion in case tryMultipleInstructions modified
              // the regions Elements vector, invalidating the iterator.
              LastBBInRegion = (*CurrentRegion)->getLastMBB();
              if (FromThisBB == LastBBInRegion)
                break;
              else {
                LLVM_DEBUG(dbgs() << "\n\tNext BB in this region\n";
                           (*FromThisBB)->dump());
                FromThisBBEnd = (*FromThisBB)->end();
                FromHere = (*FromThisBB)->begin();
                if (FromThisBBEnd == FromHere)
                  break;
              }
            }
            if ((*FromHere).isDebugInstr()) {
              ++FromHere;
              continue;
            }
            // This is a step Home.
            backtrack.push_back(&*FromHere);
            if (!findPullUpCandidates(WorkPoint, FromHere, backtrack,
                                      MainCandidateQueueSize))
              break;
            ++FromHere;
          }
          // Try to pull-up one of the selected candidates.
          if (!tryMultipleInstructions(/*output*/ WorkPoint, CurrentRegion, MI,
                                       ToThisBBEnd, FromThisBBEnd))
            break;
        }
      }
      // Refresh LastBBInRegion after potential CFG modifications.
      LastBBInRegion = (*CurrentRegion)->getLastMBB();
    }
    // AllowUnlikelyPath is on by default,
    // if we wish to disable it, we can do so here.
    if (!AllowUnlikelyPath)
      continue;

    // We have parsed the likely path through the region.
    // Now traverse the other (unlikely) path.
    //
    // Note: BasicBlockRegion uses a vector for MBB storage, so adding BBs to
    // the region while iterating could invalidate iterators. Collect the work
    // items first, then process them.
    std::vector<std::pair<MachineBasicBlock *, MachineBasicBlock *>>
        UnlikelyWork;
    UnlikelyWork.reserve((*CurrentRegion)->size());
    for (auto ToIt = (*CurrentRegion)->getRootMBB(),
              End = (*CurrentRegion)->getLastMBB();
         ToIt != End; ++ToIt) {
      MachineBasicBlock *ToBB = *ToIt;
      MachineBasicBlock *SecondBest = getNextPURBB(ToBB, true);
      if (SecondBest)
        UnlikelyWork.emplace_back(ToBB, SecondBest);
    }

    for (auto [ToBB, SecondBest] : UnlikelyWork) {
      LLVM_DEBUG(dbgs() << "\tFor BB:\n"; ToBB->dump());
      LLVM_DEBUG(dbgs() << "\tHave SecondBest:\n"; SecondBest->dump());
      // Adding this BB to the region should not be done if we
      // plan to reuse it(the region) again. For now it is OK.
      (*CurrentRegion)->addBBtoRegion(SecondBest);
      LLVM_DEBUG(dbgs() << "\tHome iterator moved to new BB("
                        << ToBB->getNumber() << ")\n";
                 ToBB->dump());
      MachineBasicBlock::iterator ToThisBBEnd = ToBB->end();
      MachineBasicBlock::iterator MI = ToBB->begin();

      // For all instructions in the BB.
      while (MI != ToThisBBEnd) {
        MachineBasicBlock::iterator WorkPoint = MI;
        ++MI;

        // Trivial check that there are unused resources
        // in the current location (cycle).
        while (ResourcesAvailableInBundle(*CurrentRegion, WorkPoint)) {
          LLVM_DEBUG(dbgs() << "\nxxxx Second visit Home in BB("
                            << ToBB->getNumber() << "):\n";
                     DumpPacket(WorkPoint.getInstrIterator()));

          FromHere = SecondBest->begin();
          FromThisBBEnd = SecondBest->end();

          // Keep the path to the candidate.
          backtrack.clear();

          // This is Home location.
          for (MachineBasicBlock::iterator I = WorkPoint, IE = ToThisBBEnd;
               I != IE; ++I)
            backtrack.push_back(&*I);

          while (true) {
            // If this BB is over, move onto the next one
            // in this region.
            if (FromHere == FromThisBBEnd) {
              LLVM_DEBUG(dbgs()
                         << "\tOnly do one successor for the second try\n");
              break;
            }
            if ((*FromHere).isDebugInstr()) {
              ++FromHere;
              continue;
            }
            // This is a step Home.
            backtrack.push_back(&*FromHere);
            if (!findPullUpCandidates(WorkPoint, FromHere, backtrack,
                                      SecondaryCandidateQueueSize))
              break;
            ++FromHere;
          }
          // Try to pull-up one of selected candidate.
          if (!tryMultipleInstructions(/*output*/ WorkPoint, CurrentRegion, MI,
                                       ToThisBBEnd, FromThisBBEnd, false))
            break;
        }
      }
    }
  }
  return true;
}

bool HexagonGlobalSchedulerImpl::incrementalAddToPacket(MachineInstr &MI) {

  LLVM_DEBUG(dbgs() << "\t[AddToPacket] (" << CurrentPacketMIs.size()
                    << ") adding:\t";
             MI.dump());

  if (!ResourceTracker->canReserveResources(MI) || !shouldAddToPacket(MI))
    return false;

  ResourceTracker->reserveResources(MI);
  CurrentPacketMIs.push_back(&MI);
  CurrentState.HomeBundle.push_back(&MI);

  if (QII->isExtended(MI) || QII->isConstExtended(MI) ||
      isJumpOutOfRange(&MI)) {
    // If at this point of time we cannot reserve resources,
    // this might mean that the packet came into the pull-up
    // pass already in danger of overflowing.
    // Nevertheless, since this is only a possibility of overflow
    // no error should be issued here.
    if (ResourceTracker->canReserveResources(*Ext)) {
      ResourceTracker->reserveResources(*Ext);
      LLVM_DEBUG(dbgs() << "\t[AddToPacket] (" << CurrentPacketMIs.size()
                        << ") adding:\t  immext_i\n");
      CurrentPacketMIs.push_back(Ext);
      CurrentState.HomeBundle.push_back(Ext);
      return true;
    } else {
      LLVM_DEBUG(dbgs() << "\t  Previous overflow possible.\n");
      return false;
    }
  }
  return true;
}

void HexagonGlobalSchedulerImpl::checkBundleCounts(MachineFunction &Fn) {
  if (DisableCheckBundles)
    return;

  unsigned BundleLimit = 4;

  for (MachineFunction::iterator MBBi = Fn.begin(), MBBe = Fn.end();
       MBBi != MBBe; ++MBBi) {

    for (MachineBasicBlock::iterator MI = MBBi->instr_begin(),
                                     ME = MBBi->instr_end();
         MI != ME; ++MI) {
      if (MI->isBundle()) {
        MachineBasicBlock::instr_iterator MII = MI.getInstrIterator();
        MachineBasicBlock::instr_iterator End = MII->getParent()->instr_end();

        unsigned InstrCount = 0;

        for (++MII; MII != End && MII->isInsideBundle(); ++MII) {
          if (MII->getOpcode() == TargetOpcode::DBG_VALUE ||
              MII->getOpcode() == TargetOpcode::IMPLICIT_DEF ||
              MII->getOpcode() == TargetOpcode::CFI_INSTRUCTION ||
              MII->isEHLabel() || QII->isEndLoopN(MII->getOpcode())) {
            continue;
          } else {
            InstrCount++;
          }
        }
        if (InstrCount > BundleLimit) {
          if (WarnOnBundleSize) {
            LLVM_DEBUG(dbgs() << "Warning bundle size exceeded " << *MI);
          } else {
            assert(0 && "Bundle size exceeded");
          }
        }
      }
    }
  }
}

/// Debugging only. Count compound and duplex opportunities.
unsigned HexagonGlobalSchedulerImpl::countCompounds(MachineFunction &Fn) {
  unsigned CompoundCount = 0;
  [[maybe_unused]] unsigned DuplexCount = 0;
  [[maybe_unused]] unsigned InstOffset = 0;

  // Loop over all basic blocks.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end(); MBB != MBBe;
       ++MBB) {
    LLVM_DEBUG(dbgs() << "\n BB#" << MBB->getNumber() << " " << MBB->getName()
                      << " in_func "
                      << MBB->getParent()->getFunction().getName() << " \n");
    for (MachineBasicBlock::iterator MI = MBB->instr_begin(),
                                     ME = MBB->instr_end();
         MI != ME; ++MI) {
      if (MI->isDebugInstr())
        continue;
      if (MI->isBundle()) {
        MachineBasicBlock::instr_iterator MII = MI.getInstrIterator();
        MachineBasicBlock::instr_iterator MIE = MI->getParent()->instr_end();
        MachineInstr *FirstCompound = NULL, *SecondCompound = NULL;
        MachineInstr *FirstDuplex = NULL, *SecondDuplex = NULL;
        LLVM_DEBUG(dbgs() << "{\n");

        for (++MII; MII != MIE && MII->isInsideBundle() && !MII->isBundle();
             ++MII) {
          if (MII->isDebugInstr())
            continue;
          LLVM_DEBUG(dbgs() << "(" << InstOffset << ")\t");
          InstOffset += QII->getSize(*MII);
          if (QII->getCompoundCandidateGroup(*MII)) {
            if (!FirstCompound) {
              FirstCompound = &*MII;
              LLVM_DEBUG(dbgs() << "XX ");
            } else {
              SecondCompound = &*MII;
              LLVM_DEBUG(dbgs() << "YY ");
            }
          }
          if (QII->getDuplexCandidateGroup(*MII)) {
            if (!FirstDuplex) {
              FirstDuplex = &*MII;
              LLVM_DEBUG(dbgs() << "AA ");
            } else {
              SecondDuplex = &*MII;
              LLVM_DEBUG(dbgs() << "VV ");
            }
          }
          LLVM_DEBUG(MII->dump());
        }
        LLVM_DEBUG(dbgs() << "}\n");
        if (SecondCompound) {
          if (isCompoundPair(FirstCompound, SecondCompound)) {
            LLVM_DEBUG(dbgs() << "Compound pair (" << CompoundCount << ")\n");
            CompoundCount++;
          }
        }
        if (SecondDuplex) {
          if (QII->isDuplexPair(*FirstDuplex, *SecondDuplex)) {
            LLVM_DEBUG(dbgs() << "Duplex pair (" << DuplexCount << ")\n");
            DuplexCount++;
          }
        }
      } else {
        LLVM_DEBUG(dbgs() << "(" << InstOffset << ")\t");
        if (QII->getCompoundCandidateGroup(*MI))
          LLVM_DEBUG(dbgs() << "XX ");
        if (QII->getDuplexCandidateGroup(*MI))
          LLVM_DEBUG(dbgs() << "AA ");
        InstOffset += QII->getSize(*MI);
        LLVM_DEBUG(MI->dump());
      }
    }
  }
  LLVM_DEBUG(dbgs() << "Total compound(" << CompoundCount << ") duplex("
                    << DuplexCount << ")\n");
  return CompoundCount;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonGlobalScheduler() {
  return new HexagonGlobalScheduler();
}
