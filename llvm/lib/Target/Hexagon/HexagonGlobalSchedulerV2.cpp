//===----- HexagonGlobalSchedulerV2.cpp - Global Scheduler ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic infrastructure for the global scheduling + Hexagon pull-up pass.
// Currently run at the very end of code generation for Hexagon, cleans
// up lost scheduling opportunities. Liveness is preserved after this pass,
// and we should not run a liveness analysis after this pass because compound
// instructions might be formed which do not fit into the usual model of
// use-after-def of regs.
//
// @note: Local pullup runs after global scheduling because global pullup
// leaves room in packets that local instructions can fill.
//
// @note: Local push-down moves instructions into the immediate successor
// packet only.
//
// Brief overview of working of this global scheduler.
// This version of Hexagon Global Scheduler is basically a greedy scheduler.
// The main purpose of this VLIW global scheduler is to increase the packet
// density (= #instructions/#packets). Since the VLIW packetizer works on a
// local scope, it cannot fill up the packets very efficiently.
// Having a global scheduler post-SSA makes it a complicated infrastructure
// because it has to deal with liveness+dependencies of physical registers.
// As a result, many strict measures have been taken to ensure correctness.
// There are assertions put at places where some assumptions were implicit.
// This also means that many optimization opportunities are not explored.
//
// The global scheduler heavily relies on HexagonLiveVariables.cpp and
// HexagonVLIWPacketizer.cpp.
// There is one function isLegalToPacketizeTogetherMI() which copies little
// bit of code from HexagonPacketizerList::isLegalToPacketizeTogether, because
// this the packetizer version takes in SchedulingUnits as arguments and
// it wasn't easy to refactor it completely. Please update both the functions
// in a coherent way. Or better, instead, add code to one
// of the following three functions:
//
// HexagonPacketizerList::hasDeadDependence
// HexagonPacketizerList::hasControlDependence
// HexagonPacketizerList::hasDualStoreDependence
//
//
// Step by step explanation of the process of global scheduling.
//
// Step1: Correct the CFG.
// The packetizer/other-pass may leave MBB in a state where there are
// non-branching instructions after the branches. This pass corrects the CFG
// so that branches are the only instructions at the very end. This helps
// reuse analyzeBranch even after packetizer.
//
// Step2: Form Scheduling regions.
// Scheduling regions are sets of basic blocks which have only one entry point
// but, possibly, multiple exit points. This is essentially a super-block.
//
// Step3: Sort Scheduling regions.
// Sort scheduling regions based on profile/branch-probability. This gives
// priority to instructions from branch targets with higher probability. In the
// case where both branches have same probability, it biases one successor by
// increasing the weight of TBB (see analyzeBranch).
//   - During this pass, AnalyzeBranchInfo of all the MBBs are
//     calculated and stored to be reused later. Care must be taken to update
//     the branch-info when a branch is modified (mostly post-pullup opts)
//
// Step4: Perform global+local scheduling of instructions.
//   Global scheduling is done first followed by local.
//   Global scheduling: globalPullup(SchedulingRegion_t &SR)
//   Speculative scheduling is done first followed by predicated scheduling.
//     for each scheduling region
//     for each MBB in a scheduling region
//     for each MI in leaf MBB
//       find a MI candidate by calling canMIBeSpeculated
//         YES=>speculate.
//           YES means it has to be speculated. There is no coming back.
//           These strict checks are necessary to ensure correctness in this
//           complicated infrastructure.
//         NO=>Try to squeeze the bundle by making a compound instruction
//           and hence, freeing up one slot. Only in special cases where
//           `bool preciousToSpeculate' returns true, the opportunities for
//           compound instructions are explored. If it is possible to create
//           a compound then the compound instruction
//           (generateCompoundIfPossible()) is created and again
//           decomposed into individual instructions. This is done because
//           it is not possible to analyzeBranch a BB with compounds (for now).
//
//           And then try again by calling canMIBeSpeculated
//           YES=>speculate. The bundle now contains five instructions. Since
//             the bundle contains five instructions, care must be taken to
//             avoid doing too much around that packet, because there is no
//             available infrastructure to support five-instruction bundles.
//           NO=>move on to the next MI.
//
//     for each MI in leaf MBB
//       find a MI candidate by calling canMIBePredicated
//         YES=>predicate.
//           YES means it has to be predicated. There is no coming back.
//         NO=>Try to squeeze the bundle by making a compound instruction
//           and hence, freeing up one slot. Only in special cases where
//           `bool preciousToPredicate' returns true, the opportunities for
//           compound instructions are explored.
//
//          And then try again by calling canMIBePredicated.
//          This is identical to the case of speculative scheduling.
//
//
//   Local scheduling: localPullup
//     Reuses some functionalities of speculative pullup.
//     The function findInsertPtSpec has been designed to be reused for both
//     local+global `speculative' scheduling. Similarly, srcDependence() and
//     dstDependence() are reused.
//
// Step5: Packetize
//  When post-sched-packet is enabled (by default), repacketize the MBBs from
//  where pull up moved instructions.
//
// Step6: Post scheduling optimizations
//  These are branch optimizations and they modify the CFG. During this stage
//  MBBs having compounds are skipped because analyzeBranch cannot handle
//  compound branches.
//  During this stage AnalyzeBranchInfo is updated whenever a branch
//  optimization was successful.
//
// Step7: Generate compound instructions.
//  The function `generateCompoundsPostSched' re-generates compound instructions
//  at places where 5-instruction packets were formed during scheduling.
//
// Important functions which are used at multiple places:
//
// MII is already a bundle. This will initialize the packetizer and update
// the DFA with existing resources so that the DFA can be queried for more
// information e.g. can we put another MI in the bundle?
// void getResourcesReadyForBundle(MIInstIterType MII, bool Squeeze = false);
//
// Returns true if resources are available for MII in the current packet
// after getResourcesReadyForBundle has been called to initialize the DFA.
// Takes into account const-extenders as well.
// bool canReserveResources(MIInstIterType MII) const;
//
// When the resources are available, as checked by canReserveResources(),
// we need to check for compatibility between MI and the destination bundle.
// This function returns true if MI is compatible. This also takes care of
// a special case when a branch and its target are too far, and
// the branch might be constant extended.
// bool isLegalToPacketizeWith(MIIterType DestPack, MachineInstr *MI);
//
// While global scheduling there are two parts.
//  1. The instruction should be able to execute as the first instruction
//    within its Home/Src MBB. This is handled in findInsertionPtSpecLocal.
//
//  2. The instruction should be able to execute in parent/dominating MBB
//    without causing side-effects. This is handled in findInsertPtSpec()
//    while speculating and findInsertPtPred() while predicating.
//
// This function will return MBB->begin() if it is possible to move this
// instruction to the beginning of MBB.
// MIIterType findInsertionPtSpecLocal(MachineInstr *MI);
//
// The function findInsertionPtSpecLocal calls srcDependence which tells if
// it is possible to `detach' the MI from its Home/Src BUNDLE. For example
// when there is a .new/duplex etc. dependency we don't want to detach the MI.
// Dependence within the same packet [SrcBeg, SrcEnd).
// int srcDependence(MIInstIterType SrcBeg, MIInstIterType SrcEnd,
//                   MIInstIterType RMII,
//                   SmallPtrSetImpl<MachineInstr *> *DepSet)
//                   const;
//
// This function checks for dependencies between a range of MIIs
// in the destination/target MBB (where MII will end up). Obviously, in the
// case of predication we first create a dummy predicate instruction
// in canMIBePredicated() which calls findInsertPtPred() which then
// calls dstDependence() to check for dependencies.
// int dstDependence(MICInstIterType DestBeg, MICInstIterType DestEnd,
//                   MICInstIterType MII, MICInstIterType &DepTrue,
//                   MICInstIterType &DepAnti, unsigned HLXUseReg = 0,
//                   bool isPullDown = false) const;
//
//===----------------------------------------------------------------------===//
#include "Hexagon.h"
#include "HexagonGlobalRegion.h"
#include "HexagonInstrInfo.h"
#include "HexagonLiveVariables.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonVLIWPacketizer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <unordered_map>

#define DEBUG_TYPE "global_schedV2"

using namespace llvm;

STATISTIC(HexagonGSLocalPullups, "Number of instructions pull-ups locally");
STATISTIC(HexagonGSGlobalPullups, "Number of instructions pull-ups globally");

static cl::opt<bool>
    EnableGlobalSched("enable-global-sched", cl::Hidden, cl::init(true),
                      cl::desc("Enable global scheduler optimizer"));

static cl::opt<bool> EnablePullupPred("enable-pullup-pred", cl::Hidden,
                                      cl::init(true),
                                      cl::desc("Use pullup predication"));

static cl::opt<bool>
    EnableLocalPullup("enable-local-pullup", cl::Hidden, cl::init(true),
                      cl::desc("Speculate local MIs in parent BUNDLEs"));

static cl::opt<bool> EnableLocalMergePackets(
    "enable-local-merge-packets", cl::Hidden, cl::init(true),
    cl::desc("Push down local MIs into the next packet"));

static cl::opt<bool> PostSchedOpt("post-sched-opt", cl::Hidden, cl::init(true),
                                  cl::desc("Optimize after global scheduling"));

static cl::opt<bool>
    PostSchedPack("post-sched-packet", cl::Hidden, cl::init(true),
                  cl::desc("Packetize unbundled MIs after global scheduling"));

static cl::opt<bool>
    NoPredInBundle("nopred-sched-bundle", cl::Hidden,
                   cl::desc("Do not put predicated instructions in bundle"));

static cl::opt<bool>
    NoSpecInBundle("nospec-sched-bundle", cl::Hidden,
                   cl::desc("Do not put speculated instructions in bundle"));

static cl::opt<bool>
    EnableSqueeze("try-squeeze-bundle", cl::Hidden, cl::init(true),
                  cl::desc("Try putting 5 instructions in bundle"));

// Defaults to true up to -O2 and to false above it.
static cl::opt<bool> PreventDuplexSep(
    "prevent-duplex-sep-v2", cl::Hidden, cl::init(true),
    cl::desc(
        "Prevent separation of duplex while pullup in global scheduler V2"));

static cl::opt<bool>
    PullUnsafeMemops("pull-unsafe-memops", cl::Hidden, cl::init(true),
                     cl::desc("Pull up loads which have no memoperands"));

// Run the machine verifier at several stages of the pass (assert builds).
static cl::opt<bool> VerifyGlobalSched(
    "verify-global-sched", cl::Hidden, cl::init(false),
    cl::desc("Run the machine verifier at several stages of global scheduler"));

static cl::opt<bool>
    PredicateAllInst("predicate-all-inst", cl::Hidden,
                     cl::desc("Try predicating all possible instructions."));

static cl::opt<bool>
    SpeculateAllInst("speculate-all-inst", cl::Hidden,
                     cl::desc("Try speculating all possible instructions."));

static cl::opt<unsigned> MaxPUBundles(
    "hexagon-pullup-distance", cl::Hidden, cl::init(10),
    cl::desc("Maximum number of bundles to check for local pull-up"));

static cl::opt<bool> EnableSingleLoopSuperblock(
    "enable-single-loop-superblock", cl::Hidden, cl::init(true),
    cl::desc("Enable superblocks consisting of single block loops."));

static cl::opt<bool> EnableHVXLocalPullupLB(
    "enable-hvx-local-pullup-load-bypass", cl::Hidden, cl::init(true),
    cl::desc("Enable the HVX local pull-up pass (load bypass only)."));

static cl::opt<bool> EnableLCGPForLoadBypassOnly(
    "enable-lcgp-for-load-bypass-only", cl::Hidden, cl::init(true),
    cl::desc("Enable the HVX last chance global pull-up for load-bypass-only"));

// Enables tryTwoLoadCurBypass: a .cur load bypass for a consumer with up to
// two producers in one packet; the producer that is not the load is moved
// away by lastChanceGlobalPullup to free a slot. Off by default.
static cl::opt<bool> EnableHVXTwoLoadCurBypass(
    "enable-hvx-two-load-cur-bypass", cl::Hidden, cl::init(false),
    cl::desc("Enable the two-load .cur load-bypass case in local pull-up."));

static cl::opt<bool>
    EnableHVXLocalPullup("enable-hvx-local-pullup", cl::Hidden, cl::init(true),
                         cl::desc("Enable the HVX local pull-up pass."));

static cl::opt<bool> EnableHVXLocalPullupInnermost(
    "enable-hvx-local-pullup-innermost", cl::Hidden, cl::init(false),
    cl::desc("Enable the HVX local pull-up pass on single block innermost "
             "loops only."));

static cl::opt<bool> DisableHVXLocalPullupAndSuperblock(
    "disable-hvx-sched-superblock", cl::Hidden, cl::init(false),
    cl::desc("Disable single block loops in superblocks and the HVX local "
             "pull-up pass."));

static cl::opt<bool>
    EnableHVXLCGP("enable-hvx-lcgp", cl::Hidden, cl::init(true),
                  cl::desc("Enable the HVX last chance global pull-up pass."));

static cl::opt<bool> EnableHVXSuperblockHeuristic(
    "enable-hvx-superblock-heuristic", cl::Hidden, cl::init(true),
    cl::desc("Run HVX scheduling only on blocks whose ratio of HVX "
             "instructions exceeds -hvx-sched-profit-ratio."));

static cl::opt<float>
    HVXSchedProfitRatio("hvx-sched-profit-ratio", cl::Hidden, cl::init(0.80),
                        cl::desc("Ratio of HVX instruction/Total instructions "
                                 "for the profitability check."));

static cl::opt<bool> EnableHLXLoadStorePullup(
    "enable-hlx-load-store-pullup", cl::Hidden, cl::init(true),
    cl::desc("Pull up HLX .tmp load uses and .new HLX stores to remove the "
             "copies inserted by HLX load/store expansion"));

static cl::opt<bool>
    EnableSparsePullup("enable-sparse-pullup", cl::Hidden, cl::init(true),
                       cl::desc("Enable pullup to/from sparse packets"));

/// Enable cluster pullup optimization.
/// Cluster pullup allows pulling up groups of dependent instructions together,
/// rather than just individual instructions. This is particularly important for
/// HVX/HLX instructions where .tmp and .cur dependencies create instruction
/// chains that must be moved atomically to maintain correctness.
static cl::opt<bool> EnableClusterPullup(
    "enable-cluster-pullup", cl::Hidden, cl::init(true),
    cl::desc("Enable pullup for dependent instruction groups"));

extern cl::opt<bool> EnableDefUseSamePacket;

extern cl::opt<bool> OneFloatPerPacket;
extern cl::opt<bool> OneComplexPerPacket;

namespace {

typedef MachineBasicBlock::iterator MIIterType;
typedef MachineBasicBlock::const_iterator MICIterType;
typedef MachineBasicBlock::instr_iterator MIInstIterType;
typedef MachineBasicBlock::const_instr_iterator MICInstIterType;
typedef MachineBasicBlock::reverse_iterator MIRIterType;

// The Scheduler Class.
class GlobalScheduler : public HexagonPacketizerList {
public:
  // Data structure to store the analyzeBranch.
  // ABI => Analyze Branch Info.
  struct ABI {
    MachineBasicBlock *TBB, *FBB;
    MachineInstr *FirstTerm, *SecondTerm;
    SmallVector<MachineOperand, 4> Cond;
    bool AnalysisValid;
    ABI()
        : TBB(NULL), FBB(NULL), FirstTerm(NULL), SecondTerm(NULL),
          AnalysisValid(false) {}
    bool analyzeBranch(MachineBasicBlock *MBB, const HexagonInstrInfo *QII,
                       bool PostSched);
    void dump() {
      LLVM_DEBUG(if (TBB) dbgs() << "\nTBB: " << TBB->getNumber();
                 if (FBB) dbgs() << ", FBB: " << FBB->getNumber();
                 if (FirstTerm) dbgs() << ", FirstTerm: " << *FirstTerm;
                 if (SecondTerm) dbgs() << ", SecondTerm: " << *SecondTerm);
    }
  };
  struct AnalyzeBranchInfo_t {
    typedef std::map<MachineBasicBlock *, ABI> BranchAnalyses_t;
    BranchAnalyses_t BranchAnalyses;

    // It always returns an ABI so the caller should check if the
    // branch analysis was valid or not.
    ABI &getOrCreate(MachineBasicBlock *MBB, const HexagonInstrInfo *QII,
                     bool PostSched = false) {
      ABI &abi = BranchAnalyses[MBB];
      if (!abi.AnalysisValid)
        abi.analyzeBranch(MBB, QII, PostSched);
      return abi;
    }
    void invalidate(MachineBasicBlock *MBB) {
      iterator it = BranchAnalyses.find(MBB);
      assert((it != BranchAnalyses.end()) &&
             "Cannot invalidate un-analyzed branch.");
      BranchAnalyses.erase(it);
    }
    bool hasABInfo(MachineBasicBlock *MBB) const {
      return BranchAnalyses.find(MBB) != BranchAnalyses.end();
    }

    typedef BranchAnalyses_t::iterator iterator;
    typedef BranchAnalyses_t::const_iterator const_iterator;
    typedef BranchAnalyses_t::size_type size_type;

    iterator begin() { return BranchAnalyses.begin(); }
    const_iterator begin() const { return BranchAnalyses.begin(); }
    iterator end() { return BranchAnalyses.end(); }
    const_iterator end() const { return BranchAnalyses.end(); }
    bool empty() const { return BranchAnalyses.empty(); }
    void clear() { BranchAnalyses.clear(); }
    size_type size() const { return BranchAnalyses.size(); }
  };

  // Data structure to store a scheduling region.
  struct SchedulingRegion_t {
    typedef SmallVector<MachineBasicBlock *, 2> SRMembers_t;
    typedef MachineBasicBlock SRAtom_t;
    BranchProbability EdgeProb;
    SRMembers_t Members;
    SchedulingRegion_t() : EdgeProb(BranchProbability::getZero()) {}

    void addSRMember(MachineBasicBlock *MBB) { Members.push_back(MBB); }

    void dump() {
      SmallVector<MachineBasicBlock *, 2>::iterator MBBI = Members.begin();
      while (MBBI != Members.end()) {
        LLVM_DEBUG(dbgs() << "\tBB#" << (*MBBI)->getNumber(););
        ++MBBI;
      }
    }
    typedef SRMembers_t::iterator iterator;
    typedef SRMembers_t::reverse_iterator reverse_iterator;
    typedef SRMembers_t::const_iterator const_iterator;
    typedef SRMembers_t::const_reverse_iterator const_reverse_iterator;
    typedef SRMembers_t::size_type size_type;

    iterator begin() { return Members.begin(); }
    const_iterator begin() const { return Members.begin(); }
    reverse_iterator rbegin() { return Members.rbegin(); }
    const_reverse_iterator rbegin() const { return Members.rbegin(); }
    iterator end() { return Members.end(); }
    const_iterator end() const { return Members.end(); }
    reverse_iterator rend() { return Members.rend(); }
    const_reverse_iterator rend() const { return Members.rend(); }
    bool empty() const { return Members.empty(); }
    void clear() { Members.clear(); }
    size_type size() const { return Members.size(); }
    bool operator<(const SchedulingRegion_t &RHS) {
      return EdgeProb < RHS.EdgeProb;
    }
    bool operator<=(const SchedulingRegion_t &RHS) const {
      return EdgeProb <= RHS.EdgeProb;
    }
    bool operator>(const SchedulingRegion_t &RHS) const {
      return EdgeProb > RHS.EdgeProb;
    }
    bool operator>=(const SchedulingRegion_t &RHS) const {
      return EdgeProb >= RHS.EdgeProb;
    }
  };

  using Dependency = HexagonInstrInfo::Dependency;

  // Typedefs.
  typedef SmallVector<SchedulingRegion_t, 2> SchedulingRegions_t;
  typedef DenseMap<SchedulingRegion_t::SRAtom_t *,
                   SmallVector<MachineInstr *, 1>>
      PulledUpII;

  // Constructor.
  GlobalScheduler(const HexagonTargetMachine *tm, const HexagonInstrInfo *qii,
                  const HexagonRegisterInfo *tri, MachineFunction &mf,
                  MachineLoopInfo &mli, AliasAnalysis *aa,
                  const MachineBranchProbabilityInfo *mbpi,
                  HexagonLiveVariables *hlv)
      : HexagonPacketizerList(mf, mli, aa, mbpi, false), TM(tm), QII(qii),
        TRI(tri), HLV(hlv), MF(mf), MLI(mli), GlobalSpeculatedIns(0),
        GlobalPredicatedIns(0), LocalPullups(0), SafetyBuffer(200),
        SafetyBufferDelta(50) {
    useHVXOps = mf.getSubtarget<HexagonSubtarget>().useHVXOps();
    useHLXOps = mf.getSubtarget<HexagonSubtarget>().useHLXOps();
    // TinyCore has different scheduling constraints and resource limits.
    // Track this to adjust scheduling heuristics accordingly.
    IsTinyCore = mf.getSubtarget<HexagonSubtarget>().isTinyCore();
  }

  // Dummy addtoPacket to emulate packetization.
  MachineBasicBlock::iterator addToPacketPseudo(MachineInstr *MI);
  void initPacketizerStateLoc();
  // Is it legal to packetize \p MI with instructions in DestPack.
  bool isLegalToPacketizeWith(MIIterType DestPack, MachineInstr *MI);
  // Mimics Packetizers isLegalToPacketizeTogether, plus some more checks.
  inline bool isLegalToPacketizeTogetherMI(MachineInstr *BI, MachineInstr *MI);
  // Instructions for which pulled up is disabled
  inline bool shouldNotBePulledUp(const MachineInstr *MI) const;

  /// Check if it is legal to pull up ToMove across MI during scheduling.
  /// This function determines if MI is a "barrier" instruction that ToMove
  /// cannot be moved past.
  /// \param MI The instruction that might block the move
  /// \param ToMove The instruction being considered for pullup (optional)
  /// \param IsMIInDestBundle True if MI is in the destination bundle where
  ///        ToMove will be inserted
  /// \return true if ToMove should NOT cross MI (i.e., MI is a barrier)
  inline bool shouldNotBeCrossed(const MachineInstr *MI,
                                 const MachineInstr *ToMove,
                                 bool IsMIInDestBundle = false) const;
  // Reserve the resources for Bundle in the resource tracker.
  // Make compound/duplex instructions when \p Squeeze=true.
  void getResourcesReadyForBundle(MIInstIterType MII, bool Squeeze = false,
                                  MachineInstr *InstrToSkip = nullptr);
  MachineInstr *generateCompoundIfPossible(MIInstIterType MII,
                                           MIInstIterType &GA,
                                           MIInstIterType &GB);
  // TODO: Once MCShuffler can accept bundles with five instructions remove
  // this function.
  void generateCompoundsPostSched();

  // Functions to optimize for auto-and predicates
  bool failsSpecialConditions(MachineInstr *BundleMI, Register DestReg);
  bool canNewifyPredReg(unsigned Reg, MICInstIterType Producer,
                        MICInstIterType Sentinel) const;
  bool getDefinitions(MachineInstr *MI, MachineInstr *&Def1,
                      MachineInstr *&Def2, MachineInstr *&Def3);
  void generateAutoAndPredicates(MachineFunction &MF);

  void packetize();
  // Returns true if resources are available for MII in the current packet.
  // Takes into account const-extenders as well.
  bool canReserveResources(MIInstIterType MII) const;
  // The usage of \p Reg is replaced by \p ScratchReg.
  void replaceUsageOfReg(unsigned Reg, unsigned ScratchReg,
                         MachineBasicBlock *MBB,
                         const MachineInstr *ExceptMI = NULL) const;
  inline void markLiveInsAsUsed(std::unique_ptr<RegScavenger> &RS,
                                MachineBasicBlock *MBB) const;
  inline void markLiveOutsAsUsed(std::unique_ptr<RegScavenger> &RS,
                                 MachineBasicBlock *MBB) const;
  void markAllRegsAsUsed(std::unique_ptr<RegScavenger> &RS,
                         const MachineBasicBlock *MBB) const;
  MachineBasicBlock *splitCriticalEdge(MachineBasicBlock *FromBB,
                                       MachineBasicBlock *ToBB,
                                       MachineBasicBlock *LP);
  bool hasOneDef(const MachineInstr *MI, unsigned &DefR) const;
  bool hasOneUse(const MachineInstr *MI, unsigned &UseR) const;
  bool hasOneDefAndUse(const MachineInstr *MI, unsigned &Def,
                       unsigned &Use) const;
  MICInstIterType laterOfTheTwo(MICInstIterType I1, MICInstIterType I2,
                                MICInstIterType Sentinel) const;
  // Supporting API to figure out if the branching instruction MII,
  // is followed by non-branching instruction.
  bool branchesInBetweenMBB(MIInstIterType MII, MachineBasicBlock *MBB);

  bool splitMBBAt(MIInstIterType MII, MachineBasicBlock *MBB);

  // Generates control flow by removing 3 way branching.
  void generateCFG(MachineFunction &MF);

  virtual void schedule() = 0;

  // Mostly branch related optimizations.
  void performExposedOptimizations();
  void removeEmptyMBB(MachineBasicBlock *MBB);
  bool makeAsLayoutSuccessor(MachineBasicBlock *BB1, MachineBasicBlock *BB2);
  bool removeRedundantBranches(MachineBasicBlock *MBB, MachineBasicBlock *TBB,
                               MachineInstr *FirstTerm, MachineBasicBlock *FBB,
                               MachineInstr *SecondTerm);
  bool optimizeDualJumps(MachineBasicBlock *MBB, MachineBasicBlock *TBB,
                         MachineInstr *FirstTerm, MachineBasicBlock *FBB,
                         MachineInstr *SecondTerm);
  MachineBasicBlock *optimizeBranches(MachineBasicBlock *MBB,
                                      MachineBasicBlock *TBB,
                                      MachineInstr *FirstTerm,
                                      MachineBasicBlock *FBB);
  // A region could be a set of MBBs (Hyperblocks)
  // where scheduling algorithms could work. Default scheduling region
  // consists of MBBs where there is a single entry point i.e., the head
  // of scheduling region. It has at least one exit point i.e., the tail
  // of the scheduling region.
  // Criteria of HEAD of a scheduling region (Superblock).
  // 1. MF.begin()
  // 2. MBB with more than one successor:
  //    - Can be part of multiple scheduling regions.
  //    For simplicity we'll make this as HEAD as well. This lets us not
  //    worry about the loop as a special case.
  // 3. MBB with more than one predecessor:
  //    - Will always be (if at all) the HEAD of the scheduling region.
  virtual void formSchedulingRegions(MachineFunction &MF);

  void printSchedulingRegions() {
    LLVM_DEBUG(for (SchedulingRegions_t::iterator SRI =
                        SchedulingRegions.begin();
                    SRI != SchedulingRegions.end(); ++SRI) {
      dbgs() << "\nScheduling Region: ";
      SRI->dump();
    } dbgs() << "\nTotal Scheduling Regions: "
             << SchedulingRegions.size() << "\n";);
  }

  // PruneCriteria returns true if that scheduling region can be removed.
  typedef bool (GlobalScheduler::*PruneCriteriaFP)(SchedulingRegion_t &) const;

  // Decides whether a scheduling region is to be kept for scheduling.
  // Returns true when it is okay to remove that region.
  bool pruneCriteria(SchedulingRegion_t &SR) const {
    assert(!SR.Members.empty() && "Empty Scheduling region.");
    if (SR.Members.size() == 1)
      return true;
    // If the members are not in the same loop then remove the SR.
    // We do not want to speculate from outside of loop into the loop body,
    // unless in a very special case which we deal with in speculateMI.
    MachineLoop *HeadLI = MLI.getLoopFor(SR.Members[0]);
    MachineLoop *LeafLI = MLI.getLoopFor(SR.Members[1]);

    bool speculateInnerToOuter = false;
    if (EnableSingleLoopSuperblock && useHVXOps &&
        !DisableHVXLocalPullupAndSuperblock) {
      if (HeadLI != nullptr && LeafLI != nullptr) {
        if (HeadLI->getLoopDepth() < LeafLI->getLoopDepth())
          speculateInnerToOuter = true;
      }
    }

    if (HeadLI != LeafLI && !speculateInnerToOuter)
      return true;
    return false;
  }

  // Removes scheduling regions which cannot/need-not be scheduled.
  virtual void pruneSchedulingRegions(PruneCriteriaFP PC);

  // Prefers scheduling regions with high branch probability.
  virtual void sortSchedulingRegions();

  bool isSuperblockLoop(MachineLoop *L);
  virtual bool isProfitableToScheduleHVX(MachineLoop *L);
  virtual bool isProfitableToScheduleHVX(MachineBasicBlock *MBB);

protected:
  // Member variables.
  const HexagonTargetMachine *TM;
  const HexagonInstrInfo *QII;
  const HexagonRegisterInfo *TRI;
  HexagonLiveVariables *HLV;
  MachineFunction &MF;
  MachineLoopInfo &MLI;

  // Since we need to use analyze branch a lot. It is better to save the
  // analyzeBranch information. Care, however, must be taken to update
  // this object whenever a basic block is changed.
  AnalyzeBranchInfo_t AnalyzeBranchInfo;
  SchedulingRegions_t SchedulingRegions;
  // Locations where more opportunities of packetization are possible.
  SmallVector<MachineInstr *, 4> SpecPacketOpp;
  // Set of bundles which have compound instructions. We need to generate
  // compound because MCShuffler rejects bundles with five instructions.
  DenseSet<MachineInstr *> ToBeSqueezed;
  // Ignore any post-sched opts on these MBBs.
  SmallVector<MachineBasicBlock *, 1> IgnorePostSchedOpt;
  // List of Speculated instructions inside each MBB (MBB is the destination)
  PulledUpII SpeculatedIns;
  // Global pull up from an MBB creates scope for local push-down.
  // LocalMergePackCand keeps the list of MBBs from where MI was pulled-up.
  std::set<MachineBasicBlock *> LocalMergePackCand;
  // Single block loops for HVX local pullup pass.
  std::unordered_map<MachineBasicBlock *, float> ProfitableHVXBlocks;
  unsigned GlobalSpeculatedIns;
  unsigned GlobalPredicatedIns;
  unsigned LocalPullups;
  const unsigned SafetyBuffer;
  // In case when MBBs are moved in between Jump and their targets.
  int SafetyBufferDelta;
  // Is HVX support enabled?
  bool useHVXOps = false;
  // Is HLX support enabled?
  bool useHLXOps = false;
  // Is target CPU TinyCore? TinyCore has reduced resources and different
  // scheduling constraints compared to full Hexagon cores.
  bool IsTinyCore = false;
};

class GreedyPullup : public GlobalScheduler {
public:
  GreedyPullup(const HexagonTargetMachine *tm, const HexagonInstrInfo *qii,
               const HexagonRegisterInfo *tri, MachineFunction &mf,
               MachineLoopInfo &mli, AliasAnalysis *aa,
               const MachineBranchProbabilityInfo *mbpi,
               HexagonLiveVariables *hlv)
      : GlobalScheduler(tm, qii, tri, mf, mli, aa, mbpi, hlv) {}

  void schedule() override {
    for (SchedulingRegions_t::iterator SRI = SchedulingRegions.begin();
         SRI != SchedulingRegions.end(); ++SRI) {
      unsigned currSpec = GlobalSpeculatedIns;
      unsigned currPred = GlobalPredicatedIns;
      globalPullup(*SRI);
      LLVM_DEBUG(SRI->dump(););
      (void)currSpec;
      (void)currPred;
      LLVM_DEBUG(
          dbgs() << "\n\nCurrent BB Global Pullup stats: Speculated Ins: "
                 << GlobalSpeculatedIns - currSpec
                 << ", Predicated Ins: " << GlobalPredicatedIns - currPred;);
    }

    MachineFunction::iterator MBBI = MF.begin();

    if (EnableLocalPullup)
      while (MBBI != MF.end()) {
        LocalPullups = 0;
        if (!MBBI->empty() && // Empty elements in a scheduling region.
            std::find(IgnorePostSchedOpt.begin(), IgnorePostSchedOpt.end(),
                      &*MBBI) == IgnorePostSchedOpt.end()) { // Ignorable MBB?
          localPullupHLX(&*MBBI);
          localPullup(&*MBBI);
          localPullupHVX(&*MBBI);
        }

        ++MBBI;
        LLVM_DEBUG(
            dbgs() << "\n\nCurrent BB Local Pullup stats: Speculated Ins: "
                   << LocalPullups;);
        // TODO: Keep track of MBBs where LocalPullup > 0. Then we can do
        // global scheduling in the regions having this MBB as more
        // opportunities might have arisen.
        HexagonGSLocalPullups += LocalPullups;
      }

    // Merge an instruction into adjacent packet.
    if (EnableLocalMergePackets)
      mergePackets();

    HexagonGSGlobalPullups = GlobalPredicatedIns + GlobalSpeculatedIns;

    LLVM_DEBUG(
        dbgs() << "\n\n**********Total Greedy Pullup stats**********"
               << "\nGlobal speculated instructions: " << GlobalSpeculatedIns
               << "\nGlobal predicated instructions: " << GlobalPredicatedIns
               << "\nLocal speculated instruction: " << HexagonGSLocalPullups
               << "\n";);
  }

  // Check if moving SrcMI to DestBundle will cause an interlock stall.
  bool canCauseInterlockStall(MIInstIterType DestBundle, MachineInstr *SrcMI,
                              const InstrItineraryData *II);
  // Generate the load-bypass (.cur) after moving InstrToMove after LoadInstr.
  void genLoadBypass(MachineInstr *LoadInstr, MachineInstr *InstrToMove);
  // Returns true if a single RAW dependence on \p DepTrue (a .cur-able HVX
  // load in DestBundle) can be satisfied via a .cur load bypass when pulling
  // up a consumer. \p PktEnd bounds the destination packet for the
  // no-other-consumer safety check.
  bool canGenCurBypass(MIInstIterType DestBundle, MachineInstr *DepTrue,
                       MIInstIterType PktEnd, int DstDep,
                       MachineInstr *Consumer,
                       const MachineInstr *Exempt = nullptr);
  // Two-load .cur bypass: try to pull \p SrcMI (a single consumer with two
  // RAW producers sharing a packet, one a .cur-able load) into \p DestBundle,
  // relocating the other producer via lastChanceGlobalPullup. Returns true if
  // the candidate was handled (moved); iterators are updated in place.
  bool tryTwoLoadCurBypass(MIIterType &BI, SchedulingRegion_t::SRAtom_t *SR,
                           MIInstIterType &DestBundle, MIIterType &SrcBundle,
                           MIIterType &NextBI, bool isNextBISrc,
                           MIIterType &Next, MachineInstr *SrcMI,
                           SmallPtrSetImpl<MachineInstr *> &PulledUpSet);
  void localPullupHVX(SchedulingRegion_t::SRAtom_t *SR);
  // Shared local pullup engine used by localPullup (HVXMode=false) and
  // localPullupHVX (HVXMode=true).
  void localPullupImpl(SchedulingRegion_t::SRAtom_t *SR, bool HVXMode);
  // Pullup HLX instructions to eliminate copies inserted for
  // .new HLX store and .tmp HLX load
  void localPullupHLX(SchedulingRegion_t::SRAtom_t *SR);
  void localPullupHLXLoadUses(SchedulingRegion_t::SRAtom_t *SR,
                              MachineInstr *HLXLoadMI, MachineInstr *HLXCopyMI);
  void localPullupHLXStore(SchedulingRegion_t::SRAtom_t *SR,
                           MachineInstr *HLXStoreMI, MachineInstr *HLXCopyMI);
  // Within an MBB
  void localPullup(SchedulingRegion_t::SRAtom_t *SR);
  // Local push-down of MI into adjacent packets.
  bool mergePacketsAround(const SmallVectorImpl<MachineInstr *> &Candidates);
  bool mergePackets();
  // Across MBBs
  void globalPullup(SchedulingRegion_t &SR);
  // Returns true if MI can be speculated (without predication)
  // from \p SrcBB to \p DstBB
  bool canMIBeSpeculated(MachineInstr *MI, MachineBasicBlock *SrcBB,
                         MachineBasicBlock *DstBB, MIIterType &TargetPacket);
  // Safety checks for instructions e.g., load, store etc.
  bool safeToSpeculate(MachineInstr *MI, MachineBasicBlock *DstBB) const;

  bool preciousMICanBeSpeculated(MachineInstr *MI, MachineBasicBlock *SrcBB,
                                 MachineBasicBlock *DstBB,
                                 MachineBasicBlock *OtherBB,
                                 MIIterType &TargetPacket,
                                 unsigned ApproachNum);
  inline bool preciousToSpeculate(const MachineInstr *MI,
                                  const MachineBasicBlock *SrcBB,
                                  const MachineBasicBlock *DstBB,
                                  const MachineBasicBlock *OtherBB) const;
  inline bool preciousToPredicate(const MachineInstr *MI,
                                  const MachineBasicBlock *SrcBB,
                                  const MachineBasicBlock *DstBB,
                                  const MachineBasicBlock *OtherBB) const;

  // Returns true if MI can be speculatively executed (with predication)
  // from \p SrcBB to \p DstBB
  bool canMIBePredicated(MachineInstr *MI, MachineBasicBlock *SrcBB,
                         MachineBasicBlock *DstBB, MIIterType &TargetPacket,
                         MICInstIterType &WithinPacket, bool &NeedToNewify,
                         bool &Precious);
  // Returns the position *before* which \p MI could be inserted.
  // In case of BUNDLE the \p MI can be inserted within that BUNDLE.
  // It will try to take an extra step in accommodating the \p SrcMI when
  // \p SpecPreciousMI=true.
  MIIterType findInsertPtSpec(MachineInstr *SrcMI, MachineBasicBlock *DstBB,
                              bool SpecPreciousMI, bool SpecPreciousMIComp,
                              bool GlobalPrecious = false);
  // Returns the position *before* which \p MI could be inserted.
  // In case of BUNDLE the \p MI can be inserted within that BUNDLE.
  // @note: The SrcMI can be modified by this function.
  MIIterType findInsertPtPred(MachineInstr *SrcMI, MachineBasicBlock *DstBB,
                              bool &NeedToNewify, MICInstIterType &WithinPacket,
                              bool PredPreciousMI = false);
  // Returns true if MI was successfully speculated.
  bool speculateMI(MachineInstr *MI, MachineBasicBlock *SrcBB,
                   MachineBasicBlock *DstBB, MIIterType DstPos);
  // Returns true if MI was successfully speculated.
  bool predicateMI(MachineInstr *MI, MachineBasicBlock *SrcBB,
                   MachineBasicBlock *DstBB, MIIterType DstPack,
                   MICInstIterType WithinPacket, bool NeedToNewify,
                   bool Precious);
  // Is it okay to execute MI as the first instruction in the MBB.
  MIIterType findInsertionPtSpecLocal(MachineInstr *MI);

  /// Check dependencies between MII and instructions in range [DestBeg,
  /// DestEnd). This is used during pullup to ensure moving MII to the
  /// destination doesn't violate dependencies. [DestBeg, DestEnd) can
  /// span multiple bundles.
  /// \param DestBeg Start of destination range
  /// \param DestEnd End of destination range
  /// \param MII Instruction being checked for dependencies
  /// \param DepTrue Output: iterator to true (RAW) dependency if found
  /// \param DepAnti Output: iterator to anti (WAR) dependency if found
  /// \param HLXUseReg Optional: specific register to check for HLX
  /// dependencies
  /// \param isPullDown True if checking for pulldown (vs pullup) operation
  /// \return Dependency flags indicating type of dependencies found
  int dstDependence(MICInstIterType DestBeg, MICInstIterType DestEnd,
                    MICInstIterType MII, MICInstIterType &DepTrue,
                    MICInstIterType &DepAnti, unsigned HLXUseReg = 0,
                    bool IsPullDown = false) const;
  // A more exhaustive dependence analysis for load bypass, more aggressive
  // moves, etc.  Used from the HVX local pullup.
  int dstDependence(MICInstIterType DestBeg, MICInstIterType DestEnd,
                    MICInstIterType MII,
                    std::vector<const MachineInstr *> &DepTrueVec,
                    std::vector<const MachineInstr *> &DepAntiVec,
                    std::vector<const MachineInstr *> &DepOutputVec,
                    std::vector<unsigned> *Regs = nullptr) const;
  // Dependence within the same packet [SrcBeg, SrcEnd).
  int srcDependence(MIInstIterType SrcBeg, MIInstIterType SrcEnd,
                    MIInstIterType RMII,
                    SmallPtrSetImpl<MachineInstr *> *DepSet = nullptr) const;
  bool singleTrueDep(MICInstIterType SrcBeg, MICInstIterType SrcEnd,
                     MICInstIterType MII, unsigned &Reg) const;
  bool lastChanceGlobalPullup(MIInstIterType &BundleHead,
                              MIInstIterType InstrToMove,
                              const MachineInstr *LoadBypassCand = nullptr);
  bool isLoopInvariant(MachineInstr &MI);
  MachineInstr *isLastChanceGlobalPullupCand(const MachineInstr *MI);
};

class HexagonGlobalSchedulerV2 : public MachineFunctionPass {
public:
  static char ID;

  HexagonGlobalSchedulerV2() : MachineFunctionPass(ID), GS(nullptr) {
    initializeHexagonGlobalSchedulerV2Pass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Hexagon Global Scheduler V2";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<HexagonLiveVariables>();
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  // Generates control flow by removing 3 way branching.
  inline void generateCFG(MachineFunction &MF);

  // Run the machine verifier when -verify-global-sched is set.
  bool verifyMachineFunction(MachineFunction &MF, StringRef msg) {
    if (VerifyGlobalSched)
      MF.verify(this, msg.data());
    return true;
  }

  // Member variables.
  const HexagonTargetMachine *TM;
  const HexagonInstrInfo *QII;
  const HexagonRegisterInfo *TRI;
  std::unique_ptr<GlobalScheduler> GS;
};

char HexagonGlobalSchedulerV2::ID = 0;

} // namespace

INITIALIZE_PASS(HexagonGlobalSchedulerV2, "global-schedV2",
                "Hexagon Global Scheduler V2", false, false)

//===----------------------------------------------------------------------===//
//                    static/inline Functions
//===----------------------------------------------------------------------===//
static inline bool isSchedBarrier(const MachineInstr *MI) {
  return MI->getOpcode() == Hexagon::Y2_barrier;
}

// Heuristic to determine whether to run the HVX local pull-up pass.
bool GlobalScheduler::isProfitableToScheduleHVX(MachineBasicBlock *MBB) {
  if (!EnableHVXSuperblockHeuristic)
    return true;
  auto [It, Inserted] = ProfitableHVXBlocks.try_emplace(MBB);
  if (!Inserted)
    return It->second > HVXSchedProfitRatio;

  unsigned NumHVXInst = 0, NumTotalInst = 0;
  for (MIInstIterType II = MBB->instr_begin(), End = MBB->instr_end();
       II != End; ++II) {
    if (II->isBundle())
      continue;
    if (QII->isHVXVec(*II))
      ++NumHVXInst;
    ++NumTotalInst;
  }
  if (NumTotalInst == 0)
    It->second = 0;
  else
    It->second = (float)NumHVXInst / (float)NumTotalInst;
  return It->second > HVXSchedProfitRatio;
}

// This heuristic can be tuned to allow/disallow formation of superblocks
// containing single block innermost loops.  Currently, it's a simple ratio
// of HVX vs non-HVX instructions.
bool GlobalScheduler::isProfitableToScheduleHVX(MachineLoop *L) {
  assert(L->getNumBlocks() == 1 && "Expecting a single block loop");
  if (!EnableHVXSuperblockHeuristic)
    return true;
  auto *BB = L->getBlocks()[0];
  return isProfitableToScheduleHVX(BB);
}

// We allow single block innermost loops to be a part of the superblock to
// attempt to schedule them,
bool GlobalScheduler::isSuperblockLoop(MachineLoop *L) {
  if (!EnableSingleLoopSuperblock || !useHVXOps ||
      DisableHVXLocalPullupAndSuperblock)
    return false;
  if (!L || L->getNumBlocks() != 1 || !L->isInnermost())
    return false;
  if (L->getBlocks()[0]->succ_size() < 2)
    return false;
  if (L->getBlocks()[0]->pred_size() != 2)
    return false;

  return isProfitableToScheduleHVX(L);
}

bool GlobalScheduler::shouldNotBePulledUp(const MachineInstr *MI) const {
  // Allow .new HLX store to be pulled up
  if (MI->getOpcode() == Hexagon::LX_S16b_new)
    return false;
  if (MI->mayStore() || MI->isCall() || MI->isBarrier() || MI->isInlineAsm() ||
      MI->hasUnmodeledSideEffects() || MI->isEHLabel() || isSchedBarrier(MI))
    return true;
  if (MI->isBundle() || MI->isDebugInstr())
    return true;
  // Loads with ordered (volatile) memoperands. Assuming loads with empty
  // memoperands are safe to pull unlike MachineInstr::hasOrderedMemoryRef.
  if (PullUnsafeMemops && MI->memoperands_empty())
    return false;
  return MI->hasOrderedMemoryRef();
}

/// Determines if MI is a barrier that prevents ToMove from being pulled up past
/// it. This function implements the core safety checks for instruction
/// reordering.
///
/// Key rules:
/// - Returns, calls, inline asm, EH labels, Y2_barrier and instructions with
///   unmodeled side effects are always barriers.
/// - A load or store is a barrier when ToMove is unknown.
/// - MI does not block ToMove when both may load.
/// - A .tmp or .cur load in the destination bundle (IsMIInDestBundle) does not
///   block a store, because it executes early in the packet.
/// - Otherwise, a memory access blocks ToMove when areMemAccessesDependent
///   says they may alias.
bool GlobalScheduler::shouldNotBeCrossed(const MachineInstr *MI,
                                         const MachineInstr *ToMove,
                                         bool IsMIInDestBundle) const {
  if (MI->isBundle() || MI->isDebugInstr())
    return false;

  if (MI->isReturn() || MI->isCall() || MI->hasUnmodeledSideEffects() ||
      MI->isInlineAsm() || MI->isEHLabel() || isSchedBarrier(MI))
    return true;

  // MI is not a load or store
  if (!MI->mayLoadOrStore())
    return false;

  // MI is load or store, but we do not know what is ToMove
  if (!ToMove)
    return true;

  // MI is load or store, ToMove is not a load or store
  if (!ToMove->mayLoadOrStore())
    return false;

  if (MI->mayLoad()) {
    // If MI is a load and ToMove are loads, MI and ToMove can be reordered
    if (ToMove->mayLoad())
      return false;

    // MI is a load and ToMove is a store
    // If MI is in DestBundle, ignore dependence if the load is a .tmp or .cur
    // .tmp and .cur loads execute early
    if (IsMIInDestBundle && (QII->isDotTmpInst(*MI) || QII->isDotCurInst(*MI)))
      return false;
  }
  return QII->areMemAccessesDependent(AA, MI, ToMove);
}

static std::vector<const MachineOperand *>
collectRegmaskMOs(const MachineBasicBlock &MBB) {
  std::vector<const MachineOperand *> regmaskMOs;
  for (MICInstIterType II = MBB.instr_begin(), End = MBB.instr_end(); II != End;
       ++II) {
    if (II->isBundle() || II->isDebugInstr())
      continue;
    for (auto &MO : II->operands())
      if (MO.isRegMask())
        regmaskMOs.push_back(&MO);
  }
  return regmaskMOs;
}

// Check if MI is invariant in a single-block loop.
// 1) All its uses are loop invariant
// 2) It's not redefined
// Call-clobbered registers are found from the regmask operands in the block.
bool GreedyPullup::isLoopInvariant(MachineInstr &MI) {
  auto *MBB = MI.getParent();
  MachineLoop *Loop = MLI.getLoopFor(MBB);
  if (!Loop)
    return true;
  if (!isSuperblockLoop(Loop))
    return false;
  MICInstIterType DefUseR = MBB->instr_end();

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    unsigned Op = MO.getReg();
    if (MO.isUse()) {
      if (HLV->isDefinedWithin(MBB->instr_begin(), MBB->instr_end(), Op,
                               DefUseR))
        return false;
    } else if (MO.isDef()) {
      if (!MO.isDead()) {
        if (HLV->isDefinedWithin(std::next(MI.getIterator()), MBB->instr_end(),
                                 Op, DefUseR) ||
            HLV->isDefinedWithin(MBB->instr_begin(), MI.getIterator(), Op,
                                 DefUseR))
          return false;

        // Not a candidate if it's live out
        for (MCRegAliasIterator AI(Op, TRI, true); AI.isValid(); ++AI)
          if (HLV->isLiveOut(MBB, *AI))
            return false;
      }
    }
    auto regmaskMOs = collectRegmaskMOs(*MBB);
    for (auto *MO : regmaskMOs)
      if (MO->clobbersPhysReg(Op))
        return false;
  }
  return true;
}

// Get the iterators corresponding to the BUNDLE enclosing MI.
// If MI is not inside the bundle it returns MI as the first element
// and the MI+1, as the second element.
static std::pair<MIInstIterType, MIInstIterType>
getParentBundle(MachineInstr *MI) {
  MIInstIterType BundleStart = getBundleStart(MI->getIterator());
  MIInstIterType BundleEnd = getBundleEnd(MI->getIterator());
  return std::make_pair(BundleStart, BundleEnd);
}

// Checks if the bundle has at least 2 instructions.
// If not then unbundles the instructions and save in SpecScope (if provided).
static MachineInstr *
updateBundle(MachineInstr *BundleHead, const HexagonInstrInfo *QII,
             SmallVectorImpl<MachineInstr *> *SpecScope = nullptr) {
  unsigned Size = QII->nonDbgBundleSize(BundleHead);
  if (Size >= 2)
    return nullptr;
  MachineInstr *MI = nullptr;
  if (Size == 1) {
    // Since bundle may have debug instructions.
    MIInstIterType BundleEnd = getBundleEnd(BundleHead->getIterator());
    MIInstIterType MIter = BundleHead->getIterator();
    ++MIter; // Skip the bundle header.
    while (MIter != BundleEnd) {
      MI = &*(MIter++);
      MI->unbundleFromPred();
      if (SpecScope)
        SpecScope->push_back(MI);
    }
  }
  // In case a bundle full of debug instructions get here.
  assert(BundleHead->getBundleSize() == 0);
  BundleHead->eraseFromParent();
  return MI; // if non-nullptr, point to the instruction
}

// When MI is removed from a two-instruction bundle, the other instruction
// becomes unbundled.  Return it in singleInstr.
static MachineInstr *
removeMI(MachineInstr *MI, const HexagonInstrInfo *QII,
         SmallVectorImpl<MachineInstr *> *SpecPacketOpp = nullptr,
         MachineInstr **singleInstr = nullptr) {
  MachineInstr *NewSingleInstr = nullptr;
  MachineInstr *MIP = nullptr;
  if (MI->isBundled()) {
    MachineInstr *Bundle = &*getBundleStart(MI->getIterator());
    MIP = MI->removeFromBundle();
    NewSingleInstr = updateBundle(Bundle, QII, SpecPacketOpp);
  } else
    MIP = MI->removeFromParent();

  if (singleInstr)
    *singleInstr = NewSingleInstr;
  return MIP;
}

// TODO: Preprocess the MachineFunction to collect these
// information so that we don't have to loop over and over again.
// Use analyzebranch data (the cond vector)
inline bool endloopAfterMI(MachineBasicBlock::instr_iterator MII,
                           MachineBasicBlock *MBB) {
  while (MII != MBB->instr_end()) {
    if ((MII->getOpcode() == Hexagon::ENDLOOP0) ||
        (MII->getOpcode() == Hexagon::ENDLOOP1))
      return true;
    ++MII;
  }
  return false;
}

static void transferLiveIns(const MachineBasicBlock *From,
                            MachineBasicBlock *To) {
  // Update liveins all at once.
  for (MachineBasicBlock::livein_iterator it = From->livein_begin();
       it != From->livein_end(); ++it) {
    if (!To->isLiveIn((*it).PhysReg))
      To->addLiveIn(*it);
  }
}

// Returns true if Iter is a barrier and if there are other barriers
// 'after' Iter.
static bool inBetweenBarriers(MICIterType Iter,
                              const MachineBasicBlock *DstBB) {
  MICIterType TestIt = DstBB->begin();
  // If Iter is after a Barrier.
  while (TestIt != Iter) {
    if (Iter->isBarrier())
      return true;
    ++TestIt;
  }
  // No barrier before Iter and Iter is not a barrier => okay.
  if (!Iter->isBarrier())
    return false;
  // Iter is a barrier, there should not be a barrier after Iter.
  while (Iter != DstBB->end()) {
    if (Iter->isBarrier())
      return true;
    ++Iter;
  }
  return false;
}

// Blocks should be considered empty if they contain only debug info;
// else the debug info would affect codegen.
static bool isEmptyBlock(MachineBasicBlock *MBB) {
  if (MBB->isEHPad() || MBB->hasAddressTaken() || MBB->pred_size() != 1 ||
      MBB->succ_empty())
    return false;
  for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
       MBBI != MBBE; ++MBBI) {
    if (!MBBI->isDebugInstr())
      return false;
  }
  return true;
}

// When the DstBB does not contain SrcMI, start from the end.
// Otherwise start from the bundle_iterator just before SrcMI.
static MIRIterType getSpeculationBeginPoint(MachineInstr *SrcMI,
                                            MachineBasicBlock *DstBB) {
  MIRIterType Rb = DstBB->rbegin();
  MIRIterType Re = DstBB->rend();

  if (SrcMI->getParent() != DstBB)
    return Rb;

  std::pair<MICInstIterType, MICInstIterType> B = getParentBundle(SrcMI);
  // TODO: Is there an advance function for reverse instr iterators?
  while ((&*Rb != &*B.first) && (Rb != Re))
    ++Rb;
  return ++Rb;
}

//===----------------------------------------------------------------------===//
//                    GlobalScheduler::ABI Functions
//===----------------------------------------------------------------------===//
// TODO: Modify this to handle cases where succ_size=1.
bool GlobalScheduler::ABI::analyzeBranch(MachineBasicBlock *MBB,
                                         const HexagonInstrInfo *QII,
                                         bool PostSched) {
  AnalysisValid = !QII->analyzeBranch(*MBB, TBB, FBB, Cond, false);
  if (AnalysisValid) {
    // According to the HexagonInstrInfo::getBranchingInstrs
    // Only the last element can be a conditional branching instruction.
    SmallVector<MachineInstr *, 2> Jumpers = QII->getBranchingInstrs(*MBB);
    if (!Jumpers.empty() && ((MBB->succ_size() == 2) || PostSched)) {
      unsigned jumpers_size = Jumpers.size();
      assert((jumpers_size == 1) || (jumpers_size == 2));
      FirstTerm = Jumpers.back();
      if (jumpers_size == 2)
        SecondTerm = Jumpers[0];
      if (MBB->succ_size() == 2) {
        FBB = *MBB->succ_begin();
        if (FBB == TBB)
          FBB = *MBB->succ_rbegin(); // Since only two successors.
      }
    } else {
      AnalysisValid = false;
    }
  }
  return AnalysisValid;
}

//===----------------------------------------------------------------------===//
//                    GlobalScheduler Functions
//===----------------------------------------------------------------------===//
bool GlobalScheduler::branchesInBetweenMBB(MIInstIterType MII,
                                           MachineBasicBlock *MBB) {
  if (!MII->isBranch())
    return false;
  // Get the beginning of the bundle containing MII.
  while (MII->isInsideBundle())
    --MII;
  // Now, skip to the next bundle.
  MII = std::next(MachineBasicBlock::iterator(MII)).getInstrIterator();
  if (MII == MBB->instr_end())
    return false;

  if (endloopAfterMI(MII, MBB))
    return false;
  if (MII->isReturn())
    return false;
  // Anything other than a direct jump => branches in between or, > 2 succs.
  // MII may be a bundle => > 2 following instructions.
  if (MII->isBranch())
    return false;
  return true;
}

bool GlobalScheduler::splitMBBAt(MIInstIterType MII, MachineBasicBlock *MBB) {
  MachineBasicBlock *OrigTBB = NULL, *OrigFBB = NULL;
  SmallVector<MachineOperand, 4> OrigCond;

  // Not storing analyzeBranch info here because this will be stale soon.
  if (QII->analyzeBranch(*MBB, OrigTBB, OrigFBB, OrigCond, false) ||
      (MBB->succ_size() != 2))
    return false;

  MachineBasicBlock *Succ1 = *MBB->succ_begin();
  MachineBasicBlock *Succ2 = *MBB->succ_rbegin();

  if (!OrigFBB) {
    if (OrigTBB == Succ1)
      OrigFBB = Succ2;
    OrigFBB = Succ1;
  }

  const BasicBlock *IR_Block = MBB->getBasicBlock();
  MachineFunction *MF = MBB->getParent();
  MachineBasicBlock *NewBlock = MF->CreateMachineBasicBlock(IR_Block);
  MachineFunction::iterator MBBI = MBB->getIterator();

  MF->insert(++MBBI, NewBlock);
  NewBlock->splice(NewBlock->end(), MBB, MII, MBB->end());

  MachineBasicBlock *FirstTBB = NULL, *FirstFBB = NULL;
  SmallVector<MachineOperand, 4> FirstCond;

  (void)FirstTBB;
  (void)FirstFBB;
  (void)FirstCond;
  assert(!QII->analyzeBranch(*MBB, FirstTBB, FirstFBB, FirstCond, false));

  MBB->removeSuccessor(OrigFBB);
  MBB->addSuccessor(NewBlock);
  NewBlock->addSuccessor(OrigFBB);
  // transfer liveins from MBB to NewBlock, just copy the parent's livein.
  transferLiveIns(MBB, NewBlock);
  return true;
}

// Ensures that there are no more than 2 successors of a MBB
// with direct branching instructions.
// Also, performs a small optimization of removing branches to immediate succ.
// TODO: Some useful CFG optimizations can be done here to aid in scheduling.
void GlobalScheduler::generateCFG(MachineFunction &MF) {
  MachineFunction::iterator MBBI = MF.begin();
  while (MBBI != MF.end()) {
    MachineBasicBlock *MBB = &*MBBI;
    MIInstIterType MII = MBBI->instr_begin();
    while (MII != MBBI->instr_end()) {
      MachineInstr *MI = &*MII;
      ++MII; // As MI might be removed.
      if (branchesInBetweenMBB(MI->getIterator(), MBB)) {
        // Splitting the MBB at MII.
        LLVM_DEBUG(dbgs() << "\nSplitting the MBB#" << MBB->getNumber()
                          << " at: ";
                   MI->dump());
        if (!splitMBBAt(std::next(MI->getIterator()), MBB))
          assert(MBB->succ_size() > 2);
        LLVM_DEBUG(dbgs() << "\nResulting BB: "; MBB->dump(););
      } else if (MI->getOpcode() == Hexagon::J2_jump &&
                 MI->getOperand(0).isMBB()) {
        // If it jumps to the layout successor, remove the J2_jump instruction.
        MachineBasicBlock *Target = MI->getOperand(0).getMBB();
        if (MBB->isLayoutSuccessor(Target)) {
          LLVM_DEBUG(dbgs() << "\nRemoving branch to layout succ." << *MI);
          if (MI->isBundled()) {
            MachineInstr *Head = &*getBundleStart(MI->getIterator());
            MI->eraseFromBundle();
            updateBundle(Head, QII);
          } else
            MI->eraseFromParent();
        }
      }
    }
    ++MBBI;
  }
}

// TODO: This needs to be improved. Is there a way to just 'copy'
// \p BB2 and emplace just after BB1?
// @brief Make BB2 as layout successor of BB1.
bool GlobalScheduler::makeAsLayoutSuccessor(MachineBasicBlock *BB1,
                                            MachineBasicBlock *BB2) {
  assert(!BB1->isLayoutSuccessor(BB2));
  // We can only deal with returning MBBs and MBBs with a direct jump.
  if (BB2->pred_size() != 1 || (BB2->succ_size() > 1))
    return false;
  // We don't want to move in MBBs if they change the CFG too much.
  // Passing same BB2 in both params to get the num_instrs*instr_size
  int InsSize = HLV->getDistanceBetween(BB2, BB2);
  if (SafetyBufferDelta < InsSize)
    return false;
  SafetyBufferDelta -= InsSize;

  SmallVector<MachineInstr *, 2> Jumpers = QII->getBranchingInstrs(*BB2);
  if (Jumpers.size() > 1 || Jumpers.empty())
    return false;
  if (QII->isPredicated(*Jumpers[0]))
    return false;
  LLVM_DEBUG(dbgs() << "\nMaking MBB#" << BB2->getNumber()
                    << " as LayoutSuccessor of " << BB1->getNumber(););
  MachineFunction::iterator BB1LS = BB1->getIterator();
  ++BB1LS; // Current layout successor of BB1
  if (BB1LS != MF.end()) {
    MF.splice(BB1LS, BB2);
    return true;
  }
  return false;
}

void GlobalScheduler::removeEmptyMBB(MachineBasicBlock *MBB) {
  assert((MBB->succ_size() == 1) && (MBB->pred_size() == 1));
  LLVM_DEBUG(dbgs() << "\nRemoving:" << *MBB);
  MachineBasicBlock *Pred = *MBB->pred_begin();
  LLVM_DEBUG(dbgs() << "\nPred:" << *Pred);
  LLVM_DEBUG(dbgs() << "\nSucc:" << **MBB->succ_begin());
  Pred->ReplaceUsesOfBlockWith(MBB, *MBB->succ_begin());
  LLVM_DEBUG(dbgs() << "\nPred after update:" << *Pred);
  MachineJumpTableInfo *MJTI = MBB->getParent()->getJumpTableInfo();
  if (MJTI) {
    MJTI->ReplaceMBBInJumpTables(MBB, *MBB->succ_begin());
  }
  MBB->removeSuccessor(MBB->succ_begin());
  LLVM_DEBUG(dbgs() << "Removing BB(" << MBB->getNumber()
                    << ") from parent.\n");
  MBB->eraseFromParent();
}

// 1. Remove jump to the layout successor.
// 2. Remove multiple (dual) jump to the same target.
bool GlobalScheduler::removeRedundantBranches(MachineBasicBlock *MBB,
                                              MachineBasicBlock *TBB,
                                              MachineInstr *FirstTerm,
                                              MachineBasicBlock *FBB,
                                              MachineInstr *SecondTerm) {
  bool Analyzed = false;
  LLVM_DEBUG(dbgs() << "\n[removeRedundantBranches]");
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
    if (++FirstTermIter == SecondTermIter &&
        !QII->isCompoundBranchInstr(*FirstTerm)) {
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
  } else if (!SecondTerm && !FirstTerm->isUnconditionalBranch() &&
             MBB->isLayoutSuccessor(TBB)) {
    // Branching+Fallthrough to the layout successor.
    // Jmp_c layout_succ_basic_block <-- Remove
    assert(MBB->succ_size() == 1 && "Malformed CFG");
    ToErase = FirstTerm;
  }
  // Remove the instruction from the BB
  if (ToErase) {
    if (ToErase->isBundled()) {
      Head = &*getBundleStart(ToErase->getIterator());
      ToErase->eraseFromBundle();
      updateBundle(Head, QII);
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
bool GlobalScheduler::optimizeDualJumps(MachineBasicBlock *MBB,
                                        MachineBasicBlock *TBB,
                                        MachineInstr *FirstTerm,
                                        MachineBasicBlock *FBB,
                                        MachineInstr *SecondTerm) {
  LLVM_DEBUG(dbgs() << "\n[optimizeDualJumps]");

  bool Analyzed = false;

  if (QII->PredOpcodeHasJMP_c(FirstTerm->getOpcode()) &&
      (SecondTerm->getOpcode() == Hexagon::J2_jump)) {

    if (TBB == FBB) {
      LLVM_DEBUG(dbgs() << "\nBoth successors are the same.");
      return Analyzed;
    }

    LLVM_DEBUG(dbgs() << "\nCandidate for jump optimization in BB("
                      << MBB->getNumber() << ").\n";);

    // Predicated jump to layout successor followed by an unconditional jump.
    if (MBB->isLayoutSuccessor(TBB)) {
      // Check if the jump in the last instruction is within range.
      unsigned Distance = HLV->getDistanceBetween(MBB, FBB) + SafetyBuffer;
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
        updateBundle(SecondHead, QII);
      } else if (FirstHead == SecondHead) {
        // 4. Both are in the same bundle
        assert((FirstHead && SecondHead) && "Unbundled Instruction");
        SecondTerm->eraseFromBundle();
        updateBundle(SecondHead, QII);
      } else {
        // 5. Both are in different bundles
        SecondTerm->eraseFromBundle();
        updateBundle(SecondHead, QII);
      }
      Analyzed = true;
    }
  }
  return Analyzed;
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
MachineBasicBlock *GlobalScheduler::optimizeBranches(MachineBasicBlock *MBB,
                                                     MachineBasicBlock *TBB,
                                                     MachineInstr *FirstTerm,
                                                     MachineBasicBlock *FBB) {
  LLVM_DEBUG(dbgs() << "\n[optimizeBranches]");
  if ((TBB == MBB) || (FBB == MBB))
    LLVM_DEBUG(dbgs() << "Cannot deal with loops in BB#" << MBB->getNumber(););

  // { if(p) jump t1; }
  // t1: { jump t2; }
  // --> { if(p) jump t2
  // remove t1: { jump t2; }, if it's address is not taken/not a landing pad.
  if (QII->nonDbgBBSize(TBB) == 1) {
    MachineInstr *TBBMIb = &*TBB->getFirstNonDebugInstr();
    if (TBBMIb->getOpcode() == Hexagon::J2_jump &&
        TBBMIb->getOperand(0).isMBB()) {
      MachineBasicBlock *NewTarget = TBBMIb->getOperand(0).getMBB();
      if (TBB->hasAddressTaken() || TBB->isEHPad() ||
          TBB == NewTarget /*Infinite loop*/ || (TBB->pred_size() != 1))
        return NULL;

      LLVM_DEBUG(dbgs() << "\nSuboptimal branching in TBB");
      // Check if the jump in the last instruction is within range.
      unsigned Distance =
          HLV->getDistanceBetween(MBB, NewTarget) + SafetyBuffer;
      if (!QII->isJumpWithinBranchRange(*FirstTerm, Distance)) {
        LLVM_DEBUG(dbgs() << "\nUnconditional jump target:" << Distance
                          << " out of range.");
        return NULL;
      }
      MBB->ReplaceUsesOfBlockWith(TBB, NewTarget);
      // TBB has only one successor since only one J2_jump instr.
      TBB->removeSuccessor(TBB->succ_begin());
      TBBMIb->eraseFromParent();
      return TBB;
    }
    if (TBBMIb->isReturn() && QII->isPredicable(*TBBMIb)) {
      SmallVector<MachineOperand, 4> PredCond;
      QII->findInstrPredicate(FirstTerm, PredCond);
      if (PredCond.empty())
        return NULL;
      unsigned Opcode = TBBMIb->getOpcode();
      if ((Opcode != Hexagon::PS_jmpret) && (Opcode != Hexagon::L4_return))
        return NULL;
      MachineInstr *NewMI = FirstTerm;
      bool NeedToNewify = QII->isPredicatedNew(*NewMI);
      // Since return instructions execute only in slot 0.
      // TODO: For bs > 2 we have to do resource tracking etc.
      if (NewMI->isBundled()) {
        if (!NeedToNewify)
          return nullptr;
        MachineBasicBlock::instr_iterator It = NewMI->getIterator();
        if (QII->nonDbgBundleSize(*getBundleStart(It)) > 2)
          return nullptr;
      }
      int rem = NewMI->getNumOperands() - 1;
      while (rem >= 0) // Removing from back is faster.
        NewMI->removeOperand(rem--);
      const MachineInstr *MI2Clone = TBBMIb;
      NewMI->setDesc(QII->get(MI2Clone->getOpcode()));
      unsigned NumOps = MI2Clone->getNumOperands();
      unsigned add = 0;
      while (add < NumOps)
        NewMI->addOperand(MI2Clone->getOperand(add++));
      if (!QII->PredicateInstruction(*NewMI, PredCond))
        llvm_unreachable("Instruction should have been predicable");
      if (NeedToNewify) {
        int NewOpcode = QII->getDotNewPredOp(*NewMI, MBPI);
        NewMI->setDesc(QII->get(NewOpcode));
      }
      MBB->removeSuccessor(TBB);
      LLVM_DEBUG(dbgs() << "ModifiedInstr:"; NewMI->dump(););
      if ((TBB != &*MBB->getParent()->begin()) && !TBB->isEHPad() &&
          !TBB->hasAddressTaken() && TBB->pred_empty() && TBB->succ_empty())
        return TBB;
    }
    return NULL;
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
          (FBB == NewTarget /*Infinite loop*/) || (FBB->pred_size() != 1))
        return NULL;
      if (!FBB->isLayoutSuccessor(TBB) && !makeAsLayoutSuccessor(FBB, TBB))
        return NULL;
      assert(FBB->isLayoutSuccessor(TBB));
      LLVM_DEBUG(dbgs() << "\nSuboptimal branching in FBB");
      // Check if the jump in the last instruction is within range.
      unsigned Distance =
          HLV->getDistanceBetween(MBB, NewTarget) + SafetyBuffer;

      if (!QII->isJumpWithinBranchRange(*FirstTerm, Distance)) {
        LLVM_DEBUG(dbgs() << "\nUnconditional jump target:" << Distance
                          << " out of range.");
        return NULL;
      }
      if (!QII->invertAndChangeJumpTarget(*FirstTerm, NewTarget))
        return NULL;
      LLVM_DEBUG(dbgs() << "\nNew instruction:"; FirstTerm->dump(););
      MBB->ReplaceUsesOfBlockWith(FBB, NewTarget);
      FBB->removeSuccessor(FBB->succ_begin());
      FBBMIb->eraseFromParent();
      return FBB;
    }
  }
  return NULL;
}

void GlobalScheduler::performExposedOptimizations() {
  LLVM_DEBUG(dbgs() << "\n[PerformExposedOptimizations]";);
  MachineFunction::iterator MBBI = MF.begin();
  for (; MBBI != MF.end();) {
    MachineBasicBlock *MBB = &*MBBI;
    ++MBBI; // MBB might be removed so incrementing early.
    // Empty BBs.
    if (isEmptyBlock(MBB)) {
      for (MachineBasicBlock::pred_iterator pi = MBB->pred_begin(),
                                            pe = MBB->pred_end();
           pi != pe; ++pi)
        if (AnalyzeBranchInfo.hasABInfo(*pi))
          AnalyzeBranchInfo.invalidate(*pi);

      if (AnalyzeBranchInfo.hasABInfo(MBB))
        AnalyzeBranchInfo.invalidate(MBB);
      removeEmptyMBB(MBB);
    }
  }
  for (MBBI = MF.begin(); MBBI != MF.end();) {
    MachineBasicBlock *MBB = &*MBBI;
    ++MBBI; // MBB might be removed so incrementing early.
    LLVM_DEBUG(dbgs() << "\nOptimizing Branches in MBB#" << MBB->getNumber(););
    // Don't perform optimizations on these MBBs.
    if (std::find(IgnorePostSchedOpt.begin(), IgnorePostSchedOpt.end(), MBB) !=
        IgnorePostSchedOpt.end())
      continue;
    // Branch Optimizations.
    ABI &abi = AnalyzeBranchInfo.getOrCreate(MBB, QII, true);
    if (!abi.AnalysisValid)
      continue;
    MachineBasicBlock *TBB = abi.TBB, *FBB = abi.FBB;
    MachineInstr *FirstTerm = abi.FirstTerm, *SecondTerm = abi.SecondTerm;

    // If there are two jump instructions then see if one can be removed.
    if (TBB && FirstTerm &&
        removeRedundantBranches(MBB, TBB, FirstTerm, FBB, SecondTerm)) {
      LLVM_DEBUG(dbgs() << "\nRemoved redundant branches in BB#"
                        << MBB->getNumber(););
    } else if (FirstTerm && SecondTerm &&
               optimizeDualJumps(MBB, TBB, FirstTerm, FBB, SecondTerm)) {
      LLVM_DEBUG(dbgs() << "\nRemoved dual jumps in BB#" << MBB->getNumber(););
    } else if (TBB && FBB && FirstTerm && !SecondTerm) {
      MachineBasicBlock *BBToErase = optimizeBranches(MBB, TBB, FirstTerm, FBB);
      if (BBToErase) {
        if (AnalyzeBranchInfo.hasABInfo(BBToErase))
          AnalyzeBranchInfo.invalidate(BBToErase);
        assert(BBToErase->succ_empty() && BBToErase->pred_empty());
        BBToErase->eraseFromParent();
        // Since we don't know from where this MBB got deleted, restart.
        // Also, this enables removal of branches which might
        // be redundant as a result of removing BBToErase.
        MBBI = MF.begin();
        LLVM_DEBUG(dbgs() << "\nOptimized jump from BB#" << MBB->getNumber());
      }
    }
    // We know that MBB was analyzed.
    AnalyzeBranchInfo.invalidate(MBB);
  }
}

// Criteria of HEAD of a scheduling region (Superblock).
// 1. MF.begin()
// 2. MBB with more than one successor:
//    - Can be part of multiple scheduling regions.
//    For simplicity we'll make this as HEAD as well. This lets us not
//    worry about the loop as a special case.
// 3. MBB with more than one predecessor:
//    - Will always be (if at all) the HEAD of the scheduling region.
// TODO: Use the dominator tree to construct regions.
void GlobalScheduler::formSchedulingRegions(MachineFunction &MF) {
  MachineFunction::iterator MBBI = MF.begin();
  // Populate the HEADs of scheduling regions.
  // We don't want to handle MBBs with >2 successors. This lets us
  // focus on most common cases and simplifies the analysis.
  while (MBBI != MF.end()) {
    unsigned MBBPredSize = MBBI->pred_size();
    unsigned MBBSuccSize = MBBI->succ_size();
    if (((MBBPredSize > 1) && ((MBBSuccSize == 1) || (MBBSuccSize == 2))) ||
        (MBBSuccSize == 2)) {
      SchedulingRegion_t SR;
      SR.addSRMember(&*MBBI);
      SchedulingRegions.push_back(SR);
      // Push two HEADs so that two scheduling regions can be created.
      if (MBBI->succ_size() == 2) {
        SchedulingRegions.push_back(SR);
      }
    }
    ++MBBI;
  }
  // For each HEAD, populate corresponding scheduling region.
  SchedulingRegions_t::iterator SRI = SchedulingRegions.begin();
  SchedulingRegions_t::iterator SRIPrev = SchedulingRegions.end();
  LLVM_DEBUG(dbgs() << "\nConstructing Scheduling Region for HEADs at";);
  while (SRI != SchedulingRegions.end()) {
    MachineBasicBlock *RegHead = SRI->Members[0];
    assert(RegHead);
    LLVM_DEBUG(dbgs() << " BB#" << RegHead->getNumber(););
    MachineBasicBlock::succ_iterator MBBSucc = RegHead->succ_begin();
    // In case of multiple entries, the first HEAD will have the first succ.
    // in its region.
    // Look at the other successor to construct the region, if the first branch
    // of Region HEAD is already included in the previous region.
    if ((SRIPrev != SchedulingRegions.end()) &&
        (SRIPrev->Members[0] == SRI->Members[0]))
      ++MBBSucc;
    assert(RegHead->succ_size() < 3);
    MachineFunction::iterator RegMBB = (*MBBSucc)->getIterator();
    while (1) {
      auto *L = MLI.getLoopFor(&*RegMBB);
      bool SuccIsSingleBlockLoop =
          isSuperblockLoop(L) && RegMBB->getNumber() != RegHead->getNumber();
      // pred_size > 1 => HEAD, cannot be included in this region.
      // Leaf MBBs should have at least one MI.
      if ((RegMBB->pred_size() > 1 && !SuccIsSingleBlockLoop) ||
          (RegMBB->succ_empty() && RegMBB->empty()))
        break;
      SRI->addSRMember(&*RegMBB);
      // succ_size = 0 => done!, =2 => HEAD
      if (RegMBB->succ_size() != 1) {
        break;
      }
      RegMBB = (*RegMBB->succ_begin())->getIterator();
    }
    SRIPrev = SRI;
    ++SRI;
  }
}

void GlobalScheduler::pruneSchedulingRegions(PruneCriteriaFP PC) {
  LLVM_DEBUG(dbgs() << "\nErasing Scheduling Region with HEAD at";);
  for (SchedulingRegions_t::iterator SRI = SchedulingRegions.begin();
       SRI != SchedulingRegions.end();) {
    if ((this->*PC)(*SRI)) {
      LLVM_DEBUG(dbgs() << "\tBB#" << SRI->Members[0]->getNumber());
      SRI = SchedulingRegions.erase(SRI);
    } else {
      ++SRI;
    }
  }
}

void GlobalScheduler::sortSchedulingRegions() {
  for (SchedulingRegions_t::iterator SRI = SchedulingRegions.begin();
       SRI != SchedulingRegions.end(); ++SRI) {
    SchedulingRegion_t &SR = *SRI;
    // The EdgeProb of a scheduling region is the same as the edge prob.
    // of the first two MBBs.
    SR.EdgeProb = MBPI->getEdgeProbability(SR.Members[0], SR.Members[1]);
    if (SR.EdgeProb.getCompl() == SR.EdgeProb) { // N = D/2;
      ABI &abi = AnalyzeBranchInfo.getOrCreate(SR.Members[0], QII);
      if (!abi.AnalysisValid)
        continue;
      // Increase the probability of TBB.
      if (abi.TBB == SR.Members[1])
        SR.EdgeProb = BranchProbability(2, 3); // N = 2D/3
      else
        SR.EdgeProb = BranchProbability(1, 3); // N = D/3
    }
    LLVM_DEBUG(dbgs() << "Edge Probability of: "; SR.dump(); dbgs() << ":\t";
               SR.EdgeProb.dump(););
  }
  // Sort in the decreasing order of edge probabilities.
  std::sort(SchedulingRegions.begin(), SchedulingRegions.end(),
            std::greater<SchedulingRegions_t::value_type>());
}

MachineBasicBlock::iterator
GlobalScheduler::addToPacketPseudo(MachineInstr *MI) {
  MachineBasicBlock::iterator MII;
  // We don't want to reserve a resource for CFI_INSTRUCTION (it has zero
  // cost).
  if (!TII->isZeroCost(MI->getOpcode()))
    ResourceTracker->reserveResources(*MI);
  if (QII->isConstExtended(*MI)) {
    assert(canReserveResourcesForConstExt());
    reserveResourcesForConstExt();
  }
  CurrentPacketMIs.push_back(MI);
  return MII;
}

void GlobalScheduler::initPacketizerStateLoc() {
  HexagonPacketizerList::initPacketizerState();
  CurrentPacketMIs.clear();
}

// Uses the packetizer checks to ensure correctness.
bool GlobalScheduler::isLegalToPacketizeWith(MIIterType DestPack,
                                             MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "\n[isLegalToPacketizeWith]";);
  assert(DestPack->isBundle());
  // Never packetize a solo instruction.
  if (isSoloInstruction(*MI))
    return false;

  // The implementation should be inline with isLegalToPacketizeTogether.
  // For all MIs in DestPack call isLegalToPacketizeTogetherMI.
  MIInstIterType BI = (&*DestPack)->getIterator();
  MachineBasicBlock *MBB = BI->getParent();
  const MachineInstr *BranchMI = NULL;
  // skipping the bundle header.
  for (++BI; (BI != MBB->instr_end()) && BI->isInsideBundle(); ++BI) {
    // Remember the branch; its target may need a constant extender (below).
    if (BI->isBranch())
      BranchMI = &*BI;
    if (!isLegalToPacketizeTogetherMI(&*BI, MI))
      return false;
  }

  unsigned NumI = QII->nonDbgBundleSize(*DestPack);
  // When there is only one slot left we want to check if there's a
  // scope for const-extender.
  if ((NumI >= (HEXAGON_PACKET_SIZE - 1)) && BranchMI &&
      (DisableHVXLocalPullupAndSuperblock || !useHVXOps ||
       !QII->isEndLoopN(BranchMI->getOpcode()))) {
    for (unsigned i = 0; i < BranchMI->getNumOperands(); ++i) {
      const MachineOperand &MO = BranchMI->getOperand(i);
      if (MO.isMBB()) {
        unsigned dist = HLV->getDistanceBetween(MBB, MO.getMBB());
        if (!QII->isJumpWithinBranchRange(*BranchMI, dist + SafetyBuffer)) {
          LLVM_DEBUG(dbgs() << "\nBranch might be constant extended later";);
          return false;
        }
      }
    }
  }
  return true;
}

// BI is the destination packet where MI may be placed.
bool GlobalScheduler::isLegalToPacketizeTogetherMI(MachineInstr *BI,
                                                   MachineInstr *MI) {
  if (cannotCoexist(*BI, *MI))
    return false;

  if (hasDeadDependence(*BI, *MI))
    return false;
  if (hasControlDependence(*BI, *MI))
    return false;
  if (hasDualStoreDependence(*BI, *MI))
    return false;

  const MachineBasicBlock *MBB = BI->getParent();
  // The implementation should be inline with isLegalToPacketizeTogether.
  // Hexagon PRM 7.3.4: a loop setup packet in loopN or spNloop0 cannot
  // contain a speculative indirect jump,
  // a new-value compare jump or a dealloc_return.
  // Speculative indirect jumps (predicate + .new + indirect):
  // if ([!]Ps.new) jumpr:t Rs
  // if ([!]Ps.new) jumpr:nt Rs
  // @note: We don't want to pull across a call to be on the safe side.
  if (QII->isLoopN(*MI)) {
    if ((QII->isPredicated(*BI) && QII->isPredicatedNew(*BI) &&
         QII->isJumpR(*BI)) ||
        QII->isNewValueJump(*BI) || QII->isDeallocRet(*BI) || BI->isCall()) {
      LLVM_DEBUG(dbgs() << "\nLoopN pull interference.");
      return false;
    }
    // loopN cannot be pulled too far.
    unsigned dist = HLV->getDistanceBetween(MBB, MI->getOperand(0).getMBB());
    if (!QII->isJumpWithinBranchRange(*MI, dist + SafetyBuffer)) {
      LLVM_DEBUG(dbgs() << "\nLoopN cannot be pulled too far";);
      return false;
    }
  }

  // The opposite is also true.
  if (QII->isLoopN(*BI) &&
      ((QII->isPredicated(*MI) && QII->isPredicatedNew(*MI) &&
        QII->isJumpR(*MI)) ||
       QII->isNewValueJump(*MI) || QII->isDeallocRet(*MI) || MI->isCall())) {
    LLVM_DEBUG(dbgs() << "\tResident LoopN.");
    return false;
  }

  // @todo Hexagon PRM 7.6.1:
  // Presence of NVJ adds more restrictions.
  if (QII->isNewValueJump(*BI) &&
      (MI->mayStore() || MI->getOpcode() == Hexagon::S2_allocframe ||
       MI->isCall())) {
    LLVM_DEBUG(dbgs() << "\nNew val Jump.");
    return false;
  }

  // For memory operations, check aliasing.
  // Be conservative with ordered (volatile/atomic) or unknown (no memoperands)
  // memory references.
  if (BI->hasOrderedMemoryRef())
    return false;

  // If packet has a new-value store, MI can't be a store instruction.
  if (QII->isNewValueStore(*BI) && MI->mayStore()) {
    LLVM_DEBUG(dbgs() << "\nNew Value Store to store.");
    return false;
  }

  if ((QII->isMemOp(*BI) && MI->mayStore()) ||
      (QII->isMemOp(*MI) && BI->mayStore())) {
    LLVM_DEBUG(dbgs() << "\nSlot 0 not available for store because of memop.");
    return false;
  }
  return true;
}

// @note: This function leaves the packet in an invalid state.
// GA and GB are left in the packet and it is the responsibility of the caller
// to move/delete them to appropriate places.
// @warning: Only works for limited set of branches handled
// in HexagonInstrInfo::getCompoundOpcode.
MachineInstr *GlobalScheduler::generateCompoundIfPossible(MIInstIterType MII,
                                                          MIInstIterType &GA,
                                                          MIInstIterType &GB) {
  assert(MII->isBundle());
  MachineInstr *NewMI = NULL;
  MachineBasicBlock *MBB = MII->getParent();
  GA = MBB->instr_end();
  GB = MBB->instr_end();
  ++MII; // Skipping Bundle header.
  while ((MII != MBB->instr_end()) && MII->isInsideBundle()) {
    if (MII->isCompare() &&
        (QII->getCompoundCandidateGroup(*MII) == HexagonII::HCG_A)) {
      if (GA != MBB->instr_end()) {
        GA = MBB->instr_end();
        break;
      }
      GA = MII;
    }
    if (MII->isBranch() &&
        (QII->getCompoundCandidateGroup(*MII) == HexagonII::HCG_B)) {
      if (GB != MBB->instr_end()) {
        GB = MBB->instr_end();
        break;
      }
      GB = MII;
    }
    ++MII;
  }
  int CmpOpcode = 0;
  if ((GA != MBB->instr_end()) && (GB != MBB->instr_end())) {
    CmpOpcode = QII->getCompoundOpcode(*GA, *GB);
    if (CmpOpcode != -1) {
      LLVM_DEBUG(dbgs() << "\nFound Compound pair: " << *GA << *GB;);
      assert(GA->getNumOperands() == 3);
      DebugLoc dl = GB->getDebugLoc();
      NewMI = BuildMI(*MBB, GB, dl, QII->get(CmpOpcode));
      Register PredReg = GA->getOperand(0).getReg();
      MachineOperand *PredUse =
          NewMI->findRegisterUseOperand(PredReg, /*TRI=*/nullptr);
      assert(PredUse && PredUse->isImplicit() &&
             "Missing compound predicate use");
      PredUse->setIsInternalRead();
      NewMI->addOperand(GA->getOperand(1)); // LHS of compare
      NewMI->addOperand(GA->getOperand(2)); // RHS of compare
      // All the MIs in the HexagonII::HCG_B category have MBB as 2nd operand.
      NewMI->addOperand(GB->getOperand(1)); // The branch target.
      unsigned dist = HLV->getDistanceBetween(MBB, GB->getOperand(1).getMBB());
      if (!QII->isJumpWithinBranchRange(*NewMI, dist + SafetyBuffer +
                                                    SafetyBufferDelta)) {
        NewMI->eraseFromBundle();
        LLVM_DEBUG(dbgs() << ".....NO (Branch too far, cannot form)";);
        return NULL;
      }
      LLVM_DEBUG(dbgs() << "\nNew Compound Instruction:" << *NewMI;);
    }
  }
  return NewMI;
}

void GlobalScheduler::generateCompoundsPostSched() {
  DenseSet<MachineInstr *>::iterator it = ToBeSqueezed.begin();
  while (it != ToBeSqueezed.end()) {
    auto MBB = (*it)->getParent();
    MIInstIterType GA = MBB->instr_end(), GB = MBB->instr_end();
    MachineInstr *NewMI =
        generateCompoundIfPossible((*it)->getIterator(), GA, GB);
    (void)NewMI;
    assert(NewMI && "Wrong Analysis of compound packets.");
    GA->eraseFromBundle();
    GB->eraseFromBundle();
    updateBundle(*it, QII);
    ++it;
  }
}

// Check if the machine instructions belong in the same bundle
static bool checkIfSameBundle(MachineInstr *MI1, MachineInstr *MI2) {
  if (MI1->isInsideBundle() && MI2->isInsideBundle() &&
      getBundleStart(MI1->getIterator()) == getBundleStart(MI2->getIterator()))
    return true;
  return false;
}

// Returns true if any of the conditions happen:
// (1) If a packet contains endloopN, it cannot perform an auto-AND with
// predicate register P3
// (2) If a packet contains a register transfer from a general register to a
// predicate register, no other instruction in the packet can write to the same
// predicate register.
// (3) Instructions spNloop0, decbin, tlbmatch, memw_locked, memd_locked,
// add:carry, sub:carry, sfcmp, and dfcmp cannot be grouped with another
// instruction that sets the same predicate register.
bool GlobalScheduler::failsSpecialConditions(MachineInstr *BundleMI,
                                             Register DestReg) {

  if (!BundleMI || !BundleMI->getParent())
    return true;
  for (MachineBasicBlock::instr_iterator I = BundleMI->getIterator(),
                                         E = std::next(I);
       E != BundleMI->getParent()->instr_end() && E->isInsideBundle(); ++E) {
    MachineInstr &MI = *E;
    auto OpC = MI.getOpcode();
    // Case 1
    if (QII->isEndLoopN(OpC) && DestReg == Hexagon::P3)
      return true;
    // Case 2
    if ((OpC == Hexagon::C2_tfrrp) && MI.getNumOperands() > 0 &&
        (DestReg == MI.getOperand(0).getReg()))
      return true;
    // Case 3
    if ((OpC == Hexagon::S2_cabacdecbin) ||
        ((QII->isSoftwarePipelineLoopInstr(MI) || OpC == Hexagon::A4_tlbmatch ||
          QII->isFloatCompare(MI)) &&
         MI.getNumOperands() > 0 && MI.getOperand(0).getReg() == DestReg) ||
        ((OpC == Hexagon::S2_storew_locked ||
          OpC == Hexagon::S4_stored_locked || OpC == Hexagon::A6_vminub_RdP ||
          OpC == Hexagon::A5_ACS || OpC == Hexagon::F2_sfinvsqrta ||
          OpC == Hexagon::F2_sfrecipa) &&
         MI.getNumOperands() > 1 && MI.getOperand(1).getReg() == DestReg) ||
        ((OpC == Hexagon::A4_addp_c || OpC == Hexagon::A4_subp_c) &&
         MI.getNumOperands() > 2 && MI.getOperand(3).getReg() == DestReg))
      return true;
  }
  return false;
}

// Returns true if Reg can be used as a .new predicate source in a packet,
// i.e. it is a non-fake predicate register whose definition (Producer) is
// not a late-predicate instruction (e.g. A4_addp_c, A4_subp_c).
// Sentinel marks the end of the valid iterator range for Producer.
bool GlobalScheduler::canNewifyPredReg(unsigned Reg, MICInstIterType Producer,
                                       MICInstIterType Sentinel) const {
  if (TRI->isFakeReg(Reg))
    return false;
  if (TRI->getMinimalPhysRegClass(Reg) != &Hexagon::PredRegsRegClass)
    return false;
  if (Producer != Sentinel && QII->isPredicateLate(Producer->getOpcode()))
    return false;
  return true;
}

// Given a machine instruction, get the defs of the uses and return them as
// reference. It uses the infrastructure created in this pass, ie. MIUseDefs
bool GlobalScheduler::getDefinitions(MachineInstr *MI, MachineInstr *&Def1,
                                     MachineInstr *&Def2, MachineInstr *&Def3) {

  auto MBB = MI->getParent();
  if (MI->getNumOperands() < 3)
    return false;
  Register Reg1 = MI->getOperand(1).getReg();
  Register Reg2 = MI->getOperand(2).getReg();
  auto Reg3 =
      (MI->getNumOperands() == 4) ? MI->getOperand(3).getReg() : Register(0);
  const MIUseDef_t &MIUseDefs = HLV->getMIUseDefs();
  if (&MBB->front() == MI)
    return false;
  for (auto DefSearch = std::prev(MI->getIterator());
       DefSearch != MBB->instr_begin(); --DefSearch) {
    if (DefSearch->isBundle())
      continue;

    MIUseDef_t::const_iterator DefIter = MIUseDefs.find(&*DefSearch);
    if (DefIter == MIUseDefs.end())
      continue;

    const BitVector &Defs = DefIter->second.second;
    if (Defs[Reg1])
      Def1 = &*DefSearch;
    if (Defs[Reg2])
      Def2 = &*DefSearch;
    if (Reg3 && Defs[Reg3])
      Def3 = &*DefSearch;

    if (Def1 && Def2 && (!Reg3 || (Reg3 && Def3)))
      return true;
  }
  return false;
}

// Exploits a feature of the Hexagon hardware that predicate instructions
// in a packet are internally auto ANDed, so a later AND is unnecessary.
// Updating liveness is not necessary.
void GlobalScheduler::generateAutoAndPredicates(MachineFunction &Fn) {

  SmallVector<MachineInstr *, 4> ToBeErased;
  for (auto &MBB : Fn) {
    for (auto MII = MBB.instr_begin(), ME = MBB.instr_end(); MII != ME; ++MII) {
      unsigned OpC = MII->getOpcode();
      // TODO check for other opcode types too
      if (OpC != Hexagon::C2_and && OpC != Hexagon::C2_andn &&
          OpC != Hexagon::C4_and_and && OpC != Hexagon::C4_and_andn)
        continue;

      Register DestPred = MII->getOperand(0).getReg();
      Register Reg1 = MII->getOperand(1).getReg();
      Register Reg2 = MII->getOperand(2).getReg();
      auto Reg3 = MII->getNumOperands() == 4 ? MII->getOperand(3).getReg()
                                             : Register(0);

      // Skip if the operands are the same register
      if (Reg1 == Reg2)
        continue;
      if (Reg3 != 0 && Reg3 == Reg2)
        continue;

      // Skip if any of the regs are not killed in the AND instruction.
      // We don't want any uses of the regs after the AND.
      if (!MII->getOperand(1).isKill() || !MII->getOperand(2).isKill() ||
          (Reg3 != 0 && !MII->getOperand(3).isKill()))
        continue;

      // Check if the result of the AND instruction (DestPred) is used by a
      // .new instruction within the same packet. If so, skip this AND.
      // TODO: This could be handled by replacing the .new use with a regular
      // use. Another option is to pullup the .new use if possible
      bool UsedByDotNew = false;
      auto AndBundleStart = getBundleStart(MII);
      auto AndBundleEnd = getBundleEnd(MII);
      for (auto BundledII = AndBundleStart; BundledII != AndBundleEnd;
           ++BundledII) {
        if (&*BundledII == &*MII)
          continue;
        int UseIdx = BundledII->findRegisterUseOperandIdx(DestPred, TRI);
        if (UseIdx >= 0 && QII->isDotNewOperand(*BundledII, UseIdx)) {
          UsedByDotNew = true;
          break;
        }
      }
      if (UsedByDotNew)
        continue;

      // Find definitions using MIIDefSet in the same MBB.
      MachineInstr *Def1 = nullptr;
      MachineInstr *Def2 = nullptr;
      MachineInstr *Def3 = nullptr;
      if (!getDefinitions(&*MII, Def1, Def2, Def3))
        continue;

      // Check if the defs are compare instructions
      if (!QII->definesPredicateReg(*Def1) ||
          !QII->definesPredicateReg(*Def2) ||
          (Def3 && !QII->definesPredicateReg(*Def3)))
        continue;

      // Check if both definitions are in the same bundle
      if (!checkIfSameBundle(Def1, Def2))
        continue;
      if (Def3 && !checkIfSameBundle(Def1, Def3))
        continue;

      // Check if there are any uses of the predicates before the use in the AND
      // instruction If there are, we cannot do this transformation.
      MICInstIterType Use;
      SmallPtrSet<MachineInstr *, 2> Exception;
      auto MIStart = getBundleStart(Def1->getIterator());
      auto MIEnd = getBundleEnd(MII);
      Exception.insert(&*MII);
      if (HLV->isUsedWithin(MIStart, MIEnd, Reg1, Use, &Exception) ||
          HLV->isUsedWithin(MIStart, MIEnd, Reg2, Use, &Exception) ||
          (Def3 && HLV->isUsedWithin(MIStart, MIEnd, Reg3, Use, &Exception)))
        continue;

      // The compares will define DestPred in their packet. Nothing between
      // them and the AND may read or write DestPred, otherwise it would see
      // or clobber the new value.
      bool DestPredTouched = false;
      for (auto I = MIStart; I != MII; ++I) {
        if (I->isBundle() || &*I == Def1 || &*I == Def2 || &*I == Def3)
          continue;
        if (I->readsRegister(DestPred, TRI) ||
            I->modifiesRegister(DestPred, TRI)) {
          DestPredTouched = true;
          break;
        }
      }
      if (DestPredTouched)
        continue;

      // Check some special conditions which makes auto-and not possible
      if (failsSpecialConditions(&*MIStart, DestPred))
        continue;

      LLVM_DEBUG(dbgs() << "\n[Found predicate auto and candidate]: ";
                 MII->dump());

      std::optional<unsigned> NewOpC;
      switch (OpC) {
      // For C2_and: Pd = and(Ps, Pt) - both compares write to DestPred
      // (1) Change the def registers
      case Hexagon::C2_and:
        Def1->getOperand(0).setReg(DestPred);
        Def2->getOperand(0).setReg(DestPred);
        break;

      // For C2_andn: Pd = and(Ps, !Pt) - Ps writes to DestPred, Pt writes
      // inverted to DestPred The second operand is the one that gets negated.
      // (1) Invert the opcode of the second def
      // (2) Change the def registers
      case Hexagon::C2_andn:
        LLVM_DEBUG(dbgs() << "Get inverted opcode for: "; Def2->dump());
        NewOpC = QII->getInvertedCompare(Def2->getOpcode());
        if (!NewOpC)
          continue;
        Def2->setDesc(QII->get(NewOpC.value()));
        Def1->getOperand(0).setReg(DestPred);
        Def2->getOperand(0).setReg(DestPred);
        break;

      // (1) Change the three def registers
      case Hexagon::C4_and_and:
        Def1->getOperand(0).setReg(DestPred);
        Def2->getOperand(0).setReg(DestPred);
        Def3->getOperand(0).setReg(DestPred);
        break;

      // (1) Change the three def registers
      // (2) Invert the last register's def opcode
      case Hexagon::C4_and_andn:
        LLVM_DEBUG(dbgs() << "Get inverted opcode for: "; Def3->dump());
        NewOpC = QII->getInvertedCompare(Def3->getOpcode());
        if (!NewOpC)
          continue;
        Def3->setDesc(QII->get(NewOpC.value()));
        Def1->getOperand(0).setReg(DestPred);
        Def2->getOperand(0).setReg(DestPred);
        Def3->getOperand(0).setReg(DestPred);
        break;

      default:
        continue;
      }

      ToBeErased.push_back(&*MII);
    }
  }

  // Erase the redundant AND instructions
  for (MachineInstr *It : ToBeErased) {
    if (It->isBundled()) {
      auto Head = &*getBundleStart(It->getIterator());
      It->eraseFromBundle();
      updateBundle(Head, QII);
    } else
      It->eraseFromParent();
  }
}

void GlobalScheduler::packetize() {
  DenseSet<unsigned> DoneMBBs;
  HexagonPacketizerList Packetizer(MF, MLI, AA, MBPI, false);
  SmallVectorImpl<MachineInstr *>::iterator PacketOpp = SpecPacketOpp.begin();
  // The Entries are the last instruction where a new MI was scheduled.
  // For now we only move up into the MBB.
  for (; PacketOpp != SpecPacketOpp.end(); ++PacketOpp) {
    MachineBasicBlock *MBB = (*PacketOpp)->getParent();
    if (!DoneMBBs.insert(MBB->getNumber()).second)
      continue;
    MachineBasicBlock::iterator RegionBeg = MBB->begin();
    MachineBasicBlock::iterator RegionEnd = std::next(RegionBeg);
    for (; RegionBeg != MBB->end(); RegionBeg = RegionEnd) {
      // Find scheduling regions and schedule / packetize each region.
      // TODO: Find all the regions (unpacketized) and make bundles.
      while (RegionBeg != MBB->end() && RegionBeg->isBundled())
        ++RegionBeg;
      if (RegionBeg == MBB->end())
        break;
      RegionEnd = std::next(RegionBeg);
      while (RegionEnd != MBB->end() && !RegionEnd->isBundled())
        ++RegionEnd;
      // Skip regions with one instruction.
      if (RegionBeg == std::prev(RegionEnd))
        continue;
      Packetizer.PacketizeMIs(MBB, RegionBeg, RegionEnd);
    }
  }
  Packetizer.unpacketizeSoloInstrs(MF);
}

// @note When generating a compound instruction, we do not want to
// update the liveness of the BUNDLE/MBB because we might decompose later.
void GlobalScheduler::getResourcesReadyForBundle(MIInstIterType MII,
                                                 bool Squeeze,
                                                 MachineInstr *InstrToSkip) {
  assert(MII->isBundle());

  bool GeneratedCompoundInstr = false;
  MachineInstr *NewMI = NULL;
  MachineBasicBlock *MBB = MII->getParent();
  MIInstIterType GA = MBB->instr_end(), GB = MBB->instr_end();
  // MIs because they might be pointing to MIs which are removed from Bundle.
  MachineInstr *GAPos, *GBPos;
  if (Squeeze && EnableSqueeze &&
      (QII->nonDbgBundleSize(*MII) == HEXAGON_PACKET_SIZE)) {
    // handle compound/duplex stuff.
    LLVM_DEBUG(dbgs() << "\nCompound can be made in this bundle.";);
    NewMI = generateCompoundIfPossible(MII, GA, GB);
    if (NewMI != NULL) {
      GeneratedCompoundInstr = true;
      MIInstIterType GANext = std::next(GA);
      MIInstIterType GBNext = std::next(GB);
      GAPos = GANext != MBB->instr_end() ? &*GANext : nullptr;
      GBPos = GBNext != MBB->instr_end() ? &*GBNext : nullptr;
      GA = GA->removeFromBundle()->getIterator();
      GB = GB->removeFromBundle()->getIterator();
    }
  }

  initPacketizerStateLoc();
  ResourceTracker->clearResources();
  // Add already packetized instructions.
  ++MII; // Skipping Bundle header.

  // For TinyCore with Duplexes, change the opcodes of MII to Big Core.
  MIInstIterType SavedMII = MII;
  auto &ST = MII->getParent()->getParent()->getSubtarget<HexagonSubtarget>();
  if (ST.isTinyCoreWithDuplex())
    QII->translateInstrsForDup(SavedMII, true);

  while ((MII != MBB->instr_end()) && MII->isInsideBundle()) {
    assert(ResourceTracker->canReserveResources(*MII) ||
           TII->isZeroCost(MII->getOpcode()));
    if (!(InstrToSkip && InstrToSkip == &*MII)) {
      addToPacketPseudo(&*MII);
    }
    ++MII;
  }

  // Restore the opcodes for Tiny Core with Duplexes.
  if (ST.isTinyCoreWithDuplex())
    QII->translateInstrsForDup(SavedMII, false);

  if (GeneratedCompoundInstr) {
    // Group B MI should occur 'after' group A MI.
    MIInstIterType GBPosMII = GBPos ? GBPos->getIterator() : MBB->instr_end();
    MBB->insert(GBPosMII, &*GB);
    if (!GB->isBundled())
      GB->bundleWithPred();
    MIInstIterType GAPosMII = GAPos ? GAPos->getIterator() : MBB->instr_end();
    MBB->insert(GAPosMII, &*GA);
    assert(GA->isBundled());
    NewMI->eraseFromBundle();
  }
}

// TODO: Deal with constant extenders later.
// For that we have to try allocate resources for two instructions.
bool GlobalScheduler::canReserveResources(MIInstIterType MII) const {
  if (QII->isExtended(*MII) || QII->isConstExtended(*MII))
    return false;
  return ResourceTracker->canReserveResources(*MII);
}

// The usage of \p Reg is replaced by \p ScratchReg.
// @note: Does not do anything with the liveness.
void GlobalScheduler::replaceUsageOfReg(unsigned Reg, unsigned ScratchReg,
                                        MachineBasicBlock *MBB,
                                        const MachineInstr *ExceptMI) const {
  assert(Reg != ScratchReg);
  for (MIInstIterType MII = MBB->instr_begin(), MIE = MBB->instr_end();
       MII != MIE; ++MII) {
    MachineInstr *MI = &*MII;
    if (MI == ExceptMI)
      continue;
    for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && (MO.getReg() == Reg))
        MO.setReg(ScratchReg);
    }
  }
}

void GlobalScheduler::markLiveInsAsUsed(std::unique_ptr<RegScavenger> &RS,
                                        MachineBasicBlock *MBB) const {
  for (MachineBasicBlock::livein_iterator lit = MBB->livein_begin();
       lit != MBB->livein_end(); ++lit) {
    RS->setRegUsed((*lit).PhysReg);
    LLVM_DEBUG(dbgs() << printReg((*lit).PhysReg, TRI););
  }
}

void GlobalScheduler::markLiveOutsAsUsed(std::unique_ptr<RegScavenger> &RS,
                                         MachineBasicBlock *MBB) const {
  const BitVector &LO = HLV->getLiveOuts(MBB);
  int Reg = LO.find_first();
  while (Reg >= 0) {
    RS->setRegUsed(Reg);
    LLVM_DEBUG(dbgs() << printReg(Reg, TRI););
    Reg = LO.find_next(Reg);
  }
}

void GlobalScheduler::markAllRegsAsUsed(std::unique_ptr<RegScavenger> &RS,
                                        const MachineBasicBlock *MBB) const {
  // We have to insert a A2_tfr just before the TargetPacket.
  // Conservative approach: Traverse all the MIs in DstBB and mark all the
  // defs and uses as 'used' in the RegScav.
  const MIUseDef_t &MIUseDefs = HLV->getMIUseDefs();
  BitVector MBBUseDef(TRI->getNumRegs());
  for (MICInstIterType I = MBB->instr_begin(); I != MBB->instr_end(); ++I) {
    if (I->isBundle() || I->isDebugInstr())
      continue;
    MIUseDef_t::const_iterator MIUseDef = MIUseDefs.find(&*I);
    assert(MIUseDef != MIUseDefs.end());
    MBBUseDef |= MIUseDef->second.first;
    MBBUseDef |= MIUseDef->second.second;
  }
  int Reg = MBBUseDef.find_first();
  while (Reg >= 0) {
    RS->setRegUsed(Reg);
    LLVM_DEBUG(dbgs() << printReg(Reg, TRI););
    Reg = MBBUseDef.find_next(Reg);
  }
}

// We want to split the critical edge only when there is a predecessor of ToBB
// which is also the layout predecessor of ToBB.
static MachineBasicBlock *ShouldSplitCriticalEdge(MachineBasicBlock *FromBB,
                                                  MachineBasicBlock *ToBB) {
  // Find the layout predecessor of ToBB.
  assert(FromBB->isSuccessor(ToBB));
  MachineBasicBlock *LP = NULL;
  for (MachineBasicBlock::pred_iterator pi = ToBB->pred_begin(),
                                        pe = ToBB->pred_end();
       pi != pe; ++pi)
    if ((*pi)->isLayoutSuccessor(ToBB))
      LP = *pi;
  return LP;
}

MachineBasicBlock *GlobalScheduler::splitCriticalEdge(MachineBasicBlock *FromBB,
                                                      MachineBasicBlock *ToBB,
                                                      MachineBasicBlock *LP) {
  assert(LP->isLayoutSuccessor(ToBB));
  MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock();
  // We want to insert just before the ToBB because then we don't have to worry
  // about the branch range of (FromBB->ToBB).
  MF.insert(MachineFunction::iterator(ToBB), NewBB);
  FromBB->ReplaceUsesOfBlockWith(ToBB, NewBB);
  NewBB->addSuccessor(ToBB);

  // Inform HLV of NewBB.
  HLV->addNewMBB(NewBB);

  // According to the HexagonInstrInfo::getBranchingInstrs
  // Only the last element can be a conditional branching instruction.
  SmallVector<MachineInstr *, 2> Jumpers = QII->getBranchingInstrs(*LP);
  // Jumpers.empty || Jumpers[0] is predicated => fall-through
  if ((LP != FromBB) &&
      (Jumpers.empty() || !Jumpers[0]->isUnconditionalBranch())) {
    DebugLoc dl;
    MachineInstr *JumpMI =
        BuildMI(*LP, LP->instr_end(), dl, QII->get(Hexagon::J2_jump))
            .addMBB(ToBB);
    // Update liveness.
    HLV->addNewMI(JumpMI, LP);
  }
  LLVM_DEBUG(dbgs() << "\nLayout Pred after update:" << *LP;);

  // We add a jump as a safe-measure, if this is redundant it will be removed.
  DebugLoc dl;
  MachineInstr *JumpMI =
      BuildMI(*NewBB, NewBB->instr_end(), dl, QII->get(Hexagon::J2_jump))
          .addMBB(ToBB);
  HLV->addNewMI(JumpMI, NewBB);
  // Update the branches in LP, invalidate the analyzeBranch of LP and of FromBB
  if (AnalyzeBranchInfo.hasABInfo(LP))
    AnalyzeBranchInfo.invalidate(LP);
  if (AnalyzeBranchInfo.hasABInfo(FromBB))
    AnalyzeBranchInfo.invalidate(FromBB);
  return NewBB;
}

// The def should not have subregs.
bool GlobalScheduler::hasOneDef(const MachineInstr *MI, unsigned &DefR) const {
  DefR = 0;
  const MIUseDef_t &MIUseDefs = HLV->getMIUseDefs();
  MIUseDef_t::const_iterator MIUseDef = MIUseDefs.find(MI);
  if (MIUseDef == MIUseDefs.end())
    return false;
  const BitVector &Defs = MIUseDef->second.second;
  unsigned DefsCount = Defs.count();
  // An MI which does not defines a reg should not be here.
  assert(DefsCount);
  // We don't want to process an MI which defines multiple/no regs.
  if (DefsCount != 1)
    return false;
  DefR = Defs.find_first();
  // If this register has a sub-reg we do not want to handle that situation.
  return !MCSubRegIterator(DefR, TRI, false).isValid();
}

// The use should not have subregs.
bool GlobalScheduler::hasOneUse(const MachineInstr *MI, unsigned &UseR) const {
  UseR = 0;
  const MIUseDef_t &MIUseDefs = HLV->getMIUseDefs();
  MIUseDef_t::const_iterator MIUseDef = MIUseDefs.find(MI);
  if (MIUseDef == MIUseDefs.end())
    return false;
  const BitVector &Uses = MIUseDef->second.first;
  unsigned UsesCount = Uses.count();
  // We don't want to process an MI which defines multiple/no regs.
  if (UsesCount != 1)
    return false;
  UseR = Uses.find_first();
  // If this register has a sub-reg we do not want to handle that situation.
  return !MCSubRegIterator(UseR, TRI, false).isValid();
}

// MI should have just one def and just one usage.
// The def and use should not have subregs.
bool GlobalScheduler::hasOneDefAndUse(const MachineInstr *MI, unsigned &Def,
                                      unsigned &Use) const {
  return hasOneDef(MI, Def) && hasOneUse(MI, Use);
}

// Returns later of the two in a linear sequence which might be
// I1 .. I2 .. Sentinel, or I2 .. I1 .. Sentinel
// If I1 == I2, returns I2.
MICInstIterType GlobalScheduler::laterOfTheTwo(MICInstIterType I1,
                                               MICInstIterType I2,
                                               MICInstIterType Sentinel) const {
  if (I2 == Sentinel)
    return I2;
  MICInstIterType Temp = I1;
  while (Temp != Sentinel) {
    if (Temp == I2)
      return I2;
    ++Temp;
  }
  return I1;
}

// A more exhaustive dependence analysis for load bypass, more aggressive
// moves, etc.  Used from the HVX local pullup.
int GreedyPullup::dstDependence(MICInstIterType DestBeg,
                                MICInstIterType DestEnd, MICInstIterType MII,
                                std::vector<const MachineInstr *> &DepTrueVec,
                                std::vector<const MachineInstr *> &DepAntiVec,
                                std::vector<const MachineInstr *> &DepOutputVec,
                                std::vector<unsigned> *Regs) const {

  if (DestBeg == DestEnd)
    return 0;

  unsigned dep = Dependency::Independent;

  SmallVector<unsigned> UseRegs;
  SmallVector<unsigned> DefRegs;
  for (const MachineOperand &MO : MII->operands()) {
    if (MO.isReg()) {
      if (MO.isUse())
        UseRegs.push_back(MO.getReg());
      else if (MO.isDef())
        DefRegs.push_back(MO.getReg());
    }
  }

  bool addedTrueDep = false, addedFalseDep = false;
  auto I = DestEnd;
  auto &MIUseDefs = HLV->getMIUseDefs();
  do {
    --I;
    if (!I->isBundle()) {
      if (shouldNotBeCrossed(&*I, &*MII)) {
        dep = Dependency::Unknown;
        break;
      }
      if (I->isDebugInstr())
        continue;
      for (auto RegI = UseRegs.begin(), RegE = UseRegs.end(); RegI != RegE;
           ++RegI) {
        for (MCRegAliasIterator AI(*RegI, TRI, true); AI.isValid(); ++AI) {
          if (MIUseDefs[&*I].second[*AI]) {
            DepTrueVec.push_back(&*I);
            if (Regs)
              Regs->push_back(*RegI);
            addedTrueDep = true;
            dep |= Dependency::RAW;
            break;
          }
        }

        if (addedTrueDep) {
          UseRegs.erase(RegI);
          addedTrueDep = false;
          break;
        }
      }

      for (auto RegI = DefRegs.begin(), RegE = DefRegs.end(); RegI != RegE;
           ++RegI) {
        for (MCRegAliasIterator AI(*RegI, TRI, true); AI.isValid(); ++AI) {
          if (MIUseDefs[&*I].second[*AI]) { // WAW case is first
            addedFalseDep = true; // WAW prevents any movement, so stop
            dep |= Dependency::WAW;
            DepOutputVec.push_back(&*I);
            break;
          }
        }

        for (MCRegAliasIterator AI(*RegI, TRI, true); AI.isValid(); ++AI) {
          if (MIUseDefs[&*I].first[*AI]) {
            DepAntiVec.push_back(&*I);
            dep |= Dependency::WAR;
            break;
          }
        }

        if (addedFalseDep) {
          DefRegs.erase(RegI);
          addedFalseDep = false;
          break;
        }
      }
    }

    if (I == DestBeg)
      break;

  } while (I != DestBeg);

  return dep;
}

//===----------------------------------------------------------------------===//
//                    GreedyPullup Functions
//===----------------------------------------------------------------------===//
// Returns true if Resources (MII) /\ Resources [DestBeg, DestEnd) != NULL
// Use(MII) /\ Def[DestBeg, DestEnd) != NULL
// Def(MII) /\ Def+Use[DestBeg, DestEnd) != NULL
// For .new MII: Use(MII) /\ Def(Bundle(MII)) == NULL
// [DestBeg, DestEnd) can span multiple bundles.
// \p DepTrue is the instruction which defines the Operand used by MII.
// \p DepAnti is the instruction which uses the Operand defined by MII.
// @note: Care must be taken because DepTrue and DepAnti will only give
// the def/use of the last operand (there might be more Deps).
int GreedyPullup::dstDependence(MICInstIterType DestBeg,
                                MICInstIterType DestEnd, MICInstIterType MII,
                                MICInstIterType &DepTrue,
                                MICInstIterType &DepAnti, unsigned HLXUseReg,
                                bool IsPullDown) const {
  LLVM_DEBUG(dbgs() << "\t[DstDependence]";);
  const MachineInstr *MI = &*MII;
  const MachineBasicBlock *SrcMBB = MII->getParent();
  // Initialize iterators to safe defaults
  MICInstIterType DepTrueP = SrcMBB->instr_end(),
                  DepTrueQ = SrcMBB->instr_end(),
                  DepAntiP = SrcMBB->instr_end();
  DepTrue = SrcMBB->instr_end();
  DepAnti = SrcMBB->instr_end();

  assert(!shouldNotBePulledUp(&*MII));
  MICInstIterType DestBundle =
      DestBeg->isBundled() ? getBundleStart(DestBeg) : DestBeg;
  MICInstIterType MIBegin = DestBeg;
  bool IsMIInDestBundle = false;
  for (; MIBegin != DestEnd; ++MIBegin) {
    // Need to pass isMIInDestBundle to create single packet HLX load-use-store
    // Only do this for local pullup
    IsMIInDestBundle = false;
    if (!IsPullDown && DestBundle->isBundle() && MIBegin->isBundled())
      IsMIInDestBundle = (DestBundle == getBundleStart(MIBegin));
    if (!shouldNotBeCrossed(&*MIBegin, &*MII, IsMIInDestBundle))
      continue;
    LLVM_DEBUG(dbgs() << "<<Unknown dependency>>";);
    return Dependency::Unknown;
  }
  // If MII points to a branch, check that there are no instructions
  // between dest bundle end and MII
  if (MII->isBranch() || MII->isTerminator() || MII->isReturn()) {
    // Only allow local pullup for unbundled endloop.
    // Do not allow endloop to be pulled down.
    // Check that there are no instructions between DestBundle and endloop.
    MICInstIterType SrcBundle = getBundleStart(MII);
    if (SrcBundle->getBundleSize() != 1 || SrcMBB != DestBeg->getParent() ||
        (laterOfTheTwo(MII, DestBeg, SrcMBB->instr_end()) == DestBeg) ||
        getBundleEnd(DestBeg) != SrcBundle) {
      LLVM_DEBUG(dbgs() << "<<Unknown dependency>>";);
      return Dependency::Unknown;
    }
  }
  unsigned dep = Dependency::Independent;

  auto CheckDep = [&](const MachineOperand &MO, unsigned Reg) -> void {
    bool IsDef = MO.isReg() ? MO.isDef() : true; // regmask clobbers are defs
    const MachineInstr *MI = MO.getParent();
    unsigned OpIdx = MO.isReg() ? MO.getOperandNo() : 0;
    // Use/Def should not be Defined.
    bool destIsBundle = (DestBeg->isBundle() || DestBeg->isBundledWithPred());
    // However, return Dependency::Independent for:
    // Reg is known to be HLX .new use and has def in Dest bundle
    // Reg is known to be use of a HLX load and has tmp def in Dest bundle
    if (HLV->isDefinedWithin(DestBeg, DestEnd, Reg, DepTrueP)) {
      bool depInDestBundle =
          destIsBundle && (getBundleStart(DepTrueP) == getBundleStart(DestBeg));

      // DestBundle could have a regular def and a .tmp def
      // Regular def could also be a redundant copy (such as l0 = l0)
      bool tmpDef = QII->isDotTmpInst(*DepTrueP);
      bool regularDef = !tmpDef;
      if (depInDestBundle &&
          HLV->isDefinedWithin(DestBeg, DepTrueP, Reg, DepTrueQ)) {
        tmpDef |= QII->isDotTmpInst(*DepTrueQ);
        regularDef |= !tmpDef;
      }

      bool hasHLXDep = depInDestBundle && (HLXUseReg == Reg);
      if (!IsDef) {
        // Pull up HLX load use or store.
        // We know that DestBundle has .tmp load that feeds the load use
        // or DestBundle has a def that feeds .new store
        if (hasHLXDep)
          return;
        LLVM_DEBUG(dbgs() << "<<RAW dependency>>";);
        dep |= Dependency::RAW;
      } else { // is Def.
        if (hasHLXDep && !regularDef)
          return;
        LLVM_DEBUG(dbgs() << "<<WAW dependency>>";);
        dep |= Dependency::WAW;
      }
      assert(DepTrueP != DestEnd);
      DepTrue = DepTrueP;
    }
    // Defs should not be used in the same packet via operand forwarding (.new,
    // .tmp or .cur).
    if (!IsDef)
      return;
    MICInstIterType LastII = DestEnd;
    while (HLV->isUsedWithin(DestBeg, LastII, Reg, DepAntiP)) {
      bool depInDestBundle =
          destIsBundle && (getBundleStart(DepAntiP) == getBundleStart(DestBeg));
      const MachineInstr *DepMI = &*DepAntiP;
      int UseIdx = DepMI->findRegisterUseOperandIdx(Reg, TRI);
      assert(UseIdx >= 0);
      // TODO: Packet can have both .cur and .new
      bool DefViaRegMask = !MO.isReg(); // regmask clobber has no def operand
      if (!EnableDefUseSamePacket || !depInDestBundle ||
          !QII->isSafeInBundleWAR(*MI, OpIdx, *DepMI, UseIdx, IsPullDown,
                                  DefViaRegMask)) {
        dep |= Dependency::WAR;
        assert(DepAntiP != DestEnd);
        DepAnti = DepAntiP;
        LLVM_DEBUG(dbgs() << "<<WAR dependency>>";);
        return;
      }
      // Safe only if inserted into same bundle as use; callers decide.
      dep |= Dependency::WAR_SameBundle;
      DepAnti = DepAntiP;
      LLVM_DEBUG(dbgs() << "<<WAR_SameBundle dependency>>";);
      LastII = DepAntiP;
    }
  };
  // TODO: Check if instruction to be pulled up can be newified
  for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg()) {
      // If this reg has subregs, check dependencies for each subreg
      if (MCSubRegIterator(MO.getReg(), TRI, false).isValid()) {
        for (MCPhysReg R : TRI->subregs(MO.getReg()))
          CheckDep(MO, R);
      } else
        CheckDep(MO, MO.getReg());
    } else if (MO.isRegMask()) {
      for (unsigned R = 1, NR = TRI->getNumRegs(); R != NR; ++R)
        if (MO.clobbersPhysReg(R))
          CheckDep(MO, R);
    }
  }
  return dep;
}

// Dependence of RMII within the same packet.
// \p SrcBeg and \p SrcEnd are begin and end of the bundle respectively.
// \p DepSet, if non-null, collects the same-packet instructions that must
// move with RMII.
int GreedyPullup::srcDependence(MIInstIterType SrcBeg, MIInstIterType SrcEnd,
                                MIInstIterType RMII,
                                SmallPtrSetImpl<MachineInstr *> *DepSet) const {
  LLVM_DEBUG(dbgs() << "\t[SrcDependence]";);
  bool PreventDuplex = PreventDuplexSep.getNumOccurrences()
                           ? PreventDuplexSep
                           : TM->getOptLevel() <= CodeGenOptLevel::Default;
  return QII->srcDependence(SrcBeg, SrcEnd, RMII, HLV->getMIUseDefs(), DepSet,
                            PreventDuplex, /*IsPullDown=*/false);
}

bool GreedyPullup::singleTrueDep(MICInstIterType SrcBeg, MICInstIterType SrcEnd,
                                 MICInstIterType MII, unsigned &Reg) const {
  const MachineInstr *MI = &*MII;
  bool TD = false;
  MICInstIterType DepTrueP = SrcEnd;
  for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isUse() && MO.getReg()) {
      if (HLV->isDefinedWithin(SrcBeg, SrcEnd, MO.getReg(), DepTrueP)) {
        if (TD)
          return false;
        Reg = MO.getReg();
        TD = true;
      }
    }
  }
  return true;
}

// Is it okay to execute MI as the first instruction in the MBB.
MIIterType GreedyPullup::findInsertionPtSpecLocal(MachineInstr *MI) {
  LLVM_DEBUG(dbgs() << "\n[findInsertionPtSpecLocal]");
  assert(!shouldNotBePulledUp(MI));
  MachineBasicBlock *MBB = MI->getParent();
  // First instruction in MBB. Bundled MI can never be the first because
  // there will be a Bundle header.
  if (MI == &*MBB->getFirstNonDebugInstr())
    return MBB->begin();
  std::pair<MIInstIterType, MIInstIterType> ParentBundle = getParentBundle(MI);

  if (MI->isBundled()) {
    // Inside the packet, check only for Src Dependency,
    // i.e. are there any consumers of MI within the same packet.
    unsigned SrcDep = srcDependence(ParentBundle.first, ParentBundle.second,
                                    MI->getIterator());
    // True/Unknown dependence => bail out.
    if (!Dependency::hasOnly(SrcDep, Dependency::Independent))
      return MBB->end();
    // MI is in the first bundle and it has no consumer in the same packet.
    if (ParentBundle.first == MBB->getFirstNonDebugInstr())
      return MBB->begin();
  }

  // If MI can execute as the first instruction.
  MIIterType InsPos = findInsertPtSpec(MI, MBB, false, false);

  return InsPos;
}

// Local pullup pass to schedule HVX instructions, while avoiding interlock
// stalls.  Schedule the load bypass opportunities in the basic block
// first.
void GreedyPullup::localPullupHVX(SchedulingRegion_t::SRAtom_t *SR) {
  if (!EnableHVXLocalPullup || !useHVXOps || DisableHVXLocalPullupAndSuperblock)
    return;

  // Only on innermost loops.
  if (EnableHVXLocalPullupInnermost) {
    auto *L = MLI.getLoopFor(SR);
    if (!L)
      return;
    if (!isSuperblockLoop(L))
      return;
  }

  // Check if the majority of instructions in the block are HVX instructions.
  if (EnableHVXSuperblockHeuristic && !isProfitableToScheduleHVX(SR))
    return;

  // Route HVX vector pullup through the shared local-pullup engine. HVXMode
  // admits plain HVX vector ops as candidates, applies the same
  // intermediate-packet-aware dependence checks as scalar pullup, and
  // generates .cur load bypasses where profitable.
  localPullupImpl(SR, /*HVXMode=*/true);
}

// InstrToMove is the instruction to be moved to the same packet as LoadInstr
// to utilize the load bypass (.cur).
// LoadInstr is inside DestBundle.
void GreedyPullup::genLoadBypass(MachineInstr *LoadInstr,
                                 MachineInstr *InstrToMove) {
  assert(LoadInstr->isBundled() && "Load bypass requires a bundle");
  MachineInstr *DestBundle = &*getBundleStart(LoadInstr->getIterator());
  int CurOpcode = QII->getDotCurOp(*LoadInstr);
  LoadInstr->setDesc(QII->get(CurOpcode));
  MIBundleBuilder Bundle(DestBundle);
  Bundle.insert(std::next(LoadInstr->getIterator()), InstrToMove);
  unsigned DestReg = LoadInstr->getOperand(0).getReg();
  for (auto &MO : InstrToMove->operands()) {
    if (!MO.isReg())
      continue;
    if (!MO.isUse())
      continue;
    for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI) {
      if (*AI == DestReg)
        MO.setIsInternalRead();
    }
  }
}

// Returns true if a single RAW dependence on \p DepTrue can be resolved via a
// .cur load bypass: \p DepTrue must be a .cur-able HVX load that lives in
// DestBundle, there must be no WAW on the destination, and no instruction
// other than the consumer being pulled up may read the load's def reg within
// the destination packet [next(DestBundle), PktEnd). \p Consumer is the
// instruction being pulled up; the cur-load must be its only true-dependence
// producer in the pulled region (see the intermediate-producer check below).
bool GreedyPullup::canGenCurBypass(MIInstIterType DestBundle,
                                   MachineInstr *DepTrue, MIInstIterType PktEnd,
                                   int DstDep, MachineInstr *Consumer,
                                   const MachineInstr *Exempt) {
  assert(Consumer && "Consumer is required for .cur bypass correctness checks");
  if (!DepTrue || Dependency::has(DstDep, Dependency::WAW))
    return false;
  // .cur load bypass is keyed on the mayBeCurLoad TSFlag, not on HVX: any load
  // with a .cur form (present or future) is eligible.
  if (!QII->mayBeCurLoad(*DepTrue))
    return false;
  if (&*getBundleStart(DepTrue->getIterator()) != &*DestBundle)
    return false;
  // If DestBundle is a real bundle, the load must be bundled inside it.
  if (DestBundle->isBundle() && !DepTrue->isBundled())
    return false;
  // No other consumer of the load's def reg may exist in the packet.
  unsigned DestReg = DepTrue->getOperand(0).getReg();
  for (auto &MI :
       iterator_range<MIInstIterType>(std::next(DestBundle), PktEnd)) {
    if (&MI == DepTrue)
      continue;
    for (const MachineOperand &MOP : MI.operands()) {
      if (MOP.isReg() && MOP.isUse() && !MOP.isDebug() && MOP.getReg() &&
          !MOP.isUndef() && TRI->regsOverlap(MOP.getReg(), DestReg))
        return false;
    }
  }
  // .cur only forwards the load's primary def; a post-increment
  // load's second (tied pointer) def is not forwarded, so masking the whole
  // RAW bit is unsound if the consumer reads that other def instead.
  if (Consumer) {
    for (const MachineOperand &Def : DepTrue->operands()) {
      if (!Def.isReg() || !Def.isDef() || !Def.getReg() ||
          Def.getReg() == DestReg)
        continue;
      for (const MachineOperand &Use : Consumer->operands()) {
        if (Use.isReg() && Use.isUse() && !Use.isUndef() && Use.getReg() &&
            TRI->regsOverlap(Use.getReg(), Def.getReg()))
          return false;
      }
    }
  }
  // The caller masks the whole RAW bit, which is only sound if the cur-load is
  // the consumer's sole producer across the hoisted region. Refuse if any
  // instruction between DestBundle and the consumer defines a reg the consumer
  // reads (use-before-def); the scalar dependence surfaces only one producer.
  for (MIInstIterType MI = std::next(DestBundle);
       MI != Consumer->getIterator() &&
       MI != Consumer->getParent()->instr_end();
       ++MI) {
    if (&*MI == DepTrue || &*MI == Exempt || MI->isBundle())
      continue;
    for (const MachineOperand &Def : MI->operands()) {
      if (!Def.isReg() || !Def.isDef() || !Def.getReg())
        continue;
      bool ConsumerReads =
          llvm::any_of(Consumer->operands(), [&](const MachineOperand &Use) {
            return Use.isReg() && Use.isUse() && !Use.isUndef() &&
                   Use.getReg() && TRI->regsOverlap(Use.getReg(), Def.getReg());
          });
      if (ConsumerReads)
        return false;
    }
  }
  return true;
}
// Check if moving SrcMI to DestBundle will cause an interlock-stall due to
// input latency.
bool GreedyPullup::canCauseInterlockStall(MIInstIterType DestBundle,
                                          MachineInstr *SrcMI,
                                          const InstrItineraryData *II) {
  bool PrevOK = true;
  auto PrevEnd = std::prev(DestBundle);
  auto *MBB = SrcMI->getParent();
  MachineInstr *PrevStart = nullptr;
  if (PrevEnd != MBB->end()) {
    if (PrevEnd->isBundled())
      PrevStart = &*getBundleStart(PrevEnd);
    else
      PrevStart = &*PrevEnd;
  }

  if (PrevEnd != MBB->end() && PrevStart != nullptr) {
    std::vector<const MachineInstr *> DepTruePrevVec, DepAntiPrevVec,
        DepOutputPrevVec;
    std::vector<unsigned> TrueRegs;
    int dep = dstDependence(PrevStart->getIterator(), DestBundle,
                            SrcMI->getIterator(), DepTruePrevVec,
                            DepAntiPrevVec, DepOutputPrevVec, &TrueRegs);
    if (Dependency::has(dep, Dependency::RAW)) {
      unsigned RegIdx = 0;
      for (auto DepTruePrev : DepTruePrevVec) {
        int DefIdx = -1, UseIdx = -1;
        for (MCRegAliasIterator AI(TrueRegs[RegIdx], TRI, true);
             AI.isValid() && DefIdx == -1; ++AI)
          DefIdx = DepTruePrev->findRegisterDefOperandIdx(*AI, /*TRI=*/nullptr);
        for (MCRegAliasIterator AI(TrueRegs[RegIdx], TRI, true);
             AI.isValid() && UseIdx == -1; ++AI)
          UseIdx = SrcMI->findRegisterUseOperandIdx(*AI, /*TRI=*/nullptr);
        std::optional<unsigned> Latency =
            QII->getOperandLatency(II, *DepTruePrev, DefIdx, *SrcMI, UseIdx);
        if (!Latency) {
          ++RegIdx;
          continue;
        }
        Latency = MF.getSubtarget<HexagonSubtarget>().updateLatency(
            *SrcMI, const_cast<MachineInstr &>(*DepTruePrev), false, *Latency);
        if (Latency > 1) {
          PrevOK = false;
          break;
        }
        ++RegIdx;
      }
    }
  }
  return !PrevOK;
}

// Find all uses of .tmp HLX load and attempt to pullup the uses
// to the load's bundle
void GreedyPullup::localPullupHLXLoadUses(SchedulingRegion_t::SRAtom_t *SR,
                                          MachineInstr *HLXLoadMI,
                                          MachineInstr *HLXCopyMI) {
  LLVM_DEBUG(dbgs() << "\n\n\t\t[LocalPullupHLXLoadUses]");
  MIIterType BI =
      MachineInstrBundleIterator<MachineInstr, false>::getAtBundleBegin(
          HLXLoadMI->getIterator());
  MIInstIterType DestBundle = BI.getInstrIterator();
  unsigned HLXLoadDestReg = HLXLoadMI->getOperand(0).getReg();
  unsigned NumLoadUsesPullups = 0;
  // List of uses of HLX load that can be pulled up
  SmallVector<MachineInstr *, 4> LoadUses;

  if (!HLXCopyMI || !HLXLoadDestReg ||
      (HLXCopyMI->getOperand(0).getReg() != HLXLoadDestReg))
    return;
  LLVM_DEBUG(dbgs() << "\nFound HLX load pseudo";
             dbgs() << "\nFinding uses of HLX load for pullup into: "
                    << *DestBundle;);
  MachineBasicBlock *MBB = HLXCopyMI->getParent();
  MIInstIterType LoadBundleEnd = getBundleEnd(DestBundle);
  // Early exit if DestBundle is last bundle in MBB
  if (LoadBundleEnd == MBB->end())
    return;

  // Find all uses of the load
  MICInstIterType NextLoadRegDef = MBB->instr_end();
  MICInstIterType DepTrueP = MBB->instr_end();
  while (HLV->isDefinedWithin(LoadBundleEnd, NextLoadRegDef, HLXLoadDestReg,
                              DepTrueP))
    NextLoadRegDef = DepTrueP;
  MICInstIterType PrevLoadRegUse = NextLoadRegDef;
  // Instruction that defines HLXLoadDestReg may also use the same register
  // Allow isUsedWithin to find it
  if (PrevLoadRegUse != MBB->instr_end())
    ++PrevLoadRegUse;
  while (HLV->isUsedWithin(LoadBundleEnd, PrevLoadRegUse, HLXLoadDestReg,
                           DepTrueP)) {
    LLVM_DEBUG(dbgs() << "\nFound use: "; DepTrueP->dump(););
    LoadUses.push_back(&*DepTrueP.getNonConst());
    PrevLoadRegUse = DepTrueP;
  }

  // Check if the load reg is live out from MBB
  bool LoadRegIsLiveOut = HLV->isLiveOut(MBB, HLXLoadDestReg) &&
                          (NextLoadRegDef == MBB->instr_end());

  // Check if the bundle contains a call that reads the loaded register
  MICInstIterType BStart = getBundleStart(DestBundle);
  MICInstIterType BEnd = getBundleEnd(DestBundle);

  bool BundleHasCallUse =
      QII->hasCallUseInBundle(BStart, BEnd, HLXLoadDestReg, *TRI);

  // Attempt to pull up each use of the load
  // Iteration order: Nearest to farthest
  for (MachineInstr *SrcMI : reverse(LoadUses)) {
    if (!HLXCopyMI && (DestBundle->getBundleSize() >= HEXAGON_PACKET_SIZE))
      continue;
    MIInstIterType SrcMII = SrcMI->getIterator();
    // If Use is not bundled, create a single instruction bundle for it
    if (!SrcMI->isBundled())
      finalizeBundle(*(SrcMI->getParent()), SrcMII, std::next(SrcMII));
    // Check if resources are available in Bundle.
    std::pair<MIInstIterType, MIInstIterType> ParentBundle =
        getParentBundle(&*SrcMII);

    LLVM_DEBUG(dbgs() << "\nChecking Resources for: " << *SrcMII;);
    if (!isLegalToPacketizeWith(BI, SrcMI) || isSoloInstruction(*SrcMI) ||
        shouldNotBePulledUp(SrcMI))
      continue;
    MICInstIterType DepTrue = MBB->instr_end(), DepAnti = MBB->instr_end();
    int DstDep = dstDependence(DestBundle, ParentBundle.first, SrcMII, DepTrue,
                               DepAnti, HLXLoadDestReg);
    int SrcDep = srcDependence(ParentBundle.first, ParentBundle.second, SrcMII);
    MIInstIterType DestPos = DestBundle;
    getResourcesReadyForBundle(DestPos);
    // Delete copy and attempt to pull up the last use
    if ((!Dependency::hasOnly(DstDep, Dependency::Independent) ||
         !canReserveResources(SrcMII)) &&
        !LoadRegIsLiveOut && (NumLoadUsesPullups == (LoadUses.size() - 1)) &&
        SrcMI == LoadUses.front() && !BundleHasCallUse) {
      LLVM_DEBUG(dbgs() << "\nAttempting to pullup last use by erasing copy: ";
                 HLXCopyMI->dump(););
      HLXCopyMI = HLXCopyMI->removeFromBundle();
      DstDep = dstDependence(DestBundle, ParentBundle.first, SrcMII, DepTrue,
                             DepAnti, HLXLoadDestReg);
      getResourcesReadyForBundle(DestPos);
    }
    // Speculate when there are no dependencies across scheduling path.
    if (!(Dependency::hasOnly(DstDep, Dependency::Independent)) ||
        !(Dependency::hasOnly(SrcDep, Dependency::Independent)))
      continue;
    LLVM_DEBUG(dbgs() << "[Can add to packet]";);
    if (!canReserveResources(SrcMII)) {
      LLVM_DEBUG(dbgs() << " .....NO");
      continue;
    }
    LLVM_DEBUG(dbgs() << "..... YES (Resource available)";);
    MachineInstr *InstrToMove = removeMI(SrcMI, QII, nullptr);

    MachineOperand *Op = nullptr;
    for (MCRegAliasIterator AI(HLXLoadDestReg, TRI, true); AI.isValid(); ++AI) {
      Op = InstrToMove->findRegisterUseOperand(*AI, /*TRI=*/nullptr);
      if (Op)
        break;
    }
    assert(Op && "Could not find use operand!");
    Op->setIsInternalRead();

    // Load use must be inserted after the HLX load instruction
    // in the bundle to avoid confusing finalizeBundle about liveness
    // of the use reg
    MIBundleBuilder Bundle(&*DestBundle);
    Bundle.insert(std::next(HLXLoadMI->getIterator()), InstrToMove);

    // Update pullup count. This might be useful in doing another
    // iteration of global pullup into these MBBs.
    ++LocalPullups;
    ++NumLoadUsesPullups;
    LLVM_DEBUG(dbgs() << "\nSpeculated Local MI:\t" << *InstrToMove
                      << "\nMBB: " << *InstrToMove->getParent());

    HLV->updateLocalLiveness(SR, true);
  }
  // Keep the copy if all uses could not be pulled up, if register is live out,
  // if there are no in-bundle uses, or if a call in the bundle reads the reg.
  if (NumLoadUsesPullups != LoadUses.size() || LoadRegIsLiveOut ||
      BundleHasCallUse) {
    // Add back copy if last use could not be pulled up
    if (!HLXCopyMI->isBundled()) {
      LLVM_DEBUG(dbgs() << "\nRestoring copy\n");
      MIBundleBuilder Bundle(&*DestBundle);
      Bundle.insert(std::next(HLXLoadMI->getIterator()), HLXCopyMI);
      HLV->updateLocalLiveness(SR, true);
    }
  } else {
    // Delete the copy if all uses have been pulled up
    if (HLXCopyMI->isBundled())
      HLXCopyMI->eraseFromBundle();
    else
      MF.deleteMachineInstr(HLXCopyMI);
    HLV->updateLocalLiveness(SR, true);
  }
}

// Find reaching def of .new HLX store and attempt to pullup the store
// to the reaching def's bundle
void GreedyPullup::localPullupHLXStore(SchedulingRegion_t::SRAtom_t *SR,
                                       MachineInstr *HLXStoreMI,
                                       MachineInstr *HLXCopyMI) {
  LLVM_DEBUG(dbgs() << "\n\n\t\t[LocalPullupHLXStore]");
  MIIterType BI =
      MachineInstrBundleIterator<MachineInstr, false>::getAtBundleBegin(
          HLXStoreMI->getIterator());
  MIInstIterType SrcBundle = BI.getInstrIterator();
  unsigned HLXStoreSrcReg = HLXStoreMI->getOperand(1).getReg();

  // Bundle with Def instruction is the destination bundle
  MachineInstr *DestMI = nullptr;
  // Check if .new HLX store has a HLX copy that feeds it
  if (!HLXCopyMI || !HLXStoreSrcReg ||
      HLXCopyMI->getOperand(0).getReg() != HLXStoreSrcReg)
    return;
  LLVM_DEBUG(dbgs() << "\nFound HLX store pseudo";
             dbgs() << "\nFinding def for HLX store to be pulled up from: "
                    << *SrcBundle;);
  MachineBasicBlock *MBB = HLXCopyMI->getParent();
  MIInstIterType StoreBundleStart = getBundleStart(SrcBundle);

  MICInstIterType DepTrueP = MBB->instr_end();
  if (HLV->isDefinedWithin(MBB->instr_front().getIterator(), StoreBundleStart,
                           HLXStoreSrcReg, DepTrueP))
    DestMI = &*DepTrueP.getNonConst();

  if (!DestMI)
    return;
  MIInstIterType DestMII = DestMI->getIterator();
  // If Def is not bundled, create a single instruction bundle for it
  if (!DestMI->isBundled())
    finalizeBundle(*(DestMI->getParent()), DestMII, std::next(DestMII));
  std::pair<MICInstIterType, MICInstIterType> ParentBundle =
      getParentBundle(DestMI);

  // Check if resources are available in Bundle.
  LLVM_DEBUG(dbgs() << "\nChecking Resources for: " << *HLXStoreMI;);
  MIInstIterType DestPos = getBundleStart(DestMII);
  if (!isLegalToPacketizeWith(&*DestPos, HLXStoreMI))
    return;
  MICInstIterType DepTrue = MBB->instr_end(), DepAnti = MBB->instr_end();
  int DstDep =
      dstDependence(ParentBundle.first, SrcBundle, HLXStoreMI->getIterator(),
                    DepTrue, DepAnti, HLXStoreSrcReg);

  // .new HLX store cannot have dependence on any instruction other than
  // feeder copy in source packet. Skip srcDependence check.

  // Speculate when there are no dependencies across scheduling path.
  if (!Dependency::hasOnly(DstDep, Dependency::Independent))
    return;
  LLVM_DEBUG(dbgs() << "[Can add to packet]";);
  getResourcesReadyForBundle(DestPos);
  if (!canReserveResources(HLXStoreMI->getIterator())) {
    LLVM_DEBUG(dbgs() << " .....NO");
    return;
  }
  LLVM_DEBUG(dbgs() << "..... YES (Resource available)";);

  // Remove the MI before creating the bundle builder, so that the
  // builder has the correct iterator bundle end.
  MachineInstr *InstrToMove = removeMI(HLXStoreMI, QII, nullptr);

  // HLX store must be inserted after the def in the bundle to avoid confusing
  // finalizeBundle about liveness of the def reg
  MIBundleBuilder Bundle(&*(DestPos));
  Bundle.insert(std::next(DestMII), InstrToMove);

  if (HLXCopyMI->isBundled())
    HLXCopyMI->eraseFromBundle();
  else
    HLXCopyMI->eraseFromParent();

  // Update pullup count. This might be useful in doing another
  // iteration of global pullup into these MBBs.
  ++LocalPullups;
  LLVM_DEBUG(dbgs() << "\nSpeculated Local MI:\t" << *InstrToMove
                    << "\nMBB: " << *InstrToMove->getParent());
  HLV->updateLocalLiveness(SR, true);
}

// Pullup HLX instructions to create more dense bundles for
// HLX load/store instructions.
// This will help eliminate HLX copy instructions inserted during pseudo
// expansion. Find uses of a .tmp HLX load and try to pullup into the load's
// bundle Pullup .new HLX store into a bundle containing its def.
void GreedyPullup::localPullupHLX(SchedulingRegion_t::SRAtom_t *SR) {
  if (!useHLXOps || !EnableHLXLoadStorePullup)
    return;

  assert(!SR->empty());
  LLVM_DEBUG(dbgs() << "\n\n\t\t[LocalPullupHLX]\tBB#" << SR->getNumber(););
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> HLXLoadPseudos;
  SmallVector<std::pair<MachineInstr *, MachineInstr *>, 4> HLXStorePseudos;
  for (MIIterType BI = SR->begin(); BI != SR->end(); ++BI) {
    // Do not count DBG instructions if they are present.
    if (BI->isDebugInstr())
      continue;
    // Only look for bundles where HLX load or HLX store pseudos
    // were expanded
    if (!BI->isBundle() || (!BI->mayLoad() && !BI->mayStore()))
      continue;
    MachineInstr *HLXLoadMI = nullptr;
    MachineInstr *HLXStoreMI = nullptr;
    SmallVector<MachineInstr *, 2> HLXCopies;
    MIInstIterType MII = BI.getInstrIterator();
    for (++MII; // Skip the bundle header.
         (MII != SR->instr_end()) && MII->isInsideBundle(); ++MII) {
      if (!QII->isHLXVec(*MII))
        continue;
      switch (MII->getOpcode()) {
      case Hexagon::LX_L16b_tmp:
        HLXLoadMI = &*MII;
        break;
      case Hexagon::LX_S16b_new:
        HLXStoreMI = &*MII;
        break;
      case Hexagon::LX_assign:
        if (MII->getOperand(0).getReg() == MII->getOperand(1).getReg()) {
          HLXCopies.push_back(&*MII);
        }
        break;
      default:
        break;
      }
    }

    // Collect all copies and match them to HLX loads/stores by checking
    // register operands.
    MachineInstr *HLXLoadCopyMI = nullptr;
    MachineInstr *HLXStoreCopyMI = nullptr;
    for (MachineInstr *CopyMI : HLXCopies) {
      Register HLXCopyReg = CopyMI->getOperand(0).getReg();
      // If source operand of LX_assign is marked internal,
      // it corresponds to HLX load
      bool IsInternal = CopyMI->getOperand(1).isInternalRead();
      // If we have both HLX load and store in same packet, with two identical
      // copies, pair one copy with load and one copy with store.
      if (HLXStoreMI && !HLXStoreCopyMI && !IsInternal &&
          HLXStoreMI->getOperand(1).getReg() == HLXCopyReg)
        HLXStoreCopyMI = CopyMI;
      else if (HLXLoadMI && !HLXLoadCopyMI &&
               HLXLoadMI->getOperand(0).getReg() == HLXCopyReg)
        HLXLoadCopyMI = CopyMI;
    }
    if (HLXLoadCopyMI)
      HLXLoadPseudos.push_back(std::make_pair(HLXLoadMI, HLXLoadCopyMI));
    if (HLXStoreCopyMI)
      HLXStorePseudos.push_back(std::make_pair(HLXStoreMI, HLXStoreCopyMI));
  }

  for (auto &LoadPseudo : HLXLoadPseudos)
    localPullupHLXLoadUses(SR, LoadPseudo.first, LoadPseudo.second);

  for (auto &StorePseudo : HLXStorePseudos)
    localPullupHLXStore(SR, StorePseudo.first, StorePseudo.second);
}

// Try the two-load .cur load-bypass pull-up for a single-instruction cluster.
//
// This handles a consumer with up to two RAW producers that share a packet,
// where one producer is a .cur-able load. The load is rewritten to its .cur
// form (forwarding its result within the packet, freeing a register-read
// slot); the second producer, if any, is relocated to a loop-predecessor
// packet via lastChanceGlobalPullup so the consumer fits.
//
// Uses the cluster pullup dependence check, which also covers the packets in
// between, so SrcMI never moves past a packet it depends on.
//
// Returns true if it moved SrcMI (caller should skip the common cluster
// machinery for this candidate).
bool GreedyPullup::tryTwoLoadCurBypass(
    MIIterType &BI, SchedulingRegion_t::SRAtom_t *SR,
    MIInstIterType &DestBundle, MIIterType &SrcBundle, MIIterType &NextBI,
    bool isNextBISrc, MIIterType &Next, MachineInstr *SrcMI,
    SmallPtrSetImpl<MachineInstr *> &PulledUpSet) {
  MIInstIterType SrcMII = SrcMI->getIterator();
  std::pair<MIInstIterType, MIInstIterType> ParentBundle =
      getParentBundle(&*SrcMII);

  // Enumerate producers via the vector dependence overload; only the vector
  // form can express two RAW producers of a single consumer.
  std::vector<const MachineInstr *> DepTrueVec, DepAntiVec, DepOutputVec;
  int VecDstDep = dstDependence(DestBundle, ParentBundle.first, SrcMII,
                                DepTrueVec, DepAntiVec, DepOutputVec);

  LLVM_DEBUG(dbgs() << "\n[TwoLoadCur] Candidate consumer: " << *SrcMI
                    << "\t\tRAW producers=" << DepTrueVec.size()
                    << " output deps=" << DepOutputVec.size() << "\n");

  if (!Dependency::has(VecDstDep, Dependency::RAW) || DepTrueVec.size() > 2) {
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: no RAW dep or >2 producers.\n");
    return false;
  }

  bool GenCur = false;
  bool PotentialGenCur = false;
  const MachineInstr *DepTrue = nullptr;
  const MachineInstr *Other = nullptr;

  auto IsCurLoad = [&](const MachineInstr *MI) {
    return QII->isHVXVec(*MI) && QII->mayBeCurLoad(*MI);
  };

  if (DepTrueVec.size() == 1) {
    const MachineInstr *V0 = DepTrueVec[0];
    if (!Dependency::has(VecDstDep, Dependency::WAW) && IsCurLoad(V0) &&
        getBundleStart(V0->getIterator()) == DestBundle) {
      GenCur = true;
      DepTrue = V0;
    }
  } else if (DepTrueVec.size() == 2 && DepOutputVec.size() <= 1) {
    const MachineInstr *V0 = DepTrueVec[0];
    const MachineInstr *V1 = DepTrueVec[1];
    if (getBundleStart(V0->getIterator()) ==
        getBundleStart(V1->getIterator())) {
      if (IsCurLoad(V0)) {
        DepTrue = V0;
        Other = V1;
      } else if (IsCurLoad(V1)) {
        DepTrue = V1;
        Other = V0;
      }
      if (Other && (!Dependency::has(VecDstDep, Dependency::WAW) ||
                    Other == DepOutputVec[0]))
        PotentialGenCur = true;
    }
  }

  if (!GenCur && !PotentialGenCur)
    return false;

  LLVM_DEBUG(dbgs() << "[TwoLoadCur] Selected "
                    << (DepTrueVec.size() == 1 ? "single-load" : "two-load")
                    << " case: GenCur=" << GenCur
                    << " PotentialGenCur=" << PotentialGenCur << "\n");

  // Validate the load actually lives at the head of the destination bundle.
  if (GenCur && DestBundle->isBundle()) {
    if (!DepTrue->isBundled() ||
        getBundleStart(DepTrue->getIterator()) != DestBundle)
      GenCur = false;
  }

  // The load result must not be consumed by any other instruction between the
  // destination bundle and the source bundle (otherwise .cur forwarding is
  // unsafe). The second producer (Other) is exempt.
  if (GenCur) {
    Register DestReg = DepTrue->getOperand(0).getReg();
    for (MIInstIterType MI = std::next(DestBundle);
         MI != NextBI.getInstrIterator(); ++MI) {
      if (&*MI == Other)
        continue;
      bool Reads = llvm::any_of(MI->operands(), [&](const MachineOperand &MO) {
        return MO.isReg() && MO.isUse() && !MO.isUndef() && MO.getReg() &&
               TRI->regsOverlap(MO.getReg(), DestReg);
      });
      if (Reads) {
        GenCur = false;
        PotentialGenCur = false;
        break;
      }
    }
  }

  if (!GenCur && !PotentialGenCur)
    return false;

  // Guard against a third, non-forwardable RAW producer the two-element vector
  // dependence misses (use-before-def). Other is relocated later, so exempt it.
  if (DepTrue &&
      !canGenCurBypass(DestBundle, const_cast<MachineInstr *>(DepTrue),
                       NextBI.getInstrIterator(), VecDstDep, SrcMI, Other)) {
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: unsafe intermediate RAW "
                         "producer (canGenCurBypass).\n");
    return false;
  }

  // Allow WAR_SameBundle (SrcMI joins that packet) and, for a .cur, RAW (the
  // load is forwarded and the other producer is moved away); any other
  // dependence blocks the move.
  MachineBasicBlock *MBB = SrcMI->getParent();
  MICInstIterType ScalarDepTrue = MBB->instr_end();
  MICInstIterType ScalarDepAnti = MBB->instr_end();
  int ScalarDstDep = dstDependence(DestBundle, ParentBundle.first, SrcMII,
                                   ScalarDepTrue, ScalarDepAnti);
  int DstDepMasked = ScalarDstDep & ~Dependency::WAR_SameBundle;
  if (GenCur || PotentialGenCur)
    DstDepMasked &= ~Dependency::RAW;
  if (!Dependency::hasOnly(DstDepMasked, Dependency::Independent)) {
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: scalar dest dependence not "
                         "independent after masking.\n");
    return false;
  }

  int SrcDep = srcDependence(ParentBundle.first, ParentBundle.second, SrcMII);
  if (!Dependency::hasOnly(SrcDep, Dependency::Independent)) {
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: source dependence not "
                         "independent.\n");
    return false;
  }

  auto &ST = MF.getSubtarget<HexagonSubtarget>();
  const InstrItineraryData *II = ST.getInstrItineraryData();
  if (canCauseInterlockStall(DestBundle, SrcMI, II)) {
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: would cause an interlock "
                         "stall.\n");
    return false;
  }

  // Ensure the destination is a real bundle so we can pull into it.
  bool UndoDestBundle = false;
  if (!DestBundle->isBundle()) {
    finalizeBundle(*DestBundle->getParent(), DestBundle, std::next(DestBundle));
    DestBundle = std::prev(DestBundle);
    UndoDestBundle = true;
  }

  MIInstIterType DestPos = DestBundle;
  getResourcesReadyForBundle(DestPos);

  // PotentialGenCur forces the resource reservation to fail so we drop into
  // lastChanceGlobalPullup, which relocates the second producer to free a slot.
  bool CanReserve = canReserveResources(SrcMII) && !PotentialGenCur;
  bool CanBeMoved = false;
  if (!CanReserve) {
    const MachineInstr *LoadBypassCand = PotentialGenCur ? Other : nullptr;
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Resources tight; trying "
                         "lastChanceGlobalPullup (LoadBypassCand="
                      << (LoadBypassCand ? "Other" : "null") << ").\n");
    if ((CanBeMoved =
             lastChanceGlobalPullup(DestPos, SrcMII, LoadBypassCand))) {
      DestBundle = DestPos;
      if (LoadBypassCand)
        GenCur = true;
    }
  } else {
    // Single-load .cur bypass whose packet already has room: just packetize.
    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Packet has room; packetizing "
                         "directly.\n");
    CanBeMoved = isLegalToPacketizeWith(DestBundle, SrcMI);
  }

  if (CanBeMoved) {
    // Keep NextBI/SrcBundle valid across the move.
    bool FixNextBI = false;
    if (SrcMII->isBundled()) {
      SrcBundle = getBundleStart(SrcMII);
      if (&*SrcBundle == &*NextBI)
        FixNextBI = true;
    } else if (&*SrcMII == &*NextBI) {
      NextBI = DestBundle;
    }

    MachineInstr *SingleInstr = nullptr;
    MachineInstr *Instr = removeMI(&*SrcMII, QII, nullptr, &SingleInstr);
    if (SingleInstr && FixNextBI)
      NextBI = SingleInstr->getIterator();

    if (GenCur)
      genLoadBypass(const_cast<MachineInstr *>(DepTrue), Instr);
    else
      MBB->insert(std::next(DestBundle), Instr);

    LLVM_DEBUG(dbgs() << "[TwoLoadCur] Pulled up (GenCur=" << GenCur
                      << "): " << *Instr);
    ++LocalPullups;
    PulledUpSet.insert(Instr);
    HLV->updateLocalLiveness(DestBundle->getParent(), true);
    return true;
  }

  LLVM_DEBUG(dbgs() << "[TwoLoadCur] Reject: candidate could not be moved.\n");
  if (UndoDestBundle) {
    MIInstIterType I = std::next(DestPos);
    I->unbundleFromPred();
    DestBundle->eraseFromParent();
    DestBundle = I;
  }
  return false;
}

// TODO: Handle non-packetized instructions.
// @note: Only speculated instructions can be done here.
void GreedyPullup::localPullup(SchedulingRegion_t::SRAtom_t *SR) {
  localPullupImpl(SR, /*HVXMode=*/false);
}

// Shared local pullup engine. When HVXMode is set, plain HVX vector ops are
// admitted as pullup candidates and .cur load-bypass is generated; otherwise
// HVX vector ops are rejected (scalar/HLX behavior).
void GreedyPullup::localPullupImpl(SchedulingRegion_t::SRAtom_t *SR,
                                   bool HVXMode) {
  assert(!SR->empty());
  LLVM_DEBUG(dbgs() << "\n\n\t\t[LocalPullup]\tBB#" << SR->getNumber(););

  // Sparse pullup and cluster pullup:
  // Enabled by default
  // Disabled for TinyCore.

  // Allow pullup to/from unbundled instructions or from packets that have <= 2
  // instructions.
  bool AllowSparsePullup = EnableSparsePullup && !IsTinyCore;

  // Allow pullup of cluster of dependent instructions.
  bool AllowClusterPullup = EnableClusterPullup && !IsTinyCore;

  MIIterType NextBI = SR->begin();
  for (MIIterType BI = NextBI; BI != std::prev(SR->end()) && BI != SR->end();
       BI = NextBI) {
    NextBI = std::next(BI);
    if (BI->isDebugInstr())
      continue;
    MIInstIterType DestBundle = BI.getInstrIterator();
    if (!DestBundle->isBundle() && AllowSparsePullup) {
      if (isSoloInstruction(*DestBundle))
        continue;
      finalizeBundle(*(DestBundle->getParent()), DestBundle,
                     std::next(DestBundle));
      DestBundle = std::prev(DestBundle);
      BI = DestBundle;
    }
    if (!DestBundle->isBundle())
      continue;
    LLVM_DEBUG(dbgs() << "\nDestBundle: " << *DestBundle;);
    MIIterType SrcBundle = BI;

    // Move to the next bundle to find candidates for local pullup.
    // Check dependencies from DestBundle through the SrcBundle.
    unsigned NumBundles = 0;
    MIIterType Next = std::next(SrcBundle);
    for (SrcBundle = Next; SrcBundle != SR->end(); SrcBundle = Next) {
      Next = std::next(SrcBundle);
      // Do not count DBG instructions if they are present.
      if (SrcBundle->isDebugInstr())
        continue;
      // Verify max pull-up distance based on non-debug values,
      // otherwise we can quit earlier in case of -g code.
      if (++NumBundles > MaxPUBundles)
        break;
      if (DestBundle->getBundleSize() >= HEXAGON_PACKET_SIZE)
        break;
      LLVM_DEBUG(dbgs() << "\nSrcBundle: " << *SrcBundle;);

      bool isNextBISrc = false;
      if (SrcBundle == std::next(BI))
        isNextBISrc = true;

      // Pull an unbundled call into the previous packet when the call does
      // not depend on that packet, e.g.:
      // BUNDLE 4 {
      //    S4_storeiri_io renamable $r16, 12
      //    renamable $r0 = L2_loadrub_io killed renamable $r17, 8
      //    renamable $r17 = A2_addi renamable $r16, 12
      //  }
      // J2_call @Foo
      // The generic local pullup does not handle this: the call is not in a
      // bundle, and shouldNotBePulledUp rejects calls.
      // TODO: Handle this in sparse pullup if shouldNotBePulledUp is relaxed.
      if (!SrcBundle->isBundle() && isNextBISrc) {
        MIInstIterType MII = SrcBundle.getInstrIterator();
        MachineInstr &MI = *MII;
        if (MI.getOpcode() == Hexagon::J2_call) {
          if (isLegalToPacketizeWith(BI, &MI)) {
            MIInstIterType DestPos = DestBundle;
            getResourcesReadyForBundle(DestPos);
            ++DestPos; // Skip the bundle header.
            while (DestPos->isInsideBundle())
              ++DestPos;
            --DestPos;
            if (canReserveResources(MII)) {
              NextBI = std::next(SrcBundle);
              Next = std::next(SrcBundle);
              MachineInstr *InstrToMove = removeMI(&MI, QII, nullptr);
              MIBundleBuilder Bundle(&*DestBundle);

              Bundle.insert(std::next(DestPos), InstrToMove);

              // Update pullup count. This might be useful in doing another
              // iteration of global pullup into these MBBs.
              ++LocalPullups;
              LLVM_DEBUG(dbgs() << "\nSpeculated Local MI:\t" << *InstrToMove
                                << "\nMBB: " << *InstrToMove->getParent());
              HLV->updateLocalLiveness(SR, true);
              continue;
            }
          }
        }
      }

      // When the source bundle has <= 2 instructions, and pullup from sparse
      // bundles is disabled, skip the bundle
      if (SrcBundle->getBundleSize() <= 2 && !AllowSparsePullup)
        continue;

      MIInstIterType SrcMII = SrcBundle.getInstrIterator();
      if (!SrcMII->isBundled() && AllowSparsePullup) {
        if (isSoloInstruction(*SrcMII) || shouldNotBePulledUp(&*SrcMII))
          continue;
        finalizeBundle(*(SrcMII->getParent()), SrcMII, std::next(SrcMII));
        MIInstIterType BundleHead = std::prev(SrcMII);
        SrcBundle = BundleHead;
        if (isNextBISrc)
          NextBI = BundleHead;
        SrcMII = BundleHead;
      }
      MIInstIterType SrcBundleEnd = getBundleEnd(SrcBundle.getInstrIterator());

      SmallVector<SmallPtrSet<MachineInstr *, 4>, 4> SrcClusterMIs;
      SmallPtrSet<MachineInstr *, 4> SrcVisitedSet;
      // Find a suitable candidate for speculation to DestBundle and speculate.
      for (++SrcMII; // Skip the bundle header.
           (SrcMII != SR->instr_end()) && SrcMII->isInsideBundle(); ++SrcMII) {
        MachineInstr *SrcMI = &*SrcMII;
        if (isSoloInstruction(*SrcMI) || SrcMI->isDebugInstr() ||
            !SrcVisitedSet.insert(SrcMI).second)
          continue;

        // Cluster pullup: collect dependent instructions in DepSet.
        // A cluster is a set of instructions with same-packet dependencies
        // (e.g., .tmp producer and .new consumer) that must be moved together.
        // DepSet will contain all instructions in SrcBundle that SrcMI depends
        // on.
        SmallPtrSet<MachineInstr *, 4> DepSet;
        int SrcDep =
            srcDependence(SrcBundle.getInstrIterator(), SrcBundleEnd, SrcMII,
                          AllowClusterPullup ? &DepSet : nullptr);
        SmallPtrSet<MachineInstr *, 4> SrcClusterSet;
        if (Dependency::hasOnly(SrcDep, Dependency::Independent)) {
          SrcClusterSet.insert(SrcMI);
          SrcClusterMIs.push_back(SrcClusterSet);
          continue;
        }
        // If cluster pullup is enabled and we found same-packet dependencies,
        // create a cluster containing all dependent instructions.
        if (AllowClusterPullup) {
          set_union(SrcVisitedSet, DepSet);
          if (Dependency::hasOnly(SrcDep, Dependency::SamePacketRAW) ||
              Dependency::hasOnly(SrcDep, Dependency::SamePacketWAR) ||
              Dependency::hasOnly(SrcDep, Dependency::SamePacket)) {
            LLVM_DEBUG(dbgs() << "\nFound cluster of dependent instructions in "
                                 "same packet\n");
            for (auto *DepMI : DepSet) {
              LLVM_DEBUG(dbgs() << "Cluster MI: "; DepMI->dump());
              SrcClusterSet.insert(DepMI);
            }
            SrcClusterMIs.push_back(SrcClusterSet);
          }
        }
      }

      // Sort the clusters by size in descending order
      if (AllowClusterPullup) {
        std::sort(SrcClusterMIs.begin(), SrcClusterMIs.end(),
                  [](SmallPtrSet<MachineInstr *, 4> ClusterA,
                     SmallPtrSet<MachineInstr *, 4> ClusterB) {
                    return ClusterA.size() > ClusterB.size();
                  });
      }

      // Track which instructions have been successfully pulled up to avoid
      // pulling the same instruction multiple times (clusters may overlap).
      // Pull up clusters from largest to smallest.
      // Example candidate:
      // {
      //   renamable $r7, renamable $r2 = L2_loadri_pi killed renamable $r2, 4
      // }
      // {
      //   $l4 = LX_L16b_tmp killed renamable $r3
      //   renamable $ll2 = LX_conv_h_b_lo internal killed renamable $l4
      // }
      // After cluster pullup:
      // {
      //   $l4 = LX_L16b_tmp killed renamable $r3
      //   renamable $ll2 = LX_conv_h_b_lo internal killed renamable $l4
      //   renamable $r7, renamable $r2 = L2_loadri_pi killed renamable $r2, 4
      // }
      SmallPtrSet<MachineInstr *, 4> PulledUpSet;
      SmallPtrSet<MachineInstr *, 4> InsertAtEndSet;
      for (auto ClusterSet : SrcClusterMIs) {
        bool RejectCluster = false;
        // .cur load bypass state for this cluster (HVX mode, single consumer).
        MachineInstr *GenCurLoad = nullptr;
        MachineInstr *GenCurConsumer = nullptr;
        LLVM_DEBUG(dbgs() << "\nChecking pullup legality for cluster: \n";
                   for (auto *SrcMI : ClusterSet) SrcMI->dump(););

        // Two-load .cur bypass (off by default); see tryTwoLoadCurBypass.
        if (EnableHVXTwoLoadCurBypass && ClusterSet.size() == 1) {
          MachineInstr *SrcMI = *ClusterSet.begin();
          if (tryTwoLoadCurBypass(BI, SR, DestBundle, SrcBundle, NextBI,
                                  isNextBISrc, Next, SrcMI, PulledUpSet)) {
            continue;
          }
        }

        // Skip this cluster if any instruction was already pulled up.
        for (auto *SrcMI : ClusterSet) {
          if (PulledUpSet.contains(SrcMI)) {
            LLVM_DEBUG(dbgs() << " .....NO, SrcMI has been pulled up: ";
                       SrcMI->dump(););
            RejectCluster = true;
            break;
          }
        }

        if (RejectCluster)
          continue;

        for (auto *SrcMI : ClusterSet) {
          LLVM_DEBUG(dbgs() << "\nChecking legality for SrcMI: ";
                     SrcMI->dump(););

          // Do not pull up instructions with multi-cycle latencies, because
          // the pull-up pass doesn't correctly
          // account for instructions that stall for more than one cycle.
          // Exceptions for this: branches, instructions that utilize operand
          // forwarding, and HVX vector ops when invoked in HVX mode (their
          // latency is checked below via canCauseInterlockStall).
          if ((!QII->isTC1(*SrcMI) && !(HVXMode && QII->isHVXVec(*SrcMI)) &&
               (!QII->isDotTmpInst(*SrcMI) && !QII->isDotNewInst(*SrcMI) &&
                !QII->isDotCurInst(*SrcMI)) &&
               (!SrcMI->isBranch() && !SrcMI->isTerminator() &&
                !SrcMI->isReturn()))) {
            LLVM_DEBUG(dbgs() << " .....NO");
            RejectCluster = true;
            break;
          }

          if (!isLegalToPacketizeWith(BI, SrcMI) || isSoloInstruction(*SrcMI) ||
              shouldNotBePulledUp(SrcMI)) {
            LLVM_DEBUG(dbgs() << " .....NO");
            RejectCluster = true;
            break;
          }
          MICInstIterType DepTrue = SR->instr_end(), DepAnti = SR->instr_end();
          int DstDep = dstDependence(DestBundle, SrcBundle.getInstrIterator(),
                                     SrcMI->getIterator(), DepTrue, DepAnti);

          // A single RAW dependence on a .cur-able load in the destination
          // packet can be resolved with a .cur load bypass instead of blocking
          // the pullup. Keyed on mayBeCurLoad, not HVX, so any current-value
          // load benefits. Only the single-consumer cluster is handled; when
          // eligible the RAW bit is masked below so the check is satisfiable.
          bool ClusterGenCur =
              EnableHVXLocalPullupLB && ClusterSet.size() == 1 &&
              Dependency::has(DstDep, Dependency::RAW) &&
              DepTrue != SR->instr_end() &&
              canGenCurBypass(DestBundle, const_cast<MachineInstr *>(&*DepTrue),
                              NextBI.getInstrIterator(), DstDep, SrcMI);

          // Reject the cluster on any dependence except WAR_SameBundle (the
          // cluster joins the bundle) and a RAW resolved by a .cur bypass.
          int DstDepMasked = DstDep & ~Dependency::WAR_SameBundle;
          if (ClusterGenCur)
            DstDepMasked &= ~Dependency::RAW;
          if (!(Dependency::hasOnly(DstDepMasked, Dependency::Independent))) {
            LLVM_DEBUG(dbgs() << " .....NO");
            RejectCluster = true;
            break;
          }
          if (DstDep & Dependency::WAR_SameBundle)
            InsertAtEndSet.insert(SrcMI);

          if (ClusterGenCur) {
            GenCurLoad = const_cast<MachineInstr *>(&*DepTrue);
            GenCurConsumer = SrcMI;
          }

          // Reject if pulling SrcMI into DestBundle violates latency from the
          // previous packet (e.g., a multiply result with 2-cycle latency).
          // Skip for HLX instructions which have different timing semantics.
          if (!QII->isHLXVec(*SrcMI)) {
            auto &ST = MF.getSubtarget<HexagonSubtarget>();
            const InstrItineraryData *II = ST.getInstrItineraryData();
            if (canCauseInterlockStall(DestBundle, SrcMI, II)) {
              LLVM_DEBUG(dbgs() << " .....NO (latency violation)");
              RejectCluster = true;
              break;
            }
          }

          LLVM_DEBUG(dbgs() << " .....YES");
        }

        // Move to the next cluster if this cluster cannot be pulled up
        if (RejectCluster)
          continue;

        // Check if all instructions in the cluster can fit in the destination
        // bundle given current resource constraints.
        LLVM_DEBUG(dbgs() << "Checking resources for cluster\n");
        MIInstIterType DestPos = DestBundle;
        getResourcesReadyForBundle(DestPos);
        for (auto *SrcMI : ClusterSet) {
          MIInstIterType SrcMII = SrcMI->getIterator();
          // Check if resources are available in Bundle.
          if (canReserveResources(SrcMII))
            addToPacketPseudo(SrcMI);
          else {
            RejectCluster = true;
            break;
          }
        }

        if (RejectCluster)
          continue;

        // All legality checks passed. Now physically move the cluster.
        LLVM_DEBUG(dbgs() << "[Can add to packet]";);
        for (auto *SrcMI : ClusterSet) {
          // If we have a single instruction bundle here, it means sparse
          // pullup is enabled
          if (!SrcBundle->isBundled() || SrcBundle->getBundleSize() == 1) {
            if (isNextBISrc)
              NextBI = std::next(SrcBundle);
            Next = std::next(SrcBundle);
          }

          SmallVector<MachineInstr *, 4> UnbundledMIs;
          MachineInstr *InstrToMove = removeMI(SrcMI, QII, &UnbundledMIs);
          MIBundleBuilder Bundle(&*DestBundle);

          // .cur load bypass: rewrite the load to .cur and let genLoadBypass
          // insert the consumer right after it (it performs its own insert and
          // marks the internal read), so skip the normal insertion below.
          if (GenCurLoad && SrcMI == GenCurConsumer) {
            genLoadBypass(GenCurLoad, InstrToMove);
          }
          // Insert at the end of the bundle: .new MIs, branches and
          // terminators go after their producers, and WAR_SameBundle defs go
          // after their reader so finalizeBundle does not mark the read
          // internal.
          else if (QII->isDotNewInst(*InstrToMove) || InstrToMove->isBranch() ||
                   InstrToMove->isTerminator() ||
                   InsertAtEndSet.contains(SrcMI))
            Bundle.insert(getBundleEnd(DestPos), InstrToMove);
          else {
            // Insert after any .tmp or .cur instructions at the beginning.
            // This maintains correct liveness: .tmp/.cur must be defined before
            // other instructions in the packet can use them.
            MIInstIterType InsertPos = std::next(DestPos);
            while (QII->isDotTmpInst(*InsertPos) ||
                   QII->isDotCurInst(*InsertPos))
              ++InsertPos;
            Bundle.insert(InsertPos, InstrToMove);
          }

          // Update pullup count. This might be useful in doing another
          // iteration of global pullup into these MBBs.
          ++LocalPullups;
          PulledUpSet.insert(InstrToMove);
          LLVM_DEBUG(dbgs() << "\nSpeculated Local MI:\t" << *InstrToMove);

          // Handle unbundled instructions left behind after removing SrcMI.
          // If sparse pullup is enabled, rebundle them; otherwise track them
          // for iteration control.
          if (UnbundledMIs.empty())
            continue;
          MIInstIterType UnbundledII = UnbundledMIs.front()->getIterator();
          if (AllowSparsePullup) {
            MIInstIterType LastUnbundledII = UnbundledMIs.back()->getIterator();
            finalizeBundle(*(UnbundledII->getParent()), UnbundledII,
                           std::next(LastUnbundledII));
            --UnbundledII;
          }
          if (isNextBISrc)
            NextBI = UnbundledII;
          SrcBundle = UnbundledII;
          Next = UnbundledII;
        } // end for
        LLVM_DEBUG(dbgs() << "\nMBB: "; DestBundle->getParent()->dump());
        HLV->updateLocalLiveness(SR, true);
      } // end for
    } // end for
    // If DestBundle is a single instruction bundle at this point,
    // unbundle the single instruction to retain compatibility with other APIs
    if (AllowSparsePullup && DestBundle->isBundle())
      updateBundle(&*DestBundle, QII);
  } // end for
  // If the last bundle in the scheduling region is a single instruction bundle,
  // unbundle the single instruction
  MIIterType LastBundle = std::prev(SR->end());
  if (AllowSparsePullup && LastBundle->isBundle())
    updateBundle(&*LastBundle, QII);
}

bool GreedyPullup::mergePacketsAround(
    const SmallVectorImpl<MachineInstr *> &Candidates) {
  bool Changed = false;
  for (auto CMII : Candidates) {
    MachineInstr *MI = CMII;
    MIIterType DestBundle = MI;
    MachineBasicBlock *MBB = MI->getParent();
    // Since local-pull-up pulls up into preceding bundles,
    // we only want to push down into local bundles (immediate next for now).
    ++DestBundle;

    while (DestBundle != MBB->end() && DestBundle->isDebugInstr())
      ++DestBundle;

    if (DestBundle == MBB->end() || !DestBundle->isBundle() ||
        // Ignore these instructions for simplicity.
        shouldNotBePulledUp(MI))
      continue;

    MICInstIterType DepTrue = MBB->instr_end(), DepAnti = MBB->instr_end();
    MIInstIterType DestEnd = getBundleEnd(DestBundle.getInstrIterator());
    int DstDep = dstDependence(DestBundle.getInstrIterator(), DestEnd,
                               MI->getIterator(), DepTrue, DepAnti,
                               /*HLXUseReg=*/0,
                               /*isPullDown=*/true);
    int DstDepMasked = DstDep & ~Dependency::WAR_SameBundle;
    if (!Dependency::hasOnly(DstDepMasked, Dependency::Independent))
      continue;

    // TODO: Only does a push-down in simple cases without any
    // analysis for .new/.cur.
    getResourcesReadyForBundle(DestBundle.getInstrIterator());
    if (canReserveResources(MI->getIterator()) &&
        isLegalToPacketizeWith(DestBundle, MI)) {
      LLVM_DEBUG(dbgs() << "\nCan push-down" << *MI);
      MachineInstr *MIToInsert = MI->removeFromParent();
      MIInstIterType InsertPos = DestBundle.getInstrIterator();
      // Insert as the first instruction in bundle (push-down).
      MBB->insert(++InsertPos, MIToInsert);
      HLV->updateLocalLiveness(MBB, false);
      Changed = true;
    }
  }
  return Changed;
}

// Merge adjacent packets in a BB when possible.
// Currently pushing down the individual MIs into next packet.
// TODO: push down into later packets as well.
bool GreedyPullup::mergePackets() {
  bool Changed = false;
  for (auto MBB : LocalMergePackCand) {
    if (!MBB)
      continue;
    LLVM_DEBUG(dbgs() << "\nMerging packets within MBB#" << MBB->getNumber(););
    SmallVector<MachineInstr *, 2> Candidates;
    for (MIIterType MII = MBB->begin(); MII != MBB->end(); ++MII) {
      if (!MII->isBundle())
        Candidates.push_back(&*MII);
    }
    Changed |= mergePacketsAround(Candidates);
  }
  return Changed;
}

// TODO: Form bigger scheduling regions more than 2 MBBs.
void GreedyPullup::globalPullup(SchedulingRegion_t &SR) {
  LLVM_DEBUG(dbgs() << "\n[GlobalPullup]";);
  // Start from the bottom of the scheduling region (Bottom-up scheduling)
  SchedulingRegion_t::reverse_iterator BlockI = SR.rbegin();
  assert(!SR.empty() && "Empty scheduling region!");
  DenseSet<const MachineBasicBlock *> IgnoreBlocks;
  for (++BlockI; BlockI != SR.rend(); ++BlockI) {
    if (IgnoreBlocks.find(*BlockI) != IgnoreBlocks.end())
      continue;
    SchedulingRegion_t::reverse_iterator SrcBlock = BlockI - 1;
    LLVM_DEBUG(dbgs() << "\n\t\t*****Speculating MIs from MBB#"
                      << (*SrcBlock)->getNumber() << " to MBB#"
                      << (*BlockI)->getNumber() << "*****");
    // Do not speculate into a SrcBlock from multiple Blocks.
    // Except: When PreciousMICanBeSpeculated.
    PulledUpII::iterator hasSpeculatedIns = SpeculatedIns.find(*BlockI);
    if (hasSpeculatedIns != SpeculatedIns.end()) {
      LLVM_DEBUG(
          dbgs() << "\nSrcMBB Already contains speculated ins. Skipping!";);
      continue;
    }
    for (MIInstIterType MII = (*SrcBlock)->instr_begin();
         MII != (*SrcBlock)->instr_end();) {
      MachineInstr *MIToMove = &*MII;
      // The MI might be moved so skip.
      ++MII;
      if (MIToMove->isDebugInstr())
        continue;
      LLVM_DEBUG(dbgs() << "\nChecking pullup for MI: "; MIToMove->dump());
      MIIterType TargetPacket = (*BlockI)->end();
      if (canMIBeSpeculated(MIToMove, *SrcBlock, *BlockI, TargetPacket)) {
        // TODO: If MI can be speculated that does not mean
        // we 'should' speculate it. Store in ScheduledMI and
        // then 'sort/prune' the list before speculating.
        LLVM_DEBUG(dbgs() << " .....YES (MICanBeSpeculated)";);
        bool done = speculateMI(MIToMove, *SrcBlock, *BlockI, TargetPacket);
        SpeculatedIns[*BlockI].push_back(MIToMove);
        ++GlobalSpeculatedIns;
        // Populate candidates for local push-down.
        LocalMergePackCand.insert(*SrcBlock);
        (void)done;
        assert(done && "Wrong Analysis");
      } else
        LLVM_DEBUG(dbgs() << " .....NO (MICanBeSpeculated)";);
    }
  }
  BlockI = SR.rbegin();
  for (++BlockI; BlockI != SR.rend(); ++BlockI) {
    if (IgnoreBlocks.find(*BlockI) != IgnoreBlocks.end())
      continue;
    SchedulingRegion_t::reverse_iterator SrcBlock = BlockI - 1;
    LLVM_DEBUG(dbgs() << "\n\t\t*****Predicating MIs from MBB#"
                      << (*SrcBlock)->getNumber() << " to MBB#"
                      << (*BlockI)->getNumber() << "*****");
    // Unlike speculation predication can be done from both successors.
    for (MIInstIterType MII = (*SrcBlock)->instr_begin();
         MII != (*SrcBlock)->instr_end();) {
      MachineInstr *MIToMove = &*MII;
      // The MI might be moved so skip.
      ++MII;
      if (MIToMove->isDebugInstr())
        continue;
      LLVM_DEBUG(dbgs() << "\nChecking pullup for MI: "; MIToMove->dump());
      MIIterType TargetPacket = (*BlockI)->end();
      MICInstIterType WithinPacket = (*BlockI)->instr_end();
      bool NeedToNewify = false, PreciousMIToInsert = false;
      if (canMIBePredicated(MIToMove, *SrcBlock, *BlockI, TargetPacket,
                            WithinPacket, NeedToNewify, PreciousMIToInsert)) {
        // TODO: If MI can be predicated that does not mean
        // we 'should' predicate it. Store in ScheduledMI and
        // then 'sort/prune' the list before predicating.
        LLVM_DEBUG(dbgs() << " .....YES (MICanBePredicated)";);
        bool done = predicateMI(MIToMove, *SrcBlock, *BlockI, TargetPacket,
                                WithinPacket, NeedToNewify, PreciousMIToInsert);
        (void)done;
        assert(done && "Wrong Analysis");
        ++GlobalPredicatedIns;
        // Populate candidates for local push-down.
        LocalMergePackCand.insert(*SrcBlock);
        if (PreciousMIToInsert) {
          // Once the destination bundle has 5 MIs we want to avoid subsequent
          // scheduling into this MBB, so that we skip different sanity checks
          // added to ensure consistency in resources available in bundle which
          // uses packetizer to track resources and the packetizer does not
          // validate more than four instructions in a bundle.
          IgnoreBlocks.insert(*BlockI);
          ToBeSqueezed.insert(&*TargetPacket);
          // Prevent any subsequent e.g. optimizations as well.
          IgnorePostSchedOpt.push_back(*BlockI);
          LLVM_DEBUG(dbgs() << "\nIgnoring MBB#" << (*BlockI)->getNumber()
                            << " from further speculation";);
          break;
        }
      } else
        LLVM_DEBUG(dbgs() << " .....NO (MICanBePredicated)";);
    }
  }
  // TODO: Look for packetization opportunities or
  // or leave this task for MergePackets.
  // TODO: Perform branch/post-pullup optimizations.
}

// Analyze a bundle/independent-instruction at a time.
// Perform a bottom up traversal on the def-use set of DstBB MIs.
// Resource available in BUNDLE && Dependency constraints are met => done!
// Returns the position *before* which \p SrcMI could be inserted.
// In case of BUNDLE the \p SrcMI can be inserted within that BUNDLE.
// TODO: Insert into a BUNDLE even if there are WAR dependencies
// See also findInsertPtPred().
// \p SpecPreciousMIComp = true, when speculating PreciousMI comp. code.
// \p SpecPreciousMI = true, when speculating PreciousMI.
// \p GlobalPrecious = true, when hoisting SrcMI from the innermost single block
//    loop.  Stop searching after the first available bundle with resources.
MIIterType GreedyPullup::findInsertPtSpec(MachineInstr *SrcMI,
                                          MachineBasicBlock *DstBB,
                                          bool SpecPreciousMI,
                                          bool SpecPreciousMIComp,
                                          bool GlobalPrecious) {
  LLVM_DEBUG(dbgs() << "\n[FindInsertPtSpec]";);
  MIIterType DstEnd = DstBB->end();
  MIInstIterType InsPos = DstBB->instr_end();
  MICInstIterType DepTrue = DstBB->instr_end(), DepAnti = DstBB->instr_end();
  if (shouldNotBePulledUp(SrcMI) ||
      (QII->isPredicated(*SrcMI) && QII->isPredicatedNew(*SrcMI)) ||
      isSoloInstruction(*SrcMI))
    return DstEnd;
  LLVM_DEBUG(dbgs() << "\nChecking Resource conflict\n";);

  // Perform a bottom up traversal on the def-use set of DstBB MIs.
  MIRIterType MII = getSpeculationBeginPoint(SrcMI, DstBB);
  for (; MII != DstBB->rend(); ++MII) {
    MachineInstr *MI = &*MII;
    if (MI->isDebugInstr())
      continue;
    LLVM_DEBUG(dbgs() << "\n\tAgainst:\t* " << *MI;);
    // Check for dependencies [RangeBeg, MII) <- in reverse order.
    if (MI->isBundle()) {
      int DstDep = dstDependence(MI->getIterator(), InsPos,
                                 SrcMI->getIterator(), DepTrue, DepAnti);
      if (!Dependency::hasOnly(DstDep, Dependency::Independent))
        break;
      // A precious MI takes this bundle as the insertion point before the
      // resource check; speculateMI() inserts it just before the bundle if it
      // does not fit. Other MIs only move into bundles with free resources.
      if (SpecPreciousMI)
        InsPos = MI->getIterator();
      // Continue searching higher up.
      LLVM_DEBUG(
          dbgs() << "\n\tNo conflict (BUNDLE)! Checking resources.....\n";);
      // Check if resources are available in BUNDLE or not.
      getResourcesReadyForBundle(MI->getIterator(), SpecPreciousMI);
      if (canReserveResources(SrcMI->getIterator())) {
        if (!isLegalToPacketizeWith(MI, SrcMI) || NoSpecInBundle)
          break;
        InsPos = MI->getIterator();
        LLVM_DEBUG(dbgs() << ".....(Resources available. Checking higher up)";);
        if (GlobalPrecious && InsPos->isBundle())
          break;
        // Check for dst dep from insertion point until the end of DstBB.
      } else {
        LLVM_DEBUG(
            dbgs() << ".....(No Resources (BUNDLE) Checking further up)";);
        // Keep searching higher up; SrcMI is not placed in this bundle.
      }
    } else {
      // Check for dst dep from insertion point until the end of DstBB.
      int DstDep = dstDependence(MI->getIterator(), InsPos,
                                 SrcMI->getIterator(), DepTrue, DepAnti);
      if (!Dependency::hasOnly(DstDep, Dependency::Independent))
        break;
      // TODO: Move this somewhere else and abstract it together with
      // the code in isLegalToPacketize.
      if (QII->isLoopN(*SrcMI)) {
        // loopN cannot be pulled too far.
        unsigned dist = HLV->getDistanceBetween(MI->getParent(),
                                                SrcMI->getOperand(0).getMBB());
        if (!QII->isJumpWithinBranchRange(*SrcMI, dist + SafetyBuffer)) {
          LLVM_DEBUG(dbgs() << "\nLoopN cannot be pulled too far into MBB";);
          break;
        }
      }
      // Continue searching higher up.
      InsPos = MI->getIterator();
      LLVM_DEBUG(dbgs() << "No conflict! Checking further up .....\n";);
    }
  }

  if (InsPos == DstBB->instr_end())
    return DstEnd;

  assert(!InsPos->isInsideBundle());

  if (inBetweenBarriers(&*InsPos, DstBB))
    return DstEnd;

  // Don't speculate outside a bundle when there are multiple execution
  // paths from the DstBB. When SpecPreciousMI = true, then in that case if
  // InsPos is a bundle then SpeculateMI will insert SrcMI 'just' before
  // the BUNDLE when the resources are not available.
  // For details see speculateMI().
  // TODO: What if resources were available in BUNDLE then higher up
  // InsPos was not in a BUNDLE, that time speculation will not happen.
  if (!InsPos->isBundle() && !SpecPreciousMIComp && (DstBB->succ_size() > 1))
    return DstEnd;

  return InsPos;
}

// @note: The SrcMI can be modified by this function.
MIIterType GreedyPullup::findInsertPtPred(MachineInstr *SrcMI,
                                          MachineBasicBlock *DstBB,
                                          bool &NeedToNewify,
                                          MICInstIterType &WithinPacket,
                                          bool PredPreciousMI) {
  LLVM_DEBUG(dbgs() << "\n[FindInsertPtPred]";);
  MIIterType DstEnd = DstBB->end();
  MIInstIterType InsPos = DstBB->instr_end();
  MICInstIterType DepTrue = DstBB->instr_end(), DepAnti = DstBB->instr_end();
  if (shouldNotBePulledUp(SrcMI) || !QII->isPredicated(*SrcMI) ||
      QII->isPredicatedNew(*SrcMI) || isSoloInstruction(*SrcMI))
    return DstEnd;
  NeedToNewify = false;
  WithinPacket = DstBB->instr_end();
  LLVM_DEBUG(dbgs() << "\n[FindInsertionPointPredicatedIns]" << *SrcMI;);
  LLVM_DEBUG(dbgs() << "\nChecking Resource conflict\n";);
  // Perform a bottom up traversal on the def-use set of DstBB MIs.
  for (MIRIterType MII = DstBB->rbegin(); MII != DstBB->rend(); ++MII) {
    MachineInstr *MI = &*MII;
    if (MI->isDebugInstr())
      continue;
    LLVM_DEBUG(dbgs() << "Against:\t* " << *MI;);
    // Check for dependencies [RangeBeg, MII) <- in reverse order.
    if (MI->isBundle()) {
      // Check if resources are available in BUNDLE or not.
      getResourcesReadyForBundle(MI->getIterator(), PredPreciousMI);
      if (canReserveResources(SrcMI->getIterator())) {
        if (!isLegalToPacketizeWith(MI, SrcMI) || NoPredInBundle)
          break;
        // Check for dst dep from insertion point until the end of DstBB.
        int DstDep = dstDependence(MI->getIterator(), InsPos,
                                   SrcMI->getIterator(), DepTrue, DepAnti);
        // check for .new and look no further.
        if (Dependency::hasOnly(DstDep, Dependency::RAW)) {
          // The dstMI should define a predicate which SrcMI should use.
          // And SrcMI should be newifiable.
          // Set NeedToNewify = true;
          // For RAW in the packet this is perfectly fine and
          // no need for further analysis.
          LLVM_DEBUG(dbgs()
                         << "Read after write dependency with: " << *DepTrue;);
          unsigned DefP = 0;
          if ((InsPos != DepTrue) && hasOneDef(&*DepTrue, DefP)) {
            if (canNewifyPredReg(DefP, DepTrue, DstBB->instr_end()) &&
                SrcMI->readsRegister(DefP, /*TRI=*/nullptr) &&
                isNewifiable(*SrcMI, &Hexagon::PredRegsRegClass)) {
              InsPos = MI->getIterator();
              WithinPacket = DepTrue;
              NeedToNewify = true;
              LLVM_DEBUG(dbgs() << "\tokay (BUNDLE)!";);
            }
          } else {
            LLVM_DEBUG(dbgs() << "\tNot okay (BUNDLE)!";);
          }
          break;
        }
        // WAR is fine in the same packet, but look no further.
        int DstDepMasked = DstDep & ~Dependency::WAR_SameBundle;
        if (Dependency::hasOnly(DstDepMasked, Dependency::RAW)) {
          unsigned DepReg = 0;
          LLVM_DEBUG(dbgs() << "WAR_SameBundle dependency with: " << *DepAnti;);
          if (singleTrueDep(MI->getIterator(), InsPos, SrcMI->getIterator(),
                            DepReg)) {
            MICIterType MIBundle = &*MII;
            MICInstIterType Sentinel =
                MIBundle == DstBB->end()
                    ? DstBB->instr_end()
                    : std::next(MIBundle).getInstrIterator();
            if (canNewifyPredReg(DepReg, DepTrue, Sentinel)) {
              // WAR for one and RAW for the predicate ins.
              InsPos = MI->getIterator();
              WithinPacket = laterOfTheTwo(DepTrue, DepAnti, Sentinel);
              assert(WithinPacket != Sentinel);
              NeedToNewify = true;
              LLVM_DEBUG(dbgs() << "Single RAW dependency with: " << *DepTrue;);
            }
          }
          // TODO: Enable this case of only WAR. WithinPacket = DepAnti;
          LLVM_DEBUG(dbgs() << "Write after read, okay (BUNDLE)!";);
        }
        // can look further up.
        if (Dependency::hasOnly(DstDep, Dependency::Independent)) {
          InsPos = MI->getIterator();
          LLVM_DEBUG(dbgs() << "No conflict (BUNDLE)! Checking further up!\n";);
          continue;
        }
        break;
      } else {
        LLVM_DEBUG(dbgs() << "No Resources";);
        // We don't want to move further up because we haven't analyzed the
        // dependencies for this packet. We might, but that will make things
        // more complicated.
        break;
      }
    } else {
      // Check for dst dep from insertion point until the end of DstBB.
      int DstDep = dstDependence(MI->getIterator(), InsPos,
                                 SrcMI->getIterator(), DepTrue, DepAnti);
      if (!Dependency::hasOnly(DstDep, Dependency::Independent))
        break;
      // Continue searching higher up.
      InsPos = MI->getIterator();
      LLVM_DEBUG(dbgs() << "No conflict! Checking further up .....\n";);
    }
  }
  if (InsPos == DstBB->instr_end())
    return DstEnd;

  assert(!InsPos->isInsideBundle());
  if (inBetweenBarriers(&*InsPos, DstBB))
    return DstEnd;
  return InsPos;
}

// Is the instruction a candidate for the "last chance global pullup"?
// That is, these are the instructions from single block innermost loops that
// could be hoisted to the loop entry AND copied to a downstream packet to
// free up a resource.
// Here, y is a candidate for the "last chance global pullup".
// {
//   x = ...
//   y = y1, y2 <--- where y1, y2 are loop invariant
// }
// z = ...
// {
//    ...
//    ... = y<killed>
// }
// br Loop
//
// Only V6_vcombine is handled.
// See GreedyPullup::lastChanceGlobalPullup for more details.
MachineInstr *
GreedyPullup::isLastChanceGlobalPullupCand(const MachineInstr *MI) {
  if (MI->getOpcode() != Hexagon::V6_vcombine)
    return nullptr;
  auto *MBB = MI->getParent();

  unsigned Op0 = MI->getOperand(0).getReg();
  unsigned Op1 = MI->getOperand(1).getReg();
  unsigned Op2 = MI->getOperand(2).getReg();
  MICInstIterType DefUseR = MBB->instr_end();

  if (MBB->isLiveIn(Op1) && MBB->isLiveIn(Op2) && // y1, y2 live in
      !HLV->isDefinedWithin(MBB->instr_begin(), MBB->instr_end(), Op1,
                            DefUseR) && // and y1, y2 are loop invariant
      !HLV->isDefinedWithin(MBB->instr_begin(), MBB->instr_end(), Op2,
                            DefUseR)) {
    if (HLV->isDefinedWithin(std::next(MI->getIterator()), MBB->instr_end(),
                             Op0, DefUseR)) {
      // Cannot be live out.
      for (MCRegAliasIterator AI(Op0, TRI, true); AI.isValid(); ++AI)
        if (HLV->isLiveOut(MBB, *AI))
          return nullptr;

      // Find the point where it's killed.
      MICInstIterType Use = MBB->instr_end();
      if (!HLV->isUsedWithin(std::next(DefUseR), MBB->instr_end(), Op0, Use))
        return nullptr;

      auto regmaskMOs = collectRegmaskMOs(*MBB);
      for (auto *MO : regmaskMOs)
        if (MO->clobbersPhysReg(Op0) || MO->clobbersPhysReg(Op1) ||
            MO->clobbersPhysReg(Op2))
          return nullptr;

      MachineInstr *InsertBundle = (MachineInstr *)(const_cast<MachineInstr *>(
          &*getBundleStart(Use->getIterator())));

      if (!InsertBundle->isBundle())
        return nullptr;

      return InsertBundle;
    }
  }
  return nullptr;
}

// Attempt to hoist/copy an instruction from BundleHead to make room for
// InstrToMove, if beneficial.
// Here, y is a candidate for the "last chance global pullup"
// Loop:
// {
//   x = ...
//   y = y1, y2 <--- where y1, y2 are loop invariant
// }
// z = ... <--- z could be moved to the packet above if not for y
// {
//    ...
//    ... = y<killed>
// }
// br Loop
//
// If the last chance global pullup is successful, we will end up with:
// {
//   y = y1, y2 <--- global pull up to make it live into the loop
// }
// Loop:
// {
//   x = ...
//   z = ...  <---- z will get moved to the packet
// }
// {
//    y = y1,y2 <--- make it live into the loop
//    ... = y<killed> <-- uses old value (3.3.1 packet execution semantics)
// }
// br Loop
//
bool GreedyPullup::lastChanceGlobalPullup(MIInstIterType &BundleHead,
                                          MIInstIterType InstrToMove,
                                          const MachineInstr *LoadBypassCand) {
  if (!EnableHVXLCGP)
    return false;
  if (EnableLCGPForLoadBypassOnly && LoadBypassCand == nullptr)
    return false;
  MachineLoop *Loop = MLI.getLoopFor(BundleHead->getParent());
  if (!isSuperblockLoop(Loop))
    return false;
  if (InstrToMove->getOpcode() == Hexagon::V6_vcombine)
    return false;
  auto MII = BundleHead->getIterator();
  MachineInstr *SingleInstr = nullptr;
  std::vector<std::pair<MachineInstr *, MachineInstr *>> candidatesForHoist;
  auto MBB = MII->getParent();

  for (++MII; MII->isInsideBundle(); ++MII)
    if (auto *InsertBundle = isLastChanceGlobalPullupCand(&*MII))
      candidatesForHoist.emplace_back(&*MII, InsertBundle);

  // Find a place to insert
  for (auto Cand : candidatesForHoist) {
    // Check if it's worth kicking it out
    getResourcesReadyForBundle(BundleHead->getIterator(), false,
                               Cand.first); // kick out
    if (!canReserveResources(InstrToMove))
      continue;

    // We have our candidate; Check if we can place it in the outer scope.
    auto *Pred = Loop->getLoopPredecessor();

    if (Pred) {
      auto TargetPacket = findInsertPtSpec(Cand.first, Pred, false, true);
      if (TargetPacket != Pred->end()) {
        // We have the packet to which to copy.
        for (MIIterType BI = Cand.second; BI != MBB->end(); ++BI) {
          if (LoadBypassCand && LoadBypassCand != &*Cand.first)
            continue;
          if (!BI->isBundle())
            continue;
          getResourcesReadyForBundle(BI->getIterator());
          if (canReserveResources(Cand.first->getIterator())) {
            if (isLegalToPacketizeWith(BI, Cand.first)) {
              // Now find a spot to copy to make it live in
              // Cand.second is the place where it's killed
              // Copy the instruction to make it live out of the loop;
              MachineInstr *ClonedInstr = MF.CloneMachineInstr(Cand.first);
              MIBundleBuilder Bundle(&*BI);
              MICInstIterType Use = MBB->instr_end();
              if (HLV->isUsedWithin(std::next(BI.getInstrIterator()),
                                    getBundleEnd(BI.getInstrIterator()),
                                    Cand.first->getOperand(0).getReg(), Use)) {
                ++Use;
                MachineInstr *UseMI = const_cast<MachineInstr *>(&*Use);
                Bundle.insert(UseMI->getIterator(), ClonedInstr);
                BI->addRegisterDefined(Cand.first->getOperand(0).getReg());
              } else {
                MIInstIterType DestPos = BI->getIterator();
                Bundle.insert(std::next(DestPos), ClonedInstr);
              }

              assert(!MBB->isLiveIn(Cand.first->getOperand(0).getReg()) &&
                     "Already live-in");
              MBB->addLiveIn(
                  Cand.first->getOperand(0).getReg()); // Make sure it's live in

              auto *Instr =
                  removeMI(Cand.first, QII, nullptr,
                           &SingleInstr); // may invalidate the bundle, DestPos
                                          // invalidated

              // Not a bundle any longer; turn into a single instruction bundle,
              // as the local pullup pass expects a bundle into which to move
              // the instruction.
              if (SingleInstr) {
                BundleHead = SingleInstr->getIterator();
                finalizeBundle(*BundleHead->getParent(), BundleHead,
                               std::next(BundleHead));
                BundleHead = std::prev(BundleHead);
              }

              // Now hoist to the predecessor (i.e. global move)
              MIInstIterType DestPos = TargetPacket.getInstrIterator();
              // Already a bundle
              if (DestPos->isBundle()) {
                MIBundleBuilder HoistBundle(&*TargetPacket);
                HoistBundle.insert(std::next(DestPos), Instr);
              } else {
                //  Check if it's safe to form a bundle
                finalizeBundle(*Pred, DestPos, std::next(DestPos));
                auto CheckBundle = std::prev(DestPos);
                getResourcesReadyForBundle(CheckBundle);
                if (canReserveResources(Instr->getIterator()) &&
                    isLegalToPacketizeWith(CheckBundle, Instr)) {
                  MIBundleBuilder HoistBundle(&*CheckBundle);
                  HoistBundle.insert(std::next(CheckBundle), Instr);
                } else {
                  DestPos->unbundleFromPred();
                  CheckBundle->eraseFromParent();
                  Pred->insertAfter(&*DestPos, Instr);
                }
              }
              HLV->updateLocalLiveness(Pred, true);
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

// Currently we are only interested in speculating to
// immediate (dominating) predecessor.
bool GreedyPullup::canMIBeSpeculated(MachineInstr *MI, MachineBasicBlock *SrcBB,
                                     MachineBasicBlock *DstBB,
                                     MIIterType &TargetPacket) {
  LLVM_DEBUG(dbgs() << "\n[canMIBeSpeculated]");
  TargetPacket = DstBB->end();

  MachineLoop *L = MLI.getLoopFor(SrcBB);
  bool isSuperBlockL = isSuperblockLoop(L);
  if (isSuperBlockL) {
    if (!isLoopInvariant(*MI))
      return false;
  }

  // No dependency in the current SrcBB
  if (shouldNotBePulledUp(MI) || findInsertionPtSpecLocal(MI) != SrcBB->begin())
    return false;
  LLVM_DEBUG(dbgs() << " .....YES";);
  ABI &abi = AnalyzeBranchInfo.getOrCreate(DstBB, QII);
  // TODO: Handle SchedulingRegions with more than 2 blocks.
  if (!abi.AnalysisValid)
    return false;
  assert(abi.TBB && abi.FBB);
  if (!DstBB->isSuccessor(SrcBB))
    return false;
  bool SrcIsTBB = (abi.TBB == SrcBB);
  MachineBasicBlock *OtherBB = abi.FBB;
  if (!SrcIsTBB) {
    assert(abi.FBB == SrcBB);
    OtherBB = abi.TBB;
  }

  assert(isSuperBlockL || SrcBB->pred_size() == 1);

  if (!safeToSpeculate(MI, DstBB))
    return false;
  LLVM_DEBUG(dbgs() << " .....YES (SafeToSpeculate)";);
  // The def(s) of MI should not be live-in in the 'Other' branch.
  // TODO: If it is live-in in the 'Other' branch, even then we can speculate
  // provided we assign the live-in value to a new register
  // and replace the usage with the new register in the 'Other' branch.
  bool DefLiveInOtherBB = HLV->isDefLiveIn(MI, OtherBB);
  MIIterType TargetPacketIntMI = DstBB->end();
  bool PreciousMIShouldBeSpeculated = false;
  if (DefLiveInOtherBB) {
    LLVM_DEBUG(dbgs() << " ..... (DefLiveInOtherBB)";);
    if (!preciousToSpeculate(MI, SrcBB, DstBB, OtherBB))
      return false;
    LLVM_DEBUG(dbgs() << " .....YES (PreciousToSpeculate)";);
    if (!preciousMICanBeSpeculated(MI, SrcBB, DstBB, OtherBB, TargetPacketIntMI,
                                   1))
      return false;
    PreciousMIShouldBeSpeculated = true;
    LLVM_DEBUG(dbgs() << " .....YES (PreciousMICanBeSpeculated)";);
  } else
    LLVM_DEBUG(dbgs() << " ..... (DefNotLiveInOtherBB)";);

  // If MI is loop invariant in the innermost single-block loop, try hard
  // to hoist during global pull-up.  This is indicated with GlobalPrecious;
  bool GlobalPrecious = false;
  if ((EnableSingleLoopSuperblock && useHVXOps && isSuperBlockL &&
       !DisableHVXLocalPullupAndSuperblock) &&
      isLoopInvariant(*MI))
    GlobalPrecious = true;

  TargetPacket = findInsertPtSpec(MI, DstBB, PreciousMIShouldBeSpeculated,
                                  false, GlobalPrecious);

  // Verify what we did was correct.
  if (DefLiveInOtherBB) {
    MICInstIterType TestIt = DstBB->instr_begin();
    // MI cannot be at or before the position of the inserted A2_tfr.
    while (TestIt != TargetPacketIntMI.getInstrIterator()) {
      assert(TestIt != TargetPacket.getInstrIterator());
      ++TestIt;
    }
  }

  if (TargetPacket != DstBB->end()) {
    LLVM_DEBUG(dbgs() << "<<Resource available for speculation>>";);
    LLVM_DEBUG(if (TargetPacket->isBundle()) dbgs()
                   << "\nInsert MI inside: " << *TargetPacket;
               else dbgs() << "\nInsert MI just before: " << *TargetPacket;);
    return true;
  }
  assert(!PreciousMIShouldBeSpeculated);
  return false;
}

bool GreedyPullup::safeToSpeculate(MachineInstr *MI,
                                   MachineBasicBlock *DstBB) const {
  const MachineFunction &MF = *MI->getParent()->getParent();
  LLVM_DEBUG(dbgs() << "\n[SafeToSpeculate]";);
  if (QII->isPredicated(*MI) || QII->isMemOp(*MI) ||
      MI->hasUnmodeledSideEffects()) // To simplify.
    return false;
  if (MI->mayStore())
    return false;
  // Do not speculate instructions that modify reserved global registers.
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const MachineOperand &Op : MI->operands()) {
    if (!Op.isReg() || !Op.isDef())
      continue;
    unsigned R = Op.getReg();
    if (MRI.isReserved(R) && TRI->isGlobalReg(R))
      return false;
  }
  if (!MI->mayLoad())
    return true;
  // For simplification we want to analyze safety of simplest load.
  // TODO: Load from absolute address can be added as well.
  unsigned DefR = 0, UseR = 0, BaseR = 0, OffR = 0;
  if (!MI->hasOneMemOperand() || (*MI->memoperands_begin())->isVolatile() ||
      !(*MI->memoperands_begin())->getValue() ||
      !hasOneDefAndUse(MI, DefR, UseR))
    return false;
  // This check is for safety if getUnderlyingObject is messed up.
  MICInstIterType DefUseR = DstBB->instr_end();
  if (!MI->getParent()->isLiveIn(UseR) ||
      HLV->isDefinedWithin(MI->getParent()->instr_begin(), MI->getIterator(),
                           UseR, DefUseR))
    return false;
  if (!QII->getBaseAndOffsetPosition(*MI, BaseR, OffR))
    return false;
  const Value *MIVal =
      getUnderlyingObject((*MI->memoperands_begin())->getValue());
  if (!MIVal)
    return false;
  bool LoadFromSameObject = false;
  assert(DefR && UseR);
  LLVM_DEBUG(dbgs() << "\nChecking if any loads as same as: " << *MI
                    << "\tValue: ";
             MIVal->dump(););
  for (MICInstIterType MIIt = DstBB->instr_begin(); MIIt != DstBB->instr_end();
       ++MIIt) {
    const MachineInstr *MII = &*MIIt;
    if (MII->isDebugInstr())
      continue;
    unsigned DefP = 0, UseP = 0, BaseP = 0, OffP = 0;
    if (MII->mayLoad() &&
        !(QII->isPredicated(*MII) || QII->isMemOp(*MII) ||
          shouldNotBeCrossed(MII, MI)) &&
        MII->hasOneMemOperand() && hasOneDefAndUse(MII, DefP, UseP)) {
      LLVM_DEBUG(dbgs() << "\nAgainst: " << *MII;);
      MachineMemOperand *mmo = *MII->memoperands_begin();
      // There is already a load from same address => safe.
      if (UseP == UseR) {
        MICInstIterType DefUseP = DstBB->instr_end();
        if (HLV->isDefinedWithin(MII->getIterator(), DstBB->instr_end(), UseP,
                                 DefUseP))
          return false;
        // We have to check the immediate part also!
        if (mmo->isVolatile() || !mmo->getValue() ||
            !QII->getBaseAndOffsetPosition(*MII, BaseP, OffP))
          continue;
        // Cannot safely speculate a load with a stricter alignment
        // requirement.
        if ((*MI->memoperands_begin())->getAlign() > mmo->getAlign())
          continue;
        assert(MII->getOperand(BaseP).getReg() ==
               MI->getOperand(BaseR).getReg());
        const Value *MIIVal = getUnderlyingObject(mmo->getValue());
        if (MIVal == MIIVal) {
          LoadFromSameObject = true;
          LLVM_DEBUG(dbgs() << "\n\tLoad from same object:"
                            << "\n\tValue: ";
                     MIIVal->dump(););
        }
      }
    }
    if (MII->modifiesRegister(UseR, TRI))
      LoadFromSameObject = false;
  }
  return LoadFromSameObject;
}

// Approach1: Check if there is a spare register available such that
// the def(MI) can be assigned to that spare register to preserve
// liveness in the OtherBB and change the usage of def(MI) in otherBB
// to the spare reg. The spare register should not be livein in otherBB.
// TODO: Implement Approach2 as well.
// Approach2: Check if there is just one MI in the OtherBB which uses
// def(MI) (account for aliasing etc.) and if that can be speculated first.
// In this approach we won't have to insert compensation code.
// Approach3: Split critical edge and insert a copy into the critical BB.
// This is a costly operation so it should be done very selectively.
bool GreedyPullup::preciousMICanBeSpeculated(MachineInstr *MI,
                                             MachineBasicBlock *SrcBB,
                                             MachineBasicBlock *DstBB,
                                             MachineBasicBlock *OtherBB,
                                             MIIterType &TargetPacket,
                                             unsigned ApproachNum) {
  LLVM_DEBUG(dbgs() << "\n[PreciousMICanBeSpeculated]";);
  assert(preciousToSpeculate(MI, SrcBB, DstBB, OtherBB));
  if (ApproachNum != 1)
    return false;

  unsigned DefR, UseR;
  if (!hasOneDefAndUse(MI, DefR, UseR))
    return false;
  // TODO: We can remove this, then we have to replace all the usage
  // of DefR recursively for OtherBB and all its successors. That is not
  // worth it for speculating a single instruction.
  bool splitEdge = false;
  if (HLV->isLiveOut(OtherBB, DefR))
    splitEdge = true;
  MachineBasicBlock *LP = nullptr;
  if (OtherBB->pred_size() > 1) {
    LP = ShouldSplitCriticalEdge(DstBB, OtherBB);
    if (!LP)
      return false;
    LLVM_DEBUG(dbgs() << "\nCritical edge will be split during speculation.";);
  }
  if (splitEdge ^ (LP != nullptr)) // Both true or both false (for now).
    return false;

  // TODO: Reconsider the design of this and the MICanBeSpeculated
  // because FindInsertionPoint is called twice for MI.
  // We need to find the insertion point of MI because we should know
  // that MI can be inserted at some position. This function inserts
  // an extra instruction so we better be sure in the first place.
  TargetPacket = findInsertPtSpec(MI, DstBB, true, false);
  if (TargetPacket == DstBB->end()) {
    LLVM_DEBUG(dbgs() << "\nPrecious MI cannot be speculated: " << *MI);
    return false;
  } else {
    LLVM_DEBUG(dbgs() << "<<Resource available for speculating PreciousMI>>";);
    LLVM_DEBUG(if (TargetPacket->isBundle()) dbgs()
                   << "\nInsert MI inside: " << *TargetPacket;
               else dbgs() << "\nInsert MI just before: " << *TargetPacket;);
  }

  std::unique_ptr<RegScavenger> RS(new RegScavenger());
  RS->enterBasicBlock(*DstBB);
  // Mark all the live-ins and their aliases of MBBs involved as used.
  LLVM_DEBUG(dbgs() << "\nMarking Liveins:";);
  markLiveInsAsUsed(RS, DstBB);
  markLiveInsAsUsed(RS, SrcBB);
  markLiveInsAsUsed(RS, OtherBB);
  markLiveOutsAsUsed(RS, DstBB);
  markLiveOutsAsUsed(RS, SrcBB);
  // Instead of advancing the scavenger with RS->forward(), conservatively mark
  // every register used in DstBB as used.
  // TODO: Only the instructions up to TargetPacket need to be considered.
  markAllRegsAsUsed(RS, DstBB);
  if (!Hexagon::IntRegsRegClass.contains(DefR))
    return false;
  // Use IntRegs: getMinimalPhysRegClass often returns IntRegsLow8 (R0-R7),
  // which would needlessly restrict the scavenged register.
  const TargetRegisterClass *RC = &Hexagon::IntRegsRegClass;
  unsigned ScratchReg = RS->FindUnusedReg(RC);
  if (ScratchReg == 0 || SrcBB->isLiveIn(ScratchReg))
    return false;
  LLVM_DEBUG(dbgs() << "\nRegister " << printReg(ScratchReg, TRI)
                    << " available for speculating PreciousMI\n\t" << *MI;);

  // Give up if ScratchReg is defined in OtherBB. This keeps the later register
  // replacement simple; scavenging another register is not worth it for a
  // single instruction.
  for (MIInstIterType OMBBI = OtherBB->instr_begin();
       OMBBI != OtherBB->instr_end(); ++OMBBI) {
    if (OMBBI->modifiesRegister(ScratchReg, TRI))
      return false;
    // We cannot replace implicit register uses.
    if (OMBBI->hasRegisterImplicitUseOperand(DefR))
      return false;
  }

  // Compensation code is created in OtherBB and then moved to DstBB.
  DebugLoc dl;
  MachineInstr *NewMI = BuildMI(*OtherBB, OtherBB->instr_begin(), dl,
                                QII->get(Hexagon::A2_tfr), ScratchReg)
                            .addReg(DefR);
  LLVM_DEBUG(dbgs() << "\nCompensation code:" << *NewMI;);
  MIIterType NewMIPos = findInsertPtSpec(NewMI, DstBB, false, true);

  // TODO: If TargetPacket==NewMIPos we bail out in this case,
  // which is sub-optimal. We should modify speculateMI just like
  // predicateMI so that it can be clearly specified that NewMI
  // can be inserted 'just' before the BUNDLE or inside.
  bool invalidSpeculation = (NewMIPos == DstBB->end());
  // Returns NewMIPos when TargetPacket = NewMIPos.
  MICInstIterType laterOne =
      laterOfTheTwo(TargetPacket.getInstrIterator(),
                    NewMIPos.getInstrIterator(), DstBB->instr_end());
  // Requirements on linear ordering of NewMIPos and TargetPacket:
  // When TargetPacket is BUNDLE then NewMIPos < TargetPacket
  // When TargetPacket is not BUNDLE then NewMIPos <= TargetPacket
  // Otherwise the Speculation has to be reverted.
  if (laterOne == NewMIPos.getInstrIterator()) {
    if (TargetPacket->isBundle() || (NewMIPos != TargetPacket))
      invalidSpeculation = true;
  }

  if (invalidSpeculation) {
    LLVM_DEBUG(
        dbgs() << "\nSupporting MI for precious MI cannot be speculated: "
               << *NewMI);
    NewMI->eraseFromParent();
    return false;
  } else {
    LLVM_DEBUG(dbgs() << "<<Resource available for speculating Comp code>>";);
    LLVM_DEBUG(if (NewMIPos->isBundle()) dbgs()
                   << "\nInsert MI inside: " << *NewMIPos;
               else dbgs() << "\nInsert MI just before: " << *NewMIPos;);
  }

  // The position (P) of compensation code should be before the
  // position (P') MI that needs to be speculated when P is a bundle.
  // When P is not bundle then P may be before or same as P'.
  assert(laterOfTheTwo(TargetPacket.getInstrIterator(),
                       NewMIPos.getInstrIterator(),
                       DstBB->instr_end()) == TargetPacket.getInstrIterator());

  // Replace usage of Reg with ScratchReg in OtherBB.
  // Speculate after substitution so that liveness update is correct.
  if (!splitEdge && !LP)
    replaceUsageOfReg(DefR, ScratchReg, OtherBB, NewMI);

  // Since we know that ScratchReg is not live-in in the SrcBB,
  // we can safely speculate NewMI into DstBB
  bool done = speculateMI(NewMI, OtherBB, DstBB, NewMIPos);
  (void)done;
  assert(done && "Wrong Analysis of speculation of precious MI");
  TargetPacket = NewMIPos;
  SpeculatedIns[DstBB].push_back(NewMI);
  if (splitEdge && LP) {
    MachineBasicBlock *NewCritBB = NULL;
    NewCritBB = splitCriticalEdge(DstBB, OtherBB, LP);
    assert(NewCritBB);
    LLVM_DEBUG(dbgs() << "\nNew critical MBB:" << *NewCritBB;);
    LLVM_DEBUG(dbgs() << *DstBB << *OtherBB;);
    DebugLoc dl;
    MachineInstr *NewMI = BuildMI(*NewCritBB, NewCritBB->instr_begin(), dl,
                                  QII->get(Hexagon::A2_tfr), DefR)
                              .addReg(ScratchReg);
    SpecPacketOpp.push_back(NewMI);
    NewCritBB->removeLiveIn(DefR);
    // Update liveness.
    HLV->addNewMI(NewMI, NewCritBB);
    LLVM_DEBUG(dbgs() << "\nAdding MI to New critical MBB:" << *NewCritBB;);
  }
  return true;
}

bool GreedyPullup::preciousToSpeculate(const MachineInstr *MI,
                                       const MachineBasicBlock *SrcBB,
                                       const MachineBasicBlock *DstBB,
                                       const MachineBasicBlock *OtherBB) const {
  if (SpeculateAllInst)
    return true;

  LLVM_DEBUG(dbgs() << "\n[PreciousToSpeculate]";);
  // When we want to speculate within a loop and the otherBB is outside loop.
  MachineLoop *SrcLoop = MLI.getLoopFor(SrcBB);
  MachineLoop *DstLoop = MLI.getLoopFor(DstBB);
  MachineLoop *OtherBBLoop = MLI.getLoopFor(OtherBB);
  return (SrcLoop && MI->mayLoad() && (SrcLoop != OtherBBLoop) &&
          (SrcLoop == DstLoop));
}

bool GreedyPullup::preciousToPredicate(const MachineInstr *MI,
                                       const MachineBasicBlock *SrcBB,
                                       const MachineBasicBlock *DstBB,
                                       const MachineBasicBlock *OtherBB) const {
  if (PredicateAllInst)
    return true;

  MachineLoop *SrcLoop = MLI.getLoopFor(SrcBB);
  MachineLoop *DstLoop = MLI.getLoopFor(DstBB);
  MachineLoop *OtherBBLoop = MLI.getLoopFor(OtherBB);
  // When we want to speculate within a loop and the otherBB is outside loop.
  return (MI->mayLoad() && SrcLoop && (SrcLoop == DstLoop) &&
          (SrcLoop != OtherBBLoop));
}

bool GreedyPullup::canMIBePredicated(MachineInstr *MI, MachineBasicBlock *SrcBB,
                                     MachineBasicBlock *DstBB,
                                     MIIterType &TargetPacket,
                                     MICInstIterType &WithinPacket,
                                     bool &NeedToNewify, bool &Precious) {
  // Initialize iterators to safe defaults
  TargetPacket = DstBB->end();
  WithinPacket = DstBB->instr_end();

  if (!EnablePullupPred)
    return false;
  LLVM_DEBUG(dbgs() << "\n[MICanBePredicated]");

  MachineLoop *L = MLI.getLoopFor(SrcBB);
  if (isSuperblockLoop(L)) {
    if (!isLoopInvariant(*MI))
      return false;
  }

  // No dependency in the current SrcBB
  if (shouldNotBePulledUp(MI) || findInsertionPtSpecLocal(MI) != SrcBB->begin())
    return false;
  LLVM_DEBUG(dbgs() << " .....YES";);
  ABI &abi = AnalyzeBranchInfo.getOrCreate(DstBB, QII);
  if (!abi.AnalysisValid)
    return false;
  assert(abi.TBB && abi.FBB);
  if (!DstBB->isSuccessor(SrcBB))
    return false;
  bool SrcIsTBB = (abi.TBB == SrcBB);
  MachineBasicBlock *OtherBB = abi.FBB;
  if (!SrcIsTBB) {
    assert(abi.FBB == SrcBB);
    OtherBB = abi.TBB;
  }
  SmallVector<MachineOperand, 4> Cond = abi.Cond;
  // Checking the first two is faster than checking isPredicable.
  if (QII->isPredicated(*MI) || QII->isMemOp(*MI) || !QII->isPredicable(*MI))
    return false;
  // Even if def is live-in we should be able to predicate as long
  // as we do not cross the existing def in the MBB which we don't anyway.

  // Clone the MI for analysis
  DebugLoc dl;
  // Manage resources properly.
  MachineInstr *PredMI =
      BuildMI(*SrcBB, SrcBB->instr_begin(), dl, QII->get(MI->getOpcode()));
  for (unsigned i = 0; i < MI->getNumOperands(); ++i)
    PredMI->addOperand(MI->getOperand(i));
  if (!SrcIsTBB) {
    // If the branch condition cannot be reversed.
    if (QII->reverseBranchCondition(Cond)) {
      PredMI->eraseFromParent();
      return false;
    }
  }
  if (!QII->PredicateInstruction(*PredMI, Cond)) {
    PredMI->eraseFromParent();
    return false;
  }
  LLVM_DEBUG(dbgs() << "\nPredicated MI for analysis:" << *PredMI);
  TargetPacket =
      findInsertPtPred(PredMI, DstBB, NeedToNewify, WithinPacket, false);
  if (TargetPacket == DstBB->end()) {
    LLVM_DEBUG(dbgs() << "\nNo resources available for MI; is it precious?";);
    Precious = preciousToPredicate(MI, SrcBB, DstBB, OtherBB);
    if (Precious) {
      LLVM_DEBUG(
          dbgs() << "\nMI is Precious, trying the squeeze the dest. BUNDLE");
      TargetPacket =
          findInsertPtPred(PredMI, DstBB, NeedToNewify, WithinPacket, Precious);
    }
  }
  PredMI->eraseFromParent();
  if (TargetPacket != DstBB->end()) {
    LLVM_DEBUG(dbgs() << "<<Resource available for predication>>";);
    LLVM_DEBUG(if (TargetPacket->isBundle()) dbgs()
                   << "\nInsert MI inside: " << *TargetPacket;
               else dbgs() << "\nInsert MI just before: " << *TargetPacket;);
    return true;
  }
  return false;
}

// When \p DstPos is a BUNDLE, MI can be moved in that
// BUNDLE, otherwise just before the DstPos
bool GreedyPullup::speculateMI(MachineInstr *MI, MachineBasicBlock *SrcBB,
                               MachineBasicBlock *DstBB, MIIterType DstPos) {
  LLVM_DEBUG(dbgs() << "\n[SpeculateMI]");
  // Remove the MI from SrcBB and put it into DstBB at TargetPacket.
  assert(DstPos->getParent() == DstBB);
  MachineInstr *InstrToMove;

  if (DstPos->isBundle()) {
    MIInstIterType DstPack = DstPos.getInstrIterator();
    getResourcesReadyForBundle(DstPack);
    if (canReserveResources(MI->getIterator()) &&
        isLegalToPacketizeWith(DstPos, MI)) {
      InstrToMove = removeMI(MI, QII, &SpecPacketOpp);
      MIBundleBuilder Bundle(&*DstPos);
      Bundle.insert(std::next(DstPack), InstrToMove);
    } else {
      // If there are no resources then insert just before the BUNDLE.
      // TODO: Do this only in special cases.
      InstrToMove = removeMI(MI, QII, &SpecPacketOpp);
      DstBB->insert(DstPos, InstrToMove);
      SpecPacketOpp.push_back(&*DstPos);
    }
  } else {
    // Move the instruction.
    InstrToMove = removeMI(MI, QII, &SpecPacketOpp);
    DstBB->insert(DstPos, InstrToMove);
    // Since an instruction is moved besides another unpacketized insn
    // we might be able to packetize them and possibly other nearby insn.
    SpecPacketOpp.push_back(&*DstPos);
    // TODO: Make the bundle if possible, or leave this task to mergePackets.
  }
  // Update Liveness
  LLVM_DEBUG(dbgs() << "\n[Liveness incrementalUpdate]";);
  HLV->incrementalUpdate(InstrToMove->getIterator(), SrcBB, DstBB);
  // Mark MI as moved with speculation.
  LLVM_DEBUG(dbgs() << "\nSpeculated MI:\t" << *MI;);
  LLVM_DEBUG(dbgs() << "\nDstMBB: " << *DstBB << "\nSrcMBB: " << *SrcBB);
  return true;
}

bool GreedyPullup::predicateMI(MachineInstr *MI, MachineBasicBlock *SrcBB,
                               MachineBasicBlock *DstBB, MIIterType DstPack,
                               MICInstIterType WithinPacket, bool NeedToNewify,
                               bool Precious) {
  LLVM_DEBUG(dbgs() << "\n[PredicateMI]");
  // Remove the MI from SrcBB and put it into DstBB at DstPos.
  // Update Liveness
  // Mark MI as moved with predication.
  assert(DstPack->getParent() == DstBB);
  ABI &abi = AnalyzeBranchInfo.getOrCreate(DstBB, QII);
  assert(abi.AnalysisValid);
  bool SrcIsTBB = (abi.TBB == SrcBB);
  SmallVector<MachineOperand, 4> Cond = abi.Cond;
  if (!SrcIsTBB) {
    // If the branch condition cannot be reversed.
    bool Reversed = !QII->reverseBranchCondition(Cond);
    (void)Reversed;
    assert(Reversed);
  }
  bool Predicated = QII->PredicateInstruction(*MI, Cond);
  (void)Predicated;
  assert(Predicated);
  // Convert to dot.new as well.
  if (NeedToNewify) {
    int NewOpcode = QII->getDotNewPredOp(*MI, MBPI);
    MI->setDesc(QII->get(NewOpcode));
  }

  MachineInstr *InstrToMove;
  if (DstPack->isBundle()) {
    MIInstIterType DstPos = DstPack.getInstrIterator();
    getResourcesReadyForBundle(DstPos);
    // When the MI is Precious, => it is being inserted into the bundle
    // by making compound i.e., five instructions into a BUNDLE.
    [[maybe_unused]] bool Reserved = canReserveResources(MI->getIterator());
    [[maybe_unused]] bool LegalPacket = isLegalToPacketizeWith(DstPack, MI);
    assert(((Reserved && LegalPacket) || Precious) && "Wrong analysis.");

    InstrToMove = removeMI(MI, QII, &SpecPacketOpp);
    MIBundleBuilder Bundle(&*DstPack);
    if (WithinPacket != DstBB->instr_end()) {
      assert(WithinPacket->isBundledWithPred());
      DstPos = const_cast<MachineInstr *>(&*WithinPacket)->getIterator();
    }
    Bundle.insert(std::next(DstPos), InstrToMove);
  } else {
    // Move the instruction.
    InstrToMove = removeMI(MI, QII, &SpecPacketOpp);
    DstBB->insert(DstPack, InstrToMove);
    // Since an instruction is moved besides another unpacketized insn
    // we might be able to packetize them and possibly other nearby insn.
    SpecPacketOpp.push_back(&*DstPack);
    // TODO: Make the bundle if possible, or leave this task to mergePackets.
  }

  // Update Liveness
  LLVM_DEBUG(dbgs() << "\n[Liveness incrementalUpdate]";);
  HLV->incrementalUpdate(InstrToMove->getIterator(), SrcBB, DstBB);
  // Mark MI as moved with speculation.
  LLVM_DEBUG(dbgs() << "\nPredicated MI:\t" << *MI;);
  LLVM_DEBUG(dbgs() << "\nDstMBB: " << *DstBB << "\nSrcMBB: " << *SrcBB);
  return true;
}

//===----------------------------------------------------------------------===//
//                    HexagonGlobalScheduler Functions
//===----------------------------------------------------------------------===//

void HexagonGlobalSchedulerV2::generateCFG(MachineFunction &MF) {
  GS->generateCFG(MF);
}

bool HexagonGlobalSchedulerV2::runOnMachineFunction(MachineFunction &MF) {
  auto &ST = MF.getSubtarget<HexagonSubtarget>();
  if (!EnableGlobalSched || !ST.usePackets() || skipFunction(MF.getFunction()))
    return false;

  TM = static_cast<const HexagonTargetMachine *>(&MF.getTarget());
  QII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  HexagonLiveVariables *HLV = &getAnalysis<HexagonLiveVariables>();
  if (!HLV->HLVComplete) {
    LLVM_DEBUG(dbgs() << "\nHexagon Live Variables Analysis unavailable."
                      << "\nExiting Hexagon Global Scheduler");
    return false;
  }
  const MachineBranchProbabilityInfo *MBPI =
      &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  GS.reset(new GreedyPullup(TM, QII, TRI, MF, *MLI, AA, MBPI, HLV));

  assert(verifyMachineFunction(MF, "VLIW Packetizer") &&
         "Invalid IR after Packetizer");
  generateCFG(MF);
  LLVM_DEBUG(MF.dump(););
  assert(verifyMachineFunction(MF, "After GenerateCFG") &&
         "Invalid CFG transformation.");
  GS->formSchedulingRegions(MF);
  // TODO: Mechanism of sorting the scheduling regions.
  GS->printSchedulingRegions();
  GS->pruneSchedulingRegions(&GlobalScheduler::pruneCriteria);
  LLVM_DEBUG(dbgs() << "\nScheduling regions before sorting";);
  GS->printSchedulingRegions();
  GS->sortSchedulingRegions();
  LLVM_DEBUG(dbgs() << "\nScheduling regions after sorting";);
  GS->printSchedulingRegions();
  LLVM_DEBUG(MF.dump(););
  GS->schedule();
  assert(verifyMachineFunction(MF, "After Scheduling") &&
         "Invalid Scheduling.");
  if (PostSchedPack)
    GS->packetize();
  assert(verifyMachineFunction(MF, "After post-schedule packetization") &&
         "Invalid Opts.");
  if (PostSchedOpt) {
    // Regenerate distance map because so many instructions have
    // been moved.
    HLV->regenerateDistanceMap(MF);
    GS->performExposedOptimizations();
    assert(verifyMachineFunction(MF, "After post-schedule branch opt") &&
           "Invalid Opts.");
  }
  GS->generateCompoundsPostSched();
  GS->generateAutoAndPredicates(MF);
  return true;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonGlobalSchedulerV2() {
  return new HexagonGlobalSchedulerV2();
}
