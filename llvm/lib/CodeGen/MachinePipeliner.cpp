//===- MachinePipeliner.cpp - Machine Software Pipeliner Pass -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An implementation of the Swing Modulo Scheduling (SMS) software pipeliner.
//
// This SMS implementation is a target-independent back-end pass. When enabled,
// the pass runs just prior to the register allocation pass, while the machine
// IR is in SSA form. If software pipelining is successful, then the original
// loop is replaced by the optimized loop. The optimized loop contains one or
// more prolog blocks, the pipelined kernel, and one or more epilog blocks. If
// the instructions cannot be scheduled in a given MII, we increase the MII by
// one and try again.
//
// The SMS implementation is an extension of the ScheduleDAGInstrs class. We
// represent loop carried dependences in the DAG as order edges to the Phi
// nodes. We also perform several passes over the DAG to eliminate unnecessary
// edges that inhibit the ability to pipeline. The implementation uses the
// DFAPacketizer class to compute the minimum initiation interval and the check
// where an instruction may be inserted in the pipelined schedule.
//
// In order for the SMS pass to work, several target specific hooks need to be
// implemented to get information about the loop structure and to rewrite
// instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePipeliner.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ModuloSchedule.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pipeliner"

STATISTIC(NumTrytoPipeline, "Number of loops that we attempt to pipeline");
STATISTIC(NumPipelined, "Number of loops software pipelined");
STATISTIC(NumNodeOrderIssues, "Number of node order issues found");
STATISTIC(NumFailBranch, "Pipeliner abort due to unknown branch");
STATISTIC(NumFailLoop, "Pipeliner abort due to unsupported loop");
STATISTIC(NumFailPreheader, "Pipeliner abort due to missing preheader");
STATISTIC(NumFailLargeMaxMII, "Pipeliner abort due to MaxMII too large");
STATISTIC(NumFailZeroMII, "Pipeliner abort due to zero MII");
STATISTIC(NumFailNoSchedule, "Pipeliner abort due to no schedule found");
STATISTIC(NumFailZeroStage, "Pipeliner abort due to zero stage");
STATISTIC(NumFailLargeMaxStage, "Pipeliner abort due to too many stages");

/// A command line option to turn software pipelining on or off.
static cl::opt<bool> EnableSWP("enable-pipeliner", cl::Hidden, cl::init(true),
                               cl::desc("Enable Software Pipelining"));

/// A command line option to enable SWP at -Os.
static cl::opt<bool> EnableSWPOptSize("enable-pipeliner-opt-size",
                                      cl::desc("Enable SWP at Os."), cl::Hidden,
                                      cl::init(false));

/// A command line argument to limit minimum initial interval for pipelining.
static cl::opt<int> SwpMaxMii("pipeliner-max-mii",
                              cl::desc("Size limit for the MII."),
                              cl::Hidden, cl::init(27));

/// A command line argument to force pipeliner to use specified initial
/// interval.
static cl::opt<int> SwpForceII("pipeliner-force-ii",
                               cl::desc("Force pipeliner to use specified II."),
                               cl::Hidden, cl::init(-1));

/// A command line argument to limit the number of stages in the pipeline.
static cl::opt<int>
    SwpMaxStages("pipeliner-max-stages",
                 cl::desc("Maximum stages allowed in the generated scheduled."),
                 cl::Hidden, cl::init(3));

/// A command line option to disable the pruning of chain dependences due to
/// an unrelated Phi.
static cl::opt<bool>
    SwpPruneDeps("pipeliner-prune-deps",
                 cl::desc("Prune dependences between unrelated Phi nodes."),
                 cl::Hidden, cl::init(true));

/// A command line option to disable the pruning of loop carried order
/// dependences.
static cl::opt<bool>
    SwpPruneLoopCarried("pipeliner-prune-loop-carried",
                        cl::desc("Prune loop carried order dependences."),
                        cl::Hidden, cl::init(true));

#ifndef NDEBUG
static cl::opt<int> SwpLoopLimit("pipeliner-max", cl::Hidden, cl::init(-1));
#endif

static cl::opt<bool> SwpIgnoreRecMII("pipeliner-ignore-recmii",
                                     cl::ReallyHidden,
                                     cl::desc("Ignore RecMII"));

static cl::opt<bool> SwpShowResMask("pipeliner-show-mask", cl::Hidden,
                                    cl::init(false));
static cl::opt<bool> SwpDebugResource("pipeliner-dbg-res", cl::Hidden,
                                      cl::init(false));

static cl::opt<bool> EmitTestAnnotations(
    "pipeliner-annotate-for-testing", cl::Hidden, cl::init(false),
    cl::desc("Instead of emitting the pipelined code, annotate instructions "
             "with the generated schedule for feeding into the "
             "-modulo-schedule-test pass"));

static cl::opt<bool> ExperimentalCodeGen(
    "pipeliner-experimental-cg", cl::Hidden, cl::init(false),
    cl::desc(
        "Use the experimental peeling code generator for software pipelining"));

static cl::opt<int> SwpIISearchRange("pipeliner-ii-search-range",
                                     cl::desc("Range to search for II"),
                                     cl::Hidden, cl::init(10));

static cl::opt<bool>
    LimitRegPressure("pipeliner-register-pressure", cl::Hidden, cl::init(false),
                     cl::desc("Limit register pressure of scheduled loop"));

static cl::opt<int>
    RegPressureMargin("pipeliner-register-pressure-margin", cl::Hidden,
                      cl::init(5),
                      cl::desc("Margin representing the unused percentage of "
                               "the register pressure limit"));

static cl::opt<bool>
    MVECodeGen("pipeliner-mve-cg", cl::Hidden, cl::init(false),
               cl::desc("Use the MVE code generator for software pipelining"));

namespace llvm {

// A command line option to enable the CopyToPhi DAG mutation.
cl::opt<bool> SwpEnableCopyToPhi("pipeliner-enable-copytophi", cl::ReallyHidden,
                                 cl::init(true),
                                 cl::desc("Enable CopyToPhi DAG Mutation"));

/// A command line argument to force pipeliner to use specified issue
/// width.
cl::opt<int> SwpForceIssueWidth(
    "pipeliner-force-issue-width",
    cl::desc("Force pipeliner to use specified issue width."), cl::Hidden,
    cl::init(-1));

/// A command line argument to set the window scheduling option.
static cl::opt<WindowSchedulingFlag> WindowSchedulingOption(
    "window-sched", cl::Hidden, cl::init(WindowSchedulingFlag::WS_On),
    cl::desc("Set how to use window scheduling algorithm."),
    cl::values(clEnumValN(WindowSchedulingFlag::WS_Off, "off",
                          "Turn off window algorithm."),
               clEnumValN(WindowSchedulingFlag::WS_On, "on",
                          "Use window algorithm after SMS algorithm fails."),
               clEnumValN(WindowSchedulingFlag::WS_Force, "force",
                          "Use window algorithm instead of SMS algorithm.")));

} // end namespace llvm

unsigned SwingSchedulerDAG::Circuits::MaxPaths = 5;
char MachinePipeliner::ID = 0;
#ifndef NDEBUG
int MachinePipeliner::NumTries = 0;
#endif
char &llvm::MachinePipelinerID = MachinePipeliner::ID;

INITIALIZE_PASS_BEGIN(MachinePipeliner, DEBUG_TYPE,
                      "Modulo Software Pipelining", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(MachinePipeliner, DEBUG_TYPE,
                    "Modulo Software Pipelining", false, false)

namespace {

/// This class holds an SUnit corresponding to a memory operation and other
/// information related to the instruction.
struct SUnitWithMemInfo {
  SUnit *SU;
  SmallVector<const Value *, 2> UnderlyingObjs;

  /// The value of a memory operand.
  const Value *MemOpValue = nullptr;

  /// The offset of a memory operand.
  int64_t MemOpOffset = 0;

  AAMDNodes AATags;

  /// True if all the underlying objects are identified.
  bool IsAllIdentified = false;

  SUnitWithMemInfo(SUnit *SU);

  bool isTriviallyDisjoint(const SUnitWithMemInfo &Other) const;

  bool isUnknown() const { return MemOpValue == nullptr; }

private:
  bool getUnderlyingObjects();
};

/// Add loop-carried chain dependencies. This class handles the same type of
/// dependencies added by `ScheduleDAGInstrs::buildSchedGraph`, but takes into
/// account dependencies across iterations.
class LoopCarriedOrderDepsTracker {
  // Type of instruction that is relevant to order-dependencies
  enum class InstrTag {
    Barrier = 0,      ///< A barrier event instruction.
    LoadOrStore = 1,  ///< An instruction that may load or store memory, but is
                      ///< not a barrier event.
    FPExceptions = 2, ///< An instruction that does not match above, but may
                      ///< raise floatin-point exceptions.
  };

  struct TaggedSUnit : PointerIntPair<SUnit *, 2> {
    TaggedSUnit(SUnit *SU, InstrTag Tag)
        : PointerIntPair<SUnit *, 2>(SU, unsigned(Tag)) {}

    InstrTag getTag() const { return InstrTag(getInt()); }
  };

  /// Holds loads and stores with memory related information.
  struct LoadStoreChunk {
    SmallVector<SUnitWithMemInfo, 4> Loads;
    SmallVector<SUnitWithMemInfo, 4> Stores;

    void append(SUnit *SU);
  };

  SwingSchedulerDAG *DAG;
  BatchAAResults *BAA;
  std::vector<SUnit> &SUnits;

  /// The size of SUnits, for convenience.
  const unsigned N;

  /// Loop-carried Edges.
  std::vector<BitVector> LoopCarried;

  /// Instructions related to chain dependencies. They are one of the
  /// following:
  ///
  ///  1. Barrier event.
  ///  2. Load, but neither a barrier event, invariant load, nor may load trap
  ///     value.
  ///  3. Store, but not a barrier event.
  ///  4. None of them, but may raise floating-point exceptions.
  ///
  /// This is used when analyzing loop-carried dependencies that access global
  /// barrier instructions.
  std::vector<TaggedSUnit> TaggedSUnits;

  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

public:
  LoopCarriedOrderDepsTracker(SwingSchedulerDAG *SSD, BatchAAResults *BAA,
                              const TargetInstrInfo *TII,
                              const TargetRegisterInfo *TRI);

  /// The main function to compute loop-carried order-dependencies.
  void computeDependencies();

  const BitVector &getLoopCarried(unsigned Idx) const {
    return LoopCarried[Idx];
  }

private:
  /// Tags to \p SU if the instruction may affect the order-dependencies.
  std::optional<InstrTag> getInstrTag(SUnit *SU) const;

  void addLoopCarriedDepenenciesForChunks(const LoadStoreChunk &From,
                                          const LoadStoreChunk &To);

  /// Add a loop-carried order dependency between \p Src and \p Dst if we
  /// cannot prove they are independent. When \p PerformCheapCheck is true, a
  /// lightweight dependency test (referred to as "cheap check" below) is
  /// performed at first. Note that the cheap check is retained to maintain the
  /// existing behavior and not expected to be used anymore.
  ///
  /// TODO: Remove \p PerformCheapCheck and the corresponding cheap check.
  void addDependenciesBetweenSUs(const SUnitWithMemInfo &Src,
                                 const SUnitWithMemInfo &Dst,
                                 bool PerformCheapCheck = false);

  void computeDependenciesAux();
};

} // end anonymous namespace

/// The "main" function for implementing Swing Modulo Scheduling.
bool MachinePipeliner::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(mf.getFunction()))
    return false;

  if (!EnableSWP)
    return false;

  if (mf.getFunction().getAttributes().hasFnAttr(Attribute::OptimizeForSize) &&
      !EnableSWPOptSize.getPosition())
    return false;

  if (!mf.getSubtarget().enableMachinePipeliner())
    return false;

  // Cannot pipeline loops without instruction itineraries if we are using
  // DFA for the pipeliner.
  if (mf.getSubtarget().useDFAforSMS() &&
      (!mf.getSubtarget().getInstrItineraryData() ||
       mf.getSubtarget().getInstrItineraryData()->isEmpty()))
    return false;

  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ORE = &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  TII = MF->getSubtarget().getInstrInfo();
  RegClassInfo.runOnMachineFunction(*MF);

  for (const auto &L : *MLI)
    scheduleLoop(*L);

  return false;
}

/// Attempt to perform the SMS algorithm on the specified loop. This function is
/// the main entry point for the algorithm.  The function identifies candidate
/// loops, calculates the minimum initiation interval, and attempts to schedule
/// the loop.
bool MachinePipeliner::scheduleLoop(MachineLoop &L) {
  bool Changed = false;
  for (const auto &InnerLoop : L)
    Changed |= scheduleLoop(*InnerLoop);

#ifndef NDEBUG
  // Stop trying after reaching the limit (if any).
  int Limit = SwpLoopLimit;
  if (Limit >= 0) {
    if (NumTries >= SwpLoopLimit)
      return Changed;
    NumTries++;
  }
#endif

  setPragmaPipelineOptions(L);
  if (!canPipelineLoop(L)) {
    LLVM_DEBUG(dbgs() << "\n!!! Can not pipeline loop.\n");
    ORE->emit([&]() {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE, "canPipelineLoop",
                                             L.getStartLoc(), L.getHeader())
             << "Failed to pipeline loop";
    });

    LI.LoopPipelinerInfo.reset();
    return Changed;
  }

  ++NumTrytoPipeline;
  if (useSwingModuloScheduler())
    Changed = swingModuloScheduler(L);

  if (useWindowScheduler(Changed))
    Changed = runWindowScheduler(L);

  LI.LoopPipelinerInfo.reset();
  return Changed;
}

void MachinePipeliner::setPragmaPipelineOptions(MachineLoop &L) {
  // Reset the pragma for the next loop in iteration.
  disabledByPragma = false;
  II_setByPragma = 0;

  MachineBasicBlock *LBLK = L.getTopBlock();

  if (LBLK == nullptr)
    return;

  const BasicBlock *BBLK = LBLK->getBasicBlock();
  if (BBLK == nullptr)
    return;

  const Instruction *TI = BBLK->getTerminator();
  if (TI == nullptr)
    return;

  MDNode *LoopID = TI->getMetadata(LLVMContext::MD_loop);
  if (LoopID == nullptr)
    return;

  assert(LoopID->getNumOperands() > 0 && "requires atleast one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop");

  for (const MDOperand &MDO : llvm::drop_begin(LoopID->operands())) {
    MDNode *MD = dyn_cast<MDNode>(MDO);

    if (MD == nullptr)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));

    if (S == nullptr)
      continue;

    if (S->getString() == "llvm.loop.pipeline.initiationinterval") {
      assert(MD->getNumOperands() == 2 &&
             "Pipeline initiation interval hint metadata should have two operands.");
      II_setByPragma =
          mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
      assert(II_setByPragma >= 1 && "Pipeline initiation interval must be positive.");
    } else if (S->getString() == "llvm.loop.pipeline.disable") {
      disabledByPragma = true;
    }
  }
}

/// Return true if the loop can be software pipelined.  The algorithm is
/// restricted to loops with a single basic block.  Make sure that the
/// branch in the loop can be analyzed.
bool MachinePipeliner::canPipelineLoop(MachineLoop &L) {
  if (L.getNumBlocks() != 1) {
    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "canPipelineLoop",
                                               L.getStartLoc(), L.getHeader())
             << "Not a single basic block: "
             << ore::NV("NumBlocks", L.getNumBlocks());
    });
    return false;
  }

  if (disabledByPragma) {
    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "canPipelineLoop",
                                               L.getStartLoc(), L.getHeader())
             << "Disabled by Pragma.";
    });
    return false;
  }

  // Check if the branch can't be understood because we can't do pipelining
  // if that's the case.
  LI.TBB = nullptr;
  LI.FBB = nullptr;
  LI.BrCond.clear();
  if (TII->analyzeBranch(*L.getHeader(), LI.TBB, LI.FBB, LI.BrCond)) {
    LLVM_DEBUG(dbgs() << "Unable to analyzeBranch, can NOT pipeline Loop\n");
    NumFailBranch++;
    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "canPipelineLoop",
                                               L.getStartLoc(), L.getHeader())
             << "The branch can't be understood";
    });
    return false;
  }

  LI.LoopInductionVar = nullptr;
  LI.LoopCompare = nullptr;
  LI.LoopPipelinerInfo = TII->analyzeLoopForPipelining(L.getTopBlock());
  if (!LI.LoopPipelinerInfo) {
    LLVM_DEBUG(dbgs() << "Unable to analyzeLoop, can NOT pipeline Loop\n");
    NumFailLoop++;
    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "canPipelineLoop",
                                               L.getStartLoc(), L.getHeader())
             << "The loop structure is not supported";
    });
    return false;
  }

  if (!L.getLoopPreheader()) {
    LLVM_DEBUG(dbgs() << "Preheader not found, can NOT pipeline Loop\n");
    NumFailPreheader++;
    ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(DEBUG_TYPE, "canPipelineLoop",
                                               L.getStartLoc(), L.getHeader())
             << "No loop preheader found";
    });
    return false;
  }

  // Remove any subregisters from inputs to phi nodes.
  preprocessPhiNodes(*L.getHeader());
  return true;
}

void MachinePipeliner::preprocessPhiNodes(MachineBasicBlock &B) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SlotIndexes &Slots =
      *getAnalysis<LiveIntervalsWrapperPass>().getLIS().getSlotIndexes();

  for (MachineInstr &PI : B.phis()) {
    MachineOperand &DefOp = PI.getOperand(0);
    assert(DefOp.getSubReg() == 0);
    auto *RC = MRI.getRegClass(DefOp.getReg());

    for (unsigned i = 1, n = PI.getNumOperands(); i != n; i += 2) {
      MachineOperand &RegOp = PI.getOperand(i);
      if (RegOp.getSubReg() == 0)
        continue;

      // If the operand uses a subregister, replace it with a new register
      // without subregisters, and generate a copy to the new register.
      Register NewReg = MRI.createVirtualRegister(RC);
      MachineBasicBlock &PredB = *PI.getOperand(i+1).getMBB();
      MachineBasicBlock::iterator At = PredB.getFirstTerminator();
      const DebugLoc &DL = PredB.findDebugLoc(At);
      auto Copy = BuildMI(PredB, At, DL, TII->get(TargetOpcode::COPY), NewReg)
                    .addReg(RegOp.getReg(), getRegState(RegOp),
                            RegOp.getSubReg());
      Slots.insertMachineInstrInMaps(*Copy);
      RegOp.setReg(NewReg);
      RegOp.setSubReg(0);
    }
  }
}

/// The SMS algorithm consists of the following main steps:
/// 1. Computation and analysis of the dependence graph.
/// 2. Ordering of the nodes (instructions).
/// 3. Attempt to Schedule the loop.
bool MachinePipeliner::swingModuloScheduler(MachineLoop &L) {
  assert(L.getBlocks().size() == 1 && "SMS works on single blocks only.");

  AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  SwingSchedulerDAG SMS(
      *this, L, getAnalysis<LiveIntervalsWrapperPass>().getLIS(), RegClassInfo,
      II_setByPragma, LI.LoopPipelinerInfo.get(), AA);

  MachineBasicBlock *MBB = L.getHeader();
  // The kernel should not include any terminator instructions.  These
  // will be added back later.
  SMS.startBlock(MBB);

  // Compute the number of 'real' instructions in the basic block by
  // ignoring terminators.
  unsigned size = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->getFirstTerminator(),
                                   E = MBB->instr_end();
       I != E; ++I, --size)
    ;

  SMS.enterRegion(MBB, MBB->begin(), MBB->getFirstTerminator(), size);
  SMS.schedule();
  SMS.exitRegion();

  SMS.finishBlock();
  return SMS.hasNewSchedule();
}

void MachinePipeliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<MachineOptimizationRemarkEmitterPass>();
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachinePipeliner::runWindowScheduler(MachineLoop &L) {
  MachineSchedContext Context;
  Context.MF = MF;
  Context.MLI = MLI;
  Context.MDT = MDT;
  Context.TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  Context.AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  Context.LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Context.RegClassInfo->runOnMachineFunction(*MF);
  WindowScheduler WS(&Context, L);
  return WS.run();
}

bool MachinePipeliner::useSwingModuloScheduler() {
  // SwingModuloScheduler does not work when WindowScheduler is forced.
  return WindowSchedulingOption != WindowSchedulingFlag::WS_Force;
}

bool MachinePipeliner::useWindowScheduler(bool Changed) {
  // WindowScheduler does not work for following cases:
  // 1. when it is off.
  // 2. when SwingModuloScheduler is successfully scheduled.
  // 3. when pragma II is enabled.
  if (II_setByPragma) {
    LLVM_DEBUG(dbgs() << "Window scheduling is disabled when "
                         "llvm.loop.pipeline.initiationinterval is set.\n");
    return false;
  }

  return WindowSchedulingOption == WindowSchedulingFlag::WS_Force ||
         (WindowSchedulingOption == WindowSchedulingFlag::WS_On && !Changed);
}

void SwingSchedulerDAG::setMII(unsigned ResMII, unsigned RecMII) {
  if (SwpForceII > 0)
    MII = SwpForceII;
  else if (II_setByPragma > 0)
    MII = II_setByPragma;
  else
    MII = std::max(ResMII, RecMII);
}

void SwingSchedulerDAG::setMAX_II() {
  if (SwpForceII > 0)
    MAX_II = SwpForceII;
  else if (II_setByPragma > 0)
    MAX_II = II_setByPragma;
  else
    MAX_II = MII + SwpIISearchRange;
}

/// We override the schedule function in ScheduleDAGInstrs to implement the
/// scheduling part of the Swing Modulo Scheduling algorithm.
void SwingSchedulerDAG::schedule() {
  buildSchedGraph(AA);
  const LoopCarriedEdges LCE = addLoopCarriedDependences();
  updatePhiDependences();
  Topo.InitDAGTopologicalSorting();
  changeDependences();
  postProcessDAG();
  DDG = std::make_unique<SwingSchedulerDDG>(SUnits, &EntrySU, &ExitSU, LCE);
  LLVM_DEBUG({
    dump();
    dbgs() << "===== Loop Carried Edges Begin =====\n";
    for (SUnit &SU : SUnits)
      LCE.dump(&SU, TRI, &MRI);
    dbgs() << "===== Loop Carried Edges End =====\n";
  });

  NodeSetType NodeSets;
  findCircuits(NodeSets);
  NodeSetType Circuits = NodeSets;

  // Calculate the MII.
  unsigned ResMII = calculateResMII();
  unsigned RecMII = calculateRecMII(NodeSets);

  fuseRecs(NodeSets);

  // This flag is used for testing and can cause correctness problems.
  if (SwpIgnoreRecMII)
    RecMII = 0;

  setMII(ResMII, RecMII);
  setMAX_II();

  LLVM_DEBUG(dbgs() << "MII = " << MII << " MAX_II = " << MAX_II
                    << " (rec=" << RecMII << ", res=" << ResMII << ")\n");

  // Can't schedule a loop without a valid MII.
  if (MII == 0) {
    LLVM_DEBUG(dbgs() << "Invalid Minimal Initiation Interval: 0\n");
    NumFailZeroMII++;
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "Invalid Minimal Initiation Interval: 0";
    });
    return;
  }

  // Don't pipeline large loops.
  if (SwpMaxMii != -1 && (int)MII > SwpMaxMii) {
    LLVM_DEBUG(dbgs() << "MII > " << SwpMaxMii
                      << ", we don't pipeline large loops\n");
    NumFailLargeMaxMII++;
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "Minimal Initiation Interval too large: "
             << ore::NV("MII", (int)MII) << " > "
             << ore::NV("SwpMaxMii", SwpMaxMii) << "."
             << "Refer to -pipeliner-max-mii.";
    });
    return;
  }

  computeNodeFunctions(NodeSets);

  registerPressureFilter(NodeSets);

  colocateNodeSets(NodeSets);

  checkNodeSets(NodeSets);

  LLVM_DEBUG({
    for (auto &I : NodeSets) {
      dbgs() << "  Rec NodeSet ";
      I.dump();
    }
  });

  llvm::stable_sort(NodeSets, std::greater<NodeSet>());

  groupRemainingNodes(NodeSets);

  removeDuplicateNodes(NodeSets);

  LLVM_DEBUG({
    for (auto &I : NodeSets) {
      dbgs() << "  NodeSet ";
      I.dump();
    }
  });

  computeNodeOrder(NodeSets);

  // check for node order issues
  checkValidNodeOrder(Circuits);

  SMSchedule Schedule(Pass.MF, this);
  Scheduled = schedulePipeline(Schedule);

  if (!Scheduled){
    LLVM_DEBUG(dbgs() << "No schedule found, return\n");
    NumFailNoSchedule++;
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "Unable to find schedule";
    });
    return;
  }

  unsigned numStages = Schedule.getMaxStageCount();
  // No need to generate pipeline if there are no overlapped iterations.
  if (numStages == 0) {
    LLVM_DEBUG(dbgs() << "No overlapped iterations, skip.\n");
    NumFailZeroStage++;
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "No need to pipeline - no overlapped iterations in schedule.";
    });
    return;
  }
  // Check that the maximum stage count is less than user-defined limit.
  if (SwpMaxStages > -1 && (int)numStages > SwpMaxStages) {
    LLVM_DEBUG(dbgs() << "numStages:" << numStages << ">" << SwpMaxStages
                      << " : too many stages, abort\n");
    NumFailLargeMaxStage++;
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "Too many stages in schedule: "
             << ore::NV("numStages", (int)numStages) << " > "
             << ore::NV("SwpMaxStages", SwpMaxStages)
             << ". Refer to -pipeliner-max-stages.";
    });
    return;
  }

  Pass.ORE->emit([&]() {
    return MachineOptimizationRemark(DEBUG_TYPE, "schedule", Loop.getStartLoc(),
                                     Loop.getHeader())
           << "Pipelined succesfully!";
  });

  // Generate the schedule as a ModuloSchedule.
  DenseMap<MachineInstr *, int> Cycles, Stages;
  std::vector<MachineInstr *> OrderedInsts;
  for (int Cycle = Schedule.getFirstCycle(); Cycle <= Schedule.getFinalCycle();
       ++Cycle) {
    for (SUnit *SU : Schedule.getInstructions(Cycle)) {
      OrderedInsts.push_back(SU->getInstr());
      Cycles[SU->getInstr()] = Cycle;
      Stages[SU->getInstr()] = Schedule.stageScheduled(SU);
    }
  }
  DenseMap<MachineInstr *, std::pair<Register, int64_t>> NewInstrChanges;
  for (auto &KV : NewMIs) {
    Cycles[KV.first] = Cycles[KV.second];
    Stages[KV.first] = Stages[KV.second];
    NewInstrChanges[KV.first] = InstrChanges[getSUnit(KV.first)];
  }

  ModuloSchedule MS(MF, &Loop, std::move(OrderedInsts), std::move(Cycles),
                    std::move(Stages));
  if (EmitTestAnnotations) {
    assert(NewInstrChanges.empty() &&
           "Cannot serialize a schedule with InstrChanges!");
    ModuloScheduleTestAnnotater MSTI(MF, MS);
    MSTI.annotate();
    return;
  }
  // The experimental code generator can't work if there are InstChanges.
  if (ExperimentalCodeGen && NewInstrChanges.empty()) {
    PeelingModuloScheduleExpander MSE(MF, MS, &LIS);
    MSE.expand();
  } else if (MVECodeGen && NewInstrChanges.empty() &&
             LoopPipelinerInfo->isMVEExpanderSupported() &&
             ModuloScheduleExpanderMVE::canApply(Loop)) {
    ModuloScheduleExpanderMVE MSE(MF, MS, LIS);
    MSE.expand();
  } else {
    ModuloScheduleExpander MSE(MF, MS, LIS, std::move(NewInstrChanges));
    MSE.expand();
    MSE.cleanup();
  }
  ++NumPipelined;
}

/// Clean up after the software pipeliner runs.
void SwingSchedulerDAG::finishBlock() {
  for (auto &KV : NewMIs)
    MF.deleteMachineInstr(KV.second);
  NewMIs.clear();

  // Call the superclass.
  ScheduleDAGInstrs::finishBlock();
}

/// Return the register values for  the operands of a Phi instruction.
/// This function assume the instruction is a Phi.
static void getPhiRegs(MachineInstr &Phi, MachineBasicBlock *Loop,
                       Register &InitVal, Register &LoopVal) {
  assert(Phi.isPHI() && "Expecting a Phi.");

  InitVal = Register();
  LoopVal = Register();
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() != Loop)
      InitVal = Phi.getOperand(i).getReg();
    else
      LoopVal = Phi.getOperand(i).getReg();

  assert(InitVal && LoopVal && "Unexpected Phi structure.");
}

/// Return the Phi register value that comes the loop block.
static Register getLoopPhiReg(const MachineInstr &Phi,
                              const MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() == LoopBB)
      return Phi.getOperand(i).getReg();
  return Register();
}

/// Return true if SUb can be reached from SUa following the chain edges.
static bool isSuccOrder(SUnit *SUa, SUnit *SUb) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SUnit *, 8> Worklist;
  Worklist.push_back(SUa);
  while (!Worklist.empty()) {
    const SUnit *SU = Worklist.pop_back_val();
    for (const auto &SI : SU->Succs) {
      SUnit *SuccSU = SI.getSUnit();
      if (SI.getKind() == SDep::Order) {
        if (Visited.count(SuccSU))
          continue;
        if (SuccSU == SUb)
          return true;
        Worklist.push_back(SuccSU);
        Visited.insert(SuccSU);
      }
    }
  }
  return false;
}

SUnitWithMemInfo::SUnitWithMemInfo(SUnit *SU) : SU(SU) {
  if (!getUnderlyingObjects())
    return;
  for (const Value *Obj : UnderlyingObjs)
    if (!isIdentifiedObject(Obj)) {
      IsAllIdentified = false;
      break;
    }
}

bool SUnitWithMemInfo::isTriviallyDisjoint(
    const SUnitWithMemInfo &Other) const {
  // If all underlying objects are identified objects and there is no overlap
  // between them, then these two instructions are disjoint.
  if (!IsAllIdentified || !Other.IsAllIdentified)
    return false;
  for (const Value *Obj : UnderlyingObjs)
    if (llvm::is_contained(Other.UnderlyingObjs, Obj))
      return false;
  return true;
}

/// Collect the underlying objects for the memory references of an instruction.
/// This function calls the code in ValueTracking, but first checks that the
/// instruction has a memory operand.
/// Returns false if we cannot find the underlying objects.
bool SUnitWithMemInfo::getUnderlyingObjects() {
  const MachineInstr *MI = SU->getInstr();
  if (!MI->hasOneMemOperand())
    return false;
  MachineMemOperand *MM = *MI->memoperands_begin();
  if (!MM->getValue())
    return false;
  MemOpValue = MM->getValue();
  MemOpOffset = MM->getOffset();
  llvm::getUnderlyingObjects(MemOpValue, UnderlyingObjs);

  // TODO: A no alias scope may be valid only in a single iteration. In this
  // case we need to peel off it like LoopAccessAnalysis does.
  AATags = MM->getAAInfo();
  return true;
}

/// Returns true if there is a loop-carried order dependency from \p Src to \p
/// Dst.
static bool
hasLoopCarriedMemDep(const SUnitWithMemInfo &Src, const SUnitWithMemInfo &Dst,
                     BatchAAResults &BAA, const TargetInstrInfo *TII,
                     const TargetRegisterInfo *TRI,
                     const SwingSchedulerDAG *SSD, bool PerformCheapCheck) {
  if (Src.isTriviallyDisjoint(Dst))
    return false;
  if (isSuccOrder(Src.SU, Dst.SU))
    return false;

  MachineInstr &SrcMI = *Src.SU->getInstr();
  MachineInstr &DstMI = *Dst.SU->getInstr();
  if (PerformCheapCheck) {
    // First, perform the cheaper check that compares the base register.
    // If they are the same and the load offset is less than the store
    // offset, then mark the dependence as loop carried potentially.
    //
    // TODO: This check will be removed.
    const MachineOperand *BaseOp1, *BaseOp2;
    int64_t Offset1, Offset2;
    bool Offset1IsScalable, Offset2IsScalable;
    if (TII->getMemOperandWithOffset(SrcMI, BaseOp1, Offset1, Offset1IsScalable,
                                     TRI) &&
        TII->getMemOperandWithOffset(DstMI, BaseOp2, Offset2, Offset2IsScalable,
                                     TRI)) {
      if (BaseOp1->isIdenticalTo(*BaseOp2) &&
          Offset1IsScalable == Offset2IsScalable &&
          (int)Offset1 < (int)Offset2) {
        assert(TII->areMemAccessesTriviallyDisjoint(SrcMI, DstMI) &&
               "What happened to the chain edge?");
        return true;
      }
    }
  }

  if (!SSD->mayOverlapInLaterIter(&SrcMI, &DstMI))
    return false;

  // Second, the more expensive check that uses alias analysis on the
  // base registers. If they alias, and the load offset is less than
  // the store offset, the mark the dependence as loop carried.
  if (Src.isUnknown() || Dst.isUnknown())
    return true;
  if (Src.MemOpValue == Dst.MemOpValue && Src.MemOpOffset <= Dst.MemOpOffset)
    return true;

  if (BAA.isNoAlias(
          MemoryLocation::getBeforeOrAfter(Src.MemOpValue, Src.AATags),
          MemoryLocation::getBeforeOrAfter(Dst.MemOpValue, Dst.AATags)))
    return false;

  // AliasAnalysis sometimes gives up on following the underlying
  // object. In such a case, separate checks for underlying objects may
  // prove that there are no aliases between two accesses.
  for (const Value *SrcObj : Src.UnderlyingObjs)
    for (const Value *DstObj : Dst.UnderlyingObjs)
      if (!BAA.isNoAlias(MemoryLocation::getBeforeOrAfter(SrcObj, Src.AATags),
                         MemoryLocation::getBeforeOrAfter(DstObj, Dst.AATags)))
        return true;

  return false;
}

void LoopCarriedOrderDepsTracker::LoadStoreChunk::append(SUnit *SU) {
  const MachineInstr *MI = SU->getInstr();
  if (!MI->mayLoadOrStore())
    return;
  (MI->mayStore() ? Stores : Loads).emplace_back(SU);
}

LoopCarriedOrderDepsTracker::LoopCarriedOrderDepsTracker(
    SwingSchedulerDAG *SSD, BatchAAResults *BAA, const TargetInstrInfo *TII,
    const TargetRegisterInfo *TRI)
    : DAG(SSD), BAA(BAA), SUnits(DAG->SUnits), N(SUnits.size()),
      LoopCarried(N, BitVector(N)), TII(TII), TRI(TRI) {}

void LoopCarriedOrderDepsTracker::computeDependencies() {
  // Traverse all instructions and extract only what we are targetting.
  for (auto &SU : SUnits) {
    auto Tagged = getInstrTag(&SU);

    // This instruction has no loop-carried order-dependencies.
    if (!Tagged)
      continue;
    TaggedSUnits.emplace_back(&SU, *Tagged);
  }

  computeDependenciesAux();
}

std::optional<LoopCarriedOrderDepsTracker::InstrTag>
LoopCarriedOrderDepsTracker::getInstrTag(SUnit *SU) const {
  MachineInstr *MI = SU->getInstr();
  if (TII->isGlobalMemoryObject(MI))
    return InstrTag::Barrier;

  if (MI->mayStore() ||
      (MI->mayLoad() && !MI->isDereferenceableInvariantLoad()))
    return InstrTag::LoadOrStore;

  if (MI->mayRaiseFPException())
    return InstrTag::FPExceptions;

  return std::nullopt;
}

void LoopCarriedOrderDepsTracker::addDependenciesBetweenSUs(
    const SUnitWithMemInfo &Src, const SUnitWithMemInfo &Dst,
    bool PerformCheapCheck) {
  // Avoid self-dependencies.
  if (Src.SU == Dst.SU)
    return;

  if (hasLoopCarriedMemDep(Src, Dst, *BAA, TII, TRI, DAG, PerformCheapCheck))
    LoopCarried[Src.SU->NodeNum].set(Dst.SU->NodeNum);
}

void LoopCarriedOrderDepsTracker::addLoopCarriedDepenenciesForChunks(
    const LoadStoreChunk &From, const LoadStoreChunk &To) {
  // Add load-to-store dependencies (WAR).
  for (const SUnitWithMemInfo &Src : From.Loads)
    for (const SUnitWithMemInfo &Dst : To.Stores)
      // Perform a cheap check first if this is a forward dependency.
      addDependenciesBetweenSUs(Src, Dst, Src.SU->NodeNum < Dst.SU->NodeNum);

  // Add store-to-load dependencies (RAW).
  for (const SUnitWithMemInfo &Src : From.Stores)
    for (const SUnitWithMemInfo &Dst : To.Loads)
      addDependenciesBetweenSUs(Src, Dst);

  // Add store-to-store dependencies (WAW).
  for (const SUnitWithMemInfo &Src : From.Stores)
    for (const SUnitWithMemInfo &Dst : To.Stores)
      addDependenciesBetweenSUs(Src, Dst);
}

void LoopCarriedOrderDepsTracker::computeDependenciesAux() {
  SmallVector<LoadStoreChunk, 2> Chunks(1);
  for (const auto &TSU : TaggedSUnits) {
    InstrTag Tag = TSU.getTag();
    SUnit *SU = TSU.getPointer();
    switch (Tag) {
    case InstrTag::Barrier:
      Chunks.emplace_back();
      break;
    case InstrTag::LoadOrStore:
      Chunks.back().append(SU);
      break;
    case InstrTag::FPExceptions:
      // TODO: Handle this properly.
      break;
    }
  }

  // Add dependencies between memory operations. If there are one or more
  // barrier events between two memory instructions, we don't add a
  // loop-carried dependence for them.
  for (const LoadStoreChunk &Chunk : Chunks)
    addLoopCarriedDepenenciesForChunks(Chunk, Chunk);

  // TODO: If there are multiple barrier instructions, dependencies from the
  // last barrier instruction (or load/store below it) to the first barrier
  // instruction (or load/store above it).
}

/// Add a chain edge between a load and store if the store can be an
/// alias of the load on a subsequent iteration, i.e., a loop carried
/// dependence. This code is very similar to the code in ScheduleDAGInstrs
/// but that code doesn't create loop carried dependences.
/// TODO: Also compute output-dependencies.
LoopCarriedEdges SwingSchedulerDAG::addLoopCarriedDependences() {
  LoopCarriedEdges LCE;

  // Add loop-carried order-dependencies
  LoopCarriedOrderDepsTracker LCODTracker(this, &BAA, TII, TRI);
  LCODTracker.computeDependencies();
  for (unsigned I = 0; I != SUnits.size(); I++)
    for (const int Succ : LCODTracker.getLoopCarried(I).set_bits())
      LCE.OrderDeps[&SUnits[I]].insert(&SUnits[Succ]);

  LCE.modifySUnits(SUnits, TII);
  return LCE;
}

/// Update the phi dependences to the DAG because ScheduleDAGInstrs no longer
/// processes dependences for PHIs. This function adds true dependences
/// from a PHI to a use, and a loop carried dependence from the use to the
/// PHI. The loop carried dependence is represented as an anti dependence
/// edge. This function also removes chain dependences between unrelated
/// PHIs.
void SwingSchedulerDAG::updatePhiDependences() {
  SmallVector<SDep, 4> RemoveDeps;
  const TargetSubtargetInfo &ST = MF.getSubtarget<TargetSubtargetInfo>();

  // Iterate over each DAG node.
  for (SUnit &I : SUnits) {
    RemoveDeps.clear();
    // Set to true if the instruction has an operand defined by a Phi.
    Register HasPhiUse;
    Register HasPhiDef;
    MachineInstr *MI = I.getInstr();
    // Iterate over each operand, and we process the definitions.
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (MO.isDef()) {
        // If the register is used by a Phi, then create an anti dependence.
        for (MachineRegisterInfo::use_instr_iterator
                 UI = MRI.use_instr_begin(Reg),
                 UE = MRI.use_instr_end();
             UI != UE; ++UI) {
          MachineInstr *UseMI = &*UI;
          SUnit *SU = getSUnit(UseMI);
          if (SU != nullptr && UseMI->isPHI()) {
            if (!MI->isPHI()) {
              SDep Dep(SU, SDep::Anti, Reg);
              Dep.setLatency(1);
              I.addPred(Dep);
            } else {
              HasPhiDef = Reg;
              // Add a chain edge to a dependent Phi that isn't an existing
              // predecessor.
              if (SU->NodeNum < I.NodeNum && !I.isPred(SU))
                I.addPred(SDep(SU, SDep::Barrier));
            }
          }
        }
      } else if (MO.isUse()) {
        // If the register is defined by a Phi, then create a true dependence.
        MachineInstr *DefMI = MRI.getUniqueVRegDef(Reg);
        if (DefMI == nullptr)
          continue;
        SUnit *SU = getSUnit(DefMI);
        if (SU != nullptr && DefMI->isPHI()) {
          if (!MI->isPHI()) {
            SDep Dep(SU, SDep::Data, Reg);
            Dep.setLatency(0);
            ST.adjustSchedDependency(SU, 0, &I, MO.getOperandNo(), Dep,
                                     &SchedModel);
            I.addPred(Dep);
          } else {
            HasPhiUse = Reg;
            // Add a chain edge to a dependent Phi that isn't an existing
            // predecessor.
            if (SU->NodeNum < I.NodeNum && !I.isPred(SU))
              I.addPred(SDep(SU, SDep::Barrier));
          }
        }
      }
    }
    // Remove order dependences from an unrelated Phi.
    if (!SwpPruneDeps)
      continue;
    for (auto &PI : I.Preds) {
      MachineInstr *PMI = PI.getSUnit()->getInstr();
      if (PMI->isPHI() && PI.getKind() == SDep::Order) {
        if (I.getInstr()->isPHI()) {
          if (PMI->getOperand(0).getReg() == HasPhiUse)
            continue;
          if (getLoopPhiReg(*PMI, PMI->getParent()) == HasPhiDef)
            continue;
        }
        RemoveDeps.push_back(PI);
      }
    }
    for (const SDep &D : RemoveDeps)
      I.removePred(D);
  }
}

/// Iterate over each DAG node and see if we can change any dependences
/// in order to reduce the recurrence MII.
void SwingSchedulerDAG::changeDependences() {
  // See if an instruction can use a value from the previous iteration.
  // If so, we update the base and offset of the instruction and change
  // the dependences.
  for (SUnit &I : SUnits) {
    unsigned BasePos = 0, OffsetPos = 0;
    Register NewBase;
    int64_t NewOffset = 0;
    if (!canUseLastOffsetValue(I.getInstr(), BasePos, OffsetPos, NewBase,
                               NewOffset))
      continue;

    // Get the MI and SUnit for the instruction that defines the original base.
    Register OrigBase = I.getInstr()->getOperand(BasePos).getReg();
    MachineInstr *DefMI = MRI.getUniqueVRegDef(OrigBase);
    if (!DefMI)
      continue;
    SUnit *DefSU = getSUnit(DefMI);
    if (!DefSU)
      continue;
    // Get the MI and SUnit for the instruction that defins the new base.
    MachineInstr *LastMI = MRI.getUniqueVRegDef(NewBase);
    if (!LastMI)
      continue;
    SUnit *LastSU = getSUnit(LastMI);
    if (!LastSU)
      continue;

    if (Topo.IsReachable(&I, LastSU))
      continue;

    // Remove the dependence. The value now depends on a prior iteration.
    SmallVector<SDep, 4> Deps;
    for (const SDep &P : I.Preds)
      if (P.getSUnit() == DefSU)
        Deps.push_back(P);
    for (const SDep &D : Deps) {
      Topo.RemovePred(&I, D.getSUnit());
      I.removePred(D);
    }
    // Remove the chain dependence between the instructions.
    Deps.clear();
    for (auto &P : LastSU->Preds)
      if (P.getSUnit() == &I && P.getKind() == SDep::Order)
        Deps.push_back(P);
    for (const SDep &D : Deps) {
      Topo.RemovePred(LastSU, D.getSUnit());
      LastSU->removePred(D);
    }

    // Add a dependence between the new instruction and the instruction
    // that defines the new base.
    SDep Dep(&I, SDep::Anti, NewBase);
    Topo.AddPred(LastSU, &I);
    LastSU->addPred(Dep);

    // Remember the base and offset information so that we can update the
    // instruction during code generation.
    InstrChanges[&I] = std::make_pair(NewBase, NewOffset);
  }
}

/// Create an instruction stream that represents a single iteration and stage of
/// each instruction. This function differs from SMSchedule::finalizeSchedule in
/// that this doesn't have any side-effect to SwingSchedulerDAG. That is, this
/// function is an approximation of SMSchedule::finalizeSchedule with all
/// non-const operations removed.
static void computeScheduledInsts(const SwingSchedulerDAG *SSD,
                                  SMSchedule &Schedule,
                                  std::vector<MachineInstr *> &OrderedInsts,
                                  DenseMap<MachineInstr *, unsigned> &Stages) {
  DenseMap<int, std::deque<SUnit *>> Instrs;

  // Move all instructions to the first stage from the later stages.
  for (int Cycle = Schedule.getFirstCycle(); Cycle <= Schedule.getFinalCycle();
       ++Cycle) {
    for (int Stage = 0, LastStage = Schedule.getMaxStageCount();
         Stage <= LastStage; ++Stage) {
      for (SUnit *SU : llvm::reverse(Schedule.getInstructions(
               Cycle + Stage * Schedule.getInitiationInterval()))) {
        Instrs[Cycle].push_front(SU);
      }
    }
  }

  for (int Cycle = Schedule.getFirstCycle(); Cycle <= Schedule.getFinalCycle();
       ++Cycle) {
    std::deque<SUnit *> &CycleInstrs = Instrs[Cycle];
    CycleInstrs = Schedule.reorderInstructions(SSD, CycleInstrs);
    for (SUnit *SU : CycleInstrs) {
      MachineInstr *MI = SU->getInstr();
      OrderedInsts.push_back(MI);
      Stages[MI] = Schedule.stageScheduled(SU);
    }
  }
}

namespace {

// FuncUnitSorter - Comparison operator used to sort instructions by
// the number of functional unit choices.
struct FuncUnitSorter {
  const InstrItineraryData *InstrItins;
  const MCSubtargetInfo *STI;
  DenseMap<InstrStage::FuncUnits, unsigned> Resources;

  FuncUnitSorter(const TargetSubtargetInfo &TSI)
      : InstrItins(TSI.getInstrItineraryData()), STI(&TSI) {}

  // Compute the number of functional unit alternatives needed
  // at each stage, and take the minimum value. We prioritize the
  // instructions by the least number of choices first.
  unsigned minFuncUnits(const MachineInstr *Inst,
                        InstrStage::FuncUnits &F) const {
    unsigned SchedClass = Inst->getDesc().getSchedClass();
    unsigned min = UINT_MAX;
    if (InstrItins && !InstrItins->isEmpty()) {
      for (const InstrStage &IS :
           make_range(InstrItins->beginStage(SchedClass),
                      InstrItins->endStage(SchedClass))) {
        InstrStage::FuncUnits funcUnits = IS.getUnits();
        unsigned numAlternatives = llvm::popcount(funcUnits);
        if (numAlternatives < min) {
          min = numAlternatives;
          F = funcUnits;
        }
      }
      return min;
    }
    if (STI && STI->getSchedModel().hasInstrSchedModel()) {
      const MCSchedClassDesc *SCDesc =
          STI->getSchedModel().getSchedClassDesc(SchedClass);
      if (!SCDesc->isValid())
        // No valid Schedule Class Desc for schedClass, should be
        // Pseudo/PostRAPseudo
        return min;

      for (const MCWriteProcResEntry &PRE :
           make_range(STI->getWriteProcResBegin(SCDesc),
                      STI->getWriteProcResEnd(SCDesc))) {
        if (!PRE.ReleaseAtCycle)
          continue;
        const MCProcResourceDesc *ProcResource =
            STI->getSchedModel().getProcResource(PRE.ProcResourceIdx);
        unsigned NumUnits = ProcResource->NumUnits;
        if (NumUnits < min) {
          min = NumUnits;
          F = PRE.ProcResourceIdx;
        }
      }
      return min;
    }
    llvm_unreachable("Should have non-empty InstrItins or hasInstrSchedModel!");
  }

  // Compute the critical resources needed by the instruction. This
  // function records the functional units needed by instructions that
  // must use only one functional unit. We use this as a tie breaker
  // for computing the resource MII. The instrutions that require
  // the same, highly used, functional unit have high priority.
  void calcCriticalResources(MachineInstr &MI) {
    unsigned SchedClass = MI.getDesc().getSchedClass();
    if (InstrItins && !InstrItins->isEmpty()) {
      for (const InstrStage &IS :
           make_range(InstrItins->beginStage(SchedClass),
                      InstrItins->endStage(SchedClass))) {
        InstrStage::FuncUnits FuncUnits = IS.getUnits();
        if (llvm::popcount(FuncUnits) == 1)
          Resources[FuncUnits]++;
      }
      return;
    }
    if (STI && STI->getSchedModel().hasInstrSchedModel()) {
      const MCSchedClassDesc *SCDesc =
          STI->getSchedModel().getSchedClassDesc(SchedClass);
      if (!SCDesc->isValid())
        // No valid Schedule Class Desc for schedClass, should be
        // Pseudo/PostRAPseudo
        return;

      for (const MCWriteProcResEntry &PRE :
           make_range(STI->getWriteProcResBegin(SCDesc),
                      STI->getWriteProcResEnd(SCDesc))) {
        if (!PRE.ReleaseAtCycle)
          continue;
        Resources[PRE.ProcResourceIdx]++;
      }
      return;
    }
    llvm_unreachable("Should have non-empty InstrItins or hasInstrSchedModel!");
  }

  /// Return true if IS1 has less priority than IS2.
  bool operator()(const MachineInstr *IS1, const MachineInstr *IS2) const {
    InstrStage::FuncUnits F1 = 0, F2 = 0;
    unsigned MFUs1 = minFuncUnits(IS1, F1);
    unsigned MFUs2 = minFuncUnits(IS2, F2);
    if (MFUs1 == MFUs2)
      return Resources.lookup(F1) < Resources.lookup(F2);
    return MFUs1 > MFUs2;
  }
};

/// Calculate the maximum register pressure of the scheduled instructions stream
class HighRegisterPressureDetector {
  MachineBasicBlock *OrigMBB;
  const MachineRegisterInfo &MRI;
  const TargetRegisterInfo *TRI;

  const unsigned PSetNum;

  // Indexed by PSet ID
  // InitSetPressure takes into account the register pressure of live-in
  // registers. It's not depend on how the loop is scheduled, so it's enough to
  // calculate them once at the beginning.
  std::vector<unsigned> InitSetPressure;

  // Indexed by PSet ID
  // Upper limit for each register pressure set
  std::vector<unsigned> PressureSetLimit;

  DenseMap<MachineInstr *, RegisterOperands> ROMap;

  using Instr2LastUsesTy = DenseMap<MachineInstr *, SmallDenseSet<Register, 4>>;

public:
  using OrderedInstsTy = std::vector<MachineInstr *>;
  using Instr2StageTy = DenseMap<MachineInstr *, unsigned>;

private:
  static void dumpRegisterPressures(const std::vector<unsigned> &Pressures) {
    if (Pressures.size() == 0) {
      dbgs() << "[]";
    } else {
      char Prefix = '[';
      for (unsigned P : Pressures) {
        dbgs() << Prefix << P;
        Prefix = ' ';
      }
      dbgs() << ']';
    }
  }

  void dumpPSet(Register Reg) const {
    dbgs() << "Reg=" << printReg(Reg, TRI, 0, &MRI) << " PSet=";
    for (auto PSetIter = MRI.getPressureSets(Reg); PSetIter.isValid();
         ++PSetIter) {
      dbgs() << *PSetIter << ' ';
    }
    dbgs() << '\n';
  }

  void increaseRegisterPressure(std::vector<unsigned> &Pressure,
                                Register Reg) const {
    auto PSetIter = MRI.getPressureSets(Reg);
    unsigned Weight = PSetIter.getWeight();
    for (; PSetIter.isValid(); ++PSetIter)
      Pressure[*PSetIter] += Weight;
  }

  void decreaseRegisterPressure(std::vector<unsigned> &Pressure,
                                Register Reg) const {
    auto PSetIter = MRI.getPressureSets(Reg);
    unsigned Weight = PSetIter.getWeight();
    for (; PSetIter.isValid(); ++PSetIter) {
      auto &P = Pressure[*PSetIter];
      assert(P >= Weight &&
             "register pressure must be greater than or equal weight");
      P -= Weight;
    }
  }

  // Return true if Reg is reserved one, for example, stack pointer
  bool isReservedRegister(Register Reg) const {
    return Reg.isPhysical() && MRI.isReserved(Reg.asMCReg());
  }

  bool isDefinedInThisLoop(Register Reg) const {
    return Reg.isVirtual() && MRI.getVRegDef(Reg)->getParent() == OrigMBB;
  }

  // Search for live-in variables. They are factored into the register pressure
  // from the begining. Live-in variables used by every iteration should be
  // considered as alive throughout the loop. For example, the variable `c` in
  // following code. \code
  //   int c = ...;
  //   for (int i = 0; i < n; i++)
  //     a[i] += b[i] + c;
  // \endcode
  void computeLiveIn() {
    DenseSet<Register> Used;
    for (auto &MI : *OrigMBB) {
      if (MI.isDebugInstr())
        continue;
      for (auto &Use : ROMap[&MI].Uses) {
        auto Reg = Use.RegUnit;
        // Ignore the variable that appears only on one side of phi instruction
        // because it's used only at the first iteration.
        if (MI.isPHI() && Reg != getLoopPhiReg(MI, OrigMBB))
          continue;
        if (isReservedRegister(Reg))
          continue;
        if (isDefinedInThisLoop(Reg))
          continue;
        Used.insert(Reg);
      }
    }

    for (auto LiveIn : Used)
      increaseRegisterPressure(InitSetPressure, LiveIn);
  }

  // Calculate the upper limit of each pressure set
  void computePressureSetLimit(const RegisterClassInfo &RCI) {
    for (unsigned PSet = 0; PSet < PSetNum; PSet++)
      PressureSetLimit[PSet] = RCI.getRegPressureSetLimit(PSet);
  }

  // There are two patterns of last-use.
  //   - by an instruction of the current iteration
  //   - by a phi instruction of the next iteration (loop carried value)
  //
  // Furthermore, following two groups of instructions are executed
  // simultaneously
  //   - next iteration's phi instructions in i-th stage
  //   - current iteration's instructions in i+1-th stage
  //
  // This function calculates the last-use of each register while taking into
  // account the above two patterns.
  Instr2LastUsesTy computeLastUses(const OrderedInstsTy &OrderedInsts,
                                   Instr2StageTy &Stages) const {
    // We treat virtual registers that are defined and used in this loop.
    // Following virtual register will be ignored
    //   - live-in one
    //   - defined but not used in the loop (potentially live-out)
    DenseSet<Register> TargetRegs;
    const auto UpdateTargetRegs = [this, &TargetRegs](Register Reg) {
      if (isDefinedInThisLoop(Reg))
        TargetRegs.insert(Reg);
    };
    for (MachineInstr *MI : OrderedInsts) {
      if (MI->isPHI()) {
        Register Reg = getLoopPhiReg(*MI, OrigMBB);
        UpdateTargetRegs(Reg);
      } else {
        for (auto &Use : ROMap.find(MI)->getSecond().Uses)
          UpdateTargetRegs(Use.RegUnit);
      }
    }

    const auto InstrScore = [&Stages](MachineInstr *MI) {
      return Stages[MI] + MI->isPHI();
    };

    DenseMap<Register, MachineInstr *> LastUseMI;
    for (MachineInstr *MI : llvm::reverse(OrderedInsts)) {
      for (auto &Use : ROMap.find(MI)->getSecond().Uses) {
        auto Reg = Use.RegUnit;
        if (!TargetRegs.contains(Reg))
          continue;
        auto [Ite, Inserted] = LastUseMI.try_emplace(Reg, MI);
        if (!Inserted) {
          MachineInstr *Orig = Ite->second;
          MachineInstr *New = MI;
          if (InstrScore(Orig) < InstrScore(New))
            Ite->second = New;
        }
      }
    }

    Instr2LastUsesTy LastUses;
    for (auto &Entry : LastUseMI)
      LastUses[Entry.second].insert(Entry.first);
    return LastUses;
  }

  // Compute the maximum register pressure of the kernel. We'll simulate #Stage
  // iterations and check the register pressure at the point where all stages
  // overlapping.
  //
  // An example of unrolled loop where #Stage is 4..
  // Iter   i+0 i+1 i+2 i+3
  // ------------------------
  // Stage   0
  // Stage   1   0
  // Stage   2   1   0
  // Stage   3   2   1   0  <- All stages overlap
  //
  std::vector<unsigned>
  computeMaxSetPressure(const OrderedInstsTy &OrderedInsts,
                        Instr2StageTy &Stages,
                        const unsigned StageCount) const {
    using RegSetTy = SmallDenseSet<Register, 16>;

    // Indexed by #Iter. To treat "local" variables of each stage separately, we
    // manage the liveness of the registers independently by iterations.
    SmallVector<RegSetTy> LiveRegSets(StageCount);

    auto CurSetPressure = InitSetPressure;
    auto MaxSetPressure = InitSetPressure;
    auto LastUses = computeLastUses(OrderedInsts, Stages);

    LLVM_DEBUG({
      dbgs() << "Ordered instructions:\n";
      for (MachineInstr *MI : OrderedInsts) {
        dbgs() << "Stage " << Stages[MI] << ": ";
        MI->dump();
      }
    });

    const auto InsertReg = [this, &CurSetPressure](RegSetTy &RegSet,
                                                   Register Reg) {
      if (!Reg.isValid() || isReservedRegister(Reg))
        return;

      bool Inserted = RegSet.insert(Reg).second;
      if (!Inserted)
        return;

      LLVM_DEBUG(dbgs() << "insert " << printReg(Reg, TRI, 0, &MRI) << "\n");
      increaseRegisterPressure(CurSetPressure, Reg);
      LLVM_DEBUG(dumpPSet(Reg));
    };

    const auto EraseReg = [this, &CurSetPressure](RegSetTy &RegSet,
                                                  Register Reg) {
      if (!Reg.isValid() || isReservedRegister(Reg))
        return;

      // live-in register
      if (!RegSet.contains(Reg))
        return;

      LLVM_DEBUG(dbgs() << "erase " << printReg(Reg, TRI, 0, &MRI) << "\n");
      RegSet.erase(Reg);
      decreaseRegisterPressure(CurSetPressure, Reg);
      LLVM_DEBUG(dumpPSet(Reg));
    };

    for (unsigned I = 0; I < StageCount; I++) {
      for (MachineInstr *MI : OrderedInsts) {
        const auto Stage = Stages[MI];
        if (I < Stage)
          continue;

        const unsigned Iter = I - Stage;

        for (auto &Def : ROMap.find(MI)->getSecond().Defs)
          InsertReg(LiveRegSets[Iter], Def.RegUnit);

        for (auto LastUse : LastUses[MI]) {
          if (MI->isPHI()) {
            if (Iter != 0)
              EraseReg(LiveRegSets[Iter - 1], LastUse);
          } else {
            EraseReg(LiveRegSets[Iter], LastUse);
          }
        }

        for (unsigned PSet = 0; PSet < PSetNum; PSet++)
          MaxSetPressure[PSet] =
              std::max(MaxSetPressure[PSet], CurSetPressure[PSet]);

        LLVM_DEBUG({
          dbgs() << "CurSetPressure=";
          dumpRegisterPressures(CurSetPressure);
          dbgs() << " iter=" << Iter << " stage=" << Stage << ":";
          MI->dump();
        });
      }
    }

    return MaxSetPressure;
  }

public:
  HighRegisterPressureDetector(MachineBasicBlock *OrigMBB,
                               const MachineFunction &MF)
      : OrigMBB(OrigMBB), MRI(MF.getRegInfo()),
        TRI(MF.getSubtarget().getRegisterInfo()),
        PSetNum(TRI->getNumRegPressureSets()), InitSetPressure(PSetNum, 0),
        PressureSetLimit(PSetNum, 0) {}

  // Used to calculate register pressure, which is independent of loop
  // scheduling.
  void init(const RegisterClassInfo &RCI) {
    for (MachineInstr &MI : *OrigMBB) {
      if (MI.isDebugInstr())
        continue;
      ROMap[&MI].collect(MI, *TRI, MRI, false, true);
    }

    computeLiveIn();
    computePressureSetLimit(RCI);
  }

  // Calculate the maximum register pressures of the loop and check if they
  // exceed the limit
  bool detect(const SwingSchedulerDAG *SSD, SMSchedule &Schedule,
              const unsigned MaxStage) const {
    assert(0 <= RegPressureMargin && RegPressureMargin <= 100 &&
           "the percentage of the margin must be between 0 to 100");

    OrderedInstsTy OrderedInsts;
    Instr2StageTy Stages;
    computeScheduledInsts(SSD, Schedule, OrderedInsts, Stages);
    const auto MaxSetPressure =
        computeMaxSetPressure(OrderedInsts, Stages, MaxStage + 1);

    LLVM_DEBUG({
      dbgs() << "Dump MaxSetPressure:\n";
      for (unsigned I = 0; I < MaxSetPressure.size(); I++) {
        dbgs() << format("MaxSetPressure[%d]=%d\n", I, MaxSetPressure[I]);
      }
      dbgs() << '\n';
    });

    for (unsigned PSet = 0; PSet < PSetNum; PSet++) {
      unsigned Limit = PressureSetLimit[PSet];
      unsigned Margin = Limit * RegPressureMargin / 100;
      LLVM_DEBUG(dbgs() << "PSet=" << PSet << " Limit=" << Limit
                        << " Margin=" << Margin << "\n");
      if (Limit < MaxSetPressure[PSet] + Margin) {
        LLVM_DEBUG(
            dbgs()
            << "Rejected the schedule because of too high register pressure\n");
        return true;
      }
    }
    return false;
  }
};

} // end anonymous namespace

/// Calculate the resource constrained minimum initiation interval for the
/// specified loop. We use the DFA to model the resources needed for
/// each instruction, and we ignore dependences. A different DFA is created
/// for each cycle that is required. When adding a new instruction, we attempt
/// to add it to each existing DFA, until a legal space is found. If the
/// instruction cannot be reserved in an existing DFA, we create a new one.
unsigned SwingSchedulerDAG::calculateResMII() {
  LLVM_DEBUG(dbgs() << "calculateResMII:\n");
  ResourceManager RM(&MF.getSubtarget(), this);
  return RM.calculateResMII();
}

/// Calculate the recurrence-constrainted minimum initiation interval.
/// Iterate over each circuit.  Compute the delay(c) and distance(c)
/// for each circuit. The II needs to satisfy the inequality
/// delay(c) - II*distance(c) <= 0. For each circuit, choose the smallest
/// II that satisfies the inequality, and the RecMII is the maximum
/// of those values.
unsigned SwingSchedulerDAG::calculateRecMII(NodeSetType &NodeSets) {
  unsigned RecMII = 0;

  for (NodeSet &Nodes : NodeSets) {
    if (Nodes.empty())
      continue;

    unsigned Delay = Nodes.getLatency();
    unsigned Distance = 1;

    // ii = ceil(delay / distance)
    unsigned CurMII = (Delay + Distance - 1) / Distance;
    Nodes.setRecMII(CurMII);
    if (CurMII > RecMII)
      RecMII = CurMII;
  }

  return RecMII;
}

/// Create the adjacency structure of the nodes in the graph.
void SwingSchedulerDAG::Circuits::createAdjacencyStructure(
    SwingSchedulerDAG *DAG) {
  BitVector Added(SUnits.size());
  DenseMap<int, int> OutputDeps;
  for (int i = 0, e = SUnits.size(); i != e; ++i) {
    Added.reset();
    // Add any successor to the adjacency matrix and exclude duplicates.
    for (auto &OE : DAG->DDG->getOutEdges(&SUnits[i])) {
      // Only create a back-edge on the first and last nodes of a dependence
      // chain. This records any chains and adds them later.
      if (OE.isOutputDep()) {
        int N = OE.getDst()->NodeNum;
        int BackEdge = i;
        auto Dep = OutputDeps.find(BackEdge);
        if (Dep != OutputDeps.end()) {
          BackEdge = Dep->second;
          OutputDeps.erase(Dep);
        }
        OutputDeps[N] = BackEdge;
      }
      // Do not process a boundary node, an artificial node.
      if (OE.getDst()->isBoundaryNode() || OE.isArtificial())
        continue;

      // This code is retained o preserve previous behavior and prevent
      // regression. This condition means that anti-dependnecies within an
      // iteration are ignored when searching circuits. Therefore it's natural
      // to consider this dependence as well.
      // FIXME: Remove this code if it doesn't have significant impact on
      // performance.
      if (OE.isAntiDep())
        continue;

      int N = OE.getDst()->NodeNum;
      if (!Added.test(N)) {
        AdjK[i].push_back(N);
        Added.set(N);
      }
    }
    // A chain edge between a store and a load is treated as a back-edge in the
    // adjacency matrix.
    for (auto &IE : DAG->DDG->getInEdges(&SUnits[i])) {
      SUnit *Src = IE.getSrc();
      SUnit *Dst = IE.getDst();
      if (!Dst->getInstr()->mayStore() || !DAG->isLoopCarriedDep(IE))
        continue;
      if (IE.isOrderDep() && Src->getInstr()->mayLoad()) {
        int N = Src->NodeNum;
        if (!Added.test(N)) {
          AdjK[i].push_back(N);
          Added.set(N);
        }
      }
    }
  }
  // Add back-edges in the adjacency matrix for the output dependences.
  for (auto &OD : OutputDeps)
    if (!Added.test(OD.second)) {
      AdjK[OD.first].push_back(OD.second);
      Added.set(OD.second);
    }
}

/// Identify an elementary circuit in the dependence graph starting at the
/// specified node.
bool SwingSchedulerDAG::Circuits::circuit(int V, int S, NodeSetType &NodeSets,
                                          const SwingSchedulerDAG *DAG,
                                          bool HasBackedge) {
  SUnit *SV = &SUnits[V];
  bool F = false;
  Stack.insert(SV);
  Blocked.set(V);

  for (auto W : AdjK[V]) {
    if (NumPaths > MaxPaths)
      break;
    if (W < S)
      continue;
    if (W == S) {
      if (!HasBackedge)
        NodeSets.push_back(NodeSet(Stack.begin(), Stack.end(), DAG));
      F = true;
      ++NumPaths;
      break;
    }
    if (!Blocked.test(W)) {
      if (circuit(W, S, NodeSets, DAG,
                  Node2Idx->at(W) < Node2Idx->at(V) ? true : HasBackedge))
        F = true;
    }
  }

  if (F)
    unblock(V);
  else {
    for (auto W : AdjK[V]) {
      if (W < S)
        continue;
      B[W].insert(SV);
    }
  }
  Stack.pop_back();
  return F;
}

/// Unblock a node in the circuit finding algorithm.
void SwingSchedulerDAG::Circuits::unblock(int U) {
  Blocked.reset(U);
  SmallPtrSet<SUnit *, 4> &BU = B[U];
  while (!BU.empty()) {
    SmallPtrSet<SUnit *, 4>::iterator SI = BU.begin();
    assert(SI != BU.end() && "Invalid B set.");
    SUnit *W = *SI;
    BU.erase(W);
    if (Blocked.test(W->NodeNum))
      unblock(W->NodeNum);
  }
}

/// Identify all the elementary circuits in the dependence graph using
/// Johnson's circuit algorithm.
void SwingSchedulerDAG::findCircuits(NodeSetType &NodeSets) {
  Circuits Cir(SUnits, Topo);
  // Create the adjacency structure.
  Cir.createAdjacencyStructure(this);
  for (int I = 0, E = SUnits.size(); I != E; ++I) {
    Cir.reset();
    Cir.circuit(I, I, NodeSets, this);
  }
}

// Create artificial dependencies between the source of COPY/REG_SEQUENCE that
// is loop-carried to the USE in next iteration. This will help pipeliner avoid
// additional copies that are needed across iterations. An artificial dependence
// edge is added from USE to SOURCE of COPY/REG_SEQUENCE.

// PHI-------Anti-Dep-----> COPY/REG_SEQUENCE (loop-carried)
// SRCOfCopY------True-Dep---> COPY/REG_SEQUENCE
// PHI-------True-Dep------> USEOfPhi

// The mutation creates
// USEOfPHI -------Artificial-Dep---> SRCOfCopy

// This overall will ensure, the USEOfPHI is scheduled before SRCOfCopy
// (since USE is a predecessor), implies, the COPY/ REG_SEQUENCE is scheduled
// late  to avoid additional copies across iterations. The possible scheduling
// order would be
// USEOfPHI --- SRCOfCopy---  COPY/REG_SEQUENCE.

void SwingSchedulerDAG::CopyToPhiMutation::apply(ScheduleDAGInstrs *DAG) {
  for (SUnit &SU : DAG->SUnits) {
    // Find the COPY/REG_SEQUENCE instruction.
    if (!SU.getInstr()->isCopy() && !SU.getInstr()->isRegSequence())
      continue;

    // Record the loop carried PHIs.
    SmallVector<SUnit *, 4> PHISUs;
    // Record the SrcSUs that feed the COPY/REG_SEQUENCE instructions.
    SmallVector<SUnit *, 4> SrcSUs;

    for (auto &Dep : SU.Preds) {
      SUnit *TmpSU = Dep.getSUnit();
      MachineInstr *TmpMI = TmpSU->getInstr();
      SDep::Kind DepKind = Dep.getKind();
      // Save the loop carried PHI.
      if (DepKind == SDep::Anti && TmpMI->isPHI())
        PHISUs.push_back(TmpSU);
      // Save the source of COPY/REG_SEQUENCE.
      // If the source has no pre-decessors, we will end up creating cycles.
      else if (DepKind == SDep::Data && !TmpMI->isPHI() && TmpSU->NumPreds > 0)
        SrcSUs.push_back(TmpSU);
    }

    if (PHISUs.size() == 0 || SrcSUs.size() == 0)
      continue;

    // Find the USEs of PHI. If the use is a PHI or REG_SEQUENCE, push back this
    // SUnit to the container.
    SmallVector<SUnit *, 8> UseSUs;
    // Do not use iterator based loop here as we are updating the container.
    for (size_t Index = 0; Index < PHISUs.size(); ++Index) {
      for (auto &Dep : PHISUs[Index]->Succs) {
        if (Dep.getKind() != SDep::Data)
          continue;

        SUnit *TmpSU = Dep.getSUnit();
        MachineInstr *TmpMI = TmpSU->getInstr();
        if (TmpMI->isPHI() || TmpMI->isRegSequence()) {
          PHISUs.push_back(TmpSU);
          continue;
        }
        UseSUs.push_back(TmpSU);
      }
    }

    if (UseSUs.size() == 0)
      continue;

    SwingSchedulerDAG *SDAG = cast<SwingSchedulerDAG>(DAG);
    // Add the artificial dependencies if it does not form a cycle.
    for (auto *I : UseSUs) {
      for (auto *Src : SrcSUs) {
        if (!SDAG->Topo.IsReachable(I, Src) && Src != I) {
          Src->addPred(SDep(I, SDep::Artificial));
          SDAG->Topo.AddPred(Src, I);
        }
      }
    }
  }
}

/// Compute several functions need to order the nodes for scheduling.
///  ASAP - Earliest time to schedule a node.
///  ALAP - Latest time to schedule a node.
///  MOV - Mobility function, difference between ALAP and ASAP.
///  D - Depth of each node.
///  H - Height of each node.
void SwingSchedulerDAG::computeNodeFunctions(NodeSetType &NodeSets) {
  ScheduleInfo.resize(SUnits.size());

  LLVM_DEBUG({
    for (int I : Topo) {
      const SUnit &SU = SUnits[I];
      dumpNode(SU);
    }
  });

  int maxASAP = 0;
  // Compute ASAP and ZeroLatencyDepth.
  for (int I : Topo) {
    int asap = 0;
    int zeroLatencyDepth = 0;
    SUnit *SU = &SUnits[I];
    for (const auto &IE : DDG->getInEdges(SU)) {
      SUnit *Pred = IE.getSrc();
      if (IE.getLatency() == 0)
        zeroLatencyDepth =
            std::max(zeroLatencyDepth, getZeroLatencyDepth(Pred) + 1);
      if (IE.ignoreDependence(true))
        continue;
      asap = std::max(asap, (int)(getASAP(Pred) + IE.getLatency() -
                                  IE.getDistance() * MII));
    }
    maxASAP = std::max(maxASAP, asap);
    ScheduleInfo[I].ASAP = asap;
    ScheduleInfo[I].ZeroLatencyDepth = zeroLatencyDepth;
  }

  // Compute ALAP, ZeroLatencyHeight, and MOV.
  for (int I : llvm::reverse(Topo)) {
    int alap = maxASAP;
    int zeroLatencyHeight = 0;
    SUnit *SU = &SUnits[I];
    for (const auto &OE : DDG->getOutEdges(SU)) {
      SUnit *Succ = OE.getDst();
      if (Succ->isBoundaryNode())
        continue;
      if (OE.getLatency() == 0)
        zeroLatencyHeight =
            std::max(zeroLatencyHeight, getZeroLatencyHeight(Succ) + 1);
      if (OE.ignoreDependence(true))
        continue;
      alap = std::min(alap, (int)(getALAP(Succ) - OE.getLatency() +
                                  OE.getDistance() * MII));
    }

    ScheduleInfo[I].ALAP = alap;
    ScheduleInfo[I].ZeroLatencyHeight = zeroLatencyHeight;
  }

  // After computing the node functions, compute the summary for each node set.
  for (NodeSet &I : NodeSets)
    I.computeNodeSetInfo(this);

  LLVM_DEBUG({
    for (unsigned i = 0; i < SUnits.size(); i++) {
      dbgs() << "\tNode " << i << ":\n";
      dbgs() << "\t   ASAP = " << getASAP(&SUnits[i]) << "\n";
      dbgs() << "\t   ALAP = " << getALAP(&SUnits[i]) << "\n";
      dbgs() << "\t   MOV  = " << getMOV(&SUnits[i]) << "\n";
      dbgs() << "\t   D    = " << getDepth(&SUnits[i]) << "\n";
      dbgs() << "\t   H    = " << getHeight(&SUnits[i]) << "\n";
      dbgs() << "\t   ZLD  = " << getZeroLatencyDepth(&SUnits[i]) << "\n";
      dbgs() << "\t   ZLH  = " << getZeroLatencyHeight(&SUnits[i]) << "\n";
    }
  });
}

/// Compute the Pred_L(O) set, as defined in the paper. The set is defined
/// as the predecessors of the elements of NodeOrder that are not also in
/// NodeOrder.
static bool pred_L(SetVector<SUnit *> &NodeOrder,
                   SmallSetVector<SUnit *, 8> &Preds, SwingSchedulerDDG *DDG,
                   const NodeSet *S = nullptr) {
  Preds.clear();

  for (SUnit *SU : NodeOrder) {
    for (const auto &IE : DDG->getInEdges(SU)) {
      SUnit *PredSU = IE.getSrc();
      if (S && S->count(PredSU) == 0)
        continue;
      if (IE.ignoreDependence(true))
        continue;
      if (NodeOrder.count(PredSU) == 0)
        Preds.insert(PredSU);
    }

    // FIXME: The following loop-carried dependencies may also need to be
    // considered.
    //   - Physical register dependencies (true-dependence and WAW).
    //   - Memory dependencies.
    for (const auto &OE : DDG->getOutEdges(SU)) {
      SUnit *SuccSU = OE.getDst();
      if (!OE.isAntiDep())
        continue;
      if (S && S->count(SuccSU) == 0)
        continue;
      if (NodeOrder.count(SuccSU) == 0)
        Preds.insert(SuccSU);
    }
  }
  return !Preds.empty();
}

/// Compute the Succ_L(O) set, as defined in the paper. The set is defined
/// as the successors of the elements of NodeOrder that are not also in
/// NodeOrder.
static bool succ_L(SetVector<SUnit *> &NodeOrder,
                   SmallSetVector<SUnit *, 8> &Succs, SwingSchedulerDDG *DDG,
                   const NodeSet *S = nullptr) {
  Succs.clear();

  for (SUnit *SU : NodeOrder) {
    for (const auto &OE : DDG->getOutEdges(SU)) {
      SUnit *SuccSU = OE.getDst();
      if (S && S->count(SuccSU) == 0)
        continue;
      if (OE.ignoreDependence(false))
        continue;
      if (NodeOrder.count(SuccSU) == 0)
        Succs.insert(SuccSU);
    }

    // FIXME: The following loop-carried dependencies may also need to be
    // considered.
    //   - Physical register dependnecies (true-dependnece and WAW).
    //   - Memory dependencies.
    for (const auto &IE : DDG->getInEdges(SU)) {
      SUnit *PredSU = IE.getSrc();
      if (!IE.isAntiDep())
        continue;
      if (S && S->count(PredSU) == 0)
        continue;
      if (NodeOrder.count(PredSU) == 0)
        Succs.insert(PredSU);
    }
  }
  return !Succs.empty();
}

/// Return true if there is a path from the specified node to any of the nodes
/// in DestNodes. Keep track and return the nodes in any path.
static bool computePath(SUnit *Cur, SetVector<SUnit *> &Path,
                        SetVector<SUnit *> &DestNodes,
                        SetVector<SUnit *> &Exclude,
                        SmallPtrSet<SUnit *, 8> &Visited,
                        SwingSchedulerDDG *DDG) {
  if (Cur->isBoundaryNode())
    return false;
  if (Exclude.contains(Cur))
    return false;
  if (DestNodes.contains(Cur))
    return true;
  if (!Visited.insert(Cur).second)
    return Path.contains(Cur);
  bool FoundPath = false;
  for (const auto &OE : DDG->getOutEdges(Cur))
    if (!OE.ignoreDependence(false))
      FoundPath |=
          computePath(OE.getDst(), Path, DestNodes, Exclude, Visited, DDG);
  for (const auto &IE : DDG->getInEdges(Cur))
    if (IE.isAntiDep() && IE.getDistance() == 0)
      FoundPath |=
          computePath(IE.getSrc(), Path, DestNodes, Exclude, Visited, DDG);
  if (FoundPath)
    Path.insert(Cur);
  return FoundPath;
}

/// Compute the live-out registers for the instructions in a node-set.
/// The live-out registers are those that are defined in the node-set,
/// but not used. Except for use operands of Phis.
static void computeLiveOuts(MachineFunction &MF, RegPressureTracker &RPTracker,
                            NodeSet &NS) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SmallVector<VRegMaskOrUnit, 8> LiveOutRegs;
  SmallSet<Register, 4> Uses;
  for (SUnit *SU : NS) {
    const MachineInstr *MI = SU->getInstr();
    if (MI->isPHI())
      continue;
    for (const MachineOperand &MO : MI->all_uses()) {
      Register Reg = MO.getReg();
      if (Reg.isVirtual())
        Uses.insert(Reg);
      else if (MRI.isAllocatable(Reg))
        Uses.insert_range(TRI->regunits(Reg.asMCReg()));
    }
  }
  for (SUnit *SU : NS)
    for (const MachineOperand &MO : SU->getInstr()->all_defs())
      if (!MO.isDead()) {
        Register Reg = MO.getReg();
        if (Reg.isVirtual()) {
          if (!Uses.count(Reg))
            LiveOutRegs.emplace_back(Reg, LaneBitmask::getNone());
        } else if (MRI.isAllocatable(Reg)) {
          for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
            if (!Uses.count(Unit))
              LiveOutRegs.emplace_back(Unit, LaneBitmask::getNone());
        }
      }
  RPTracker.addLiveRegs(LiveOutRegs);
}

/// A heuristic to filter nodes in recurrent node-sets if the register
/// pressure of a set is too high.
void SwingSchedulerDAG::registerPressureFilter(NodeSetType &NodeSets) {
  for (auto &NS : NodeSets) {
    // Skip small node-sets since they won't cause register pressure problems.
    if (NS.size() <= 2)
      continue;
    IntervalPressure RecRegPressure;
    RegPressureTracker RecRPTracker(RecRegPressure);
    RecRPTracker.init(&MF, &RegClassInfo, &LIS, BB, BB->end(), false, true);
    computeLiveOuts(MF, RecRPTracker, NS);
    RecRPTracker.closeBottom();

    std::vector<SUnit *> SUnits(NS.begin(), NS.end());
    llvm::sort(SUnits, [](const SUnit *A, const SUnit *B) {
      return A->NodeNum > B->NodeNum;
    });

    for (auto &SU : SUnits) {
      // Since we're computing the register pressure for a subset of the
      // instructions in a block, we need to set the tracker for each
      // instruction in the node-set. The tracker is set to the instruction
      // just after the one we're interested in.
      MachineBasicBlock::const_iterator CurInstI = SU->getInstr();
      RecRPTracker.setPos(std::next(CurInstI));

      RegPressureDelta RPDelta;
      ArrayRef<PressureChange> CriticalPSets;
      RecRPTracker.getMaxUpwardPressureDelta(SU->getInstr(), nullptr, RPDelta,
                                             CriticalPSets,
                                             RecRegPressure.MaxSetPressure);
      if (RPDelta.Excess.isValid()) {
        LLVM_DEBUG(
            dbgs() << "Excess register pressure: SU(" << SU->NodeNum << ") "
                   << TRI->getRegPressureSetName(RPDelta.Excess.getPSet())
                   << ":" << RPDelta.Excess.getUnitInc() << "\n");
        NS.setExceedPressure(SU);
        break;
      }
      RecRPTracker.recede();
    }
  }
}

/// A heuristic to colocate node sets that have the same set of
/// successors.
void SwingSchedulerDAG::colocateNodeSets(NodeSetType &NodeSets) {
  unsigned Colocate = 0;
  for (int i = 0, e = NodeSets.size(); i < e; ++i) {
    NodeSet &N1 = NodeSets[i];
    SmallSetVector<SUnit *, 8> S1;
    if (N1.empty() || !succ_L(N1, S1, DDG.get()))
      continue;
    for (int j = i + 1; j < e; ++j) {
      NodeSet &N2 = NodeSets[j];
      if (N1.compareRecMII(N2) != 0)
        continue;
      SmallSetVector<SUnit *, 8> S2;
      if (N2.empty() || !succ_L(N2, S2, DDG.get()))
        continue;
      if (llvm::set_is_subset(S1, S2) && S1.size() == S2.size()) {
        N1.setColocate(++Colocate);
        N2.setColocate(Colocate);
        break;
      }
    }
  }
}

/// Check if the existing node-sets are profitable. If not, then ignore the
/// recurrent node-sets, and attempt to schedule all nodes together. This is
/// a heuristic. If the MII is large and all the recurrent node-sets are small,
/// then it's best to try to schedule all instructions together instead of
/// starting with the recurrent node-sets.
void SwingSchedulerDAG::checkNodeSets(NodeSetType &NodeSets) {
  // Look for loops with a large MII.
  if (MII < 17)
    return;
  // Check if the node-set contains only a simple add recurrence.
  for (auto &NS : NodeSets) {
    if (NS.getRecMII() > 2)
      return;
    if (NS.getMaxDepth() > MII)
      return;
  }
  NodeSets.clear();
  LLVM_DEBUG(dbgs() << "Clear recurrence node-sets\n");
}

/// Add the nodes that do not belong to a recurrence set into groups
/// based upon connected components.
void SwingSchedulerDAG::groupRemainingNodes(NodeSetType &NodeSets) {
  SetVector<SUnit *> NodesAdded;
  SmallPtrSet<SUnit *, 8> Visited;
  // Add the nodes that are on a path between the previous node sets and
  // the current node set.
  for (NodeSet &I : NodeSets) {
    SmallSetVector<SUnit *, 8> N;
    // Add the nodes from the current node set to the previous node set.
    if (succ_L(I, N, DDG.get())) {
      SetVector<SUnit *> Path;
      for (SUnit *NI : N) {
        Visited.clear();
        computePath(NI, Path, NodesAdded, I, Visited, DDG.get());
      }
      if (!Path.empty())
        I.insert(Path.begin(), Path.end());
    }
    // Add the nodes from the previous node set to the current node set.
    N.clear();
    if (succ_L(NodesAdded, N, DDG.get())) {
      SetVector<SUnit *> Path;
      for (SUnit *NI : N) {
        Visited.clear();
        computePath(NI, Path, I, NodesAdded, Visited, DDG.get());
      }
      if (!Path.empty())
        I.insert(Path.begin(), Path.end());
    }
    NodesAdded.insert_range(I);
  }

  // Create a new node set with the connected nodes of any successor of a node
  // in a recurrent set.
  NodeSet NewSet;
  SmallSetVector<SUnit *, 8> N;
  if (succ_L(NodesAdded, N, DDG.get()))
    for (SUnit *I : N)
      addConnectedNodes(I, NewSet, NodesAdded);
  if (!NewSet.empty())
    NodeSets.push_back(NewSet);

  // Create a new node set with the connected nodes of any predecessor of a node
  // in a recurrent set.
  NewSet.clear();
  if (pred_L(NodesAdded, N, DDG.get()))
    for (SUnit *I : N)
      addConnectedNodes(I, NewSet, NodesAdded);
  if (!NewSet.empty())
    NodeSets.push_back(NewSet);

  // Create new nodes sets with the connected nodes any remaining node that
  // has no predecessor.
  for (SUnit &SU : SUnits) {
    if (NodesAdded.count(&SU) == 0) {
      NewSet.clear();
      addConnectedNodes(&SU, NewSet, NodesAdded);
      if (!NewSet.empty())
        NodeSets.push_back(NewSet);
    }
  }
}

/// Add the node to the set, and add all of its connected nodes to the set.
void SwingSchedulerDAG::addConnectedNodes(SUnit *SU, NodeSet &NewSet,
                                          SetVector<SUnit *> &NodesAdded) {
  NewSet.insert(SU);
  NodesAdded.insert(SU);
  for (auto &OE : DDG->getOutEdges(SU)) {
    SUnit *Successor = OE.getDst();
    if (!OE.isArtificial() && !Successor->isBoundaryNode() &&
        NodesAdded.count(Successor) == 0)
      addConnectedNodes(Successor, NewSet, NodesAdded);
  }
  for (auto &IE : DDG->getInEdges(SU)) {
    SUnit *Predecessor = IE.getSrc();
    if (!IE.isArtificial() && NodesAdded.count(Predecessor) == 0)
      addConnectedNodes(Predecessor, NewSet, NodesAdded);
  }
}

/// Return true if Set1 contains elements in Set2. The elements in common
/// are returned in a different container.
static bool isIntersect(SmallSetVector<SUnit *, 8> &Set1, const NodeSet &Set2,
                        SmallSetVector<SUnit *, 8> &Result) {
  Result.clear();
  for (SUnit *SU : Set1) {
    if (Set2.count(SU) != 0)
      Result.insert(SU);
  }
  return !Result.empty();
}

/// Merge the recurrence node sets that have the same initial node.
void SwingSchedulerDAG::fuseRecs(NodeSetType &NodeSets) {
  for (NodeSetType::iterator I = NodeSets.begin(), E = NodeSets.end(); I != E;
       ++I) {
    NodeSet &NI = *I;
    for (NodeSetType::iterator J = I + 1; J != E;) {
      NodeSet &NJ = *J;
      if (NI.getNode(0)->NodeNum == NJ.getNode(0)->NodeNum) {
        if (NJ.compareRecMII(NI) > 0)
          NI.setRecMII(NJ.getRecMII());
        for (SUnit *SU : *J)
          I->insert(SU);
        NodeSets.erase(J);
        E = NodeSets.end();
      } else {
        ++J;
      }
    }
  }
}

/// Remove nodes that have been scheduled in previous NodeSets.
void SwingSchedulerDAG::removeDuplicateNodes(NodeSetType &NodeSets) {
  for (NodeSetType::iterator I = NodeSets.begin(), E = NodeSets.end(); I != E;
       ++I)
    for (NodeSetType::iterator J = I + 1; J != E;) {
      J->remove_if([&](SUnit *SUJ) { return I->count(SUJ); });

      if (J->empty()) {
        NodeSets.erase(J);
        E = NodeSets.end();
      } else {
        ++J;
      }
    }
}

/// Compute an ordered list of the dependence graph nodes, which
/// indicates the order that the nodes will be scheduled.  This is a
/// two-level algorithm. First, a partial order is created, which
/// consists of a list of sets ordered from highest to lowest priority.
void SwingSchedulerDAG::computeNodeOrder(NodeSetType &NodeSets) {
  SmallSetVector<SUnit *, 8> R;
  NodeOrder.clear();

  for (auto &Nodes : NodeSets) {
    LLVM_DEBUG(dbgs() << "NodeSet size " << Nodes.size() << "\n");
    OrderKind Order;
    SmallSetVector<SUnit *, 8> N;
    if (pred_L(NodeOrder, N, DDG.get()) && llvm::set_is_subset(N, Nodes)) {
      R.insert_range(N);
      Order = BottomUp;
      LLVM_DEBUG(dbgs() << "  Bottom up (preds) ");
    } else if (succ_L(NodeOrder, N, DDG.get()) &&
               llvm::set_is_subset(N, Nodes)) {
      R.insert_range(N);
      Order = TopDown;
      LLVM_DEBUG(dbgs() << "  Top down (succs) ");
    } else if (isIntersect(N, Nodes, R)) {
      // If some of the successors are in the existing node-set, then use the
      // top-down ordering.
      Order = TopDown;
      LLVM_DEBUG(dbgs() << "  Top down (intersect) ");
    } else if (NodeSets.size() == 1) {
      for (const auto &N : Nodes)
        if (N->Succs.size() == 0)
          R.insert(N);
      Order = BottomUp;
      LLVM_DEBUG(dbgs() << "  Bottom up (all) ");
    } else {
      // Find the node with the highest ASAP.
      SUnit *maxASAP = nullptr;
      for (SUnit *SU : Nodes) {
        if (maxASAP == nullptr || getASAP(SU) > getASAP(maxASAP) ||
            (getASAP(SU) == getASAP(maxASAP) && SU->NodeNum > maxASAP->NodeNum))
          maxASAP = SU;
      }
      R.insert(maxASAP);
      Order = BottomUp;
      LLVM_DEBUG(dbgs() << "  Bottom up (default) ");
    }

    while (!R.empty()) {
      if (Order == TopDown) {
        // Choose the node with the maximum height.  If more than one, choose
        // the node wiTH the maximum ZeroLatencyHeight. If still more than one,
        // choose the node with the lowest MOV.
        while (!R.empty()) {
          SUnit *maxHeight = nullptr;
          for (SUnit *I : R) {
            if (maxHeight == nullptr || getHeight(I) > getHeight(maxHeight))
              maxHeight = I;
            else if (getHeight(I) == getHeight(maxHeight) &&
                     getZeroLatencyHeight(I) > getZeroLatencyHeight(maxHeight))
              maxHeight = I;
            else if (getHeight(I) == getHeight(maxHeight) &&
                     getZeroLatencyHeight(I) ==
                         getZeroLatencyHeight(maxHeight) &&
                     getMOV(I) < getMOV(maxHeight))
              maxHeight = I;
          }
          NodeOrder.insert(maxHeight);
          LLVM_DEBUG(dbgs() << maxHeight->NodeNum << " ");
          R.remove(maxHeight);
          for (const auto &OE : DDG->getOutEdges(maxHeight)) {
            SUnit *SU = OE.getDst();
            if (Nodes.count(SU) == 0)
              continue;
            if (NodeOrder.contains(SU))
              continue;
            if (OE.ignoreDependence(false))
              continue;
            R.insert(SU);
          }

          // FIXME: The following loop-carried dependencies may also need to be
          // considered.
          //   - Physical register dependnecies (true-dependnece and WAW).
          //   - Memory dependencies.
          for (const auto &IE : DDG->getInEdges(maxHeight)) {
            SUnit *SU = IE.getSrc();
            if (!IE.isAntiDep())
              continue;
            if (Nodes.count(SU) == 0)
              continue;
            if (NodeOrder.contains(SU))
              continue;
            R.insert(SU);
          }
        }
        Order = BottomUp;
        LLVM_DEBUG(dbgs() << "\n   Switching order to bottom up ");
        SmallSetVector<SUnit *, 8> N;
        if (pred_L(NodeOrder, N, DDG.get(), &Nodes))
          R.insert_range(N);
      } else {
        // Choose the node with the maximum depth.  If more than one, choose
        // the node with the maximum ZeroLatencyDepth. If still more than one,
        // choose the node with the lowest MOV.
        while (!R.empty()) {
          SUnit *maxDepth = nullptr;
          for (SUnit *I : R) {
            if (maxDepth == nullptr || getDepth(I) > getDepth(maxDepth))
              maxDepth = I;
            else if (getDepth(I) == getDepth(maxDepth) &&
                     getZeroLatencyDepth(I) > getZeroLatencyDepth(maxDepth))
              maxDepth = I;
            else if (getDepth(I) == getDepth(maxDepth) &&
                     getZeroLatencyDepth(I) == getZeroLatencyDepth(maxDepth) &&
                     getMOV(I) < getMOV(maxDepth))
              maxDepth = I;
          }
          NodeOrder.insert(maxDepth);
          LLVM_DEBUG(dbgs() << maxDepth->NodeNum << " ");
          R.remove(maxDepth);
          if (Nodes.isExceedSU(maxDepth)) {
            Order = TopDown;
            R.clear();
            R.insert(Nodes.getNode(0));
            break;
          }
          for (const auto &IE : DDG->getInEdges(maxDepth)) {
            SUnit *SU = IE.getSrc();
            if (Nodes.count(SU) == 0)
              continue;
            if (NodeOrder.contains(SU))
              continue;
            R.insert(SU);
          }

          // FIXME: The following loop-carried dependencies may also need to be
          // considered.
          //   - Physical register dependnecies (true-dependnece and WAW).
          //   - Memory dependencies.
          for (const auto &OE : DDG->getOutEdges(maxDepth)) {
            SUnit *SU = OE.getDst();
            if (!OE.isAntiDep())
              continue;
            if (Nodes.count(SU) == 0)
              continue;
            if (NodeOrder.contains(SU))
              continue;
            R.insert(SU);
          }
        }
        Order = TopDown;
        LLVM_DEBUG(dbgs() << "\n   Switching order to top down ");
        SmallSetVector<SUnit *, 8> N;
        if (succ_L(NodeOrder, N, DDG.get(), &Nodes))
          R.insert_range(N);
      }
    }
    LLVM_DEBUG(dbgs() << "\nDone with Nodeset\n");
  }

  LLVM_DEBUG({
    dbgs() << "Node order: ";
    for (SUnit *I : NodeOrder)
      dbgs() << " " << I->NodeNum << " ";
    dbgs() << "\n";
  });
}

/// Process the nodes in the computed order and create the pipelined schedule
/// of the instructions, if possible. Return true if a schedule is found.
bool SwingSchedulerDAG::schedulePipeline(SMSchedule &Schedule) {

  if (NodeOrder.empty()){
    LLVM_DEBUG(dbgs() << "NodeOrder is empty! abort scheduling\n" );
    return false;
  }

  bool scheduleFound = false;
  std::unique_ptr<HighRegisterPressureDetector> HRPDetector;
  if (LimitRegPressure) {
    HRPDetector =
        std::make_unique<HighRegisterPressureDetector>(Loop.getHeader(), MF);
    HRPDetector->init(RegClassInfo);
  }
  // Keep increasing II until a valid schedule is found.
  for (unsigned II = MII; II <= MAX_II && !scheduleFound; ++II) {
    Schedule.reset();
    Schedule.setInitiationInterval(II);
    LLVM_DEBUG(dbgs() << "Try to schedule with " << II << "\n");

    SetVector<SUnit *>::iterator NI = NodeOrder.begin();
    SetVector<SUnit *>::iterator NE = NodeOrder.end();
    do {
      SUnit *SU = *NI;

      // Compute the schedule time for the instruction, which is based
      // upon the scheduled time for any predecessors/successors.
      int EarlyStart = INT_MIN;
      int LateStart = INT_MAX;
      Schedule.computeStart(SU, &EarlyStart, &LateStart, II, this);
      LLVM_DEBUG({
        dbgs() << "\n";
        dbgs() << "Inst (" << SU->NodeNum << ") ";
        SU->getInstr()->dump();
        dbgs() << "\n";
      });
      LLVM_DEBUG(
          dbgs() << format("\tes: %8x ls: %8x\n", EarlyStart, LateStart));

      if (EarlyStart > LateStart)
        scheduleFound = false;
      else if (EarlyStart != INT_MIN && LateStart == INT_MAX)
        scheduleFound =
            Schedule.insert(SU, EarlyStart, EarlyStart + (int)II - 1, II);
      else if (EarlyStart == INT_MIN && LateStart != INT_MAX)
        scheduleFound =
            Schedule.insert(SU, LateStart, LateStart - (int)II + 1, II);
      else if (EarlyStart != INT_MIN && LateStart != INT_MAX) {
        LateStart = std::min(LateStart, EarlyStart + (int)II - 1);
        // When scheduling a Phi it is better to start at the late cycle and
        // go backwards. The default order may insert the Phi too far away
        // from its first dependence.
        // Also, do backward search when all scheduled predecessors are
        // loop-carried output/order dependencies. Empirically, there are also
        // cases where scheduling becomes possible with backward search.
        if (SU->getInstr()->isPHI() ||
            Schedule.onlyHasLoopCarriedOutputOrOrderPreds(SU, this->getDDG()))
          scheduleFound = Schedule.insert(SU, LateStart, EarlyStart, II);
        else
          scheduleFound = Schedule.insert(SU, EarlyStart, LateStart, II);
      } else {
        int FirstCycle = Schedule.getFirstCycle();
        scheduleFound = Schedule.insert(SU, FirstCycle + getASAP(SU),
                                        FirstCycle + getASAP(SU) + II - 1, II);
      }

      // Even if we find a schedule, make sure the schedule doesn't exceed the
      // allowable number of stages. We keep trying if this happens.
      if (scheduleFound)
        if (SwpMaxStages > -1 &&
            Schedule.getMaxStageCount() > (unsigned)SwpMaxStages)
          scheduleFound = false;

      LLVM_DEBUG({
        if (!scheduleFound)
          dbgs() << "\tCan't schedule\n";
      });
    } while (++NI != NE && scheduleFound);

    // If a schedule is found, validate it against the validation-only
    // dependencies.
    if (scheduleFound)
      scheduleFound = DDG->isValidSchedule(Schedule);

    // If a schedule is found, ensure non-pipelined instructions are in stage 0
    if (scheduleFound)
      scheduleFound =
          Schedule.normalizeNonPipelinedInstructions(this, LoopPipelinerInfo);

    // If a schedule is found, check if it is a valid schedule too.
    if (scheduleFound)
      scheduleFound = Schedule.isValidSchedule(this);

    // If a schedule was found and the option is enabled, check if the schedule
    // might generate additional register spills/fills.
    if (scheduleFound && LimitRegPressure)
      scheduleFound =
          !HRPDetector->detect(this, Schedule, Schedule.getMaxStageCount());
  }

  LLVM_DEBUG(dbgs() << "Schedule Found? " << scheduleFound
                    << " (II=" << Schedule.getInitiationInterval()
                    << ")\n");

  if (scheduleFound) {
    scheduleFound = LoopPipelinerInfo->shouldUseSchedule(*this, Schedule);
    if (!scheduleFound)
      LLVM_DEBUG(dbgs() << "Target rejected schedule\n");
  }

  if (scheduleFound) {
    Schedule.finalizeSchedule(this);
    Pass.ORE->emit([&]() {
      return MachineOptimizationRemarkAnalysis(
                 DEBUG_TYPE, "schedule", Loop.getStartLoc(), Loop.getHeader())
             << "Schedule found with Initiation Interval: "
             << ore::NV("II", Schedule.getInitiationInterval())
             << ", MaxStageCount: "
             << ore::NV("MaxStageCount", Schedule.getMaxStageCount());
    });
  } else
    Schedule.reset();

  return scheduleFound && Schedule.getMaxStageCount() > 0;
}

static Register findUniqueOperandDefinedInLoop(const MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  Register Result;
  for (const MachineOperand &Use : MI.all_uses()) {
    Register Reg = Use.getReg();
    if (!Reg.isVirtual())
      return Register();
    if (MRI.getVRegDef(Reg)->getParent() != MI.getParent())
      continue;
    if (Result)
      return Register();
    Result = Reg;
  }
  return Result;
}

/// When Op is a value that is incremented recursively in a loop and there is a
/// unique instruction that increments it, returns true and sets Value.
static bool findLoopIncrementValue(const MachineOperand &Op, int &Value) {
  if (!Op.isReg() || !Op.getReg().isVirtual())
    return false;

  Register OrgReg = Op.getReg();
  Register CurReg = OrgReg;
  const MachineBasicBlock *LoopBB = Op.getParent()->getParent();
  const MachineRegisterInfo &MRI = LoopBB->getParent()->getRegInfo();

  const TargetInstrInfo *TII =
      LoopBB->getParent()->getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI =
      LoopBB->getParent()->getSubtarget().getRegisterInfo();

  MachineInstr *Phi = nullptr;
  MachineInstr *Increment = nullptr;

  // Traverse definitions until it reaches Op or an instruction that does not
  // satisfy the condition.
  // Acceptable example:
  //   bb.0:
  //     %0 = PHI %3, %bb.0, ...
  //     %2 = ADD %0, Value
  //     ... = LOAD %2(Op)
  //     %3 = COPY %2
  while (true) {
    if (!CurReg.isValid() || !CurReg.isVirtual())
      return false;
    MachineInstr *Def = MRI.getVRegDef(CurReg);
    if (Def->getParent() != LoopBB)
      return false;

    if (Def->isCopy()) {
      // Ignore copy instructions unless they contain subregisters
      if (Def->getOperand(0).getSubReg() || Def->getOperand(1).getSubReg())
        return false;
      CurReg = Def->getOperand(1).getReg();
    } else if (Def->isPHI()) {
      // There must be just one Phi
      if (Phi)
        return false;
      Phi = Def;
      CurReg = getLoopPhiReg(*Def, LoopBB);
    } else if (TII->getIncrementValue(*Def, Value)) {
      // Potentially a unique increment
      if (Increment)
        // Multiple increments exist
        return false;

      const MachineOperand *BaseOp;
      int64_t Offset;
      bool OffsetIsScalable;
      if (TII->getMemOperandWithOffset(*Def, BaseOp, Offset, OffsetIsScalable,
                                       TRI)) {
        // Pre/post increment instruction
        CurReg = BaseOp->getReg();
      } else {
        // If only one of the operands is defined within the loop, it is assumed
        // to be an incremented value.
        CurReg = findUniqueOperandDefinedInLoop(*Def);
        if (!CurReg.isValid())
          return false;
      }
      Increment = Def;
    } else {
      return false;
    }
    if (CurReg == OrgReg)
      break;
  }

  if (!Phi || !Increment)
    return false;

  return true;
}

/// Return true if we can compute the amount the instruction changes
/// during each iteration. Set Delta to the amount of the change.
bool SwingSchedulerDAG::computeDelta(const MachineInstr &MI, int &Delta) const {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const MachineOperand *BaseOp;
  int64_t Offset;
  bool OffsetIsScalable;
  if (!TII->getMemOperandWithOffset(MI, BaseOp, Offset, OffsetIsScalable, TRI))
    return false;

  // FIXME: This algorithm assumes instructions have fixed-size offsets.
  if (OffsetIsScalable)
    return false;

  if (!BaseOp->isReg())
    return false;

  return findLoopIncrementValue(*BaseOp, Delta);
}

/// Check if we can change the instruction to use an offset value from the
/// previous iteration. If so, return true and set the base and offset values
/// so that we can rewrite the load, if necessary.
///   v1 = Phi(v0, v3)
///   v2 = load v1, 0
///   v3 = post_store v1, 4, x
/// This function enables the load to be rewritten as v2 = load v3, 4.
bool SwingSchedulerDAG::canUseLastOffsetValue(MachineInstr *MI,
                                              unsigned &BasePos,
                                              unsigned &OffsetPos,
                                              Register &NewBase,
                                              int64_t &Offset) {
  // Get the load instruction.
  if (TII->isPostIncrement(*MI))
    return false;
  unsigned BasePosLd, OffsetPosLd;
  if (!TII->getBaseAndOffsetPosition(*MI, BasePosLd, OffsetPosLd))
    return false;
  Register BaseReg = MI->getOperand(BasePosLd).getReg();

  // Look for the Phi instruction.
  MachineRegisterInfo &MRI = MI->getMF()->getRegInfo();
  MachineInstr *Phi = MRI.getVRegDef(BaseReg);
  if (!Phi || !Phi->isPHI())
    return false;
  // Get the register defined in the loop block.
  Register PrevReg = getLoopPhiReg(*Phi, MI->getParent());
  if (!PrevReg)
    return false;

  // Check for the post-increment load/store instruction.
  MachineInstr *PrevDef = MRI.getVRegDef(PrevReg);
  if (!PrevDef || PrevDef == MI)
    return false;

  if (!TII->isPostIncrement(*PrevDef))
    return false;

  unsigned BasePos1 = 0, OffsetPos1 = 0;
  if (!TII->getBaseAndOffsetPosition(*PrevDef, BasePos1, OffsetPos1))
    return false;

  // Make sure that the instructions do not access the same memory location in
  // the next iteration.
  int64_t LoadOffset = MI->getOperand(OffsetPosLd).getImm();
  int64_t StoreOffset = PrevDef->getOperand(OffsetPos1).getImm();
  MachineInstr *NewMI = MF.CloneMachineInstr(MI);
  NewMI->getOperand(OffsetPosLd).setImm(LoadOffset + StoreOffset);
  bool Disjoint = TII->areMemAccessesTriviallyDisjoint(*NewMI, *PrevDef);
  MF.deleteMachineInstr(NewMI);
  if (!Disjoint)
    return false;

  // Set the return value once we determine that we return true.
  BasePos = BasePosLd;
  OffsetPos = OffsetPosLd;
  NewBase = PrevReg;
  Offset = StoreOffset;
  return true;
}

/// Apply changes to the instruction if needed. The changes are need
/// to improve the scheduling and depend up on the final schedule.
void SwingSchedulerDAG::applyInstrChange(MachineInstr *MI,
                                         SMSchedule &Schedule) {
  SUnit *SU = getSUnit(MI);
  DenseMap<SUnit *, std::pair<Register, int64_t>>::iterator It =
      InstrChanges.find(SU);
  if (It != InstrChanges.end()) {
    std::pair<Register, int64_t> RegAndOffset = It->second;
    unsigned BasePos, OffsetPos;
    if (!TII->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos))
      return;
    Register BaseReg = MI->getOperand(BasePos).getReg();
    MachineInstr *LoopDef = findDefInLoop(BaseReg);
    int DefStageNum = Schedule.stageScheduled(getSUnit(LoopDef));
    int DefCycleNum = Schedule.cycleScheduled(getSUnit(LoopDef));
    int BaseStageNum = Schedule.stageScheduled(SU);
    int BaseCycleNum = Schedule.cycleScheduled(SU);
    if (BaseStageNum < DefStageNum) {
      MachineInstr *NewMI = MF.CloneMachineInstr(MI);
      int OffsetDiff = DefStageNum - BaseStageNum;
      if (DefCycleNum < BaseCycleNum) {
        NewMI->getOperand(BasePos).setReg(RegAndOffset.first);
        if (OffsetDiff > 0)
          --OffsetDiff;
      }
      int64_t NewOffset =
          MI->getOperand(OffsetPos).getImm() + RegAndOffset.second * OffsetDiff;
      NewMI->getOperand(OffsetPos).setImm(NewOffset);
      SU->setInstr(NewMI);
      MISUnitMap[NewMI] = SU;
      NewMIs[MI] = NewMI;
    }
  }
}

/// Return the instruction in the loop that defines the register.
/// If the definition is a Phi, then follow the Phi operand to
/// the instruction in the loop.
MachineInstr *SwingSchedulerDAG::findDefInLoop(Register Reg) {
  SmallPtrSet<MachineInstr *, 8> Visited;
  MachineInstr *Def = MRI.getVRegDef(Reg);
  while (Def->isPHI()) {
    if (!Visited.insert(Def).second)
      break;
    for (unsigned i = 1, e = Def->getNumOperands(); i < e; i += 2)
      if (Def->getOperand(i + 1).getMBB() == BB) {
        Def = MRI.getVRegDef(Def->getOperand(i).getReg());
        break;
      }
  }
  return Def;
}

/// Return false if there is no overlap between the region accessed by BaseMI in
/// an iteration and the region accessed by OtherMI in subsequent iterations.
bool SwingSchedulerDAG::mayOverlapInLaterIter(
    const MachineInstr *BaseMI, const MachineInstr *OtherMI) const {
  int DeltaB, DeltaO, Delta;
  if (!computeDelta(*BaseMI, DeltaB) || !computeDelta(*OtherMI, DeltaO) ||
      DeltaB != DeltaO)
    return true;
  Delta = DeltaB;

  const MachineOperand *BaseOpB, *BaseOpO;
  int64_t OffsetB, OffsetO;
  bool OffsetBIsScalable, OffsetOIsScalable;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  if (!TII->getMemOperandWithOffset(*BaseMI, BaseOpB, OffsetB,
                                    OffsetBIsScalable, TRI) ||
      !TII->getMemOperandWithOffset(*OtherMI, BaseOpO, OffsetO,
                                    OffsetOIsScalable, TRI))
    return true;

  if (OffsetBIsScalable || OffsetOIsScalable)
    return true;

  if (!BaseOpB->isIdenticalTo(*BaseOpO)) {
    // Pass cases with different base operands but same initial values.
    // Typically for when pre/post increment is used.

    if (!BaseOpB->isReg() || !BaseOpO->isReg())
      return true;
    Register RegB = BaseOpB->getReg(), RegO = BaseOpO->getReg();
    if (!RegB.isVirtual() || !RegO.isVirtual())
      return true;

    MachineInstr *DefB = MRI.getVRegDef(BaseOpB->getReg());
    MachineInstr *DefO = MRI.getVRegDef(BaseOpO->getReg());
    if (!DefB || !DefO || !DefB->isPHI() || !DefO->isPHI())
      return true;

    Register InitValB;
    Register LoopValB;
    Register InitValO;
    Register LoopValO;
    getPhiRegs(*DefB, BB, InitValB, LoopValB);
    getPhiRegs(*DefO, BB, InitValO, LoopValO);
    MachineInstr *InitDefB = MRI.getVRegDef(InitValB);
    MachineInstr *InitDefO = MRI.getVRegDef(InitValO);

    if (!InitDefB->isIdenticalTo(*InitDefO))
      return true;
  }

  LocationSize AccessSizeB = (*BaseMI->memoperands_begin())->getSize();
  LocationSize AccessSizeO = (*OtherMI->memoperands_begin())->getSize();

  // This is the main test, which checks the offset values and the loop
  // increment value to determine if the accesses may be loop carried.
  if (!AccessSizeB.hasValue() || !AccessSizeO.hasValue())
    return true;

  LLVM_DEBUG({
    dbgs() << "Overlap check:\n";
    dbgs() << "  BaseMI: ";
    BaseMI->dump();
    dbgs() << "    Base + " << OffsetB << " + I * " << Delta
           << ", Len: " << AccessSizeB.getValue() << "\n";
    dbgs() << "  OtherMI: ";
    OtherMI->dump();
    dbgs() << "    Base + " << OffsetO << " + I * " << Delta
           << ", Len: " << AccessSizeO.getValue() << "\n";
  });

  // Excessive overlap may be detected in strided patterns.
  // For example, the memory addresses of the store and the load in
  //   for (i=0; i<n; i+=2) a[i+1] = a[i];
  // are assumed to overlap.
  if (Delta < 0) {
    int64_t BaseMinAddr = OffsetB;
    int64_t OhterNextIterMaxAddr = OffsetO + Delta + AccessSizeO.getValue() - 1;
    if (BaseMinAddr > OhterNextIterMaxAddr) {
      LLVM_DEBUG(dbgs() << "  Result: No overlap\n");
      return false;
    }
  } else {
    int64_t BaseMaxAddr = OffsetB + AccessSizeB.getValue() - 1;
    int64_t OtherNextIterMinAddr = OffsetO + Delta;
    if (BaseMaxAddr < OtherNextIterMinAddr) {
      LLVM_DEBUG(dbgs() << "  Result: No overlap\n");
      return false;
    }
  }
  LLVM_DEBUG(dbgs() << "  Result: Overlap\n");
  return true;
}

/// Return true for an order or output dependence that is loop carried
/// potentially. A dependence is loop carried if the destination defines a value
/// that may be used or defined by the source in a subsequent iteration.
bool SwingSchedulerDAG::isLoopCarriedDep(
    const SwingSchedulerDDGEdge &Edge) const {
  if ((!Edge.isOrderDep() && !Edge.isOutputDep()) || Edge.isArtificial() ||
      Edge.getDst()->isBoundaryNode())
    return false;

  if (!SwpPruneLoopCarried)
    return true;

  if (Edge.isOutputDep())
    return true;

  MachineInstr *SI = Edge.getSrc()->getInstr();
  MachineInstr *DI = Edge.getDst()->getInstr();
  assert(SI != nullptr && DI != nullptr && "Expecting SUnit with an MI.");

  // Assume ordered loads and stores may have a loop carried dependence.
  if (SI->hasUnmodeledSideEffects() || DI->hasUnmodeledSideEffects() ||
      SI->mayRaiseFPException() || DI->mayRaiseFPException() ||
      SI->hasOrderedMemoryRef() || DI->hasOrderedMemoryRef())
    return true;

  if (!DI->mayLoadOrStore() || !SI->mayLoadOrStore())
    return false;

  // The conservative assumption is that a dependence between memory operations
  // may be loop carried. The following code checks when it can be proved that
  // there is no loop carried dependence.
  return mayOverlapInLaterIter(DI, SI);
}

void SwingSchedulerDAG::postProcessDAG() {
  for (auto &M : Mutations)
    M->apply(this);
}

/// Try to schedule the node at the specified StartCycle and continue
/// until the node is schedule or the EndCycle is reached.  This function
/// returns true if the node is scheduled.  This routine may search either
/// forward or backward for a place to insert the instruction based upon
/// the relative values of StartCycle and EndCycle.
bool SMSchedule::insert(SUnit *SU, int StartCycle, int EndCycle, int II) {
  bool forward = true;
  LLVM_DEBUG({
    dbgs() << "Trying to insert node between " << StartCycle << " and "
           << EndCycle << " II: " << II << "\n";
  });
  if (StartCycle > EndCycle)
    forward = false;

  // The terminating condition depends on the direction.
  int termCycle = forward ? EndCycle + 1 : EndCycle - 1;
  for (int curCycle = StartCycle; curCycle != termCycle;
       forward ? ++curCycle : --curCycle) {

    if (ST.getInstrInfo()->isZeroCost(SU->getInstr()->getOpcode()) ||
        ProcItinResources.canReserveResources(*SU, curCycle)) {
      LLVM_DEBUG({
        dbgs() << "\tinsert at cycle " << curCycle << " ";
        SU->getInstr()->dump();
      });

      if (!ST.getInstrInfo()->isZeroCost(SU->getInstr()->getOpcode()))
        ProcItinResources.reserveResources(*SU, curCycle);
      ScheduledInstrs[curCycle].push_back(SU);
      InstrToCycle.insert(std::make_pair(SU, curCycle));
      if (curCycle > LastCycle)
        LastCycle = curCycle;
      if (curCycle < FirstCycle)
        FirstCycle = curCycle;
      return true;
    }
    LLVM_DEBUG({
      dbgs() << "\tfailed to insert at cycle " << curCycle << " ";
      SU->getInstr()->dump();
    });
  }
  return false;
}

// Return the cycle of the earliest scheduled instruction in the chain.
int SMSchedule::earliestCycleInChain(const SwingSchedulerDDGEdge &Dep,
                                     const SwingSchedulerDDG *DDG) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SwingSchedulerDDGEdge, 8> Worklist;
  Worklist.push_back(Dep);
  int EarlyCycle = INT_MAX;
  while (!Worklist.empty()) {
    const SwingSchedulerDDGEdge &Cur = Worklist.pop_back_val();
    SUnit *PrevSU = Cur.getSrc();
    if (Visited.count(PrevSU))
      continue;
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(PrevSU);
    if (it == InstrToCycle.end())
      continue;
    EarlyCycle = std::min(EarlyCycle, it->second);
    for (const auto &IE : DDG->getInEdges(PrevSU))
      if (IE.isOrderDep() || IE.isOutputDep())
        Worklist.push_back(IE);
    Visited.insert(PrevSU);
  }
  return EarlyCycle;
}

// Return the cycle of the latest scheduled instruction in the chain.
int SMSchedule::latestCycleInChain(const SwingSchedulerDDGEdge &Dep,
                                   const SwingSchedulerDDG *DDG) {
  SmallPtrSet<SUnit *, 8> Visited;
  SmallVector<SwingSchedulerDDGEdge, 8> Worklist;
  Worklist.push_back(Dep);
  int LateCycle = INT_MIN;
  while (!Worklist.empty()) {
    const SwingSchedulerDDGEdge &Cur = Worklist.pop_back_val();
    SUnit *SuccSU = Cur.getDst();
    if (Visited.count(SuccSU) || SuccSU->isBoundaryNode())
      continue;
    std::map<SUnit *, int>::const_iterator it = InstrToCycle.find(SuccSU);
    if (it == InstrToCycle.end())
      continue;
    LateCycle = std::max(LateCycle, it->second);
    for (const auto &OE : DDG->getOutEdges(SuccSU))
      if (OE.isOrderDep() || OE.isOutputDep())
        Worklist.push_back(OE);
    Visited.insert(SuccSU);
  }
  return LateCycle;
}

/// If an instruction has a use that spans multiple iterations, then
/// return true. These instructions are characterized by having a back-ege
/// to a Phi, which contains a reference to another Phi.
static SUnit *multipleIterations(SUnit *SU, SwingSchedulerDAG *DAG) {
  for (auto &P : SU->Preds)
    if (P.getKind() == SDep::Anti && P.getSUnit()->getInstr()->isPHI())
      for (auto &S : P.getSUnit()->Succs)
        if (S.getKind() == SDep::Data && S.getSUnit()->getInstr()->isPHI())
          return P.getSUnit();
  return nullptr;
}

/// Compute the scheduling start slot for the instruction.  The start slot
/// depends on any predecessor or successor nodes scheduled already.
void SMSchedule::computeStart(SUnit *SU, int *MaxEarlyStart, int *MinLateStart,
                              int II, SwingSchedulerDAG *DAG) {
  const SwingSchedulerDDG *DDG = DAG->getDDG();

  // Iterate over each instruction that has been scheduled already.  The start
  // slot computation depends on whether the previously scheduled instruction
  // is a predecessor or successor of the specified instruction.
  for (int cycle = getFirstCycle(); cycle <= LastCycle; ++cycle) {
    for (SUnit *I : getInstructions(cycle)) {
      for (const auto &IE : DDG->getInEdges(SU)) {
        if (IE.getSrc() == I) {
          // FIXME: Add reverse edge to `DDG` instead of calling
          // `isLoopCarriedDep`
          if (DAG->isLoopCarriedDep(IE)) {
            int End = earliestCycleInChain(IE, DDG) + (II - 1);
            *MinLateStart = std::min(*MinLateStart, End);
          }
          int EarlyStart = cycle + IE.getLatency() - IE.getDistance() * II;
          *MaxEarlyStart = std::max(*MaxEarlyStart, EarlyStart);
        }
      }

      for (const auto &OE : DDG->getOutEdges(SU)) {
        if (OE.getDst() == I) {
          // FIXME: Add reverse edge to `DDG` instead of calling
          // `isLoopCarriedDep`
          if (DAG->isLoopCarriedDep(OE)) {
            int Start = latestCycleInChain(OE, DDG) + 1 - II;
            *MaxEarlyStart = std::max(*MaxEarlyStart, Start);
          }
          int LateStart = cycle - OE.getLatency() + OE.getDistance() * II;
          *MinLateStart = std::min(*MinLateStart, LateStart);
        }
      }

      SUnit *BE = multipleIterations(I, DAG);
      for (const auto &Dep : SU->Preds) {
        // For instruction that requires multiple iterations, make sure that
        // the dependent instruction is not scheduled past the definition.
        if (BE && Dep.getSUnit() == BE && !SU->getInstr()->isPHI() &&
            !SU->isPred(I))
          *MinLateStart = std::min(*MinLateStart, cycle);
      }
    }
  }
}

/// Order the instructions within a cycle so that the definitions occur
/// before the uses. Returns true if the instruction is added to the start
/// of the list, or false if added to the end.
void SMSchedule::orderDependence(const SwingSchedulerDAG *SSD, SUnit *SU,
                                 std::deque<SUnit *> &Insts) const {
  MachineInstr *MI = SU->getInstr();
  bool OrderBeforeUse = false;
  bool OrderAfterDef = false;
  bool OrderBeforeDef = false;
  unsigned MoveDef = 0;
  unsigned MoveUse = 0;
  int StageInst1 = stageScheduled(SU);
  const SwingSchedulerDDG *DDG = SSD->getDDG();

  unsigned Pos = 0;
  for (std::deque<SUnit *>::iterator I = Insts.begin(), E = Insts.end(); I != E;
       ++I, ++Pos) {
    for (MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;

      Register Reg = MO.getReg();
      unsigned BasePos, OffsetPos;
      if (ST.getInstrInfo()->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos))
        if (MI->getOperand(BasePos).getReg() == Reg)
          if (Register NewReg = SSD->getInstrBaseReg(SU))
            Reg = NewReg;
      bool Reads, Writes;
      std::tie(Reads, Writes) =
          (*I)->getInstr()->readsWritesVirtualRegister(Reg);
      if (MO.isDef() && Reads && stageScheduled(*I) <= StageInst1) {
        OrderBeforeUse = true;
        if (MoveUse == 0)
          MoveUse = Pos;
      } else if (MO.isDef() && Reads && stageScheduled(*I) > StageInst1) {
        // Add the instruction after the scheduled instruction.
        OrderAfterDef = true;
        MoveDef = Pos;
      } else if (MO.isUse() && Writes && stageScheduled(*I) == StageInst1) {
        if (cycleScheduled(*I) == cycleScheduled(SU) && !(*I)->isSucc(SU)) {
          OrderBeforeUse = true;
          if (MoveUse == 0)
            MoveUse = Pos;
        } else {
          OrderAfterDef = true;
          MoveDef = Pos;
        }
      } else if (MO.isUse() && Writes && stageScheduled(*I) > StageInst1) {
        OrderBeforeUse = true;
        if (MoveUse == 0)
          MoveUse = Pos;
        if (MoveUse != 0) {
          OrderAfterDef = true;
          MoveDef = Pos - 1;
        }
      } else if (MO.isUse() && Writes && stageScheduled(*I) < StageInst1) {
        // Add the instruction before the scheduled instruction.
        OrderBeforeUse = true;
        if (MoveUse == 0)
          MoveUse = Pos;
      } else if (MO.isUse() && stageScheduled(*I) == StageInst1 &&
                 isLoopCarriedDefOfUse(SSD, (*I)->getInstr(), MO)) {
        if (MoveUse == 0) {
          OrderBeforeDef = true;
          MoveUse = Pos;
        }
      }
    }
    // Check for order dependences between instructions. Make sure the source
    // is ordered before the destination.
    for (auto &OE : DDG->getOutEdges(SU)) {
      if (OE.getDst() != *I)
        continue;
      if (OE.isOrderDep() && stageScheduled(*I) == StageInst1) {
        OrderBeforeUse = true;
        if (Pos < MoveUse)
          MoveUse = Pos;
      }
      // We did not handle HW dependences in previous for loop,
      // and we normally set Latency = 0 for Anti/Output deps,
      // so may have nodes in same cycle with Anti/Output dependent on HW regs.
      else if ((OE.isAntiDep() || OE.isOutputDep()) &&
               stageScheduled(*I) == StageInst1) {
        OrderBeforeUse = true;
        if ((MoveUse == 0) || (Pos < MoveUse))
          MoveUse = Pos;
      }
    }
    for (auto &IE : DDG->getInEdges(SU)) {
      if (IE.getSrc() != *I)
        continue;
      if ((IE.isAntiDep() || IE.isOutputDep() || IE.isOrderDep()) &&
          stageScheduled(*I) == StageInst1) {
        OrderAfterDef = true;
        MoveDef = Pos;
      }
    }
  }

  // A circular dependence.
  if (OrderAfterDef && OrderBeforeUse && MoveUse == MoveDef)
    OrderBeforeUse = false;

  // OrderAfterDef takes precedences over OrderBeforeDef. The latter is due
  // to a loop-carried dependence.
  if (OrderBeforeDef)
    OrderBeforeUse = !OrderAfterDef || (MoveUse > MoveDef);

  // The uncommon case when the instruction order needs to be updated because
  // there is both a use and def.
  if (OrderBeforeUse && OrderAfterDef) {
    SUnit *UseSU = Insts.at(MoveUse);
    SUnit *DefSU = Insts.at(MoveDef);
    if (MoveUse > MoveDef) {
      Insts.erase(Insts.begin() + MoveUse);
      Insts.erase(Insts.begin() + MoveDef);
    } else {
      Insts.erase(Insts.begin() + MoveDef);
      Insts.erase(Insts.begin() + MoveUse);
    }
    orderDependence(SSD, UseSU, Insts);
    orderDependence(SSD, SU, Insts);
    orderDependence(SSD, DefSU, Insts);
    return;
  }
  // Put the new instruction first if there is a use in the list. Otherwise,
  // put it at the end of the list.
  if (OrderBeforeUse)
    Insts.push_front(SU);
  else
    Insts.push_back(SU);
}

/// Return true if the scheduled Phi has a loop carried operand.
bool SMSchedule::isLoopCarried(const SwingSchedulerDAG *SSD,
                               MachineInstr &Phi) const {
  if (!Phi.isPHI())
    return false;
  assert(Phi.isPHI() && "Expecting a Phi.");
  SUnit *DefSU = SSD->getSUnit(&Phi);
  unsigned DefCycle = cycleScheduled(DefSU);
  int DefStage = stageScheduled(DefSU);

  Register InitVal;
  Register LoopVal;
  getPhiRegs(Phi, Phi.getParent(), InitVal, LoopVal);
  SUnit *UseSU = SSD->getSUnit(MRI.getVRegDef(LoopVal));
  if (!UseSU)
    return true;
  if (UseSU->getInstr()->isPHI())
    return true;
  unsigned LoopCycle = cycleScheduled(UseSU);
  int LoopStage = stageScheduled(UseSU);
  return (LoopCycle > DefCycle) || (LoopStage <= DefStage);
}

/// Return true if the instruction is a definition that is loop carried
/// and defines the use on the next iteration.
///        v1 = phi(v2, v3)
///  (Def) v3 = op v1
///  (MO)   = v1
/// If MO appears before Def, then v1 and v3 may get assigned to the same
/// register.
bool SMSchedule::isLoopCarriedDefOfUse(const SwingSchedulerDAG *SSD,
                                       MachineInstr *Def,
                                       MachineOperand &MO) const {
  if (!MO.isReg())
    return false;
  if (Def->isPHI())
    return false;
  MachineInstr *Phi = MRI.getVRegDef(MO.getReg());
  if (!Phi || !Phi->isPHI() || Phi->getParent() != Def->getParent())
    return false;
  if (!isLoopCarried(SSD, *Phi))
    return false;
  Register LoopReg = getLoopPhiReg(*Phi, Phi->getParent());
  for (MachineOperand &DMO : Def->all_defs()) {
    if (DMO.getReg() == LoopReg)
      return true;
  }
  return false;
}

/// Return true if all scheduled predecessors are loop-carried output/order
/// dependencies.
bool SMSchedule::onlyHasLoopCarriedOutputOrOrderPreds(
    SUnit *SU, const SwingSchedulerDDG *DDG) const {
  for (const auto &IE : DDG->getInEdges(SU))
    if (InstrToCycle.count(IE.getSrc()))
      return false;
  return true;
}

/// Determine transitive dependences of unpipelineable instructions
SmallSet<SUnit *, 8> SMSchedule::computeUnpipelineableNodes(
    SwingSchedulerDAG *SSD, TargetInstrInfo::PipelinerLoopInfo *PLI) {
  SmallSet<SUnit *, 8> DoNotPipeline;
  SmallVector<SUnit *, 8> Worklist;

  for (auto &SU : SSD->SUnits)
    if (SU.isInstr() && PLI->shouldIgnoreForPipelining(SU.getInstr()))
      Worklist.push_back(&SU);

  const SwingSchedulerDDG *DDG = SSD->getDDG();
  while (!Worklist.empty()) {
    auto SU = Worklist.pop_back_val();
    if (DoNotPipeline.count(SU))
      continue;
    LLVM_DEBUG(dbgs() << "Do not pipeline SU(" << SU->NodeNum << ")\n");
    DoNotPipeline.insert(SU);
    for (const auto &IE : DDG->getInEdges(SU))
      Worklist.push_back(IE.getSrc());

    // To preserve previous behavior and prevent regression
    // FIXME: Remove if this doesn't have significant impact on
    for (const auto &OE : DDG->getOutEdges(SU))
      if (OE.getDistance() == 1)
        Worklist.push_back(OE.getDst());
  }
  return DoNotPipeline;
}

// Determine all instructions upon which any unpipelineable instruction depends
// and ensure that they are in stage 0.  If unable to do so, return false.
bool SMSchedule::normalizeNonPipelinedInstructions(
    SwingSchedulerDAG *SSD, TargetInstrInfo::PipelinerLoopInfo *PLI) {
  SmallSet<SUnit *, 8> DNP = computeUnpipelineableNodes(SSD, PLI);

  int NewLastCycle = INT_MIN;
  for (SUnit &SU : SSD->SUnits) {
    if (!SU.isInstr())
      continue;
    if (!DNP.contains(&SU) || stageScheduled(&SU) == 0) {
      NewLastCycle = std::max(NewLastCycle, InstrToCycle[&SU]);
      continue;
    }

    // Put the non-pipelined instruction as early as possible in the schedule
    int NewCycle = getFirstCycle();
    for (const auto &IE : SSD->getDDG()->getInEdges(&SU))
      if (IE.getDistance() == 0)
        NewCycle = std::max(InstrToCycle[IE.getSrc()], NewCycle);

    // To preserve previous behavior and prevent regression
    // FIXME: Remove if this doesn't have significant impact on performance
    for (auto &OE : SSD->getDDG()->getOutEdges(&SU))
      if (OE.getDistance() == 1)
        NewCycle = std::max(InstrToCycle[OE.getDst()], NewCycle);

    int OldCycle = InstrToCycle[&SU];
    if (OldCycle != NewCycle) {
      InstrToCycle[&SU] = NewCycle;
      auto &OldS = getInstructions(OldCycle);
      llvm::erase(OldS, &SU);
      getInstructions(NewCycle).emplace_back(&SU);
      LLVM_DEBUG(dbgs() << "SU(" << SU.NodeNum
                        << ") is not pipelined; moving from cycle " << OldCycle
                        << " to " << NewCycle << " Instr:" << *SU.getInstr());
    }

    // We traverse the SUs in the order of the original basic block. Computing
    // NewCycle in this order normally works fine because all dependencies
    // (except for loop-carried dependencies) don't violate the original order.
    // However, an artificial dependency (e.g., added by CopyToPhiMutation) can
    // break it. That is, there may be exist an artificial dependency from
    // bottom to top. In such a case, NewCycle may become too large to be
    // scheduled in Stage 0. For example, assume that Inst0 is in DNP in the
    // following case:
    //
    //             |  Inst0  <-+
    //   SU order  |           | artificial dep
    //             |  Inst1  --+
    //             v
    //
    // If Inst1 is scheduled at cycle N and is not at Stage 0, then NewCycle of
    // Inst0 must be greater than or equal to N so that Inst0 is not be
    // scheduled at Stage 0. In such cases, we reject this schedule at this
    // time.
    // FIXME: The reason for this is the existence of artificial dependencies
    // that are contradict to the original SU order. If ignoring artificial
    // dependencies does not affect correctness, then it is better to ignore
    // them.
    if (FirstCycle + InitiationInterval <= NewCycle)
      return false;

    NewLastCycle = std::max(NewLastCycle, NewCycle);
  }
  LastCycle = NewLastCycle;
  return true;
}

// Check if the generated schedule is valid. This function checks if
// an instruction that uses a physical register is scheduled in a
// different stage than the definition. The pipeliner does not handle
// physical register values that may cross a basic block boundary.
// Furthermore, if a physical def/use pair is assigned to the same
// cycle, orderDependence does not guarantee def/use ordering, so that
// case should be considered invalid.  (The test checks for both
// earlier and same-cycle use to be more robust.)
bool SMSchedule::isValidSchedule(SwingSchedulerDAG *SSD) {
  for (SUnit &SU : SSD->SUnits) {
    if (!SU.hasPhysRegDefs)
      continue;
    int StageDef = stageScheduled(&SU);
    int CycleDef = InstrToCycle[&SU];
    assert(StageDef != -1 && "Instruction should have been scheduled.");
    for (auto &OE : SSD->getDDG()->getOutEdges(&SU)) {
      SUnit *Dst = OE.getDst();
      if (OE.isAssignedRegDep() && !Dst->isBoundaryNode())
        if (OE.getReg().isPhysical()) {
          if (stageScheduled(Dst) != StageDef)
            return false;
          if (InstrToCycle[Dst] <= CycleDef)
            return false;
        }
    }
  }
  return true;
}

/// A property of the node order in swing-modulo-scheduling is
/// that for nodes outside circuits the following holds:
/// none of them is scheduled after both a successor and a
/// predecessor.
/// The method below checks whether the property is met.
/// If not, debug information is printed and statistics information updated.
/// Note that we do not use an assert statement.
/// The reason is that although an invalid node order may prevent
/// the pipeliner from finding a pipelined schedule for arbitrary II,
/// it does not lead to the generation of incorrect code.
void SwingSchedulerDAG::checkValidNodeOrder(const NodeSetType &Circuits) const {

  // a sorted vector that maps each SUnit to its index in the NodeOrder
  typedef std::pair<SUnit *, unsigned> UnitIndex;
  std::vector<UnitIndex> Indices(NodeOrder.size(), std::make_pair(nullptr, 0));

  for (unsigned i = 0, s = NodeOrder.size(); i < s; ++i)
    Indices.push_back(std::make_pair(NodeOrder[i], i));

  auto CompareKey = [](UnitIndex i1, UnitIndex i2) {
    return std::get<0>(i1) < std::get<0>(i2);
  };

  // sort, so that we can perform a binary search
  llvm::sort(Indices, CompareKey);

  bool Valid = true;
  (void)Valid;
  // for each SUnit in the NodeOrder, check whether
  // it appears after both a successor and a predecessor
  // of the SUnit. If this is the case, and the SUnit
  // is not part of circuit, then the NodeOrder is not
  // valid.
  for (unsigned i = 0, s = NodeOrder.size(); i < s; ++i) {
    SUnit *SU = NodeOrder[i];
    unsigned Index = i;

    bool PredBefore = false;
    bool SuccBefore = false;

    SUnit *Succ;
    SUnit *Pred;
    (void)Succ;
    (void)Pred;

    for (const auto &IE : DDG->getInEdges(SU)) {
      SUnit *PredSU = IE.getSrc();
      unsigned PredIndex = std::get<1>(
          *llvm::lower_bound(Indices, std::make_pair(PredSU, 0), CompareKey));
      if (!PredSU->getInstr()->isPHI() && PredIndex < Index) {
        PredBefore = true;
        Pred = PredSU;
        break;
      }
    }

    for (const auto &OE : DDG->getOutEdges(SU)) {
      SUnit *SuccSU = OE.getDst();
      // Do not process a boundary node, it was not included in NodeOrder,
      // hence not in Indices either, call to std::lower_bound() below will
      // return Indices.end().
      if (SuccSU->isBoundaryNode())
        continue;
      unsigned SuccIndex = std::get<1>(
          *llvm::lower_bound(Indices, std::make_pair(SuccSU, 0), CompareKey));
      if (!SuccSU->getInstr()->isPHI() && SuccIndex < Index) {
        SuccBefore = true;
        Succ = SuccSU;
        break;
      }
    }

    if (PredBefore && SuccBefore && !SU->getInstr()->isPHI()) {
      // instructions in circuits are allowed to be scheduled
      // after both a successor and predecessor.
      bool InCircuit = llvm::any_of(
          Circuits, [SU](const NodeSet &Circuit) { return Circuit.count(SU); });
      if (InCircuit)
        LLVM_DEBUG(dbgs() << "In a circuit, predecessor ");
      else {
        Valid = false;
        NumNodeOrderIssues++;
        LLVM_DEBUG(dbgs() << "Predecessor ");
      }
      LLVM_DEBUG(dbgs() << Pred->NodeNum << " and successor " << Succ->NodeNum
                        << " are scheduled before node " << SU->NodeNum
                        << "\n");
    }
  }

  LLVM_DEBUG({
    if (!Valid)
      dbgs() << "Invalid node order found!\n";
  });
}

/// Attempt to fix the degenerate cases when the instruction serialization
/// causes the register lifetimes to overlap. For example,
///   p' = store_pi(p, b)
///      = load p, offset
/// In this case p and p' overlap, which means that two registers are needed.
/// Instead, this function changes the load to use p' and updates the offset.
void SwingSchedulerDAG::fixupRegisterOverlaps(std::deque<SUnit *> &Instrs) {
  Register OverlapReg;
  Register NewBaseReg;
  for (SUnit *SU : Instrs) {
    MachineInstr *MI = SU->getInstr();
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      // Look for an instruction that uses p. The instruction occurs in the
      // same cycle but occurs later in the serialized order.
      if (MO.isReg() && MO.isUse() && MO.getReg() == OverlapReg) {
        // Check that the instruction appears in the InstrChanges structure,
        // which contains instructions that can have the offset updated.
        DenseMap<SUnit *, std::pair<Register, int64_t>>::iterator It =
            InstrChanges.find(SU);
        if (It != InstrChanges.end()) {
          unsigned BasePos, OffsetPos;
          // Update the base register and adjust the offset.
          if (TII->getBaseAndOffsetPosition(*MI, BasePos, OffsetPos)) {
            MachineInstr *NewMI = MF.CloneMachineInstr(MI);
            NewMI->getOperand(BasePos).setReg(NewBaseReg);
            int64_t NewOffset =
                MI->getOperand(OffsetPos).getImm() - It->second.second;
            NewMI->getOperand(OffsetPos).setImm(NewOffset);
            SU->setInstr(NewMI);
            MISUnitMap[NewMI] = SU;
            NewMIs[MI] = NewMI;
          }
        }
        OverlapReg = Register();
        NewBaseReg = Register();
        break;
      }
      // Look for an instruction of the form p' = op(p), which uses and defines
      // two virtual registers that get allocated to the same physical register.
      unsigned TiedUseIdx = 0;
      if (MI->isRegTiedToUseOperand(i, &TiedUseIdx)) {
        // OverlapReg is p in the example above.
        OverlapReg = MI->getOperand(TiedUseIdx).getReg();
        // NewBaseReg is p' in the example above.
        NewBaseReg = MI->getOperand(i).getReg();
        break;
      }
    }
  }
}

std::deque<SUnit *>
SMSchedule::reorderInstructions(const SwingSchedulerDAG *SSD,
                                const std::deque<SUnit *> &Instrs) const {
  std::deque<SUnit *> NewOrderPhi;
  for (SUnit *SU : Instrs) {
    if (SU->getInstr()->isPHI())
      NewOrderPhi.push_back(SU);
  }
  std::deque<SUnit *> NewOrderI;
  for (SUnit *SU : Instrs) {
    if (!SU->getInstr()->isPHI())
      orderDependence(SSD, SU, NewOrderI);
  }
  llvm::append_range(NewOrderPhi, NewOrderI);
  return NewOrderPhi;
}

/// After the schedule has been formed, call this function to combine
/// the instructions from the different stages/cycles.  That is, this
/// function creates a schedule that represents a single iteration.
void SMSchedule::finalizeSchedule(SwingSchedulerDAG *SSD) {
  // Move all instructions to the first stage from later stages.
  for (int cycle = getFirstCycle(); cycle <= getFinalCycle(); ++cycle) {
    for (int stage = 1, lastStage = getMaxStageCount(); stage <= lastStage;
         ++stage) {
      std::deque<SUnit *> &cycleInstrs =
          ScheduledInstrs[cycle + (stage * InitiationInterval)];
      for (SUnit *SU : llvm::reverse(cycleInstrs))
        ScheduledInstrs[cycle].push_front(SU);
    }
  }

  // Erase all the elements in the later stages. Only one iteration should
  // remain in the scheduled list, and it contains all the instructions.
  for (int cycle = getFinalCycle() + 1; cycle <= LastCycle; ++cycle)
    ScheduledInstrs.erase(cycle);

  // Change the registers in instruction as specified in the InstrChanges
  // map. We need to use the new registers to create the correct order.
  for (const SUnit &SU : SSD->SUnits)
    SSD->applyInstrChange(SU.getInstr(), *this);

  // Reorder the instructions in each cycle to fix and improve the
  // generated code.
  for (int Cycle = getFirstCycle(), E = getFinalCycle(); Cycle <= E; ++Cycle) {
    std::deque<SUnit *> &cycleInstrs = ScheduledInstrs[Cycle];
    cycleInstrs = reorderInstructions(SSD, cycleInstrs);
    SSD->fixupRegisterOverlaps(cycleInstrs);
  }

  LLVM_DEBUG(dump(););
}

void NodeSet::print(raw_ostream &os) const {
  os << "Num nodes " << size() << " rec " << RecMII << " mov " << MaxMOV
     << " depth " << MaxDepth << " col " << Colocate << "\n";
  for (const auto &I : Nodes)
    os << "   SU(" << I->NodeNum << ") " << *(I->getInstr());
  os << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// Print the schedule information to the given output.
void SMSchedule::print(raw_ostream &os) const {
  // Iterate over each cycle.
  for (int cycle = getFirstCycle(); cycle <= getFinalCycle(); ++cycle) {
    // Iterate over each instruction in the cycle.
    const_sched_iterator cycleInstrs = ScheduledInstrs.find(cycle);
    for (SUnit *CI : cycleInstrs->second) {
      os << "cycle " << cycle << " (" << stageScheduled(CI) << ") ";
      os << "(" << CI->NodeNum << ") ";
      CI->getInstr()->print(os);
      os << "\n";
    }
  }
}

/// Utility function used for debugging to print the schedule.
LLVM_DUMP_METHOD void SMSchedule::dump() const { print(dbgs()); }
LLVM_DUMP_METHOD void NodeSet::dump() const { print(dbgs()); }

void ResourceManager::dumpMRT() const {
  LLVM_DEBUG({
    if (UseDFA)
      return;
    std::stringstream SS;
    SS << "MRT:\n";
    SS << std::setw(4) << "Slot";
    for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I)
      SS << std::setw(3) << I;
    SS << std::setw(7) << "#Mops"
       << "\n";
    for (int Slot = 0; Slot < InitiationInterval; ++Slot) {
      SS << std::setw(4) << Slot;
      for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I)
        SS << std::setw(3) << MRT[Slot][I];
      SS << std::setw(7) << NumScheduledMops[Slot] << "\n";
    }
    dbgs() << SS.str();
  });
}
#endif

void ResourceManager::initProcResourceVectors(
    const MCSchedModel &SM, SmallVectorImpl<uint64_t> &Masks) {
  unsigned ProcResourceID = 0;

  // We currently limit the resource kinds to 64 and below so that we can use
  // uint64_t for Masks
  assert(SM.getNumProcResourceKinds() < 64 &&
         "Too many kinds of resources, unsupported");
  // Create a unique bitmask for every processor resource unit.
  // Skip resource at index 0, since it always references 'InvalidUnit'.
  Masks.resize(SM.getNumProcResourceKinds());
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &Desc = *SM.getProcResource(I);
    if (Desc.SubUnitsIdxBegin)
      continue;
    Masks[I] = 1ULL << ProcResourceID;
    ProcResourceID++;
  }
  // Create a unique bitmask for every processor resource group.
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &Desc = *SM.getProcResource(I);
    if (!Desc.SubUnitsIdxBegin)
      continue;
    Masks[I] = 1ULL << ProcResourceID;
    for (unsigned U = 0; U < Desc.NumUnits; ++U)
      Masks[I] |= Masks[Desc.SubUnitsIdxBegin[U]];
    ProcResourceID++;
  }
  LLVM_DEBUG({
    if (SwpShowResMask) {
      dbgs() << "ProcResourceDesc:\n";
      for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
        const MCProcResourceDesc *ProcResource = SM.getProcResource(I);
        dbgs() << format(" %16s(%2d): Mask: 0x%08x, NumUnits:%2d\n",
                         ProcResource->Name, I, Masks[I],
                         ProcResource->NumUnits);
      }
      dbgs() << " -----------------\n";
    }
  });
}

bool ResourceManager::canReserveResources(SUnit &SU, int Cycle) {
  LLVM_DEBUG({
    if (SwpDebugResource)
      dbgs() << "canReserveResources:\n";
  });
  if (UseDFA)
    return DFAResources[positiveModulo(Cycle, InitiationInterval)]
        ->canReserveResources(&SU.getInstr()->getDesc());

  const MCSchedClassDesc *SCDesc = DAG->getSchedClass(&SU);
  if (!SCDesc->isValid()) {
    LLVM_DEBUG({
      dbgs() << "No valid Schedule Class Desc for schedClass!\n";
      dbgs() << "isPseudo:" << SU.getInstr()->isPseudo() << "\n";
    });
    return true;
  }

  reserveResources(SCDesc, Cycle);
  bool Result = !isOverbooked();
  unreserveResources(SCDesc, Cycle);

  LLVM_DEBUG(if (SwpDebugResource) dbgs() << "return " << Result << "\n\n");
  return Result;
}

void ResourceManager::reserveResources(SUnit &SU, int Cycle) {
  LLVM_DEBUG({
    if (SwpDebugResource)
      dbgs() << "reserveResources:\n";
  });
  if (UseDFA)
    return DFAResources[positiveModulo(Cycle, InitiationInterval)]
        ->reserveResources(&SU.getInstr()->getDesc());

  const MCSchedClassDesc *SCDesc = DAG->getSchedClass(&SU);
  if (!SCDesc->isValid()) {
    LLVM_DEBUG({
      dbgs() << "No valid Schedule Class Desc for schedClass!\n";
      dbgs() << "isPseudo:" << SU.getInstr()->isPseudo() << "\n";
    });
    return;
  }

  reserveResources(SCDesc, Cycle);

  LLVM_DEBUG({
    if (SwpDebugResource) {
      dumpMRT();
      dbgs() << "reserveResources: done!\n\n";
    }
  });
}

void ResourceManager::reserveResources(const MCSchedClassDesc *SCDesc,
                                       int Cycle) {
  assert(!UseDFA);
  for (const MCWriteProcResEntry &PRE : make_range(
           STI->getWriteProcResBegin(SCDesc), STI->getWriteProcResEnd(SCDesc)))
    for (int C = Cycle; C < Cycle + PRE.ReleaseAtCycle; ++C)
      ++MRT[positiveModulo(C, InitiationInterval)][PRE.ProcResourceIdx];

  for (int C = Cycle; C < Cycle + SCDesc->NumMicroOps; ++C)
    ++NumScheduledMops[positiveModulo(C, InitiationInterval)];
}

void ResourceManager::unreserveResources(const MCSchedClassDesc *SCDesc,
                                         int Cycle) {
  assert(!UseDFA);
  for (const MCWriteProcResEntry &PRE : make_range(
           STI->getWriteProcResBegin(SCDesc), STI->getWriteProcResEnd(SCDesc)))
    for (int C = Cycle; C < Cycle + PRE.ReleaseAtCycle; ++C)
      --MRT[positiveModulo(C, InitiationInterval)][PRE.ProcResourceIdx];

  for (int C = Cycle; C < Cycle + SCDesc->NumMicroOps; ++C)
    --NumScheduledMops[positiveModulo(C, InitiationInterval)];
}

bool ResourceManager::isOverbooked() const {
  assert(!UseDFA);
  for (int Slot = 0; Slot < InitiationInterval; ++Slot) {
    for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
      const MCProcResourceDesc *Desc = SM.getProcResource(I);
      if (MRT[Slot][I] > Desc->NumUnits)
        return true;
    }
    if (NumScheduledMops[Slot] > IssueWidth)
      return true;
  }
  return false;
}

int ResourceManager::calculateResMIIDFA() const {
  assert(UseDFA);

  // Sort the instructions by the number of available choices for scheduling,
  // least to most. Use the number of critical resources as the tie breaker.
  FuncUnitSorter FUS = FuncUnitSorter(*ST);
  for (SUnit &SU : DAG->SUnits)
    FUS.calcCriticalResources(*SU.getInstr());
  PriorityQueue<MachineInstr *, std::vector<MachineInstr *>, FuncUnitSorter>
      FuncUnitOrder(FUS);

  for (SUnit &SU : DAG->SUnits)
    FuncUnitOrder.push(SU.getInstr());

  SmallVector<std::unique_ptr<DFAPacketizer>, 8> Resources;
  Resources.push_back(
      std::unique_ptr<DFAPacketizer>(TII->CreateTargetScheduleState(*ST)));

  while (!FuncUnitOrder.empty()) {
    MachineInstr *MI = FuncUnitOrder.top();
    FuncUnitOrder.pop();
    if (TII->isZeroCost(MI->getOpcode()))
      continue;

    // Attempt to reserve the instruction in an existing DFA. At least one
    // DFA is needed for each cycle.
    unsigned NumCycles = DAG->getSUnit(MI)->Latency;
    unsigned ReservedCycles = 0;
    auto *RI = Resources.begin();
    auto *RE = Resources.end();
    LLVM_DEBUG({
      dbgs() << "Trying to reserve resource for " << NumCycles
             << " cycles for \n";
      MI->dump();
    });
    for (unsigned C = 0; C < NumCycles; ++C)
      while (RI != RE) {
        if ((*RI)->canReserveResources(*MI)) {
          (*RI)->reserveResources(*MI);
          ++ReservedCycles;
          break;
        }
        RI++;
      }
    LLVM_DEBUG(dbgs() << "ReservedCycles:" << ReservedCycles
                      << ", NumCycles:" << NumCycles << "\n");
    // Add new DFAs, if needed, to reserve resources.
    for (unsigned C = ReservedCycles; C < NumCycles; ++C) {
      LLVM_DEBUG(if (SwpDebugResource) dbgs()
                 << "NewResource created to reserve resources"
                 << "\n");
      auto *NewResource = TII->CreateTargetScheduleState(*ST);
      assert(NewResource->canReserveResources(*MI) && "Reserve error.");
      NewResource->reserveResources(*MI);
      Resources.push_back(std::unique_ptr<DFAPacketizer>(NewResource));
    }
  }

  int Resmii = Resources.size();
  LLVM_DEBUG(dbgs() << "Return Res MII:" << Resmii << "\n");
  return Resmii;
}

int ResourceManager::calculateResMII() const {
  if (UseDFA)
    return calculateResMIIDFA();

  // Count each resource consumption and divide it by the number of units.
  // ResMII is the max value among them.

  int NumMops = 0;
  SmallVector<uint64_t> ResourceCount(SM.getNumProcResourceKinds());
  for (SUnit &SU : DAG->SUnits) {
    if (TII->isZeroCost(SU.getInstr()->getOpcode()))
      continue;

    const MCSchedClassDesc *SCDesc = DAG->getSchedClass(&SU);
    if (!SCDesc->isValid())
      continue;

    LLVM_DEBUG({
      if (SwpDebugResource) {
        DAG->dumpNode(SU);
        dbgs() << "  #Mops: " << SCDesc->NumMicroOps << "\n"
               << "  WriteProcRes: ";
      }
    });
    NumMops += SCDesc->NumMicroOps;
    for (const MCWriteProcResEntry &PRE :
         make_range(STI->getWriteProcResBegin(SCDesc),
                    STI->getWriteProcResEnd(SCDesc))) {
      LLVM_DEBUG({
        if (SwpDebugResource) {
          const MCProcResourceDesc *Desc =
              SM.getProcResource(PRE.ProcResourceIdx);
          dbgs() << Desc->Name << ": " << PRE.ReleaseAtCycle << ", ";
        }
      });
      ResourceCount[PRE.ProcResourceIdx] += PRE.ReleaseAtCycle;
    }
    LLVM_DEBUG(if (SwpDebugResource) dbgs() << "\n");
  }

  int Result = (NumMops + IssueWidth - 1) / IssueWidth;
  LLVM_DEBUG({
    if (SwpDebugResource)
      dbgs() << "#Mops: " << NumMops << ", "
             << "IssueWidth: " << IssueWidth << ", "
             << "Cycles: " << Result << "\n";
  });

  LLVM_DEBUG({
    if (SwpDebugResource) {
      std::stringstream SS;
      SS << std::setw(2) << "ID" << std::setw(16) << "Name" << std::setw(10)
         << "Units" << std::setw(10) << "Consumed" << std::setw(10) << "Cycles"
         << "\n";
      dbgs() << SS.str();
    }
  });
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc *Desc = SM.getProcResource(I);
    int Cycles = (ResourceCount[I] + Desc->NumUnits - 1) / Desc->NumUnits;
    LLVM_DEBUG({
      if (SwpDebugResource) {
        std::stringstream SS;
        SS << std::setw(2) << I << std::setw(16) << Desc->Name << std::setw(10)
           << Desc->NumUnits << std::setw(10) << ResourceCount[I]
           << std::setw(10) << Cycles << "\n";
        dbgs() << SS.str();
      }
    });
    if (Cycles > Result)
      Result = Cycles;
  }
  return Result;
}

void ResourceManager::init(int II) {
  InitiationInterval = II;
  DFAResources.clear();
  DFAResources.resize(II);
  for (auto &I : DFAResources)
    I.reset(ST->getInstrInfo()->CreateTargetScheduleState(*ST));
  MRT.clear();
  MRT.resize(II, SmallVector<uint64_t>(SM.getNumProcResourceKinds()));
  NumScheduledMops.clear();
  NumScheduledMops.resize(II);
}

bool SwingSchedulerDDGEdge::ignoreDependence(bool IgnoreAnti) const {
  if (Pred.isArtificial() || Dst->isBoundaryNode())
    return true;
  // Currently, dependence that is an anti-dependences but not a loop-carried is
  // also ignored. This behavior is preserved to prevent regression.
  // FIXME: Remove if this doesn't have significant impact on performance
  return IgnoreAnti && (Pred.getKind() == SDep::Kind::Anti || Distance != 0);
}

SwingSchedulerDDG::SwingSchedulerDDGEdges &
SwingSchedulerDDG::getEdges(const SUnit *SU) {
  if (SU == EntrySU)
    return EntrySUEdges;
  if (SU == ExitSU)
    return ExitSUEdges;
  return EdgesVec[SU->NodeNum];
}

const SwingSchedulerDDG::SwingSchedulerDDGEdges &
SwingSchedulerDDG::getEdges(const SUnit *SU) const {
  if (SU == EntrySU)
    return EntrySUEdges;
  if (SU == ExitSU)
    return ExitSUEdges;
  return EdgesVec[SU->NodeNum];
}

void SwingSchedulerDDG::addEdge(const SUnit *SU,
                                const SwingSchedulerDDGEdge &Edge) {
  assert(!Edge.isValidationOnly() &&
         "Validation-only edges are not expected here.");
  auto &Edges = getEdges(SU);
  if (Edge.getSrc() == SU)
    Edges.Succs.push_back(Edge);
  else
    Edges.Preds.push_back(Edge);
}

void SwingSchedulerDDG::initEdges(SUnit *SU) {
  for (const auto &PI : SU->Preds) {
    SwingSchedulerDDGEdge Edge(SU, PI, /*IsSucc=*/false,
                               /*IsValidationOnly=*/false);
    addEdge(SU, Edge);
  }

  for (const auto &SI : SU->Succs) {
    SwingSchedulerDDGEdge Edge(SU, SI, /*IsSucc=*/true,
                               /*IsValidationOnly=*/false);
    addEdge(SU, Edge);
  }
}

SwingSchedulerDDG::SwingSchedulerDDG(std::vector<SUnit> &SUnits, SUnit *EntrySU,
                                     SUnit *ExitSU, const LoopCarriedEdges &LCE)
    : EntrySU(EntrySU), ExitSU(ExitSU) {
  EdgesVec.resize(SUnits.size());

  // Add non-loop-carried edges based on the DAG.
  initEdges(EntrySU);
  initEdges(ExitSU);
  for (auto &SU : SUnits)
    initEdges(&SU);

  // Add loop-carried edges, which are not represented in the DAG.
  for (SUnit &SU : SUnits) {
    SUnit *Src = &SU;
    if (const LoopCarriedEdges::OrderDep *OD = LCE.getOrderDepOrNull(Src)) {
      SDep Base(Src, SDep::Barrier);
      Base.setLatency(1);
      for (SUnit *Dst : *OD) {
        SwingSchedulerDDGEdge Edge(Dst, Base, /*IsSucc=*/false,
                                   /*IsValidationOnly=*/true);
        Edge.setDistance(1);
        ValidationOnlyEdges.push_back(Edge);
      }
    }
  }
}

const SwingSchedulerDDG::EdgesType &
SwingSchedulerDDG::getInEdges(const SUnit *SU) const {
  return getEdges(SU).Preds;
}

const SwingSchedulerDDG::EdgesType &
SwingSchedulerDDG::getOutEdges(const SUnit *SU) const {
  return getEdges(SU).Succs;
}

/// Check if \p Schedule doesn't violate the validation-only dependencies.
bool SwingSchedulerDDG::isValidSchedule(const SMSchedule &Schedule) const {
  unsigned II = Schedule.getInitiationInterval();

  auto ExpandCycle = [&](SUnit *SU) {
    int Stage = Schedule.stageScheduled(SU);
    int Cycle = Schedule.cycleScheduled(SU);
    return Cycle + (Stage * II);
  };

  for (const SwingSchedulerDDGEdge &Edge : ValidationOnlyEdges) {
    SUnit *Src = Edge.getSrc();
    SUnit *Dst = Edge.getDst();
    if (!Src->isInstr() || !Dst->isInstr())
      continue;
    int CycleSrc = ExpandCycle(Src);
    int CycleDst = ExpandCycle(Dst);
    int MaxLateStart = CycleDst + Edge.getDistance() * II - Edge.getLatency();
    if (CycleSrc > MaxLateStart) {
      LLVM_DEBUG({
        dbgs() << "Validation failed for edge from " << Src->NodeNum << " to "
               << Dst->NodeNum << "\n";
      });
      return false;
    }
  }
  return true;
}

void LoopCarriedEdges::modifySUnits(std::vector<SUnit> &SUnits,
                                    const TargetInstrInfo *TII) {
  for (SUnit &SU : SUnits) {
    SUnit *Src = &SU;
    if (auto *OrderDep = getOrderDepOrNull(Src)) {
      SDep Dep(Src, SDep::Barrier);
      Dep.setLatency(1);
      for (SUnit *Dst : *OrderDep) {
        SUnit *From = Src;
        SUnit *To = Dst;
        if (From->NodeNum > To->NodeNum)
          std::swap(From, To);

        // Add a forward edge if the following conditions are met:
        //
        // - The instruction of the source node (FromMI) may read memory.
        // - The instruction of the target node (ToMI) may modify memory, but
        //   does not read it.
        // - Neither instruction is a global barrier.
        // - The load appears before the store in the original basic block.
        // - There are no barrier or store instructions between the two nodes.
        // - The target node is unreachable from the source node in the current
        //   DAG.
        //
        // TODO: These conditions are inherited from a previous implementation,
        // and some may no longer be necessary. For now, we conservatively
        // retain all of them to avoid regressions, but the logic could
        // potentially be simplified
        MachineInstr *FromMI = From->getInstr();
        MachineInstr *ToMI = To->getInstr();
        if (FromMI->mayLoad() && !ToMI->mayLoad() && ToMI->mayStore() &&
            !TII->isGlobalMemoryObject(FromMI) &&
            !TII->isGlobalMemoryObject(ToMI) && !isSuccOrder(From, To)) {
          SDep Pred = Dep;
          Pred.setSUnit(From);
          To->addPred(Pred);
        }
      }
    }
  }
}

void LoopCarriedEdges::dump(SUnit *SU, const TargetRegisterInfo *TRI,
                            const MachineRegisterInfo *MRI) const {
  const auto *Order = getOrderDepOrNull(SU);

  if (!Order)
    return;

  const auto DumpSU = [](const SUnit *SU) {
    std::ostringstream OSS;
    OSS << "SU(" << SU->NodeNum << ")";
    return OSS.str();
  };

  dbgs() << "  Loop carried edges from " << DumpSU(SU) << "\n"
         << "    Order\n";
  for (SUnit *Dst : *Order)
    dbgs() << "      " << DumpSU(Dst) << "\n";
}
