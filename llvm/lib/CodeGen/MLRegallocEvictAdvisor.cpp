//===- MLRegAllocEvictAdvisor.cpp - ML eviction advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the ML eviction advisor and reward injection pass
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "RegAllocEvictionAdvisor.h"
#include "RegAllocGreedy.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#if defined(LLVM_HAVE_TF_AOT_REGALLOCEVICTMODEL) || defined(LLVM_HAVE_TF_API)
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#endif
#include "llvm/Analysis/ReleaseModeModelRunner.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "ml-regalloc"

// Generated header in release (AOT) mode
#if defined(LLVM_HAVE_TF_AOT_REGALLOCEVICTMODEL)
#include "RegallocEvictModel.h"
using CompiledModelType = RegallocEvictModel;
#else
using CompiledModelType = NoopSavedModelImpl;
#endif

// Options that only make sense in development mode
#ifdef LLVM_HAVE_TF_API
#include "RegAllocScore.h"
#include "llvm/Analysis/Utils/TFUtils.h"

static cl::opt<std::string> TrainingLog(
    "regalloc-training-log", cl::Hidden,
    cl::desc("Training log for the register allocator eviction model"));

static cl::opt<std::string> ModelUnderTraining(
    "regalloc-model", cl::Hidden,
    cl::desc("The model being trained for register allocation eviction"));

#endif // #ifdef LLVM_HAVE_TF_API

extern cl::opt<unsigned> EvictInterferenceCutoff;

/// The score injection pass.
/// This pass calculates the score for a function and inserts it in the log, but
/// this happens only in development mode. It's a no-op otherwise.
namespace llvm {
class RegAllocScoring : public MachineFunctionPass {
public:
  static char ID;

  RegAllocScoring() : MachineFunctionPass(ID) {
    initializeRegAllocScoringPass(*PassRegistry::getPassRegistry());
  }

  ~RegAllocScoring() override = default;

  StringRef getPassName() const override {
    return "Register Allocation Pass Scoring";
  }

  /// RegAllocReward analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<RegAllocEvictionAdvisorAnalysis>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// Performs this pass
  bool runOnMachineFunction(MachineFunction &) override;
};

char RegAllocScoring::ID = 0;
FunctionPass *createRegAllocScoringPass() { return new RegAllocScoring(); }

} // namespace llvm

INITIALIZE_PASS(RegAllocScoring, "regallocscoringpass",
                "Register Allocation Scoring Pass", false, false)

// ===================================
// Common ML Advisor declarations
// ===================================
namespace {
// This is the maximum number of interfererring ranges. That's the number of
// distinct AllocationOrder values, which comes from MCRegisterClass::RegsSize.
// For X86, that's 32.
// TODO: find a way to get this, statically, in a programmatic way.
static const int64_t MaxInterferences = 32;

// Logically, we can think of the feature set given to the evaluator as a 2D
// matrix. The rows are the features (see next). The columns correspond to the
// interferences. We treat the candidate virt reg as an 'interference', too, as
// its feature set is the same as that of the interferring ranges. So we'll have
// MaxInterferences + 1 columns and by convention, we will use the last column
// for the virt reg seeking allocation.
static const int64_t CandidateVirtRegPos = MaxInterferences;
static const int64_t NumberOfInterferences = CandidateVirtRegPos + 1;

// Most features are as described above, so we'll reuse this vector in defining
// them.
static const std::vector<int64_t> PerLiveRangeShape{1, NumberOfInterferences};

// --------------
// Features table
// --------------
// For each interfering live range (incl. the candidate) we collect a number of
// features. However, because the features are of different types (and because
// of ML best practices), we organize the tensors per feature, not per
// candidate. Each such tensor has a scalar value corresponding to the
// interferring live range at that position, in the order in AllocationOrder.
// The last position corresponds to the virt reg seeking allocation.
// Exception to all that is the progression feature, which is just a scalar (see
// its documentation for details).
// Note on naming: the "_by_max" are normalized using the largest value of that
// tensor, as observed in the current decision making stage (i.e. for the
// current call to the advisor's tryFindEvictionCandidate)
//
// The feature list format: type, name, shape, documentation.
// Note: we can really just use int64 and float, hence the modeling of some
// bools as int64 values.
#define RA_EVICT_FEATURES_LIST(M)                                              \
  M(int64_t, mask, PerLiveRangeShape,                                          \
    "boolean values, 0 for unavailable candidates (i.e. if a position is 0, "  \
    "it "                                                                      \
    "can't be evicted)")                                                       \
  M(int64_t, is_free, PerLiveRangeShape,                                       \
    "boolean values, 1 if this phys reg is actually free (no interferences)")  \
  M(float, nr_urgent, PerLiveRangeShape,                                       \
    "number of 'urgent' intervals, normalized. Urgent are those that are OK "  \
    "to break cascades")                                                       \
  M(float, nr_broken_hints, PerLiveRangeShape,                                 \
    "if this position were evicted, how many broken hints would there be")     \
  M(int64_t, is_hint, PerLiveRangeShape,                                       \
    "is this a preferred phys reg for the candidate")                          \
  M(int64_t, is_local, PerLiveRangeShape,                                      \
    "is this live range local to a basic block")                               \
  M(float, nr_rematerializable, PerLiveRangeShape,                             \
    "nr rematerializable ranges")                                              \
  M(float, nr_defs_and_uses, PerLiveRangeShape,                                \
    "bb freq - weighed nr defs and uses")                                      \
  M(float, weighed_reads_by_max, PerLiveRangeShape,                            \
    "bb freq - weighed nr of reads, normalized")                               \
  M(float, weighed_writes_by_max, PerLiveRangeShape,                           \
    "bb feq - weighed nr of writes, normalized")                               \
  M(float, weighed_read_writes_by_max, PerLiveRangeShape,                      \
    "bb freq - weighed nr of uses that are both read and writes, normalized")  \
  M(float, weighed_indvars_by_max, PerLiveRangeShape,                          \
    "bb freq - weighed nr of uses that are indvars, normalized")               \
  M(float, hint_weights_by_max, PerLiveRangeShape,                             \
    "bb freq - weighed nr of uses that are hints, normalized")                 \
  M(float, start_bb_freq_by_max, PerLiveRangeShape,                            \
    "the freq in the start block, normalized")                                 \
  M(float, end_bb_freq_by_max, PerLiveRangeShape,                              \
    "freq of end block, normalized")                                           \
  M(float, hottest_bb_freq_by_max, PerLiveRangeShape,                          \
    "hottest BB freq, normalized")                                             \
  M(float, liverange_size, PerLiveRangeShape,                                  \
    "size (instr index diff) of the LR")                                       \
  M(float, use_def_density, PerLiveRangeShape,                                 \
    "the max weight, as computed by the manual heuristic")                     \
  M(int64_t, max_stage, PerLiveRangeShape,                                     \
    "largest stage of an interval in this LR")                                 \
  M(int64_t, min_stage, PerLiveRangeShape,                                     \
    "lowest stage of an interval in this LR")                                  \
  M(float, progress, {1}, "ratio of current queue size to initial size")

// The model learns to pick one of the mask == 1 interferences. This is the name
// of the output tensor.
// The contract with the model is that the output will be guaranteed to be to a
// mask == 1 position.
// Using a macro here to avoid 'not used' warnings (and keep cond compilation to
// a minimum)
#define DecisionName "index_to_evict"

// Named features index.
enum FeatureIDs {
#define _FEATURE_IDX(_, name, __, ___) name,
  RA_EVICT_FEATURES_LIST(_FEATURE_IDX)
#undef _FEATURE_IDX
      FeatureCount
};

// The ML advisor will typically have a sparse input to the evaluator, because
// various phys regs won't be available. It's easier (maintenance-wise) to
// bulk-reset the state of the evaluator each time we are about to use it again.
template <typename T> size_t getTotalSize(const std::vector<int64_t> &Shape) {
  size_t Ret = sizeof(T);
  for (const auto V : Shape)
    Ret *= V;
  return Ret;
}

void resetInputs(MLModelRunner &Runner) {
#define _RESET(TYPE, NAME, SHAPE, __)                                          \
  std::memset(Runner.getTensorUntyped(FeatureIDs::NAME), 0,                    \
              getTotalSize<TYPE>(SHAPE));
  RA_EVICT_FEATURES_LIST(_RESET)
#undef _RESET
}

// Per-live interval components that get aggregated into the feature values that
// will be passed to the evaluator.
struct LIFeatureComponents {
  double R = 0;
  double W = 0;
  double RW = 0;
  double IndVarUpdates = 0;
  double HintWeights = 0.0;
  int64_t NrDefsAndUses = 0;
  float HottestBlockFreq = 0.0;
  bool IsRemat = false;
};

using CandidateRegList =
    std::array<std::pair<MCRegister, bool>, NumberOfInterferences>;
using FeaturesListNormalizer = std::array<float, FeatureIDs::FeatureCount>;

/// The ML evictor (commonalities between release and development mode)
class MLEvictAdvisor : public RegAllocEvictionAdvisor {
public:
  MLEvictAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                 MLModelRunner *Runner, const MachineBlockFrequencyInfo &MBFI,
                 const MachineLoopInfo &Loops);

protected:
  const RegAllocEvictionAdvisor &getDefaultAdvisor() const {
    return static_cast<const RegAllocEvictionAdvisor &>(DefaultAdvisor);
  }

  // The assumption is that if the Runner could not be constructed, we emit-ed
  // error, and we shouldn't be asking for it here.
  const MLModelRunner &getRunner() const { return *Runner; }

  /// This just calls Evaluate on the Runner, but in the development mode case,
  /// if we're just capturing the log of the default advisor, it needs to call
  /// the latter instead, so we need to pass all the necessary parameters for
  /// it. In the development case, it will also log.
  virtual int64_t
  tryFindEvictionCandidatePosition(const LiveInterval &VirtReg,
                                   const AllocationOrder &Order,
                                   unsigned OrderLimit, uint8_t CostPerUseLimit,
                                   const SmallVirtRegSet &FixedRegisters) const;

  /// Load the features of the given VirtReg (allocated or not) at column Pos,
  /// but if  that can't be evicted, return false instead.
  bool
  loadInterferenceFeatures(const LiveInterval &VirtReg, MCRegister PhysReg,
                           bool IsHint, const SmallVirtRegSet &FixedRegisters,
                           std::array<float, FeatureIDs::FeatureCount> &Largest,
                           size_t Pos) const;

private:
  static float getInitialQueueSize(const MachineFunction &MF);

  MCRegister tryFindEvictionCandidate(
      const LiveInterval &VirtReg, const AllocationOrder &Order,
      uint8_t CostPerUseLimit,
      const SmallVirtRegSet &FixedRegisters) const override;

  void extractFeatures(const SmallVectorImpl<const LiveInterval *> &Intervals,
                       std::array<float, FeatureIDs::FeatureCount> &Largest,
                       size_t Pos, int64_t IsHint, int64_t LocalIntfsCount,
                       float NrUrgent) const;

  // Point-in-time: we didn't learn this, so we always delegate to the default.
  bool canEvictHintInterference(
      const LiveInterval &VirtReg, MCRegister PhysReg,
      const SmallVirtRegSet &FixedRegisters) const override {
    return getDefaultAdvisor().canEvictHintInterference(VirtReg, PhysReg,
                                                        FixedRegisters);
  }

  const LIFeatureComponents &
  getLIFeatureComponents(const LiveInterval &LI) const;

  // Hold on to a default advisor for:
  // 1) the implementation of canEvictHintInterference, because we didn't learn
  // that nuance yet;
  // 2) for bootstrapping (logging) in the development mode case.
  const DefaultEvictionAdvisor DefaultAdvisor;
  MLModelRunner *const Runner;
  const MachineBlockFrequencyInfo &MBFI;
  const MachineLoopInfo &Loops;

  // Indices of those features we don't want to normalize.
  // This could be static and shared, but its initialization is non-trivial.
  std::bitset<FeatureIDs::FeatureCount> DoNotNormalize;
  const float InitialQSize;

  using RegID = unsigned;
  mutable DenseMap<RegID, LIFeatureComponents> CachedFeatures;
};

#define _DECL_FEATURES(type, name, shape, _)                                   \
  TensorSpec::createSpec<type>(#name, shape),

static const std::vector<TensorSpec> InputFeatures{
    {RA_EVICT_FEATURES_LIST(_DECL_FEATURES)},
};
#undef _DECL_FEATURES
// ===================================
// Release (AOT) - specifics
// ===================================
class ReleaseModeEvictionAdvisorAnalysis final
    : public RegAllocEvictionAdvisorAnalysis {
public:
  ReleaseModeEvictionAdvisorAnalysis()
      : RegAllocEvictionAdvisorAnalysis(AdvisorMode::Release) {}
  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocEvictionAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Release;
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineLoopInfo>();
    RegAllocEvictionAdvisorAnalysis::getAnalysisUsage(AU);
  }

  std::unique_ptr<RegAllocEvictionAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {
    if (!Runner)
      Runner = std::make_unique<ReleaseModeModelRunner<CompiledModelType>>(
          MF.getFunction().getContext(), InputFeatures, DecisionName);
    return std::make_unique<MLEvictAdvisor>(
        MF, RA, Runner.get(), getAnalysis<MachineBlockFrequencyInfo>(),
        getAnalysis<MachineLoopInfo>());
  }
  std::unique_ptr<ReleaseModeModelRunner<CompiledModelType>> Runner;
};

// ===================================
// Development mode-specifics
// ===================================
//
// Features we log
#ifdef LLVM_HAVE_TF_API
static const TensorSpec Output =
    TensorSpec::createSpec<int64_t>(DecisionName, {1});
static const TensorSpec Reward = TensorSpec::createSpec<float>("reward", {1});

// Features we bind on the model. The tensor names have a prefix, and we also
// need to include some tensors that are expected to be present by the training
// algo.
// TODO: can we just get rid of these?
#define _DECL_TRAIN_FEATURES(type, name, shape, _)                             \
  TensorSpec::createSpec<type>(std::string("action_") + #name, shape),

static const std::vector<TensorSpec> TrainingInputFeatures{
    {RA_EVICT_FEATURES_LIST(_DECL_TRAIN_FEATURES)
         TensorSpec::createSpec<float>("action_discount", {1}),
     TensorSpec::createSpec<int32_t>("action_step_type", {1}),
     TensorSpec::createSpec<float>("action_reward", {1})}};
#undef _DECL_TRAIN_FEATURES

class DevelopmentModeEvictAdvisor : public MLEvictAdvisor {
public:
  DevelopmentModeEvictAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                              MLModelRunner *Runner,
                              const MachineBlockFrequencyInfo &MBFI,
                              const MachineLoopInfo &Loops, Logger *Log)
      : MLEvictAdvisor(MF, RA, Runner, MBFI, Loops), Log(Log) {}

private:
  int64_t tryFindEvictionCandidatePosition(
      const LiveInterval &VirtReg, const AllocationOrder &Order,
      unsigned OrderLimit, uint8_t CostPerUseLimit,
      const SmallVirtRegSet &FixedRegisters) const override;

  Logger *const Log;
};

class DevelopmentModeEvictionAdvisorAnalysis final
    : public RegAllocEvictionAdvisorAnalysis {
public:
  DevelopmentModeEvictionAdvisorAnalysis()
      : RegAllocEvictionAdvisorAnalysis(AdvisorMode::Development) {}
  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocEvictionAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Development;
  }

  /// get the logger for the given function, or nullptr if we didn't collect
  /// one. This is used to inject the score by the RegAllocScoring pass.
  Logger *getLogger(const MachineFunction &MF) const {
    auto I = LogMap.find(MF.getName());
    if (I == LogMap.end())
      return nullptr;
    return I->second.get();
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineLoopInfo>();
    RegAllocEvictionAdvisorAnalysis::getAnalysisUsage(AU);
  }

  // Save all the logs (when requested).
  bool doFinalization(Module &M) override {
    if (TrainingLog.empty())
      return false;
    std::error_code EC;
    auto OS = std::make_unique<raw_fd_ostream>(TrainingLog, EC);
    if (EC) {
      M.getContext().emitError(EC.message() + ":" + TrainingLog);
      return false;
    }
    Logger::flushLogs(*OS, LogMap);
    return false;
  }

  std::unique_ptr<RegAllocEvictionAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {
    LLVMContext &Ctx = MF.getFunction().getContext();
    if (ModelUnderTraining.empty() && TrainingLog.empty()) {
      Ctx.emitError("Regalloc development mode should be requested with at "
                    "least logging enabled and/or a training model");
      return nullptr;
    }
    if (!Runner) {
      if (ModelUnderTraining.empty())
        Runner = std::make_unique<NoInferenceModelRunner>(Ctx, InputFeatures);
      else
        Runner = ModelUnderTrainingRunner::createAndEnsureValid(
            Ctx, ModelUnderTraining, DecisionName, TrainingInputFeatures);
      if (!Runner) {
        Ctx.emitError("Regalloc: could not set up the model runner");
        return nullptr;
      }
    }

    Logger *Log = nullptr;
    if (!TrainingLog.empty()) {
      std::vector<LoggedFeatureSpec> LFS;
      for (const auto &FS : InputFeatures)
        LFS.push_back({FS, None});
      if (auto *MUTR = dyn_cast<ModelUnderTrainingRunner>(Runner.get()))
        if (MUTR->outputLoggedFeatureSpecs().size() > 1)
          append_range(LFS, drop_begin(MUTR->outputLoggedFeatureSpecs()));
      // We always log the output; in particular, if we're not evaluating, we
      // don't have an output spec json file. That's why we handle the
      // 'normal' output separately.
      LFS.push_back({Output, None});
      auto I = LogMap.insert(std::make_pair(
          MF.getFunction().getName(),
          std::make_unique<Logger>(LFS, Reward, /*IncludeReward*/ true)));
      assert(I.second);
      Log = I.first->second.get();
    }
    return std::make_unique<DevelopmentModeEvictAdvisor>(
        MF, RA, Runner.get(), getAnalysis<MachineBlockFrequencyInfo>(),
        getAnalysis<MachineLoopInfo>(), Log);
  }

  std::unique_ptr<MLModelRunner> Runner;
  StringMap<std::unique_ptr<Logger>> LogMap;
};
#endif //#ifdef LLVM_HAVE_TF_API
} // namespace

float MLEvictAdvisor::getInitialQueueSize(const MachineFunction &MF) {
  auto &MRI = MF.getRegInfo();
  float Ret = 0.0;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    ++Ret;
  }
  return Ret;
}

MLEvictAdvisor::MLEvictAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                               MLModelRunner *Runner,
                               const MachineBlockFrequencyInfo &MBFI,
                               const MachineLoopInfo &Loops)
    : RegAllocEvictionAdvisor(MF, RA), DefaultAdvisor(MF, RA),
      Runner(std::move(Runner)), MBFI(MBFI), Loops(Loops),
      InitialQSize(MLEvictAdvisor::getInitialQueueSize(MF)) {
  assert(this->Runner);
  DoNotNormalize.set(FeatureIDs::mask);
  DoNotNormalize.set(FeatureIDs::is_free);
  DoNotNormalize.set(FeatureIDs::is_hint);
  DoNotNormalize.set(FeatureIDs::is_local);
  DoNotNormalize.set(FeatureIDs::min_stage);
  DoNotNormalize.set(FeatureIDs::max_stage);
  DoNotNormalize.set(FeatureIDs::progress);
}

int64_t MLEvictAdvisor::tryFindEvictionCandidatePosition(
    const LiveInterval &, const AllocationOrder &, unsigned, uint8_t,
    const SmallVirtRegSet &) const {
  int64_t Ret = Runner->evaluate<int64_t>();
  assert(Ret >= 0);
  assert(Ret <= CandidateVirtRegPos);
  return Ret;
}

bool MLEvictAdvisor::loadInterferenceFeatures(
    const LiveInterval &VirtReg, MCRegister PhysReg, bool IsHint,
    const SmallVirtRegSet &FixedRegisters, FeaturesListNormalizer &Largest,
    size_t Pos) const {
  // It is only possible to evict virtual register interference.
  if (Matrix->checkInterference(VirtReg, PhysReg) > LiveRegMatrix::IK_VirtReg) {
    // leave unavailable
    return false;
  }

  const bool IsLocal = LIS->intervalIsInOneMBB(VirtReg);
  int64_t LocalIntfs = 0;
  float NrUrgent = 0.0f;

  // The cascade tracking is the same as in the default advisor
  unsigned Cascade = RA.getExtraInfo().getCascadeOrCurrentNext(VirtReg.reg());

  SmallVector<const LiveInterval *, MaxInterferences> InterferingIntervals;
  for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, *Units);
    // Different from the default heuristic, we don't make any assumptions about
    // what having more than 10 results in the query may mean.
    const auto &IFIntervals = Q.interferingVRegs(EvictInterferenceCutoff);
    if (IFIntervals.empty() && InterferingIntervals.empty())
      continue;
    if (IFIntervals.size() >= EvictInterferenceCutoff)
      return false;
    InterferingIntervals.append(IFIntervals.begin(), IFIntervals.end());
    for (const LiveInterval *Intf : reverse(IFIntervals)) {
      assert(Register::isVirtualRegister(Intf->reg()) &&
             "Only expecting virtual register interference from query");
      // This is the same set of legality checks as in the default case: don't
      // try to evict fixed regs or 'done' ones. Also don't break cascades,
      // except in the urgent case, with the same nuances used in the default
      // heuristic.
      // We could try sharing this between the advisors, but it may end up
      // more complex than it is right now.
      if (FixedRegisters.count(Intf->reg()))
        return false;
      if (RA.getExtraInfo().getStage(*Intf) == RS_Done)
        return false;
      bool Urgent =
          !VirtReg.isSpillable() &&
          (Intf->isSpillable() ||
           RegClassInfo.getNumAllocatableRegs(MRI->getRegClass(VirtReg.reg())) <
               RegClassInfo.getNumAllocatableRegs(
                   MRI->getRegClass(Intf->reg())));
      // Only evict older cascades or live ranges without a cascade.
      unsigned IntfCascade = RA.getExtraInfo().getCascade(Intf->reg());
      if (Cascade <= IntfCascade) {
        if (!Urgent)
          return false;
        ++NrUrgent;
      }

      LocalIntfs += (IsLocal && LIS->intervalIsInOneMBB(*Intf) &&
                     (!EnableLocalReassign || !canReassign(*Intf, PhysReg)));
    }
  }
  // OK, so if we made it this far, this LR is an eviction candidate, load its
  // features.
  extractFeatures(InterferingIntervals, Largest, Pos, IsHint, LocalIntfs,
                  NrUrgent);
  return true;
}

MCRegister MLEvictAdvisor::tryFindEvictionCandidate(
    const LiveInterval &VirtReg, const AllocationOrder &Order,
    uint8_t CostPerUseLimit, const SmallVirtRegSet &FixedRegisters) const {
  auto MaybeOrderLimit = getOrderLimit(VirtReg, Order, CostPerUseLimit);
  if (!MaybeOrderLimit)
    return MCRegister::NoRegister;
  unsigned OrderLimit = *MaybeOrderLimit;

  // The heuristic sets initial costs such as, if CostPerUseLimit is
  // max<uint8_t>, then any of the costs of the legally-evictable intervals
  // would be lower. When that happens, one of those will be selected.
  // Therefore, we allow the candidate be selected, unless the candidate is
  // unspillable, in which case it would be incorrect to not find a register for
  // it.
  const bool MustFindEviction =
      (!VirtReg.isSpillable() && CostPerUseLimit == static_cast<uint8_t>(~0u));
  // Number of available candidates - if 0, no need to continue.
  size_t Available = 0;
  // Make sure we don't have leftover partial state from an attempt where we had
  // no available candidates and bailed out early.
  resetInputs(*Runner);

  // Track the index->register mapping because AllocationOrder doesn't do that
  // and we'd have to scan it.
  // Also track their mask, to write asserts/debug.
  CandidateRegList Regs;
  Regs.fill({0, false});

  // Track the largest value of features seen during this eviction session. We
  // only normalize (some of) the float features, but it's just simpler to
  // dimension 'Largest' to all the features, especially since we have the
  // 'DoNotNormalize' list.
  FeaturesListNormalizer Largest;
  Largest.fill(0.0);

  // Same overal idea as in the default eviction policy - we visit the values of
  // AllocationOrder one at a time. If it's not legally available, we mask off
  // the corresponding feature column (==do nothing because we already reset all
  // the features to 0)
  // Use Pos to capture the column we load features at - in AllocationOrder
  // order.
  size_t Pos = 0;
  for (auto I = Order.begin(), E = Order.getOrderLimitEnd(OrderLimit); I != E;
       ++I, ++Pos) {
    MCRegister PhysReg = *I;
    assert(!Regs[Pos].second);
    assert(PhysReg);
    if (!canAllocatePhysReg(CostPerUseLimit, PhysReg)) {
      continue;
    }
    if (loadInterferenceFeatures(VirtReg, PhysReg, I.isHint(), FixedRegisters,
                                 Largest, Pos)) {
      ++Available;
      Regs[Pos] = std::make_pair(PhysReg, true);
    }
  }
  if (Available == 0) {
    // Nothing to decide, nothing to learn.
    assert(!MustFindEviction);
    return MCRegister::NoRegister;
  }
  const size_t ValidPosLimit = Pos;
  // If we must find eviction, the candidate should be masked out of the
  // decision making process.
  Regs[CandidateVirtRegPos].second = !MustFindEviction;
  if (!MustFindEviction)
    extractFeatures(SmallVector<const LiveInterval *, 1>(1, &VirtReg), Largest,
                    CandidateVirtRegPos, /*IsHint*/ 0, /*LocalIntfsCount*/ 0,
                    /*NrUrgent*/ 0.0);
  assert(InitialQSize > 0.0 && "We couldn't have gotten here if we had "
                               "nothing to allocate initially.");
  // Normalize the features.
  for (auto &V : Largest)
    V = V ? V : 1.0;
  for (size_t FeatureIndex = 0; FeatureIndex < FeatureIDs::FeatureCount;
       ++FeatureIndex) {
    if (DoNotNormalize.test(FeatureIndex))
      continue;
    for (size_t Pos = 0; Pos < NumberOfInterferences; ++Pos) {
      Runner->getTensor<float>(FeatureIndex)[Pos] /= Largest[FeatureIndex];
    }
  }
  *Runner->getTensor<float>(FeatureIDs::progress) =
      static_cast<float>(RA.getQueueSize()) / InitialQSize;

  // Get a decision.
  size_t CandidatePos = tryFindEvictionCandidatePosition(
      VirtReg, Order, OrderLimit, CostPerUseLimit, FixedRegisters);
  // The contract with the ML side is that CandidatePos is mask == 1 (i.e.
  // Regs[CandidatePos].second)
  assert(Regs[CandidatePos].second);
  if (CandidatePos == CandidateVirtRegPos) {
    assert(!MustFindEviction);
    return MCRegister::NoRegister;
  }
  assert(CandidatePos < ValidPosLimit);
  (void)ValidPosLimit;
  return Regs[CandidatePos].first;
}

const LIFeatureComponents &
MLEvictAdvisor::getLIFeatureComponents(const LiveInterval &LI) const {
  RegID ID = LI.reg().id();
  LIFeatureComponents Empty;
  auto I = CachedFeatures.insert(std::make_pair(ID, Empty));
  LIFeatureComponents &Ret = I.first->getSecond();
  if (!I.second)
    return Ret;

  SmallPtrSet<MachineInstr *, 8> Visited;
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  for (MachineRegisterInfo::reg_instr_nodbg_iterator
           I = MRI->reg_instr_nodbg_begin(LI.reg()),
           E = MRI->reg_instr_nodbg_end();
       I != E;) {
    MachineInstr *MI = &*(I++);

    ++Ret.NrDefsAndUses;
    if (!Visited.insert(MI).second)
      continue;

    if (MI->isIdentityCopy() || MI->isImplicitDef())
      continue;

    bool Reads, Writes;
    std::tie(Reads, Writes) = MI->readsWritesVirtualRegister(LI.reg());

    float Freq = MBFI.getBlockFreqRelativeToEntryBlock(MI->getParent());
    Ret.HottestBlockFreq = std::max(Freq, Ret.HottestBlockFreq);

    Ret.R += (Reads && !Writes) * Freq;
    Ret.W += (!Reads && Writes) * Freq;
    Ret.RW += (Reads && Writes) * Freq;

    auto *MBB = MI->getParent();
    auto *Loop = Loops.getLoopFor(MBB);
    bool IsExiting = Loop ? Loop->isLoopExiting(MBB) : false;

    if (Writes && IsExiting && LIS->isLiveOutOfMBB(LI, MBB))
      Ret.IndVarUpdates += Freq;

    if (MI->isCopy() && VirtRegAuxInfo::copyHint(MI, LI.reg(), TRI, *MRI))
      Ret.HintWeights += Freq;
  }
  Ret.IsRemat = VirtRegAuxInfo::isRematerializable(
      LI, *LIS, *VRM, *MF.getSubtarget().getInstrInfo());
  return Ret;
}

// Overall, this currently mimics what we do for weight calculation, but instead
// of accummulating the various features, we keep them separate.
void MLEvictAdvisor::extractFeatures(
    const SmallVectorImpl<const LiveInterval *> &Intervals,
    std::array<float, FeatureIDs::FeatureCount> &Largest, size_t Pos,
    int64_t IsHint, int64_t LocalIntfsCount, float NrUrgent) const {
  int64_t NrDefsAndUses = 0;
  int64_t NrBrokenHints = 0;
  double R = 0.0;
  double W = 0.0;
  double RW = 0.0;
  double IndVarUpdates = 0.0;
  double HintWeights = 0.0;
  float StartBBFreq = 0.0;
  float EndBBFreq = 0.0;
  float HottestBlockFreq = 0.0;
  int32_t NrRematerializable = 0;
  float TotalWeight = 0.0;

  SlotIndex EndSI = LIS->getSlotIndexes()->getZeroIndex();
  SlotIndex StartSI = LIS->getSlotIndexes()->getLastIndex();
  int64_t MaxStage = 0;
  int64_t MinStage =
      Intervals.empty() ? 0 : std::numeric_limits<int64_t>::max();

  for (const auto *L : Intervals) {
    const LiveInterval &LI = *L;
    MaxStage = std::max<int64_t>(
        MaxStage, static_cast<int64_t>(RA.getExtraInfo().getStage(LI)));
    MinStage = std::min<int64_t>(
        MinStage, static_cast<int64_t>(RA.getExtraInfo().getStage(LI)));

    TotalWeight = std::max(TotalWeight, LI.weight());

    if (LI.beginIndex() < StartSI)
      StartSI = LI.beginIndex();

    if (LI.endIndex() > EndSI)
      EndSI = LI.endIndex();
    const LIFeatureComponents &LIFC = getLIFeatureComponents(LI);
    NrBrokenHints += VRM->hasPreferredPhys(LI.reg());

    NrDefsAndUses += LIFC.NrDefsAndUses;
    HottestBlockFreq = std::max(HottestBlockFreq, LIFC.HottestBlockFreq);
    R += LIFC.R;
    W += LIFC.W;
    RW += LIFC.RW;

    IndVarUpdates += LIFC.IndVarUpdates;

    HintWeights += LIFC.HintWeights;
    NrRematerializable += LIFC.IsRemat;
  }
  size_t Size = 0;
  if (!Intervals.empty()) {
    StartBBFreq =
        MBFI.getBlockFreqRelativeToEntryBlock(LIS->getMBBFromIndex(StartSI));
    if (EndSI >= LIS->getSlotIndexes()->getLastIndex())
      EndSI = LIS->getSlotIndexes()->getLastIndex().getPrevIndex();
    EndBBFreq =
        MBFI.getBlockFreqRelativeToEntryBlock(LIS->getMBBFromIndex(EndSI));
    Size = StartSI.distance(EndSI);
  }
  // Set the features at the column 'Pos'.
#define SET(ID, TYPE, VAL)                                                     \
  do {                                                                         \
    Runner->getTensor<TYPE>(FeatureIDs::ID)[Pos] = static_cast<TYPE>(VAL);     \
    if (!DoNotNormalize.test(FeatureIDs::ID))                                  \
      Largest[FeatureIDs::ID] =                                                \
          std::max(Largest[FeatureIDs::ID], static_cast<float>(VAL));          \
  } while (false)
  SET(mask, int64_t, 1);
  SET(is_free, int64_t, Intervals.empty());
  SET(nr_urgent, float, NrUrgent);
  SET(nr_broken_hints, float, NrBrokenHints);
  SET(is_hint, int64_t, IsHint);
  SET(is_local, int64_t, LocalIntfsCount);
  SET(nr_rematerializable, float, NrRematerializable);
  SET(nr_defs_and_uses, float, NrDefsAndUses);
  SET(weighed_reads_by_max, float, R);
  SET(weighed_writes_by_max, float, W);
  SET(weighed_read_writes_by_max, float, RW);
  SET(weighed_indvars_by_max, float, IndVarUpdates);
  SET(hint_weights_by_max, float, HintWeights);
  SET(start_bb_freq_by_max, float, StartBBFreq);
  SET(end_bb_freq_by_max, float, EndBBFreq);
  SET(hottest_bb_freq_by_max, float, HottestBlockFreq);
  SET(liverange_size, float, Size);
  SET(use_def_density, float, TotalWeight);
  SET(max_stage, int64_t, MaxStage);
  SET(min_stage, int64_t, MinStage);
#undef SET
}

// Development mode-specific implementations
#ifdef LLVM_HAVE_TF_API
RegAllocEvictionAdvisorAnalysis *llvm::createDevelopmentModeAdvisor() {
  return new DevelopmentModeEvictionAdvisorAnalysis();
}

int64_t DevelopmentModeEvictAdvisor::tryFindEvictionCandidatePosition(
    const LiveInterval &VirtReg, const AllocationOrder &Order,
    unsigned OrderLimit, uint8_t CostPerUseLimit,
    const SmallVirtRegSet &FixedRegisters) const {
  int64_t Ret = 0;
  if (isa<ModelUnderTrainingRunner>(getRunner())) {
    Ret = MLEvictAdvisor::tryFindEvictionCandidatePosition(
        VirtReg, Order, OrderLimit, CostPerUseLimit, FixedRegisters);
  } else {
    MCRegister PhysReg = getDefaultAdvisor().tryFindEvictionCandidate(
        VirtReg, Order, CostPerUseLimit, FixedRegisters);
    // Find the index of the selected PhysReg. We need it for logging, otherwise
    // this is wasted cycles (but so would starting development mode without a
    // model nor logging)
    if (!PhysReg)
      Ret = CandidateVirtRegPos;
    else
      for (auto I = Order.begin(), E = Order.getOrderLimitEnd(OrderLimit);
           I != E; ++I, ++Ret)
        if (*I == PhysReg)
          break;
  }
  if (TrainingLog.empty())
    return Ret;
  size_t CurrentFeature = 0;
  for (; CurrentFeature < FeatureIDs::FeatureCount; ++CurrentFeature) {
    Log->logSpecifiedTensorValue(
        CurrentFeature, reinterpret_cast<const char *>(
                            getRunner().getTensorUntyped(CurrentFeature)));
  }
  if (auto *MUTR = dyn_cast<ModelUnderTrainingRunner>(&getRunner()))
    for (size_t I = 1; I < MUTR->outputLoggedFeatureSpecs().size();
         ++I, ++CurrentFeature)
      Log->logSpecifiedTensorValue(
          CurrentFeature,
          reinterpret_cast<const char *>(
              MUTR->lastEvaluationResult()->getUntypedTensorValue(I)));
  // The output is right after the features and the extra outputs
  Log->logInt64Value(CurrentFeature, &Ret);
  return Ret;
}

bool RegAllocScoring::runOnMachineFunction(MachineFunction &MF) {
  if (auto *DevModeAnalysis = dyn_cast<DevelopmentModeEvictionAdvisorAnalysis>(
          &getAnalysis<RegAllocEvictionAdvisorAnalysis>()))
    if (auto *Log = DevModeAnalysis->getLogger(MF))
      Log->logFloatFinalReward(static_cast<float>(
          calculateRegAllocScore(MF, getAnalysis<MachineBlockFrequencyInfo>())
              .getScore()));

  return false;
}
#endif // #ifdef LLVM_HAVE_TF_API

RegAllocEvictionAdvisorAnalysis *llvm::createReleaseModeAdvisor() {
  return new ReleaseModeEvictionAdvisorAnalysis();
}

// In all cases except development mode, we don't need scoring.
#if !defined(LLVM_HAVE_TF_API)
bool RegAllocScoring::runOnMachineFunction(MachineFunction &) { return false; }
#endif
