//===- RegAllocEvictionAdvisor.cpp - eviction advisor ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the default eviction advisor and of the Analysis pass.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/RegAllocEvictionAdvisor.h"
#include "AllocationOrder.h"
#include "FailedReassignCache.h"
#include "RegAllocGreedy.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegAllocPriorityAdvisor.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumCanReassignCalls, "Number of canReassign calls");
STATISTIC(NumCanReassignSuccess,
          "Number of canReassign calls that found a free register");
STATISTIC(NumCanReassignFail, "Number of canReassign calls that failed");
STATISTIC(NumCanReassignCandidates,
          "Number of physical registers examined by canReassign");
STATISTIC(NumCanReassignCacheHits,
          "Number of failed canReassign answers reused without a search");
STATISTIC(NumCanReassignRechecks,
          "Number of failed canReassign answers rechecked against freed "
          "registers");

static cl::opt<RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode> Mode(
    "regalloc-enable-advisor", cl::Hidden,
    cl::init(RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Default),
    cl::desc("Enable regalloc advisor mode"),
    cl::values(
        clEnumValN(RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Default,
                   "default", "Default"),
        clEnumValN(RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Release,
                   "release", "precompiled"),
        clEnumValN(
            RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Development,
            "development", "for training")));

static cl::opt<unsigned> ReassignCacheMinOrder(
    "regalloc-reassign-cache-min-order", cl::Hidden,
    cl::desc("Cache a failed canReassign only when the scan examined more "
             "physical registers than this"),
    cl::init(32));

static cl::opt<unsigned> ReassignCacheMaxFreed(
    "regalloc-reassign-cache-max-freed", cl::Hidden,
    cl::desc("Number of freed physical registers recorded for one cached "
             "canReassign failure before the entry is saturated"),
    cl::init(reassign::Cache::DefaultMaxFreedRegs));

static cl::opt<cl::boolOrDefault> EnableLocalReassignment(
    "enable-local-reassign", cl::Hidden,
    cl::desc("Local reassignment can yield better allocation decisions, but "
             "may be compile time intensive"));

namespace llvm {
cl::opt<unsigned> EvictInterferenceCutoff(
    "regalloc-eviction-max-interference-cutoff", cl::Hidden,
    cl::desc("Number of interferences after which we declare "
             "an interference unevictable and bail out. This "
             "is a compilation cost-saving consideration. To "
             "disable, pass a very large number."),
    cl::init(10));
}

#define DEBUG_TYPE "regalloc"
#ifdef LLVM_HAVE_TF_AOT_REGALLOCEVICTMODEL
#define LLVM_HAVE_TF_AOT
#endif

char RegAllocEvictionAdvisorAnalysisLegacy::ID = 0;
INITIALIZE_PASS(RegAllocEvictionAdvisorAnalysisLegacy, "regalloc-evict",
                "Regalloc eviction policy", false, true)

namespace {
class DefaultEvictionAdvisorProvider final
    : public RegAllocEvictionAdvisorProvider {
public:
  DefaultEvictionAdvisorProvider(bool NotAsRequested, LLVMContext &Ctx)
      : RegAllocEvictionAdvisorProvider(AdvisorMode::Default, Ctx) {
    if (NotAsRequested)
      Ctx.emitError("Requested regalloc eviction advisor analysis "
                    "could not be created. Using default");
  }

  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocEvictionAdvisorProvider *R) {
    return R->getAdvisorMode() == AdvisorMode::Default;
  }

  std::unique_ptr<RegAllocEvictionAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA,
             MachineBlockFrequencyInfo *, MachineLoopInfo *) override {
    return std::make_unique<DefaultEvictionAdvisor>(MF, RA);
  }
};

class DefaultEvictionAdvisorAnalysisLegacy final
    : public RegAllocEvictionAdvisorAnalysisLegacy {
public:
  DefaultEvictionAdvisorAnalysisLegacy(bool NotAsRequested)
      : RegAllocEvictionAdvisorAnalysisLegacy(AdvisorMode::Default),
        NotAsRequested(NotAsRequested) {}

  bool doInitialization(Module &M) override {
    Provider.reset(
        new DefaultEvictionAdvisorProvider(NotAsRequested, M.getContext()));
    return false;
  }

  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocEvictionAdvisorAnalysisLegacy *R) {
    return R->getAdvisorMode() == AdvisorMode::Default;
  }

private:
  const bool NotAsRequested;
};
} // namespace

AnalysisKey RegAllocEvictionAdvisorAnalysis::Key;

void RegAllocEvictionAdvisorAnalysis::initializeProvider(
    RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode Mode, LLVMContext &Ctx) {
  if (Provider)
    return;
  switch (Mode) {
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Default:
    Provider.reset(
        new DefaultEvictionAdvisorProvider(/*NotAsRequested=*/false, Ctx));
    return;
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Development:
    Provider.reset(createDevelopmentModeAdvisorProvider(Ctx));
    break;
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Release:
    Provider.reset(createReleaseModeAdvisorProvider(Ctx));
    break;
  }
  if (!Provider)
    Provider.reset(
        new DefaultEvictionAdvisorProvider(/*NotAsRequested=*/true, Ctx));
}

RegAllocEvictionAdvisorAnalysis::Result
RegAllocEvictionAdvisorAnalysis::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  // Lazy initialization of the provider.
  initializeProvider(::Mode, MF.getFunction().getContext());
  return Result{Provider.get()};
}

template <>
Pass *llvm::callDefaultCtor<RegAllocEvictionAdvisorAnalysisLegacy>() {
  switch (Mode) {
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Default:
    return new DefaultEvictionAdvisorAnalysisLegacy(/*NotAsRequested=*/false);
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Release: {
    Pass *Ret = createReleaseModeAdvisorAnalysisLegacy();
    // release mode advisor may not be supported
    if (Ret)
      return Ret;
    return new DefaultEvictionAdvisorAnalysisLegacy(/*NotAsRequested=*/true);
  }
  case RegAllocEvictionAdvisorAnalysisLegacy::AdvisorMode::Development:
#if defined(LLVM_HAVE_TFLITE)
    return createDevelopmentModeAdvisorAnalysisLegacy();
#else
    return new DefaultEvictionAdvisorAnalysisLegacy(/*NotAsRequested=*/true);
#endif
  }
  llvm_unreachable("unexpected advisor mode");
}

StringRef RegAllocEvictionAdvisorAnalysisLegacy::getPassName() const {
  switch (getAdvisorMode()) {
  case AdvisorMode::Default:
    return "Default Regalloc Eviction Advisor";
  case AdvisorMode::Release:
    return "Release mode Regalloc Eviction Advisor";
  case AdvisorMode::Development:
    return "Development mode Regalloc Eviction Advisor";
  }
  llvm_unreachable("Unknown advisor kind");
}

RegAllocEvictionAdvisor::RegAllocEvictionAdvisor(const MachineFunction &MF,
                                                 const RAGreedy &RA)
    : MF(MF), RA(RA), Matrix(RA.getInterferenceMatrix()),
      LIS(RA.getLiveIntervals()), VRM(RA.getVirtRegMap()),
      MRI(&VRM->getRegInfo()), TRI(MF.getSubtarget().getRegisterInfo()),
      RegClassInfo(RA.getRegClassInfo()), RegCosts(TRI->getRegisterCosts(MF)),
      EnableLocalReassign(
          EnableLocalReassignment == cl::boolOrDefault::BOU_TRUE ||
          (EnableLocalReassignment != cl::boolOrDefault::BOU_FALSE &&
           MF.getSubtarget().enableRALocalReassignment(
               MF.getTarget().getOptLevel()))) {
  assert(Matrix && LIS && "Eviction advisor requires LiveRegMatrix and LIS");
  FailedCache = std::make_unique<reassign::Cache>(
      *LIS, *Matrix, *MRI, *TRI, RegClassInfo, ReassignCacheMaxFreed);
  Matrix->setUnassignListener(&RegAllocEvictionAdvisor::unassignListener, this);
}

RegAllocEvictionAdvisor::~RegAllocEvictionAdvisor() {
  if (Matrix)
    Matrix->clearUnassignListener(this);
}

void RegAllocEvictionAdvisor::unassignListener(
    void *Ctx, const LiveInterval &Unassigned, MCRegister PhysReg,
    bool ClearAllReferencingSegments) {
  static_cast<RegAllocEvictionAdvisor *>(Ctx)->noteUnassign(
      Unassigned, PhysReg, ClearAllReferencingSegments);
}

void RegAllocEvictionAdvisor::noteUnassign(
    const LiveInterval &Unassigned, MCRegister PhysReg,
    bool ClearAllReferencingSegments) const {
  FailedCache->noteUnassign(Unassigned, PhysReg, ClearAllReferencingSegments);
}

void RegAllocEvictionAdvisor::rememberFailedReassign(
    const LiveInterval &VirtReg, bool HintOnly) const {
  FailedCache->remember(VirtReg, HintOnly);
}

void RegAllocEvictionAdvisor::forgetFailedReassign(
    const LiveInterval &VirtReg) const {
  FailedCache->erase(VirtReg.reg().id());
}

bool RegAllocEvictionAdvisor::canReassign(const LiveInterval &VirtReg,
                                          MCRegister FromReg) const {
  ++NumCanReassignCalls;
  switch (FailedCache->reuse(VirtReg, FromReg)) {
  case reassign::Reuse::HitFalse:
    ++NumCanReassignFail;
    ++NumCanReassignCacheHits;
    return false;
  case reassign::Reuse::RecheckFalse:
    ++NumCanReassignFail;
    ++NumCanReassignRechecks;
    return false;
  case reassign::Reuse::RecheckTrue:
    ++NumCanReassignSuccess;
    ++NumCanReassignRechecks;
    return true;
  case reassign::Reuse::Miss:
    break;
  }

  auto HasRegUnitInterference = [&](MCRegUnit Unit) {
    // Instantiate a "subquery", not to be confused with the Queries array.
    LiveIntervalUnion::Query SubQ(
        VirtReg, Matrix->getLiveUnions()[static_cast<unsigned>(Unit)]);
    return SubQ.checkInterference();
  };

  AllocationOrder Order =
      AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  unsigned Examined = 0;
  for (MCRegister Reg : Order) {
    if (Reg == FromReg)
      continue;
    ++Examined;
    ++NumCanReassignCandidates;
    // If no units have interference, reassignment is possible.
    if (none_of(TRI->regunits(Reg), HasRegUnitInterference)) {
      LLVM_DEBUG(dbgs() << "can reassign: " << VirtReg << " from "
                        << printReg(FromReg, TRI) << " to "
                        << printReg(Reg, TRI) << '\n');
      forgetFailedReassign(VirtReg);
      ++NumCanReassignSuccess;
      return true;
    }
  }
  // Short orders do not repay the unassign bookkeeping. The scan above is
  // their whole cost.
  if (Examined > ReassignCacheMinOrder)
    rememberFailedReassign(VirtReg, Order.hintsOnly());
  ++NumCanReassignFail;
  return false;
}

/// isUrgentEviction - Returns true if this is an urgent eviction. Once a live
/// range becomes small enough, it is urgent that we find a register for it.
/// This is indicated by an infinite spill weight. These urgent live ranges
/// get to evict almost anything.
///
/// Also allow urgent evictions of unspillable ranges from a strictly larger
/// allocation order.
bool RegAllocEvictionAdvisor::isUrgentEviction(const LiveInterval &VirtReg,
                                               const LiveInterval &Intf) const {
  return !VirtReg.isSpillable() &&
         (Intf.isSpillable() ||
          RegClassInfo.getNumAllocatableRegs(MRI->getRegClass(VirtReg.reg())) <
              RegClassInfo.getNumAllocatableRegs(MRI->getRegClass(Intf.reg())));
}

/// shouldEvict - determine if A should evict the assigned live range B. The
/// eviction policy defined by this function together with the allocation order
/// defined by enqueue() decides which registers ultimately end up being split
/// and spilled.
///
/// Cascade numbers are used to prevent infinite loops if this function is a
/// cyclic relation.
///
/// @param A          The live range to be assigned.
/// @param IsHint     True when A is about to be assigned to its preferred
///                   register.
/// @param B          The live range to be evicted.
/// @param BreaksHint True when B is already assigned to its preferred register.
bool DefaultEvictionAdvisor::shouldEvict(const LiveInterval &A, bool IsHint,
                                         const LiveInterval &B,
                                         bool BreaksHint) const {
  bool CanSplit = RA.getExtraInfo().getStage(B) < RS_Spill;

  // Be fairly aggressive about following hints as long as the evictee can be
  // split.
  if (CanSplit && IsHint && !BreaksHint)
    return true;

  if (A.weight() > B.weight()) {
    LLVM_DEBUG(dbgs() << "should evict: " << B << '\n');
    return true;
  }
  return false;
}

/// canEvictHintInterference - return true if the interference for VirtReg
/// on the PhysReg, which is VirtReg's hint, can be evicted in favor of VirtReg.
bool DefaultEvictionAdvisor::canEvictHintInterference(
    const LiveInterval &VirtReg, MCRegister PhysReg,
    const SmallVirtRegSet &FixedRegisters) const {
  EvictionCost MaxCost;
  MaxCost.setBrokenHints(MRI->getRegClass(VirtReg.reg())->getCopyCost());
  return canEvictInterferenceBasedOnCost(VirtReg, PhysReg, true, MaxCost,
                                         FixedRegisters);
}

/// canEvictInterferenceBasedOnCost - Return true if all interferences between
/// VirtReg and PhysReg can be evicted.
///
/// @param VirtReg Live range that is about to be assigned.
/// @param PhysReg Desired register for assignment.
/// @param IsHint  True when PhysReg is VirtReg's preferred register.
/// @param MaxCost Only look for cheaper candidates and update with new cost
///                when returning true.
/// @returns True when interference can be evicted cheaper than MaxCost.
bool DefaultEvictionAdvisor::canEvictInterferenceBasedOnCost(
    const LiveInterval &VirtReg, MCRegister PhysReg, bool IsHint,
    EvictionCost &MaxCost, const SmallVirtRegSet &FixedRegisters) const {
  // It is only possible to evict virtual register interference.
  if (Matrix->checkInterference(VirtReg, PhysReg) > LiveRegMatrix::IK_VirtReg)
    return false;

  bool IsLocal = VirtReg.empty() || LIS->intervalIsInOneMBB(VirtReg);

  // Find VirtReg's cascade number. This will be unassigned if VirtReg was never
  // involved in an eviction before. If a cascade number was assigned, deny
  // evicting anything with the same or a newer cascade number. This prevents
  // infinite eviction loops.
  //
  // This works out so a register without a cascade number is allowed to evict
  // anything, and it can be evicted by anything.
  unsigned Cascade = RA.getExtraInfo().getCascadeOrCurrentNext(VirtReg.reg());

  EvictionCost Cost;
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    // If there is 10 or more interferences, chances are one is heavier.
    const auto &Interferences = Q.interferingVRegs(EvictInterferenceCutoff);
    if (Interferences.size() >= EvictInterferenceCutoff)
      return false;

    // Check if any interfering live range is heavier than MaxWeight.
    for (const LiveInterval *Intf : reverse(Interferences)) {
      assert(Intf->reg().isVirtual() &&
             "Only expecting virtual register interference from query");

      // Do not allow eviction of a virtual register if we are in the middle
      // of last-chance recoloring and this virtual register is one that we
      // have scavenged a physical register for.
      if (FixedRegisters.count(Intf->reg()))
        return false;

      // Never evict spill products. They cannot split or spill.
      if (RA.getExtraInfo().getStage(*Intf) == RS_Done)
        return false;

      bool Urgent = isUrgentEviction(VirtReg, *Intf);
      // Only evict older cascades or live ranges without a cascade.
      unsigned IntfCascade = RA.getExtraInfo().getCascade(Intf->reg());
      if (Cascade == IntfCascade)
        return false;

      if (Cascade < IntfCascade) {
        if (!Urgent)
          return false;
        // We permit breaking cascades for urgent evictions. It should be the
        // last resort, though, so make it really expensive.
        Cost.BrokenHints += 10 * MRI->getRegClass(Intf->reg())->getCopyCost();
      }
      // Would this break a satisfied hint?
      bool BreaksHint = VRM->hasPreferredPhys(Intf->reg());
      // Update eviction cost.
      if (BreaksHint)
        Cost.BrokenHints += MRI->getRegClass(Intf->reg())->getCopyCost();

      Cost.MaxWeight = std::max(Cost.MaxWeight, Intf->weight());
      // Abort if this would be too expensive.
      if (Cost >= MaxCost)
        return false;
      if (Urgent)
        continue;
      // Apply the eviction policy for non-urgent evictions.
      if (!shouldEvict(VirtReg, IsHint, *Intf, BreaksHint))
        return false;
      // If !MaxCost.isMax(), then we're just looking for a cheap register.
      // Evicting another local live range in this case could lead to suboptimal
      // coloring.
      if (!MaxCost.isMax() && IsLocal && LIS->intervalIsInOneMBB(*Intf) &&
          (!EnableLocalReassign || !canReassign(*Intf, PhysReg))) {
        return false;
      }
    }
  }
  MaxCost = Cost;
  return true;
}

MCRegister DefaultEvictionAdvisor::tryFindEvictionCandidate(
    const LiveInterval &VirtReg, const AllocationOrder &Order,
    uint8_t CostPerUseLimit, const SmallVirtRegSet &FixedRegisters) const {
  // Keep track of the cheapest interference seen so far.
  EvictionCost BestCost;
  BestCost.setMax();
  MCRegister BestPhys;
  auto MaybeOrderLimit = getOrderLimit(VirtReg, Order, CostPerUseLimit);
  if (!MaybeOrderLimit)
    return MCRegister::NoRegister;
  unsigned OrderLimit = *MaybeOrderLimit;

  // When we are just looking for a reduced cost per use, don't break any
  // hints, and only evict smaller spill weights.
  if (CostPerUseLimit < uint8_t(~0u)) {
    BestCost.BrokenHints = 0;
    BestCost.MaxWeight = VirtReg.weight();
  }

  for (auto I = Order.begin(), E = Order.getOrderLimitEnd(OrderLimit); I != E;
       ++I) {
    MCRegister PhysReg = *I;
    assert(PhysReg);
    if (!canAllocatePhysReg(VirtReg, CostPerUseLimit, PhysReg) ||
        !canEvictInterferenceBasedOnCost(VirtReg, PhysReg, false, BestCost,
                                         FixedRegisters))
      continue;

    // Best so far.
    BestPhys = PhysReg;

    // Stop if the hint can be used.
    if (I.isHint())
      break;
  }
  return BestPhys;
}
