//===- RegAllocPriorityAdvisor.h - live ranges priority advisor -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H
#define LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/RegAllocEvictionAdvisor.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;
class VirtRegMap;
class RAGreedy;

/// Interface to the priority advisor, which is responsible for prioritizing
/// live ranges.
class RegAllocPriorityAdvisor {
public:
  RegAllocPriorityAdvisor(const RegAllocPriorityAdvisor &) = delete;
  RegAllocPriorityAdvisor(RegAllocPriorityAdvisor &&) = delete;
  virtual ~RegAllocPriorityAdvisor() = default;

  /// Find the priority value for a live range. A float value is used since ML
  /// prefers it.
  virtual unsigned getPriority(const LiveInterval &LI) const = 0;

  RegAllocPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                          SlotIndexes *const Indexes);

protected:
  const RAGreedy &RA;
  LiveIntervals *const LIS;
  VirtRegMap *const VRM;
  MachineRegisterInfo *const MRI;
  const TargetRegisterInfo *const TRI;
  const RegisterClassInfo &RegClassInfo;
  SlotIndexes *const Indexes;
  const bool RegClassPriorityTrumpsGlobalness;
  const bool ReverseLocalAssignment;
};

class DefaultPriorityAdvisor : public RegAllocPriorityAdvisor {
public:
  DefaultPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                         SlotIndexes *const Indexes)
      : RegAllocPriorityAdvisor(MF, RA, Indexes) {}

private:
  unsigned getPriority(const LiveInterval &LI) const override;
};

/// Stupid priority advisor which just enqueues in virtual register number
/// order, for debug purposes only.
class DummyPriorityAdvisor : public RegAllocPriorityAdvisor {
public:
  DummyPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA,
                       SlotIndexes *const Indexes)
      : RegAllocPriorityAdvisor(MF, RA, Indexes) {}

private:
  unsigned getPriority(const LiveInterval &LI) const override;
};

/// Common provider for getting the priority advisor and logging rewards.
/// Legacy analysis forwards all calls to this provider.
/// New analysis serves the provider as the analysis result.
/// Expensive setup is done in the constructor, so that the advisor can be
/// created quickly for every machine function.
/// TODO: Remove once legacy PM support is dropped.
class RegAllocPriorityAdvisorProvider {
public:
  enum class AdvisorMode : int { Default, Release, Development, Dummy };

  RegAllocPriorityAdvisorProvider(AdvisorMode Mode) : Mode(Mode) {}

  virtual ~RegAllocPriorityAdvisorProvider() = default;

  virtual void logRewardIfNeeded(const MachineFunction &MF,
                                 function_ref<float()> GetReward) {};

  virtual std::unique_ptr<RegAllocPriorityAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA,
             SlotIndexes &SI) = 0;

  AdvisorMode getAdvisorMode() const { return Mode; }

private:
  const AdvisorMode Mode;
};

class RegAllocPriorityAdvisorAnalysis
    : public AnalysisInfoMixin<RegAllocPriorityAdvisorAnalysis> {
  static AnalysisKey Key;
  friend AnalysisInfoMixin<RegAllocPriorityAdvisorAnalysis>;

public:
  struct Result {
    // Owned by this analysis.
    RegAllocPriorityAdvisorProvider *Provider;

    bool invalidate(MachineFunction &MF, const PreservedAnalyses &PA,
                    MachineFunctionAnalysisManager::Invalidator &Inv) {
      auto PAC = PA.getChecker<RegAllocPriorityAdvisorAnalysis>();
      return !PAC.preservedWhenStateless() ||
             Inv.invalidate<SlotIndexesAnalysis>(MF, PA);
    }
  };

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);

private:
  void initializeProvider(LLVMContext &Ctx);
  void initializeMLProvider(RegAllocPriorityAdvisorProvider::AdvisorMode Mode,
                            LLVMContext &Ctx);
  std::unique_ptr<RegAllocPriorityAdvisorProvider> Provider;
};

class RegAllocPriorityAdvisorAnalysisLegacy : public ImmutablePass {
public:
  using AdvisorMode = RegAllocPriorityAdvisorProvider::AdvisorMode;
  RegAllocPriorityAdvisorAnalysisLegacy(AdvisorMode Mode)
      : ImmutablePass(ID), Mode(Mode) {};
  static char ID;

  /// Get an advisor for the given context (i.e. machine function, etc)
  RegAllocPriorityAdvisorProvider &getProvider() { return *Provider; }
  AdvisorMode getAdvisorMode() const { return Mode; }
  virtual void logRewardIfNeeded(const MachineFunction &MF,
                                 llvm::function_ref<float()> GetReward) {};

protected:
  // This analysis preserves everything, and subclasses may have additional
  // requirements.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  std::unique_ptr<RegAllocPriorityAdvisorProvider> Provider;

private:
  StringRef getPassName() const override;
  const AdvisorMode Mode;
};

/// Specialization for the API used by the analysis infrastructure to create
/// an instance of the priority advisor.
template <> Pass *callDefaultCtor<RegAllocPriorityAdvisorAnalysisLegacy>();

RegAllocPriorityAdvisorAnalysisLegacy *
createReleaseModePriorityAdvisorAnalysis();

RegAllocPriorityAdvisorAnalysisLegacy *
createDevelopmentModePriorityAdvisorAnalysis();

} // namespace llvm

#endif // LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H
