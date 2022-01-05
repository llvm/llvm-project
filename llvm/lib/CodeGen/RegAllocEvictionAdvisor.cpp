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

#include "RegAllocEvictionAdvisor.h"
#include "RegAllocGreedy.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<RegAllocEvictionAdvisorAnalysis::AdvisorMode> Mode(
    "regalloc-enable-advisor", cl::Hidden,
    cl::init(RegAllocEvictionAdvisorAnalysis::AdvisorMode::Default),
    cl::desc("Enable regalloc advisor mode"),
    cl::values(
        clEnumValN(RegAllocEvictionAdvisorAnalysis::AdvisorMode::Default,
                   "default", "Default"),
        clEnumValN(RegAllocEvictionAdvisorAnalysis::AdvisorMode::Release,
                   "release", "precompiled"),
        clEnumValN(RegAllocEvictionAdvisorAnalysis::AdvisorMode::Development,
                   "development", "for training")));

static cl::opt<bool> EnableLocalReassignment(
    "enable-local-reassign", cl::Hidden,
    cl::desc("Local reassignment can yield better allocation decisions, but "
             "may be compile time intensive"),
    cl::init(false));

#define DEBUG_TYPE "regalloc"

char RegAllocEvictionAdvisorAnalysis::ID = 0;
INITIALIZE_PASS(RegAllocEvictionAdvisorAnalysis, "regalloc-evict",
                "Regalloc eviction policy", false, true)

namespace {
class DefaultEvictionAdvisorAnalysis final
    : public RegAllocEvictionAdvisorAnalysis {
public:
  DefaultEvictionAdvisorAnalysis(bool NotAsRequested)
      : RegAllocEvictionAdvisorAnalysis(AdvisorMode::Default),
        NotAsRequested(NotAsRequested) {}

  // support for isa<> and dyn_cast.
  static bool classof(const RegAllocEvictionAdvisorAnalysis *R) {
    return R->getAdvisorMode() == AdvisorMode::Default;
  }

private:
  std::unique_ptr<RegAllocEvictionAdvisor>
  getAdvisor(const MachineFunction &MF, const RAGreedy &RA) override {
    return std::make_unique<DefaultEvictionAdvisor>(MF, RA);
  }
  bool doInitialization(Module &M) override {
    if (NotAsRequested)
      M.getContext().emitError("Requested regalloc eviction advisor analysis "
                               "could be created. Using default");
    return RegAllocEvictionAdvisorAnalysis::doInitialization(M);
  }
  const bool NotAsRequested;
};
} // namespace

template <> Pass *llvm::callDefaultCtor<RegAllocEvictionAdvisorAnalysis>() {
  Pass *Ret = nullptr;
  switch (Mode) {
  case RegAllocEvictionAdvisorAnalysis::AdvisorMode::Default:
    Ret = new DefaultEvictionAdvisorAnalysis(/*NotAsRequested*/ false);
    break;
  case RegAllocEvictionAdvisorAnalysis::AdvisorMode::Development:
    // TODO(mtrofin): add implementation
    break;
  case RegAllocEvictionAdvisorAnalysis::AdvisorMode::Release:
    // TODO(mtrofin): add implementation
    break;
  }
  if (Ret)
    return Ret;
  return new DefaultEvictionAdvisorAnalysis(/*NotAsRequested*/ true);
}

StringRef RegAllocEvictionAdvisorAnalysis::getPassName() const {
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
      EnableLocalReassign(EnableLocalReassignment ||
                          MF.getSubtarget().enableRALocalReassignment(
                              MF.getTarget().getOptLevel())) {}
