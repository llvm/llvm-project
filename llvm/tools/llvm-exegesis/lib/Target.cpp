//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Target.h"

#include "Latency.h"
#include "Uops.h"

namespace llvm {
namespace exegesis {

ExegesisTarget::~ExegesisTarget() {} // anchor.

static ExegesisTarget *FirstTarget = nullptr;

const ExegesisTarget *ExegesisTarget::lookup(llvm::Triple TT) {
  for (const ExegesisTarget *T = FirstTarget; T != nullptr; T = T->Next) {
    if (T->matchesArch(TT.getArch()))
      return T;
  }
  return nullptr;
}

void ExegesisTarget::registerTarget(ExegesisTarget *Target) {
  if (FirstTarget == nullptr) {
    FirstTarget = Target;
    return;
  }
  if (Target->Next != nullptr)
    return; // Already registered.
  Target->Next = FirstTarget;
  FirstTarget = Target;
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createSnippetGenerator(InstructionBenchmark::ModeE Mode,
                                       const LLVMState &State) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createLatencySnippetGenerator(State);
  case InstructionBenchmark::Uops:
    return createUopsSnippetGenerator(State);
  }
  return nullptr;
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createBenchmarkRunner(InstructionBenchmark::ModeE Mode,
                                      const LLVMState &State) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createLatencyBenchmarkRunner(State);
  case InstructionBenchmark::Uops:
    return createUopsBenchmarkRunner(State);
  }
  return nullptr;
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createLatencySnippetGenerator(const LLVMState &State) const {
  return llvm::make_unique<LatencySnippetGenerator>(State);
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createUopsSnippetGenerator(const LLVMState &State) const {
  return llvm::make_unique<UopsSnippetGenerator>(State);
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createLatencyBenchmarkRunner(const LLVMState &State) const {
  return llvm::make_unique<LatencyBenchmarkRunner>(State);
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createUopsBenchmarkRunner(const LLVMState &State) const {
  return llvm::make_unique<UopsBenchmarkRunner>(State);
}

static_assert(std::is_pod<PfmCountersInfo>::value,
              "We shouldn't have dynamic initialization here");
const PfmCountersInfo PfmCountersInfo::Default = {nullptr, nullptr, nullptr,
                                                  0u};

const PfmCountersInfo &
ExegesisTarget::getPfmCounters(llvm::StringRef CpuName) const {
  assert(std::is_sorted(
             CpuPfmCounters.begin(), CpuPfmCounters.end(),
             [](const CpuAndPfmCounters &LHS, const CpuAndPfmCounters &RHS) {
               return strcmp(LHS.CpuName, RHS.CpuName) < 0;
             }) &&
         "CpuPfmCounters table is not sorted");

  // Find entry
  auto Found =
      std::lower_bound(CpuPfmCounters.begin(), CpuPfmCounters.end(), CpuName);
  if (Found == CpuPfmCounters.end() ||
      llvm::StringRef(Found->CpuName) != CpuName) {
    // Use the default.
    if (CpuPfmCounters.begin() != CpuPfmCounters.end() &&
        CpuPfmCounters.begin()->CpuName[0] == '\0') {
      Found = CpuPfmCounters.begin(); // The target specifies a default.
    } else {
      return PfmCountersInfo::Default; // No default for the target.
    }
  }
  assert(Found->PCI && "Missing counters");
  return *Found->PCI;
}

namespace {

// Default implementation.
class ExegesisDefaultTarget : public ExegesisTarget {
public:
  ExegesisDefaultTarget() : ExegesisTarget({}) {}

private:
  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     unsigned Reg,
                                     const llvm::APInt &Value) const override {
    llvm_unreachable("Not yet implemented");
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    llvm_unreachable("never called");
    return false;
  }
};

} // namespace

const ExegesisTarget &ExegesisTarget::getDefault() {
  static ExegesisDefaultTarget Target;
  return Target;
}

} // namespace exegesis
} // namespace llvm
