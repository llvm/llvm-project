//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Target.h"

#include "LatencyBenchmarkRunner.h"
#include "ParallelSnippetGenerator.h"
#include "SerialSnippetGenerator.h"
#include "UopsBenchmarkRunner.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace exegesis {

cl::OptionCategory Options("llvm-exegesis options");
cl::OptionCategory BenchmarkOptions("llvm-exegesis benchmark options");
cl::OptionCategory AnalysisOptions("llvm-exegesis analysis options");

ExegesisTarget::~ExegesisTarget() {} // anchor.

static ExegesisTarget *FirstTarget = nullptr;

const ExegesisTarget *ExegesisTarget::lookup(Triple TT) {
  for (const ExegesisTarget *T = FirstTarget; T != nullptr; T = T->Next) {
    if (T->matchesArch(TT.getArch()))
      return T;
  }
  return nullptr;
}

Expected<std::unique_ptr<pfm::Counter>>
ExegesisTarget::createCounter(StringRef CounterName, const LLVMState &) const {
  pfm::PerfEvent Event(CounterName);
  if (!Event.valid())
    return llvm::make_error<Failure>(
        llvm::Twine("Unable to create counter with name '")
            .concat(CounterName)
            .concat("'"));

  return std::make_unique<pfm::Counter>(std::move(Event));
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

std::unique_ptr<SnippetGenerator> ExegesisTarget::createSnippetGenerator(
    InstructionBenchmark::ModeE Mode, const LLVMState &State,
    const SnippetGenerator::Options &Opts) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createSerialSnippetGenerator(State, Opts);
  case InstructionBenchmark::Uops:
  case InstructionBenchmark::InverseThroughput:
    return createParallelSnippetGenerator(State, Opts);
  }
  return nullptr;
}

Expected<std::unique_ptr<BenchmarkRunner>>
ExegesisTarget::createBenchmarkRunner(
    InstructionBenchmark::ModeE Mode, const LLVMState &State,
    BenchmarkPhaseSelectorE BenchmarkPhaseSelector,
    InstructionBenchmark::ResultAggregationModeE ResultAggMode) const {
  PfmCountersInfo PfmCounters = State.getPfmCounters();
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
  case InstructionBenchmark::InverseThroughput:
    if (BenchmarkPhaseSelector == BenchmarkPhaseSelectorE::Measure &&
        !PfmCounters.CycleCounter) {
      const char *ModeName = Mode == InstructionBenchmark::Latency
                                 ? "latency"
                                 : "inverse_throughput";
      return make_error<Failure>(
          Twine("can't run '")
              .concat(ModeName)
              .concat(
                  "' mode, sched model does not define a cycle counter. You "
                  "can pass --skip-measurements to skip the actual "
                  "benchmarking."));
    }
    return createLatencyBenchmarkRunner(State, Mode, BenchmarkPhaseSelector,
                                        ResultAggMode);
  case InstructionBenchmark::Uops:
    if (BenchmarkPhaseSelector == BenchmarkPhaseSelectorE::Measure &&
        !PfmCounters.UopsCounter && !PfmCounters.IssueCounters)
      return make_error<Failure>(
          "can't run 'uops' mode, sched model does not define uops or issue "
          "counters. You can pass --skip-measurements to skip the actual "
          "benchmarking.");
    return createUopsBenchmarkRunner(State, BenchmarkPhaseSelector,
                                     ResultAggMode);
  }
  return nullptr;
}

std::unique_ptr<SnippetGenerator> ExegesisTarget::createSerialSnippetGenerator(
    const LLVMState &State, const SnippetGenerator::Options &Opts) const {
  return std::make_unique<SerialSnippetGenerator>(State, Opts);
}

std::unique_ptr<SnippetGenerator> ExegesisTarget::createParallelSnippetGenerator(
    const LLVMState &State, const SnippetGenerator::Options &Opts) const {
  return std::make_unique<ParallelSnippetGenerator>(State, Opts);
}

std::unique_ptr<BenchmarkRunner> ExegesisTarget::createLatencyBenchmarkRunner(
    const LLVMState &State, InstructionBenchmark::ModeE Mode,
    BenchmarkPhaseSelectorE BenchmarkPhaseSelector,
    InstructionBenchmark::ResultAggregationModeE ResultAggMode) const {
  return std::make_unique<LatencyBenchmarkRunner>(
      State, Mode, BenchmarkPhaseSelector, ResultAggMode);
}

std::unique_ptr<BenchmarkRunner> ExegesisTarget::createUopsBenchmarkRunner(
    const LLVMState &State, BenchmarkPhaseSelectorE BenchmarkPhaseSelector,
    InstructionBenchmark::ResultAggregationModeE /*unused*/) const {
  return std::make_unique<UopsBenchmarkRunner>(State, BenchmarkPhaseSelector);
}

static_assert(std::is_trivial_v<PfmCountersInfo>,
              "We shouldn't have dynamic initialization here");
const PfmCountersInfo PfmCountersInfo::Default = {nullptr, nullptr, nullptr,
                                                  0u};

const PfmCountersInfo &ExegesisTarget::getPfmCounters(StringRef CpuName) const {
  assert(llvm::is_sorted(
             CpuPfmCounters,
             [](const CpuAndPfmCounters &LHS, const CpuAndPfmCounters &RHS) {
               return strcmp(LHS.CpuName, RHS.CpuName) < 0;
             }) &&
         "CpuPfmCounters table is not sorted");

  // Find entry
  auto Found = llvm::lower_bound(CpuPfmCounters, CpuName);
  if (Found == CpuPfmCounters.end() || StringRef(Found->CpuName) != CpuName) {
    // Use the default.
    if (!CpuPfmCounters.empty() && CpuPfmCounters.begin()->CpuName[0] == '\0') {
      Found = CpuPfmCounters.begin(); // The target specifies a default.
    } else {
      return PfmCountersInfo::Default; // No default for the target.
    }
  }
  assert(Found->PCI && "Missing counters");
  return *Found->PCI;
}

ExegesisTarget::SavedState::~SavedState() {} // anchor.

namespace {

// Default implementation.
class ExegesisDefaultTarget : public ExegesisTarget {
public:
  ExegesisDefaultTarget() : ExegesisTarget({}) {}

private:
  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, unsigned Reg,
                               const APInt &Value) const override {
    llvm_unreachable("Not yet implemented");
  }

  bool matchesArch(Triple::ArchType Arch) const override {
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
