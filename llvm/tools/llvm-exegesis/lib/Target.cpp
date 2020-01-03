//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Target.h"

#include "Latency.h"
#include "Uops.h"

namespace llvm {
namespace exegesis {

ExegesisTarget::~ExegesisTarget() {} // anchor.

static ExegesisTarget *FirstTarget = nullptr;

const ExegesisTarget *ExegesisTarget::lookup(Triple TT) {
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

std::unique_ptr<SnippetGenerator> ExegesisTarget::createSnippetGenerator(
    InstructionBenchmark::ModeE Mode, const LLVMState &State,
    const SnippetGenerator::Options &Opts) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createLatencySnippetGenerator(State, Opts);
  case InstructionBenchmark::Uops:
  case InstructionBenchmark::InverseThroughput:
    return createUopsSnippetGenerator(State, Opts);
  }
  return nullptr;
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createBenchmarkRunner(InstructionBenchmark::ModeE Mode,
                                      const LLVMState &State) const {
  PfmCountersInfo PfmCounters = State.getPfmCounters();
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
  case InstructionBenchmark::InverseThroughput:
    if (!PfmCounters.CycleCounter) {
      const char *ModeName = Mode == InstructionBenchmark::Latency
                                 ? "latency"
                                 : "inverse_throughput";
      report_fatal_error(Twine("can't run '").concat(ModeName).concat("' mode, "
                               "sched model does not define a cycle counter."));
    }
    return createLatencyBenchmarkRunner(State, Mode);
  case InstructionBenchmark::Uops:
    if (!PfmCounters.UopsCounter && !PfmCounters.IssueCounters)
      report_fatal_error("can't run 'uops' mode, sched model does not define "
                         "uops or issue counters.");
    return createUopsBenchmarkRunner(State);
  }
  return nullptr;
}

std::unique_ptr<SnippetGenerator> ExegesisTarget::createLatencySnippetGenerator(
    const LLVMState &State, const SnippetGenerator::Options &Opts) const {
  return std::make_unique<LatencySnippetGenerator>(State, Opts);
}

std::unique_ptr<SnippetGenerator> ExegesisTarget::createUopsSnippetGenerator(
    const LLVMState &State, const SnippetGenerator::Options &Opts) const {
  return std::make_unique<UopsSnippetGenerator>(State, Opts);
}

std::unique_ptr<BenchmarkRunner> ExegesisTarget::createLatencyBenchmarkRunner(
    const LLVMState &State, InstructionBenchmark::ModeE Mode) const {
  return std::make_unique<LatencyBenchmarkRunner>(State, Mode);
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createUopsBenchmarkRunner(const LLVMState &State) const {
  return std::make_unique<UopsBenchmarkRunner>(State);
}

void ExegesisTarget::randomizeMCOperand(const Instruction &Instr,
                                        const Variable &Var,
                                        MCOperand &AssignedValue,
                                        const BitVector &ForbiddenRegs) const {
  const Operand &Op = Instr.getPrimaryOperand(Var);
  switch (Op.getExplicitOperandInfo().OperandType) {
  case MCOI::OperandType::OPERAND_IMMEDIATE:
    // FIXME: explore immediate values too.
    AssignedValue = MCOperand::createImm(1);
    break;
  case MCOI::OperandType::OPERAND_REGISTER: {
    assert(Op.isReg());
    auto AllowedRegs = Op.getRegisterAliasing().sourceBits();
    assert(AllowedRegs.size() == ForbiddenRegs.size());
    for (auto I : ForbiddenRegs.set_bits())
      AllowedRegs.reset(I);
    AssignedValue = MCOperand::createReg(randomBit(AllowedRegs));
    break;
  }
  default:
    break;
  }
}

static_assert(std::is_pod<PfmCountersInfo>::value,
              "We shouldn't have dynamic initialization here");
const PfmCountersInfo PfmCountersInfo::Default = {nullptr, nullptr, nullptr,
                                                  0u};

const PfmCountersInfo &ExegesisTarget::getPfmCounters(StringRef CpuName) const {
  assert(std::is_sorted(
             CpuPfmCounters.begin(), CpuPfmCounters.end(),
             [](const CpuAndPfmCounters &LHS, const CpuAndPfmCounters &RHS) {
               return strcmp(LHS.CpuName, RHS.CpuName) < 0;
             }) &&
         "CpuPfmCounters table is not sorted");

  // Find entry
  auto Found =
      std::lower_bound(CpuPfmCounters.begin(), CpuPfmCounters.end(), CpuName);
  if (Found == CpuPfmCounters.end() || StringRef(Found->CpuName) != CpuName) {
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
