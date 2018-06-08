//===-- Latency.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Latency.h"

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "PerfHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"

namespace exegesis {

static bool HasUnknownOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_UNKNOWN;
}

// FIXME: Handle memory, see PR36905.
static bool HasMemoryOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_MEMORY;
}

static bool IsInfeasible(const Instruction &Instruction, std::string &Error) {
  const auto &MCInstrDesc = Instruction.Description;
  if (MCInstrDesc.isPseudo()) {
    Error = "is pseudo";
    return true;
  }
  if (llvm::any_of(MCInstrDesc.operands(), HasUnknownOperand)) {
    Error = "has unknown operands";
    return true;
  }
  if (llvm::any_of(MCInstrDesc.operands(), HasMemoryOperand)) {
    Error = "has memory operands";
    return true;
  }
  return false;
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

InstructionBenchmark::ModeE LatencyBenchmarkRunner::getMode() const {
  return InstructionBenchmark::Latency;
}

llvm::Expected<std::vector<BenchmarkConfiguration>>
LatencyBenchmarkRunner::createConfigurations(RegisterAliasingTrackerCache &RATC,
                                             unsigned Opcode) const {
  const llvm::MCInstrDesc &MCInstrDesc = MCInstrInfo.get(Opcode);
  const Instruction ThisInstruction(MCInstrDesc, RATC);

  std::string Error;
  if (IsInfeasible(ThisInstruction, Error))
    return llvm::make_error<llvm::StringError>(
        llvm::Twine("Infeasible : ").concat(Error),
        llvm::inconvertibleErrorCode());

  BenchmarkConfiguration Conf;
  const AliasingConfigurations SelfAliasing(ThisInstruction, ThisInstruction);
  if (!SelfAliasing.empty()) {
    if (!SelfAliasing.hasImplicitAliasing()) {
      Conf.Info = "explicit self cycles, selecting one aliasing Conf.";
      setRandomAliasing(SelfAliasing);
    } else {
      Conf.Info = "implicit Self cycles, picking random values.";
    }
    Conf.Snippet = {randomizeUnsetVariablesAndBuild(ThisInstruction)};
    return std::vector<BenchmarkConfiguration>{Conf};
  }

  // Let's try to create a dependency through another opcode.
  std::vector<unsigned> Opcodes;
  Opcodes.resize(MCInstrInfo.getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  std::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());
  for (const unsigned OtherOpcode : Opcodes) {
    clearVariableAssignments(ThisInstruction);
    if (OtherOpcode == Opcode)
      continue;
    const Instruction OtherInstruction(MCInstrInfo.get(OtherOpcode), RATC);
    if (IsInfeasible(OtherInstruction, Error))
      continue;
    const AliasingConfigurations Forward(ThisInstruction, OtherInstruction);
    const AliasingConfigurations Back(OtherInstruction, ThisInstruction);
    if (Forward.empty() || Back.empty())
      continue;
    setRandomAliasing(Forward);
    setRandomAliasing(Back);
    Conf.Info = llvm::Twine("creating cycle through ")
                    .concat(MCInstrInfo.getName(OtherOpcode))
                    .concat(".")
                    .str();
    Conf.Snippet.push_back(randomizeUnsetVariablesAndBuild(ThisInstruction));
    Conf.Snippet.push_back(randomizeUnsetVariablesAndBuild(OtherInstruction));
    return std::vector<BenchmarkConfiguration>{Conf};
  }

  return llvm::make_error<llvm::StringError>(
      "Infeasible : Didn't find any scheme to make the instruction serial",
      llvm::inconvertibleErrorCode());
}

std::vector<BenchmarkMeasure>
LatencyBenchmarkRunner::runMeasurements(const ExecutableFunction &Function,
                                        const unsigned NumRepetitions) const {
  // Cycle measurements include some overhead from the kernel. Repeat the
  // measure several times and take the minimum value.
  constexpr const int NumMeasurements = 30;
  int64_t MinLatency = std::numeric_limits<int64_t>::max();
  const char *CounterName = State.getSubtargetInfo()
                                .getSchedModel()
                                .getExtraProcessorInfo()
                                .PfmCounters.CycleCounter;
  if (!CounterName)
    llvm::report_fatal_error("sched model does not define a cycle counter");
  const pfm::PerfEvent CyclesPerfEvent(CounterName);
  if (!CyclesPerfEvent.valid())
    llvm::report_fatal_error("invalid perf event");
  for (size_t I = 0; I < NumMeasurements; ++I) {
    pfm::Counter Counter(CyclesPerfEvent);
    Counter.start();
    Function();
    Counter.stop();
    const int64_t Value = Counter.read();
    if (Value < MinLatency)
      MinLatency = Value;
  }
  return {{"latency", static_cast<double>(MinLatency) / NumRepetitions, ""}};
}

} // namespace exegesis
