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

static bool hasUnknownOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_UNKNOWN;
}

// FIXME: Handle memory, see PR36905.
static bool hasMemoryOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_MEMORY;
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

InstructionBenchmark::ModeE LatencyBenchmarkRunner::getMode() const {
  return InstructionBenchmark::Latency;
}

llvm::Error LatencyBenchmarkRunner::isInfeasible(
    const llvm::MCInstrDesc &MCInstrDesc) const {
  if (MCInstrDesc.isPseudo())
    return llvm::make_error<BenchmarkFailure>("Infeasible : is pseudo");
  if (llvm::any_of(MCInstrDesc.operands(), hasUnknownOperand))
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has unknown operands");
  if (llvm::any_of(MCInstrDesc.operands(), hasMemoryOperand))
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has memory operands");
  return llvm::Error::success();
}

llvm::Expected<BenchmarkConfiguration>
LatencyBenchmarkRunner::generateSelfAliasingConfiguration(
    const Instruction &Instr,
    const AliasingConfigurations &SelfAliasing) const {
  BenchmarkConfiguration Conf;
  InstructionInstance II(Instr);
  if (SelfAliasing.hasImplicitAliasing()) {
    Conf.Info = "implicit Self cycles, picking random values.";
  } else {
    Conf.Info = "explicit self cycles, selecting one aliasing Conf.";
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice II in the following call.
    setRandomAliasing(SelfAliasing, II, II);
  }
  Conf.Snippet = {II.randomizeUnsetVariablesAndBuild()};
  return Conf;
}

llvm::Expected<BenchmarkConfiguration>
LatencyBenchmarkRunner::generateTwoInstructionConfiguration(
    const Instruction &Instr,
    const AliasingConfigurations &SelfAliasing) const {
  std::vector<unsigned> Opcodes;
  Opcodes.resize(MCInstrInfo.getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  std::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());
  for (const unsigned OtherOpcode : Opcodes) {
    if (OtherOpcode == Instr.Description.Opcode)
      continue;
    const auto &OtherInstrDesc = MCInstrInfo.get(OtherOpcode);
    if (auto E = isInfeasible(OtherInstrDesc)) {
      llvm::consumeError(std::move(E));
      continue;
    }
    const Instruction OtherInstr(OtherInstrDesc, RATC);
    const AliasingConfigurations Forward(Instr, OtherInstr);
    const AliasingConfigurations Back(OtherInstr, Instr);
    if (Forward.empty() || Back.empty())
      continue;
    InstructionInstance ThisII(Instr);
    InstructionInstance OtherII(OtherInstr);
    if (!Forward.hasImplicitAliasing())
      setRandomAliasing(Forward, ThisII, OtherII);
    if (!Back.hasImplicitAliasing())
      setRandomAliasing(Back, OtherII, ThisII);
    BenchmarkConfiguration Conf;
    Conf.Info = llvm::Twine("creating cycle through ")
                    .concat(MCInstrInfo.getName(OtherOpcode))
                    .concat(".")
                    .str();
    Conf.Snippet.push_back(ThisII.randomizeUnsetVariablesAndBuild());
    Conf.Snippet.push_back(OtherII.randomizeUnsetVariablesAndBuild());
    return Conf;
  }
  return llvm::make_error<BenchmarkFailure>(
      "Infeasible : Didn't find any scheme to make the instruction serial");
}

llvm::Expected<BenchmarkConfiguration>
LatencyBenchmarkRunner::generateConfiguration(unsigned Opcode) const {
  const auto &InstrDesc = MCInstrInfo.get(Opcode);
  if (auto E = isInfeasible(InstrDesc))
    return std::move(E);
  const Instruction Instr(InstrDesc, RATC);
  const AliasingConfigurations SelfAliasing(Instr, Instr);
  if (SelfAliasing.empty()) {
    // No self aliasing, trying to create a dependency through another opcode.
    return generateTwoInstructionConfiguration(Instr, SelfAliasing);
  } else {
    return generateSelfAliasingConfiguration(Instr, SelfAliasing);
  }
}

llvm::Expected<std::vector<BenchmarkConfiguration>>
LatencyBenchmarkRunner::createConfigurations(unsigned Opcode) const {
  if (auto E = generateConfiguration(Opcode))
    return std::vector<BenchmarkConfiguration>{E.get()};
  else
    return E.takeError();
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
