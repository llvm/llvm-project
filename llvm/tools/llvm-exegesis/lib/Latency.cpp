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
#include "llvm/Support/FormatVariadic.h"

namespace exegesis {

static bool hasUnknownOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_UNKNOWN;
}

// FIXME: Handle memory, see PR36905.
static bool hasMemoryOperand(const llvm::MCOperandInfo &OpInfo) {
  return OpInfo.OperandType == llvm::MCOI::OPERAND_MEMORY;
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

llvm::Error LatencyBenchmarkRunner::isInfeasible(
    const llvm::MCInstrDesc &MCInstrDesc) const {
  if (llvm::any_of(MCInstrDesc.operands(), hasUnknownOperand))
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has unknown operands");
  if (llvm::any_of(MCInstrDesc.operands(), hasMemoryOperand))
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has memory operands");
  return llvm::Error::success();
}

llvm::Expected<SnippetPrototype>
LatencyBenchmarkRunner::generateTwoInstructionPrototype(
    const Instruction &Instr) const {
  std::vector<unsigned> Opcodes;
  Opcodes.resize(State.getInstrInfo().getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  std::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());
  for (const unsigned OtherOpcode : Opcodes) {
    if (OtherOpcode == Instr.Description->Opcode)
      continue;
    const auto &OtherInstrDesc = State.getInstrInfo().get(OtherOpcode);
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
    SnippetPrototype Prototype;
    Prototype.Explanation =
        llvm::formatv("creating cycle through {0}.",
                      State.getInstrInfo().getName(OtherOpcode));
    Prototype.Snippet.push_back(std::move(ThisII));
    Prototype.Snippet.push_back(std::move(OtherII));
    return std::move(Prototype);
  }
  return llvm::make_error<BenchmarkFailure>(
      "Infeasible : Didn't find any scheme to make the instruction serial");
}

llvm::Expected<SnippetPrototype>
LatencyBenchmarkRunner::generatePrototype(unsigned Opcode) const {
  const auto &InstrDesc = State.getInstrInfo().get(Opcode);
  if (auto E = isInfeasible(InstrDesc))
    return std::move(E);
  const Instruction Instr(InstrDesc, RATC);
  if (auto SelfAliasingPrototype = generateSelfAliasingPrototype(Instr))
    return SelfAliasingPrototype;
  else
    llvm::consumeError(SelfAliasingPrototype.takeError());
  // No self aliasing, trying to create a dependency through another opcode.
  return generateTwoInstructionPrototype(Instr);
}

const char *LatencyBenchmarkRunner::getCounterName() const {
  if (!State.getSubtargetInfo().getSchedModel().hasExtraProcessorInfo())
    llvm::report_fatal_error("sched model is missing extra processor info!");
  const char *CounterName = State.getSubtargetInfo()
                                .getSchedModel()
                                .getExtraProcessorInfo()
                                .PfmCounters.CycleCounter;
  if (!CounterName)
    llvm::report_fatal_error("sched model does not define a cycle counter");
  return CounterName;
}

std::vector<BenchmarkMeasure>
LatencyBenchmarkRunner::runMeasurements(const ExecutableFunction &Function,
                                        const unsigned NumRepetitions) const {
  // Cycle measurements include some overhead from the kernel. Repeat the
  // measure several times and take the minimum value.
  constexpr const int NumMeasurements = 30;
  int64_t MinLatency = std::numeric_limits<int64_t>::max();
  const char *CounterName = getCounterName();
  if (!CounterName)
    llvm::report_fatal_error("could not determine cycle counter name");
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
