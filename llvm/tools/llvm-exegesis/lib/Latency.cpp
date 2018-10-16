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

LatencySnippetGenerator::~LatencySnippetGenerator() = default;

llvm::Expected<std::vector<CodeTemplate>>
generateTwoInstructionPrototypes(const LLVMState &State,
                                 const Instruction &Instr) {
  std::vector<unsigned> Opcodes;
  Opcodes.resize(State.getInstrInfo().getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  std::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());
  for (const unsigned OtherOpcode : Opcodes) {
    if (OtherOpcode == Instr.Description->Opcode)
      continue;
    const Instruction OtherInstr(State, OtherOpcode);
    if (OtherInstr.hasMemoryOperands())
      continue;
    const AliasingConfigurations Forward(Instr, OtherInstr);
    const AliasingConfigurations Back(OtherInstr, Instr);
    if (Forward.empty() || Back.empty())
      continue;
    InstructionTemplate ThisIT(Instr);
    InstructionTemplate OtherIT(OtherInstr);
    if (!Forward.hasImplicitAliasing())
      setRandomAliasing(Forward, ThisIT, OtherIT);
    if (!Back.hasImplicitAliasing())
      setRandomAliasing(Back, OtherIT, ThisIT);
    CodeTemplate CT;
    CT.Info = llvm::formatv("creating cycle through {0}.",
                            State.getInstrInfo().getName(OtherOpcode));
    CT.Instructions.push_back(std::move(ThisIT));
    CT.Instructions.push_back(std::move(OtherIT));
    return getSingleton(CT);
  }
  return llvm::make_error<BenchmarkFailure>(
      "Infeasible : Didn't find any scheme to make the instruction serial");
}

llvm::Expected<std::vector<CodeTemplate>>
LatencySnippetGenerator::generateCodeTemplates(const Instruction &Instr) const {
  if (Instr.hasMemoryOperands())
    return llvm::make_error<BenchmarkFailure>(
        "Infeasible : has memory operands");
  return llvm::handleExpected( //
      generateSelfAliasingCodeTemplates(Instr),
      [this, &Instr]() {
        return generateTwoInstructionPrototypes(State, Instr);
      },
      [](const BenchmarkFailure &) { /*Consume Error*/ });
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

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

std::vector<BenchmarkMeasure>
LatencyBenchmarkRunner::runMeasurements(const ExecutableFunction &Function,
                                        ScratchSpace &Scratch) const {
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
    Scratch.clear();
    Counter.start();
    Function(Scratch.ptr());
    Counter.stop();
    const int64_t Value = Counter.read();
    if (Value < MinLatency)
      MinLatency = Value;
  }
  return {BenchmarkMeasure::Create("latency", MinLatency)};
}

} // namespace exegesis
