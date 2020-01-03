//===-- Latency.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Latency.h"

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "PerfHelper.h"
#include "Target.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
namespace exegesis {

struct ExecutionClass {
  ExecutionMode Mask;
  const char *Description;
} static const kExecutionClasses[] = {
    {ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS |
         ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS,
     "Repeating a single implicitly serial instruction"},
    {ExecutionMode::SERIAL_VIA_EXPLICIT_REGS,
     "Repeating a single explicitly serial instruction"},
    {ExecutionMode::SERIAL_VIA_MEMORY_INSTR |
         ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR,
     "Repeating two instructions"},
};

static constexpr size_t kMaxAliasingInstructions = 10;

static std::vector<const Instruction *>
computeAliasingInstructions(const LLVMState &State, const Instruction *Instr,
                            size_t MaxAliasingInstructions,
                            const BitVector &ForbiddenRegisters) {
  // Randomly iterate the set of instructions.
  std::vector<unsigned> Opcodes;
  Opcodes.resize(State.getInstrInfo().getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  std::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());

  std::vector<const Instruction *> AliasingInstructions;
  for (const unsigned OtherOpcode : Opcodes) {
    if (OtherOpcode == Instr->Description.getOpcode())
      continue;
    const Instruction &OtherInstr = State.getIC().getInstr(OtherOpcode);
    if (OtherInstr.hasMemoryOperands())
      continue;
    if (Instr->hasAliasingRegistersThrough(OtherInstr, ForbiddenRegisters))
      AliasingInstructions.push_back(&OtherInstr);
    if (AliasingInstructions.size() >= MaxAliasingInstructions)
      break;
  }
  return AliasingInstructions;
}

static ExecutionMode getExecutionModes(const Instruction &Instr,
                                       const BitVector &ForbiddenRegisters) {
  ExecutionMode EM = ExecutionMode::UNKNOWN;
  if (Instr.hasAliasingImplicitRegisters())
    EM |= ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS;
  if (Instr.hasTiedRegisters())
    EM |= ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS;
  if (Instr.hasMemoryOperands())
    EM |= ExecutionMode::SERIAL_VIA_MEMORY_INSTR;
  else {
    if (Instr.hasAliasingRegisters(ForbiddenRegisters))
      EM |= ExecutionMode::SERIAL_VIA_EXPLICIT_REGS;
    if (Instr.hasOneUseOrOneDef())
      EM |= ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR;
  }
  return EM;
}

static void appendCodeTemplates(const LLVMState &State,
                                const Instruction *Instr,
                                const BitVector &ForbiddenRegisters,
                                ExecutionMode ExecutionModeBit,
                                StringRef ExecutionClassDescription,
                                std::vector<CodeTemplate> &CodeTemplates) {
  assert(isEnumValue(ExecutionModeBit) && "Bit must be a power of two");
  switch (ExecutionModeBit) {
  case ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS:
    // Nothing to do, the instruction is always serial.
    LLVM_FALLTHROUGH;
  case ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS: {
    // Picking whatever value for the tied variable will make the instruction
    // serial.
    CodeTemplate CT;
    CT.Execution = ExecutionModeBit;
    CT.Info = ExecutionClassDescription;
    CT.Instructions.push_back(Instr);
    CodeTemplates.push_back(std::move(CT));
    return;
  }
  case ExecutionMode::SERIAL_VIA_MEMORY_INSTR: {
    // Select back-to-back memory instruction.
    // TODO: Implement me.
    return;
  }
  case ExecutionMode::SERIAL_VIA_EXPLICIT_REGS: {
    // Making the execution of this instruction serial by selecting one def
    // register to alias with one use register.
    const AliasingConfigurations SelfAliasing(*Instr, *Instr);
    assert(!SelfAliasing.empty() && !SelfAliasing.hasImplicitAliasing() &&
           "Instr must alias itself explicitly");
    InstructionTemplate IT(Instr);
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice IT in the following call.
    setRandomAliasing(SelfAliasing, IT, IT);
    CodeTemplate CT;
    CT.Execution = ExecutionModeBit;
    CT.Info = ExecutionClassDescription;
    CT.Instructions.push_back(std::move(IT));
    CodeTemplates.push_back(std::move(CT));
    return;
  }
  case ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR: {
    // Select back-to-back non-memory instruction.
    for (const auto *OtherInstr : computeAliasingInstructions(
             State, Instr, kMaxAliasingInstructions, ForbiddenRegisters)) {
      const AliasingConfigurations Forward(*Instr, *OtherInstr);
      const AliasingConfigurations Back(*OtherInstr, *Instr);
      InstructionTemplate ThisIT(Instr);
      InstructionTemplate OtherIT(OtherInstr);
      if (!Forward.hasImplicitAliasing())
        setRandomAliasing(Forward, ThisIT, OtherIT);
      if (!Back.hasImplicitAliasing())
        setRandomAliasing(Back, OtherIT, ThisIT);
      CodeTemplate CT;
      CT.Execution = ExecutionModeBit;
      CT.Info = ExecutionClassDescription;
      CT.Instructions.push_back(std::move(ThisIT));
      CT.Instructions.push_back(std::move(OtherIT));
      CodeTemplates.push_back(std::move(CT));
    }
    return;
  }
  default:
    llvm_unreachable("Unhandled enum value");
  }
}

LatencySnippetGenerator::~LatencySnippetGenerator() = default;

Expected<std::vector<CodeTemplate>>
LatencySnippetGenerator::generateCodeTemplates(
    const Instruction &Instr, const BitVector &ForbiddenRegisters) const {
  std::vector<CodeTemplate> Results;
  const ExecutionMode EM = getExecutionModes(Instr, ForbiddenRegisters);
  for (const auto EC : kExecutionClasses) {
    for (const auto ExecutionModeBit : getExecutionModeBits(EM & EC.Mask))
      appendCodeTemplates(State, &Instr, ForbiddenRegisters, ExecutionModeBit,
                          EC.Description, Results);
    if (!Results.empty())
      break;
  }
  if (Results.empty())
    return make_error<Failure>(
        "No strategy found to make the execution serial");
  return std::move(Results);
}

LatencyBenchmarkRunner::LatencyBenchmarkRunner(const LLVMState &State,
                                               InstructionBenchmark::ModeE Mode)
    : BenchmarkRunner(State, Mode) {
  assert((Mode == InstructionBenchmark::Latency ||
          Mode == InstructionBenchmark::InverseThroughput) &&
         "invalid mode");
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

Expected<std::vector<BenchmarkMeasure>> LatencyBenchmarkRunner::runMeasurements(
    const FunctionExecutor &Executor) const {
  // Cycle measurements include some overhead from the kernel. Repeat the
  // measure several times and take the minimum value.
  constexpr const int NumMeasurements = 30;
  int64_t MinValue = std::numeric_limits<int64_t>::max();
  const char *CounterName = State.getPfmCounters().CycleCounter;
  for (size_t I = 0; I < NumMeasurements; ++I) {
    auto ExpectedCounterValue = Executor.runAndMeasure(CounterName);
    if (!ExpectedCounterValue)
      return ExpectedCounterValue.takeError();
    if (*ExpectedCounterValue < MinValue)
      MinValue = *ExpectedCounterValue;
  }
  std::vector<BenchmarkMeasure> Result;
  switch (Mode) {
  case InstructionBenchmark::Latency:
    Result = {BenchmarkMeasure::Create("latency", MinValue)};
    break;
  case InstructionBenchmark::InverseThroughput:
    Result = {BenchmarkMeasure::Create("inverse_throughput", MinValue)};
    break;
  default:
    break;
  }
  return std::move(Result);
}

} // namespace exegesis
} // namespace llvm
