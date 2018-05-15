//===-- BenchmarkRunner.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BenchmarkRunner.h"
#include "InMemoryAssembler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace exegesis {

BenchmarkRunner::InstructionFilter::~InstructionFilter() = default;

BenchmarkRunner::~BenchmarkRunner() = default;

InstructionBenchmark
BenchmarkRunner::run(const LLVMState &State, const unsigned Opcode,
                     unsigned NumRepetitions,
                     const InstructionFilter &Filter) const {
  InstructionBenchmark InstrBenchmark;

  InstrBenchmark.Key.OpcodeName = State.getInstrInfo().getName(Opcode);
  InstrBenchmark.Key.Mode = getDisplayName();
  InstrBenchmark.CpuName = State.getCpuName();
  InstrBenchmark.LLVMTriple = State.getTriple();
  InstrBenchmark.NumRepetitions = NumRepetitions;

  // Ignore instructions that we cannot run.
  if (State.getInstrInfo().get(Opcode).isPseudo()) {
    InstrBenchmark.Error = "Unsupported opcode: isPseudo";
    return InstrBenchmark;
  }
  if (llvm::Error E = Filter.shouldRun(State, Opcode)) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }

  JitFunctionContext Context(State.createTargetMachine());
  auto ExpectedInstructions =
      createCode(State, Opcode, NumRepetitions, Context);
  if (llvm::Error E = ExpectedInstructions.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }

  const std::vector<llvm::MCInst> Instructions = *ExpectedInstructions;
  const JitFunction Function(std::move(Context), Instructions);
  const llvm::StringRef CodeBytes = Function.getFunctionBytes();

  std::string AsmExcerpt;
  constexpr const int ExcerptSize = 100;
  constexpr const int ExcerptTailSize = 10;
  if (CodeBytes.size() <= ExcerptSize) {
    AsmExcerpt = llvm::toHex(CodeBytes);
  } else {
    AsmExcerpt =
        llvm::toHex(CodeBytes.take_front(ExcerptSize - ExcerptTailSize + 3));
    AsmExcerpt += "...";
    AsmExcerpt += llvm::toHex(CodeBytes.take_back(ExcerptTailSize));
  }
  llvm::outs() << "# Asm excerpt: " << AsmExcerpt << "\n";
  llvm::outs().flush(); // In case we crash.

  InstrBenchmark.Measurements =
      runMeasurements(State, Function, NumRepetitions);
  return InstrBenchmark;
}

} // namespace exegesis
