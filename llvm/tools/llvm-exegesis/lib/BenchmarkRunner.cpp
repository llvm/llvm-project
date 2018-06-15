//===-- BenchmarkRunner.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

namespace exegesis {

BenchmarkFailure::BenchmarkFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

BenchmarkRunner::InstructionFilter::~InstructionFilter() = default;

BenchmarkRunner::BenchmarkRunner(const LLVMState &State)
    : State(State), MCInstrInfo(State.getInstrInfo()),
      MCRegisterInfo(State.getRegInfo()),
      RATC(MCRegisterInfo,
           getFunctionReservedRegs(*State.createTargetMachine())) {}

BenchmarkRunner::~BenchmarkRunner() = default;

llvm::Expected<std::vector<InstructionBenchmark>>
BenchmarkRunner::run(unsigned Opcode, const InstructionFilter &Filter,
                     unsigned NumRepetitions) {
  // Ignore instructions that we cannot run.
  if (State.getInstrInfo().get(Opcode).isPseudo())
    return llvm::make_error<BenchmarkFailure>("Unsupported opcode: isPseudo");

  if (llvm::Error E = Filter.shouldRun(State, Opcode))
    return std::move(E);

  llvm::Expected<std::vector<BenchmarkConfiguration>> ConfigurationOrError =
      createConfigurations(Opcode);

  if (llvm::Error E = ConfigurationOrError.takeError())
    return std::move(E);

  std::vector<InstructionBenchmark> InstrBenchmarks;
  for (const BenchmarkConfiguration &Conf : ConfigurationOrError.get())
    InstrBenchmarks.push_back(runOne(Conf, Opcode, NumRepetitions));
  return InstrBenchmarks;
}

InstructionBenchmark
BenchmarkRunner::runOne(const BenchmarkConfiguration &Configuration,
                        unsigned Opcode, unsigned NumRepetitions) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = getMode();
  InstrBenchmark.CpuName = State.getCpuName();
  InstrBenchmark.LLVMTriple = State.getTriple();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = Configuration.Info;

  const std::vector<llvm::MCInst> &Snippet = Configuration.Snippet;
  if (Snippet.empty()) {
    InstrBenchmark.Error = "Empty snippet";
    return InstrBenchmark;
  }

  InstrBenchmark.Key.Instructions = Snippet;

  // Repeat the snippet until there are at least NumInstructions in the
  // resulting code. The snippet is always repeated at least once.
  const auto GenerateInstructions = [&Snippet](const int MinInstructions) {
    std::vector<llvm::MCInst> Code = Snippet;
    for (int I = 0; I < MinInstructions; ++I)
      Code.push_back(Snippet[I % Snippet.size()]);
    return Code;
  };

  // Assemble at least kMinInstructionsForSnippet instructions by repeating the
  // snippet for debug/analysis. This is so that the user clearly understands
  // that the inside instructions are repeated.
  constexpr const int kMinInstructionsForSnippet = 16;
  {
    auto EF = createExecutableFunction(
        GenerateInstructions(kMinInstructionsForSnippet));
    if (llvm::Error E = EF.takeError()) {
      InstrBenchmark.Error = llvm::toString(std::move(E));
      return InstrBenchmark;
    }
    const auto FnBytes = EF->getFunctionBytes();
    InstrBenchmark.AssembledSnippet.assign(FnBytes.begin(), FnBytes.end());
  }

  // Assemble NumRepetitions instructions repetitions of the snippet for
  // measurements.
  auto EF = createExecutableFunction(
      GenerateInstructions(InstrBenchmark.NumRepetitions));
  if (llvm::Error E = EF.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }
  InstrBenchmark.Measurements = runMeasurements(*EF, NumRepetitions);

  return InstrBenchmark;
}

llvm::Expected<std::string>
BenchmarkRunner::writeObjectFile(llvm::ArrayRef<llvm::MCInst> Code) const {
  int ResultFD = 0;
  llvm::SmallString<256> ResultPath;
  if (llvm::Error E = llvm::errorCodeToError(llvm::sys::fs::createTemporaryFile(
          "snippet", "o", ResultFD, ResultPath)))
    return std::move(E);
  llvm::raw_fd_ostream OFS(ResultFD, true /*ShouldClose*/);
  assembleToStream(State.createTargetMachine(), Code, OFS);
  llvm::outs() << "Check generated assembly with: /usr/bin/objdump -d "
               << ResultPath << "\n";
  return ResultPath.str();
}

llvm::Expected<ExecutableFunction> BenchmarkRunner::createExecutableFunction(
    llvm::ArrayRef<llvm::MCInst> Code) const {
  auto ExpectedObjectPath = writeObjectFile(Code);
  if (llvm::Error E = ExpectedObjectPath.takeError()) {
    return std::move(E);
  }

  // FIXME: Check if TargetMachine or ExecutionEngine can be reused instead of
  // creating one everytime.
  return ExecutableFunction(State.createTargetMachine(),
                            getObjectFromFile(*ExpectedObjectPath));
}

} // namespace exegesis
