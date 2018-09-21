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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

namespace exegesis {

BenchmarkFailure::BenchmarkFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

BenchmarkRunner::BenchmarkRunner(const LLVMState &State,
                                 InstructionBenchmark::ModeE Mode)
    : State(State), Mode(Mode), Scratch(llvm::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

// Repeat the snippet until there are at least NumInstructions in the resulting
// code.
static std::vector<llvm::MCInst>
GenerateInstructions(const BenchmarkCode &BC, const int MinInstructions) {
  std::vector<llvm::MCInst> Code = BC.Instructions;
  for (int I = 0; I < MinInstructions; ++I)
    Code.push_back(BC.Instructions[I % BC.Instructions.size()]);
  return Code;
}

InstructionBenchmark
BenchmarkRunner::runConfiguration(const BenchmarkCode &BC,
                                  unsigned NumRepetitions) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = State.getTargetMachine().getTargetCPU();
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<llvm::MCInst> &Instructions = BC.Instructions;
  if (Instructions.empty()) {
    InstrBenchmark.Error = "Empty snippet";
    return InstrBenchmark;
  }

  InstrBenchmark.Key.Instructions = Instructions;

  // Assemble at least kMinInstructionsForSnippet instructions by repeating the
  // snippet for debug/analysis. This is so that the user clearly understands
  // that the inside instructions are repeated.
  constexpr const int kMinInstructionsForSnippet = 16;
  {
    auto ObjectFilePath = writeObjectFile(
        BC, GenerateInstructions(BC, kMinInstructionsForSnippet));
    if (llvm::Error E = ObjectFilePath.takeError()) {
      InstrBenchmark.Error = llvm::toString(std::move(E));
      return InstrBenchmark;
    }
    const ExecutableFunction EF(State.createTargetMachine(),
                                getObjectFromFile(*ObjectFilePath));
    const auto FnBytes = EF.getFunctionBytes();
    InstrBenchmark.AssembledSnippet.assign(FnBytes.begin(), FnBytes.end());
  }

  // Assemble NumRepetitions instructions repetitions of the snippet for
  // measurements.
  auto ObjectFilePath = writeObjectFile(
      BC, GenerateInstructions(BC, InstrBenchmark.NumRepetitions));
  if (llvm::Error E = ObjectFilePath.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }
  llvm::outs() << "Check generated assembly with: /usr/bin/objdump -d "
               << *ObjectFilePath << "\n";
  const ExecutableFunction EF(State.createTargetMachine(),
                              getObjectFromFile(*ObjectFilePath));
  InstrBenchmark.Measurements = runMeasurements(EF, *Scratch, NumRepetitions);

  return InstrBenchmark;
}

llvm::Expected<std::string>
BenchmarkRunner::writeObjectFile(const BenchmarkCode &BC,
                                 llvm::ArrayRef<llvm::MCInst> Code) const {
  int ResultFD = 0;
  llvm::SmallString<256> ResultPath;
  if (llvm::Error E = llvm::errorCodeToError(llvm::sys::fs::createTemporaryFile(
          "snippet", "o", ResultFD, ResultPath)))
    return std::move(E);
  llvm::raw_fd_ostream OFS(ResultFD, true /*ShouldClose*/);
  assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                   BC.LiveIns, BC.RegisterInitialValues, Code, OFS);
  return ResultPath.str();
}

} // namespace exegesis
