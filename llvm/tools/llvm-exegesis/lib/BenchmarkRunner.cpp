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

BenchmarkRunner::InstructionFilter::~InstructionFilter() = default;
BenchmarkRunner::BenchmarkRunner(const LLVMState &State)
    : State(State), MCInstrInfo(State.getInstrInfo()),
      MCRegisterInfo(State.getRegInfo()),
      RATC(MCRegisterInfo,
           getFunctionReservedRegs(*State.createTargetMachine())) {}
BenchmarkRunner::~BenchmarkRunner() = default;

InstructionBenchmark BenchmarkRunner::run(unsigned Opcode,
                                          const InstructionFilter &Filter,
                                          unsigned NumRepetitions) {
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
  llvm::raw_string_ostream InfoStream(InstrBenchmark.Info);
  llvm::Expected<std::vector<llvm::MCInst>> SnippetOrError =
      createSnippet(RATC, Opcode, InfoStream);
  if (llvm::Error E = SnippetOrError.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }
  std::vector<llvm::MCInst> &Snippet = SnippetOrError.get();
  if (Snippet.empty()) {
    InstrBenchmark.Error = "Empty snippet";
    return InstrBenchmark;
  }

  InfoStream << "Snippet:\n";
  for (const auto &MCInst : Snippet) {
    DumpMCInst(MCRegisterInfo, MCInstrInfo, MCInst, InfoStream);
    InfoStream << "\n";
  }

  std::vector<llvm::MCInst> Code;
  for (int I = 0; I < InstrBenchmark.NumRepetitions; ++I)
    Code.push_back(Snippet[I % Snippet.size()]);

  auto ExpectedObjectPath = writeObjectFile(Code);
  if (llvm::Error E = ExpectedObjectPath.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }

  // FIXME: Check if TargetMachine or ExecutionEngine can be reused instead of
  // creating one everytime.
  const ExecutableFunction EF(State.createTargetMachine(),
                              getObjectFromFile(*ExpectedObjectPath));
  InstrBenchmark.Measurements = runMeasurements(EF, NumRepetitions);

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

} // namespace exegesis
