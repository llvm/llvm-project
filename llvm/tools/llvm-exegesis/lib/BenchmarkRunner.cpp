//===-- BenchmarkRunner.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "MCInstrDescView.h"
#include "PerfHelper.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace exegesis {

BenchmarkFailure::BenchmarkFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

BenchmarkRunner::BenchmarkRunner(const LLVMState &State,
                                 InstructionBenchmark::ModeE Mode)
    : State(State), Mode(Mode), Scratch(llvm::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

// Repeat the snippet until there are at least MinInstructions in the resulting
// code.
static std::vector<llvm::MCInst>
GenerateInstructions(const BenchmarkCode &BC, const size_t MinInstructions) {
  if (BC.Instructions.empty())
    return {};
  std::vector<llvm::MCInst> Code = BC.Instructions;
  for (int I = 0; Code.size() < MinInstructions; ++I)
    Code.push_back(BC.Instructions[I % BC.Instructions.size()]);
  return Code;
}

namespace {
class FunctionExecutorImpl : public BenchmarkRunner::FunctionExecutor {
public:
  FunctionExecutorImpl(const LLVMState &State,
                       llvm::object::OwningBinary<llvm::object::ObjectFile> Obj,
                       BenchmarkRunner::ScratchSpace *Scratch)
      : Function(State.createTargetMachine(), std::move(Obj)),
        Scratch(Scratch) {}

private:
  llvm::Expected<int64_t> runAndMeasure(const char *Counters) const override {
    // We sum counts when there are several counters for a single ProcRes
    // (e.g. P23 on SandyBridge).
    int64_t CounterValue = 0;
    llvm::SmallVector<llvm::StringRef, 2> CounterNames;
    llvm::StringRef(Counters).split(CounterNames, '+');
    char *const ScratchPtr = Scratch->ptr();
    for (auto &CounterName : CounterNames) {
      CounterName = CounterName.trim();
      pfm::PerfEvent PerfEvent(CounterName);
      if (!PerfEvent.valid())
        llvm::report_fatal_error(llvm::Twine("invalid perf event '")
                                     .concat(CounterName)
                                     .concat("'"));
      pfm::Counter Counter(PerfEvent);
      Scratch->clear();
      {
        llvm::CrashRecoveryContext CRC;
        llvm::CrashRecoveryContext::Enable();
        const bool Crashed = !CRC.RunSafely([this, &Counter, ScratchPtr]() {
          Counter.start();
          this->Function(ScratchPtr);
          Counter.stop();
        });
        llvm::CrashRecoveryContext::Disable();
        // FIXME: Better diagnosis.
        if (Crashed)
          return llvm::make_error<BenchmarkFailure>(
              "snippet crashed while running");
      }
      CounterValue += Counter.read();
    }
    return CounterValue;
  }

  const ExecutableFunction Function;
  BenchmarkRunner::ScratchSpace *const Scratch;
};
} // namespace

InstructionBenchmark
BenchmarkRunner::runConfiguration(const BenchmarkCode &BC,
                                  unsigned NumRepetitions,
                                  bool DumpObjectToDisk) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = State.getTargetMachine().getTargetCPU();
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<llvm::MCInst> &Instructions = BC.Instructions;

  InstrBenchmark.Key.Instructions = Instructions;
  InstrBenchmark.Key.RegisterInitialValues = BC.RegisterInitialValues;

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
  const auto Code = GenerateInstructions(BC, InstrBenchmark.NumRepetitions);

  llvm::object::OwningBinary<llvm::object::ObjectFile> ObjectFile;
  if (DumpObjectToDisk) {
    auto ObjectFilePath = writeObjectFile(BC, Code);
    if (llvm::Error E = ObjectFilePath.takeError()) {
      InstrBenchmark.Error = llvm::toString(std::move(E));
      return InstrBenchmark;
    }
    llvm::outs() << "Check generated assembly with: /usr/bin/objdump -d "
                 << *ObjectFilePath << "\n";
    ObjectFile = getObjectFromFile(*ObjectFilePath);
  } else {
    llvm::SmallString<0> Buffer;
    llvm::raw_svector_ostream OS(Buffer);
    assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                     BC.LiveIns, BC.RegisterInitialValues, Code, OS);
    ObjectFile = getObjectFromBuffer(OS.str());
  }

  const FunctionExecutorImpl Executor(State, std::move(ObjectFile),
                                      Scratch.get());
  auto Measurements = runMeasurements(Executor);
  if (llvm::Error E = Measurements.takeError()) {
    InstrBenchmark.Error = llvm::toString(std::move(E));
    return InstrBenchmark;
  }
  InstrBenchmark.Measurements = std::move(*Measurements);
  assert(InstrBenchmark.NumRepetitions > 0 && "invalid NumRepetitions");
  for (BenchmarkMeasure &BM : InstrBenchmark.Measurements) {
    // Scale the measurements by instruction.
    BM.PerInstructionValue /= InstrBenchmark.NumRepetitions;
    // Scale the measurements by snippet.
    BM.PerSnippetValue *= static_cast<double>(BC.Instructions.size()) /
                          InstrBenchmark.NumRepetitions;
  }

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

BenchmarkRunner::FunctionExecutor::~FunctionExecutor() {}

} // namespace exegesis
} // namespace llvm
