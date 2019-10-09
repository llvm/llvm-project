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
#include "Error.h"
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

BenchmarkRunner::BenchmarkRunner(const LLVMState &State,
                                 InstructionBenchmark::ModeE Mode)
    : State(State), Mode(Mode), Scratch(std::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

namespace {
class FunctionExecutorImpl : public BenchmarkRunner::FunctionExecutor {
public:
  FunctionExecutorImpl(const LLVMState &State,
                       object::OwningBinary<object::ObjectFile> Obj,
                       BenchmarkRunner::ScratchSpace *Scratch)
      : Function(State.createTargetMachine(), std::move(Obj)),
        Scratch(Scratch) {}

private:
  Expected<int64_t> runAndMeasure(const char *Counters) const override {
    // We sum counts when there are several counters for a single ProcRes
    // (e.g. P23 on SandyBridge).
    int64_t CounterValue = 0;
    SmallVector<StringRef, 2> CounterNames;
    StringRef(Counters).split(CounterNames, '+');
    char *const ScratchPtr = Scratch->ptr();
    for (auto &CounterName : CounterNames) {
      CounterName = CounterName.trim();
      pfm::PerfEvent PerfEvent(CounterName);
      if (!PerfEvent.valid())
        report_fatal_error(
            Twine("invalid perf event '").concat(CounterName).concat("'"));
      pfm::Counter Counter(PerfEvent);
      Scratch->clear();
      {
        CrashRecoveryContext CRC;
        CrashRecoveryContext::Enable();
        const bool Crashed = !CRC.RunSafely([this, &Counter, ScratchPtr]() {
          Counter.start();
          this->Function(ScratchPtr);
          Counter.stop();
        });
        CrashRecoveryContext::Disable();
        // FIXME: Better diagnosis.
        if (Crashed)
          return make_error<Failure>("snippet crashed while running");
      }
      CounterValue += Counter.read();
    }
    return CounterValue;
  }

  const ExecutableFunction Function;
  BenchmarkRunner::ScratchSpace *const Scratch;
};
} // namespace

InstructionBenchmark BenchmarkRunner::runConfiguration(
    const BenchmarkCode &BC, unsigned NumRepetitions,
    const SnippetRepetitor &Repetitor, bool DumpObjectToDisk) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = State.getTargetMachine().getTargetCPU();
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<MCInst> &Instructions = BC.Key.Instructions;

  InstrBenchmark.Key = BC.Key;

  // Assemble at least kMinInstructionsForSnippet instructions by repeating the
  // snippet for debug/analysis. This is so that the user clearly understands
  // that the inside instructions are repeated.
  constexpr const int kMinInstructionsForSnippet = 16;
  {
    SmallString<0> Buffer;
    raw_svector_ostream OS(Buffer);
    assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                     BC.LiveIns, BC.Key.RegisterInitialValues,
                     Repetitor.Repeat(Instructions, kMinInstructionsForSnippet),
                     OS);
    const ExecutableFunction EF(State.createTargetMachine(),
                                getObjectFromBuffer(OS.str()));
    const auto FnBytes = EF.getFunctionBytes();
    InstrBenchmark.AssembledSnippet.assign(FnBytes.begin(), FnBytes.end());
  }

  // Assemble NumRepetitions instructions repetitions of the snippet for
  // measurements.
  const auto Filler =
      Repetitor.Repeat(Instructions, InstrBenchmark.NumRepetitions);

  object::OwningBinary<object::ObjectFile> ObjectFile;
  if (DumpObjectToDisk) {
    auto ObjectFilePath = writeObjectFile(BC, Filler);
    if (Error E = ObjectFilePath.takeError()) {
      InstrBenchmark.Error = toString(std::move(E));
      return InstrBenchmark;
    }
    outs() << "Check generated assembly with: /usr/bin/objdump -d "
           << *ObjectFilePath << "\n";
    ObjectFile = getObjectFromFile(*ObjectFilePath);
  } else {
    SmallString<0> Buffer;
    raw_svector_ostream OS(Buffer);
    assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                     BC.LiveIns, BC.Key.RegisterInitialValues, Filler, OS);
    ObjectFile = getObjectFromBuffer(OS.str());
  }

  const FunctionExecutorImpl Executor(State, std::move(ObjectFile),
                                      Scratch.get());
  auto Measurements = runMeasurements(Executor);
  if (Error E = Measurements.takeError()) {
    InstrBenchmark.Error = toString(std::move(E));
    return InstrBenchmark;
  }
  InstrBenchmark.Measurements = std::move(*Measurements);
  assert(InstrBenchmark.NumRepetitions > 0 && "invalid NumRepetitions");
  for (BenchmarkMeasure &BM : InstrBenchmark.Measurements) {
    // Scale the measurements by instruction.
    BM.PerInstructionValue /= InstrBenchmark.NumRepetitions;
    // Scale the measurements by snippet.
    BM.PerSnippetValue *= static_cast<double>(Instructions.size()) /
                          InstrBenchmark.NumRepetitions;
  }

  return InstrBenchmark;
}

Expected<std::string>
BenchmarkRunner::writeObjectFile(const BenchmarkCode &BC,
                                 const FillFunction &FillFunction) const {
  int ResultFD = 0;
  SmallString<256> ResultPath;
  if (Error E = errorCodeToError(
          sys::fs::createTemporaryFile("snippet", "o", ResultFD, ResultPath)))
    return std::move(E);
  raw_fd_ostream OFS(ResultFD, true /*ShouldClose*/);
  assembleToStream(State.getExegesisTarget(), State.createTargetMachine(),
                   BC.LiveIns, BC.Key.RegisterInitialValues, FillFunction, OFS);
  return ResultPath.str();
}

BenchmarkRunner::FunctionExecutor::~FunctionExecutor() {}

} // namespace exegesis
} // namespace llvm
