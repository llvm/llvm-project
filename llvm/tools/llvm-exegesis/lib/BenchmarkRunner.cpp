//===-- BenchmarkRunner.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <memory>
#include <string>

#include "Assembler.h"
#include "BenchmarkRunner.h"
#include "Error.h"
#include "MCInstrDescView.h"
#include "PerfHelper.h"
#include "Target.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace exegesis {

BenchmarkRunner::BenchmarkRunner(const LLVMState &State,
                                 InstructionBenchmark::ModeE Mode,
                                 bool BenchmarkSkipMeasurements)
    : State(State), Mode(Mode),
      BenchmarkSkipMeasurements(BenchmarkSkipMeasurements),
      Scratch(std::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

namespace {
class FunctionExecutorImpl : public BenchmarkRunner::FunctionExecutor {
public:
  FunctionExecutorImpl(const LLVMState &State,
                       object::OwningBinary<object::ObjectFile> Obj,
                       BenchmarkRunner::ScratchSpace *Scratch)
      : State(State), Function(State.createTargetMachine(), std::move(Obj)),
        Scratch(Scratch) {}

private:
  Expected<int64_t> runAndMeasure(const char *Counters) const override {
    auto ResultOrError = runAndSample(Counters);
    if (ResultOrError)
      return ResultOrError.get()[0];
    return ResultOrError.takeError();
  }

  static void
  accumulateCounterValues(const llvm::SmallVector<int64_t, 4> &NewValues,
                          llvm::SmallVector<int64_t, 4> *Result) {
    const size_t NumValues = std::max(NewValues.size(), Result->size());
    if (NumValues > Result->size())
      Result->resize(NumValues, 0);
    for (size_t I = 0, End = NewValues.size(); I < End; ++I)
      (*Result)[I] += NewValues[I];
  }

  Expected<llvm::SmallVector<int64_t, 4>>
  runAndSample(const char *Counters) const override {
    // We sum counts when there are several counters for a single ProcRes
    // (e.g. P23 on SandyBridge).
    llvm::SmallVector<int64_t, 4> CounterValues;
    int Reserved = 0;
    SmallVector<StringRef, 2> CounterNames;
    StringRef(Counters).split(CounterNames, '+');
    char *const ScratchPtr = Scratch->ptr();
    const ExegesisTarget &ET = State.getExegesisTarget();
    for (auto &CounterName : CounterNames) {
      CounterName = CounterName.trim();
      auto CounterOrError = ET.createCounter(CounterName, State);

      if (!CounterOrError)
        return CounterOrError.takeError();

      pfm::Counter *Counter = CounterOrError.get().get();
      if (Reserved == 0) {
        Reserved = Counter->numValues();
        CounterValues.reserve(Reserved);
      } else if (Reserved != Counter->numValues())
        // It'd be wrong to accumulate vectors of different sizes.
        return make_error<Failure>(
            llvm::Twine("Inconsistent number of values for counter ")
                .concat(CounterName)
                .concat(std::to_string(Counter->numValues()))
                .concat(" vs expected of ")
                .concat(std::to_string(Reserved)));
      Scratch->clear();
      {
        auto PS = ET.withSavedState();
        CrashRecoveryContext CRC;
        CrashRecoveryContext::Enable();
        const bool Crashed = !CRC.RunSafely([this, Counter, ScratchPtr]() {
          Counter->start();
          this->Function(ScratchPtr);
          Counter->stop();
        });
        CrashRecoveryContext::Disable();
        PS.reset();
        if (Crashed) {
          std::string Msg = "snippet crashed while running";
#ifdef LLVM_ON_UNIX
          // See "Exit Status for Commands":
          // https://pubs.opengroup.org/onlinepubs/9699919799/xrat/V4_xcu_chap02.html
          constexpr const int kSigOffset = 128;
          if (const char *const SigName = strsignal(CRC.RetCode - kSigOffset)) {
            Msg += ": ";
            Msg += SigName;
          }
#endif
          return make_error<SnippetCrash>(std::move(Msg));
        }
      }

      auto ValueOrError = Counter->readOrError(Function.getFunctionBytes());
      if (!ValueOrError)
        return ValueOrError.takeError();
      accumulateCounterValues(ValueOrError.get(), &CounterValues);
    }
    return CounterValues;
  }

  const LLVMState &State;
  const ExecutableFunction Function;
  BenchmarkRunner::ScratchSpace *const Scratch;
};
} // namespace

Expected<InstructionBenchmark> BenchmarkRunner::runConfiguration(
    const BenchmarkCode &BC, unsigned NumRepetitions, unsigned LoopBodySize,
    ArrayRef<std::unique_ptr<const SnippetRepetitor>> Repetitors,
    bool DumpObjectToDisk) const {
  InstructionBenchmark InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = std::string(State.getTargetMachine().getTargetCPU());
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<MCInst> &Instructions = BC.Key.Instructions;

  InstrBenchmark.Key = BC.Key;

  // If we end up having an error, and we've previously succeeded with
  // some other Repetitor, we want to discard the previous measurements.
  struct ClearBenchmarkOnReturn {
    ClearBenchmarkOnReturn(InstructionBenchmark *IB) : IB(IB) {}
    ~ClearBenchmarkOnReturn() {
      if (Clear)
        IB->Measurements.clear();
    }
    void disarm() { Clear = false; }

  private:
    InstructionBenchmark *const IB;
    bool Clear = true;
  };
  ClearBenchmarkOnReturn CBOR(&InstrBenchmark);

  for (const std::unique_ptr<const SnippetRepetitor> &Repetitor : Repetitors) {
    // Assemble at least kMinInstructionsForSnippet instructions by repeating
    // the snippet for debug/analysis. This is so that the user clearly
    // understands that the inside instructions are repeated.
    const int MinInstructionsForSnippet = 4 * Instructions.size();
    const int LoopBodySizeForSnippet = 2 * Instructions.size();
    {
      SmallString<0> Buffer;
      raw_svector_ostream OS(Buffer);
      if (Error E = assembleToStream(
              State.getExegesisTarget(), State.createTargetMachine(),
              BC.LiveIns, BC.Key.RegisterInitialValues,
              Repetitor->Repeat(Instructions, MinInstructionsForSnippet,
                                LoopBodySizeForSnippet),
              OS)) {
        return std::move(E);
      }
      const ExecutableFunction EF(State.createTargetMachine(),
                                  getObjectFromBuffer(OS.str()));
      const auto FnBytes = EF.getFunctionBytes();
      llvm::append_range(InstrBenchmark.AssembledSnippet, FnBytes);
    }

    // Assemble NumRepetitions instructions repetitions of the snippet for
    // measurements.
    const auto Filler = Repetitor->Repeat(
        Instructions, InstrBenchmark.NumRepetitions, LoopBodySize);

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
      if (Error E = assembleToStream(
              State.getExegesisTarget(), State.createTargetMachine(),
              BC.LiveIns, BC.Key.RegisterInitialValues, Filler, OS)) {
        return std::move(E);
      }
      ObjectFile = getObjectFromBuffer(OS.str());
    }

    if (BenchmarkSkipMeasurements) {
      InstrBenchmark.Error =
          "in --skip-measurements mode, actual measurements skipped.";
      continue;
    }

    const FunctionExecutorImpl Executor(State, std::move(ObjectFile),
                                        Scratch.get());
    auto NewMeasurements = runMeasurements(Executor);
    if (Error E = NewMeasurements.takeError()) {
      if (!E.isA<SnippetCrash>())
        return std::move(E);
      InstrBenchmark.Error = toString(std::move(E));
      return InstrBenchmark;
    }
    assert(InstrBenchmark.NumRepetitions > 0 && "invalid NumRepetitions");
    for (BenchmarkMeasure &BM : *NewMeasurements) {
      // Scale the measurements by instruction.
      BM.PerInstructionValue /= InstrBenchmark.NumRepetitions;
      // Scale the measurements by snippet.
      BM.PerSnippetValue *= static_cast<double>(Instructions.size()) /
                            InstrBenchmark.NumRepetitions;
    }
    if (InstrBenchmark.Measurements.empty()) {
      InstrBenchmark.Measurements = std::move(*NewMeasurements);
      continue;
    }

    assert(Repetitors.size() > 1 && !InstrBenchmark.Measurements.empty() &&
           "We're in an 'min' repetition mode, and need to aggregate new "
           "result to the existing result.");
    assert(InstrBenchmark.Measurements.size() == NewMeasurements->size() &&
           "Expected to have identical number of measurements.");
    for (auto I : zip(InstrBenchmark.Measurements, *NewMeasurements)) {
      BenchmarkMeasure &Measurement = std::get<0>(I);
      BenchmarkMeasure &NewMeasurement = std::get<1>(I);
      assert(Measurement.Key == NewMeasurement.Key &&
             "Expected measurements to be symmetric");

      Measurement.PerInstructionValue = std::min(
          Measurement.PerInstructionValue, NewMeasurement.PerInstructionValue);
      Measurement.PerSnippetValue =
          std::min(Measurement.PerSnippetValue, NewMeasurement.PerSnippetValue);
    }
  }

  // We successfully measured everything, so don't discard the results.
  CBOR.disarm();
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
  if (Error E = assembleToStream(
          State.getExegesisTarget(), State.createTargetMachine(), BC.LiveIns,
          BC.Key.RegisterInitialValues, FillFunction, OFS)) {
    return std::move(E);
  }
  return std::string(ResultPath.str());
}

BenchmarkRunner::FunctionExecutor::~FunctionExecutor() {}

} // namespace exegesis
} // namespace llvm
