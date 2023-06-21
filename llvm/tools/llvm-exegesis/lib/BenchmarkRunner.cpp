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
#include "llvm/Support/Signals.h"

#ifdef __linux__
#ifdef HAVE_LIBPFM
#include <perfmon/perf_event.h>
#endif
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#endif // __linux__

namespace llvm {
namespace exegesis {

BenchmarkRunner::BenchmarkRunner(const LLVMState &State, Benchmark::ModeE Mode,
                                 BenchmarkPhaseSelectorE BenchmarkPhaseSelector,
                                 ExecutionModeE ExecutionMode)
    : State(State), Mode(Mode), BenchmarkPhaseSelector(BenchmarkPhaseSelector),
      ExecutionMode(ExecutionMode), Scratch(std::make_unique<ScratchSpace>()) {}

BenchmarkRunner::~BenchmarkRunner() = default;

void BenchmarkRunner::FunctionExecutor::accumulateCounterValues(
    const llvm::SmallVectorImpl<int64_t> &NewValues,
    llvm::SmallVectorImpl<int64_t> *Result) {
  const size_t NumValues = std::max(NewValues.size(), Result->size());
  if (NumValues > Result->size())
    Result->resize(NumValues, 0);
  for (size_t I = 0, End = NewValues.size(); I < End; ++I)
    (*Result)[I] += NewValues[I];
}

Expected<llvm::SmallVector<int64_t, 4>>
BenchmarkRunner::FunctionExecutor::runAndSample(const char *Counters) const {
  // We sum counts when there are several counters for a single ProcRes
  // (e.g. P23 on SandyBridge).
  llvm::SmallVector<int64_t, 4> CounterValues;
  SmallVector<StringRef, 2> CounterNames;
  StringRef(Counters).split(CounterNames, '+');
  for (auto &CounterName : CounterNames) {
    CounterName = CounterName.trim();
    Expected<SmallVector<int64_t, 4>> ValueOrError =
        runWithCounter(CounterName);
    if (!ValueOrError)
      return ValueOrError.takeError();
    accumulateCounterValues(ValueOrError.get(), &CounterValues);
  }
  return CounterValues;
}

namespace {
class InProcessFunctionExecutorImpl : public BenchmarkRunner::FunctionExecutor {
public:
  InProcessFunctionExecutorImpl(const LLVMState &State,
                                object::OwningBinary<object::ObjectFile> Obj,
                                BenchmarkRunner::ScratchSpace *Scratch)
      : State(State), Function(State.createTargetMachine(), std::move(Obj)),
        Scratch(Scratch) {}

private:
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
  runWithCounter(StringRef CounterName) const override {
    const ExegesisTarget &ET = State.getExegesisTarget();
    char *const ScratchPtr = Scratch->ptr();
    auto CounterOrError = ET.createCounter(CounterName, State);

    if (!CounterOrError)
      return CounterOrError.takeError();

    pfm::Counter *Counter = CounterOrError.get().get();
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

    return Counter->readOrError(Function.getFunctionBytes());
  }

  const LLVMState &State;
  const ExecutableFunction Function;
  BenchmarkRunner::ScratchSpace *const Scratch;
};

#ifdef __linux__
// The following class implements a function executor that executes the
// benchmark code within a subprocess rather than within the main llvm-exegesis
// process. This allows for much more control over the execution context of the
// snippet, particularly with regard to memory. This class performs all the
// necessary functions to create the subprocess, execute the snippet in the
// subprocess, and report results/handle errors.
class SubProcessFunctionExecutorImpl
    : public BenchmarkRunner::FunctionExecutor {
public:
  SubProcessFunctionExecutorImpl(const LLVMState &State,
                                 object::OwningBinary<object::ObjectFile> Obj,
                                 const BenchmarkKey &Key)
      : State(State), Function(State.createTargetMachine(), std::move(Obj)),
        Key(Key) {}

private:
  enum ChildProcessExitCodeE {
    CounterFDReadFailed = 1,
    TranslatingCounterFDFailed
  };

  StringRef childProcessExitCodeToString(int ExitCode) const {
    switch (ExitCode) {
    case ChildProcessExitCodeE::CounterFDReadFailed:
      return "Counter file descriptor read failed";
    case ChildProcessExitCodeE::TranslatingCounterFDFailed:
      return "Translating counter file descriptor into a file descriptor in "
             "the child process failed. This might be due running an older "
             "Linux kernel that doesn't support the pidfd_getfd system call "
             "(anything before Linux 5.6).";
    default:
      return "Child process returned with unknown exit code";
    }
  }

  Error createSubProcessAndRunBenchmark(
      StringRef CounterName, SmallVectorImpl<int64_t> &CounterValues) const {
    int PipeFiles[2];
    int PipeSuccessOrErr = pipe(PipeFiles);
    if (PipeSuccessOrErr != 0) {
      return make_error<Failure>(
          "Failed to create a pipe for interprocess communication between "
          "llvm-exegesis and the benchmarking subprocess");
    }

    pid_t ParentOrChildPID = fork();
    if (ParentOrChildPID == 0) {
      // We are in the child process, close the write end of the pipe
      close(PipeFiles[1]);
      // Unregister handlers, signal handling is now handled through ptrace in
      // the host process
      llvm::sys::unregisterHandlers();
      prepareAndRunBenchmark(PipeFiles[0], Key);
      // The child process terminates in the above function, so we should never
      // get to this point.
      llvm_unreachable("Child process didn't exit when expected.");
    }

    const ExegesisTarget &ET = State.getExegesisTarget();
    auto CounterOrError =
        ET.createCounter(CounterName, State, ParentOrChildPID);

    if (!CounterOrError)
      return CounterOrError.takeError();

    pfm::Counter *Counter = CounterOrError.get().get();

    close(PipeFiles[0]);

    int CounterFileDescriptor = Counter->getFileDescriptor();
    ssize_t BytesWritten =
        write(PipeFiles[1], &CounterFileDescriptor, sizeof(int));

    if (BytesWritten != sizeof(int))
      return make_error<Failure>("Writing peformance counter file descriptor "
                                 "to child process failed: " +
                                 Twine(strerror(errno)));

    if (ptrace(PTRACE_SEIZE, ParentOrChildPID, NULL, NULL) != 0)
      return make_error<Failure>("Failed to seize the child process: " +
                                 Twine(strerror(errno)));

    int ChildStatus;
    if (wait(&ChildStatus) == -1) {
      return make_error<Failure>(
          "Waiting for the child process to complete failed: " +
          Twine(strerror(errno)));
    }

    if (WIFEXITED(ChildStatus)) {
      int ChildExitCode = WEXITSTATUS(ChildStatus);
      if (ChildExitCode == 0) {
        // The child exited succesfully, read counter values and return
        // success
        CounterValues[0] = Counter->read();
        return Error::success();
      }
      // The child exited, but not successfully
      return make_error<SnippetCrash>(
          "Child benchmarking process exited with non-zero exit code: " +
          childProcessExitCodeToString(ChildExitCode));
    }

    // An error was encountered running the snippet, process it
    siginfo_t ChildSignalInfo;
    if (ptrace(PTRACE_GETSIGINFO, ParentOrChildPID, NULL, &ChildSignalInfo) ==
        -1) {
      return make_error<Failure>("Getting signal info from the child failed: " +
                                 Twine(strerror(errno)));
    }

    return make_error<SnippetCrash>(
        "The benchmarking subprocess sent unexpected signal: " +
        Twine(strsignal(ChildSignalInfo.si_signo)));
  }

  [[noreturn]] void prepareAndRunBenchmark(int Pipe,
                                           const BenchmarkKey &Key) const {
    // The following occurs within the benchmarking subprocess

    int ParentCounterFileDescriptor = -1;
    ssize_t BytesRead = read(Pipe, &ParentCounterFileDescriptor, sizeof(int));

    if (BytesRead != sizeof(int)) {
      exit(ChildProcessExitCodeE::CounterFDReadFailed);
    }

#ifdef SYS_pidfd_getfd
    pid_t ParentPID = getppid();
    int ParentPIDFD = syscall(SYS_pidfd_open, ParentPID, 0);
    int CounterFileDescriptor =
        syscall(SYS_pidfd_getfd, ParentPIDFD, ParentCounterFileDescriptor, 0);
#else
    int CounterFileDescriptor = 0;
    exit(ChildProcessExitCodeE::TranslatingCounterFDFailed);
#endif

    if (CounterFileDescriptor == -1) {
      exit(ChildProcessExitCodeE::TranslatingCounterFDFailed);
    }

#ifdef HAVE_LIBPFM
    ioctl(CounterFileDescriptor, PERF_EVENT_IOC_RESET);
#endif
    this->Function(nullptr);
#ifdef HAVE_LIBPFM
    ioctl(CounterFileDescriptor, PERF_EVENT_IOC_DISABLE);
#endif

    exit(0);
  }

  Expected<llvm::SmallVector<int64_t, 4>>
  runWithCounter(StringRef CounterName) const override {
    SmallVector<int64_t, 4> Value(1, 0);
    Error PossibleBenchmarkError =
        createSubProcessAndRunBenchmark(CounterName, Value);

    if (PossibleBenchmarkError) {
      return std::move(PossibleBenchmarkError);
    }

    return Value;
  }

  const LLVMState &State;
  const ExecutableFunction Function;
  const BenchmarkKey &Key;
};
#endif // __linux__
} // namespace

Expected<SmallString<0>> BenchmarkRunner::assembleSnippet(
    const BenchmarkCode &BC, const SnippetRepetitor &Repetitor,
    unsigned MinInstructions, unsigned LoopBodySize) const {
  const std::vector<MCInst> &Instructions = BC.Key.Instructions;
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  if (Error E = assembleToStream(
          State.getExegesisTarget(), State.createTargetMachine(), BC.LiveIns,
          BC.Key.RegisterInitialValues,
          Repetitor.Repeat(Instructions, MinInstructions, LoopBodySize), OS)) {
    return std::move(E);
  }
  return Buffer;
}

Expected<BenchmarkRunner::RunnableConfiguration>
BenchmarkRunner::getRunnableConfiguration(
    const BenchmarkCode &BC, unsigned NumRepetitions, unsigned LoopBodySize,
    const SnippetRepetitor &Repetitor) const {
  RunnableConfiguration RC;

  Benchmark &InstrBenchmark = RC.InstrBenchmark;
  InstrBenchmark.Mode = Mode;
  InstrBenchmark.CpuName = std::string(State.getTargetMachine().getTargetCPU());
  InstrBenchmark.LLVMTriple =
      State.getTargetMachine().getTargetTriple().normalize();
  InstrBenchmark.NumRepetitions = NumRepetitions;
  InstrBenchmark.Info = BC.Info;

  const std::vector<MCInst> &Instructions = BC.Key.Instructions;

  InstrBenchmark.Key = BC.Key;

  // Assemble at least kMinInstructionsForSnippet instructions by repeating
  // the snippet for debug/analysis. This is so that the user clearly
  // understands that the inside instructions are repeated.
  if (BenchmarkPhaseSelector > BenchmarkPhaseSelectorE::PrepareSnippet) {
    const int MinInstructionsForSnippet = 4 * Instructions.size();
    const int LoopBodySizeForSnippet = 2 * Instructions.size();
    auto Snippet = assembleSnippet(BC, Repetitor, MinInstructionsForSnippet,
                                   LoopBodySizeForSnippet);
    if (Error E = Snippet.takeError())
      return std::move(E);

    if (auto Err = getBenchmarkFunctionBytes(*Snippet,
                                             InstrBenchmark.AssembledSnippet))
      return std::move(Err);
  }

  // Assemble NumRepetitions instructions repetitions of the snippet for
  // measurements.
  if (BenchmarkPhaseSelector > BenchmarkPhaseSelectorE::PrepareAndAssembleSnippet) {
    auto Snippet = assembleSnippet(BC, Repetitor, InstrBenchmark.NumRepetitions,
                                   LoopBodySize);
    if (Error E = Snippet.takeError())
      return std::move(E);
    RC.ObjectFile = getObjectFromBuffer(*Snippet);
  }

  return std::move(RC);
}

Expected<std::unique_ptr<BenchmarkRunner::FunctionExecutor>>
BenchmarkRunner::createFunctionExecutor(
    object::OwningBinary<object::ObjectFile> ObjectFile,
    const BenchmarkKey &Key) const {
  switch (ExecutionMode) {
  case ExecutionModeE::InProcess:
    return std::make_unique<InProcessFunctionExecutorImpl>(
        State, std::move(ObjectFile), Scratch.get());
  case ExecutionModeE::SubProcess:
#ifdef __linux__
    return std::make_unique<SubProcessFunctionExecutorImpl>(
        State, std::move(ObjectFile), Key);
#else
    return make_error<Failure>(
        "The subprocess execution mode is only supported on Linux");
#endif
  }
  llvm_unreachable("ExecutionMode is outside expected range");
}

Expected<Benchmark> BenchmarkRunner::runConfiguration(
    RunnableConfiguration &&RC,
    const std::optional<StringRef> &DumpFile) const {
  Benchmark &InstrBenchmark = RC.InstrBenchmark;
  object::OwningBinary<object::ObjectFile> &ObjectFile = RC.ObjectFile;

  if (DumpFile && BenchmarkPhaseSelector >
                      BenchmarkPhaseSelectorE::PrepareAndAssembleSnippet) {
    auto ObjectFilePath =
        writeObjectFile(ObjectFile.getBinary()->getData(), *DumpFile);
    if (Error E = ObjectFilePath.takeError()) {
      InstrBenchmark.Error = toString(std::move(E));
      return std::move(InstrBenchmark);
    }
    outs() << "Check generated assembly with: /usr/bin/objdump -d "
           << *ObjectFilePath << "\n";
  }

  if (BenchmarkPhaseSelector < BenchmarkPhaseSelectorE::Measure) {
    InstrBenchmark.Error = "actual measurements skipped.";
    return std::move(InstrBenchmark);
  }

  Expected<std::unique_ptr<BenchmarkRunner::FunctionExecutor>> Executor =
      createFunctionExecutor(std::move(ObjectFile), RC.InstrBenchmark.Key);
  if (!Executor)
    return Executor.takeError();
  auto NewMeasurements = runMeasurements(**Executor);

  if (Error E = NewMeasurements.takeError()) {
    if (!E.isA<SnippetCrash>())
      return std::move(E);
    InstrBenchmark.Error = toString(std::move(E));
    return std::move(InstrBenchmark);
  }
  assert(InstrBenchmark.NumRepetitions > 0 && "invalid NumRepetitions");
  for (BenchmarkMeasure &BM : *NewMeasurements) {
    // Scale the measurements by instruction.
    BM.PerInstructionValue /= InstrBenchmark.NumRepetitions;
    // Scale the measurements by snippet.
    BM.PerSnippetValue *=
        static_cast<double>(InstrBenchmark.Key.Instructions.size()) /
        InstrBenchmark.NumRepetitions;
  }
  InstrBenchmark.Measurements = std::move(*NewMeasurements);

  return std::move(InstrBenchmark);
}

Expected<std::string>
BenchmarkRunner::writeObjectFile(StringRef Buffer, StringRef FileName) const {
  int ResultFD = 0;
  SmallString<256> ResultPath = FileName;
  if (Error E = errorCodeToError(
          FileName.empty() ? sys::fs::createTemporaryFile("snippet", "o",
                                                          ResultFD, ResultPath)
                           : sys::fs::openFileForReadWrite(
                                 FileName, ResultFD, sys::fs::CD_CreateAlways,
                                 sys::fs::OF_None)))
    return std::move(E);
  raw_fd_ostream OFS(ResultFD, true /*ShouldClose*/);
  OFS.write(Buffer.data(), Buffer.size());
  OFS.flush();
  return std::string(ResultPath.str());
}

BenchmarkRunner::FunctionExecutor::~FunctionExecutor() {}

} // namespace exegesis
} // namespace llvm
