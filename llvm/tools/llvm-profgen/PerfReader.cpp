//===-- PerfReader.cpp - perfscript reader  ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PerfReader.h"
#include "ErrorHandling.h"
#include "Options.h"
#include "ProfileGenerator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/ProfileData/ETMTraceDecoder.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cctype>
#include <cmath>
#include <limits>

#define DEBUG_TYPE "perf-reader"

namespace llvm {

cl::opt<bool> SkipSymbolization("skip-symbolization",
                                cl::desc("Dump the unsymbolized profile to the "
                                         "output file. It will show unwinder "
                                         "output for CS profile generation."),
                                cl::cat(ProfGenCategory));

static cl::opt<bool> ShowMmapEvents("show-mmap-events",
                                    cl::desc("Print binary load events."),
                                    cl::cat(ProfGenCategory));

static cl::opt<bool>
    UseOffset("use-offset", cl::init(true),
              cl::desc("Work with `--skip-symbolization` or "
                       "`--unsymbolized-profile` to write/read the "
                       "offset instead of virtual address."),
              cl::cat(ProfGenCategory));

static cl::opt<bool> UseLoadableSegmentAsBase(
    "use-first-loadable-segment-as-base",
    cl::desc("Use first loadable segment address as base address "
             "for offsets in unsymbolized profile. By default "
             "first executable segment address is used"),
    cl::cat(ProfGenCategory));

static cl::opt<bool>
    IgnoreStackSamples("ignore-stack-samples",
                       cl::desc("Ignore call stack samples for hybrid samples "
                                "and produce context-insensitive profile."),
                       cl::cat(ProfGenCategory));
cl::opt<bool> ShowDetailedWarning("show-detailed-warning",
                                  cl::desc("Show detailed warning message."),
                                  cl::cat(ProfGenCategory));

static cl::opt<int> CSProfMaxUnsymbolizedCtxDepth(
    "csprof-max-unsymbolized-context-depth", cl::init(-1),
    cl::desc("Keep the last K contexts while merging unsymbolized profile. -1 "
             "means no depth limit."),
    cl::cat(ProfGenCategory));

cl::opt<bool> TimeProfGen("time-profgen", cl::desc("Time llvm-profgen phases"),
                          cl::init(false), cl::cat(ProfGenCategory));

static const char *TimerGroupName = "profgen";
static const char *TimerGroupDesc = "llvm-profgen";

namespace sampleprof {

void VirtualUnwinder::unwindCall(UnwindState &State) {
  uint64_t Source = State.getCurrentLBRSource();
  auto *ParentFrame = State.getParentFrame();
  // The 2nd frame after leaf could be missing if stack sample is
  // taken when IP is within prolog/epilog, as frame chain isn't
  // setup yet. Fill in the missing frame in that case.
  // TODO: Currently we just assume all the addr that can't match the
  // 2nd frame is in prolog/epilog. In the future, we will switch to
  // pro/epi tracker(Dwarf CFI) for the precise check.
  if (ParentFrame == State.getDummyRootPtr() ||
      ParentFrame->Address != Source) {
    State.switchToFrame(Source);
    if (ParentFrame != State.getDummyRootPtr()) {
      if (Source == ExternalAddr)
        NumMismatchedExtCallBranch++;
      else
        NumMismatchedProEpiBranch++;
    }
  } else {
    State.popFrame();
  }
  State.InstPtr.update(Source);
}

void VirtualUnwinder::unwindLinear(UnwindState &State, uint64_t Repeat) {
  InstructionPointer &IP = State.InstPtr;
  uint64_t Target = State.getCurrentLBRTarget();
  uint64_t End = IP.Address;

  if (End == ExternalAddr && Target == ExternalAddr) {
    // Filter out the case when leaf external frame matches the external LBR
    // target, this is a valid state, it happens that the code run into external
    // address then return back.  The call frame under the external frame
    // remains valid and can be unwound later, just skip recording this range.
    NumPairedExtAddr++;
    return;
  }

  if (End == ExternalAddr || Target == ExternalAddr) {
    // Range is invalid if only one point is external address. This means LBR
    // traces contains a standalone external address failing to pair another
    // one, likely due to interrupt jmp or broken perf script. Set the
    // state to invalid.
    NumUnpairedExtAddr++;
    State.setInvalid();
    return;
  }

  if (!isValidFallThroughRange(Target, End, Binary)) {
    // Skip unwinding the rest of LBR trace when a bogus range is seen.
    State.setInvalid();
    return;
  }

  if (Binary->usePseudoProbes()) {
    // We don't need to top frame probe since it should be extracted
    // from the range.
    // The outcome of the virtual unwinding with pseudo probes is a
    // map from a context key to the address range being unwound.
    // This means basically linear unwinding is not needed for pseudo
    // probes. The range will be simply recorded here and will be
    // converted to a list of pseudo probes to report in ProfileGenerator.
    State.getParentFrame()->recordRangeCount(Target, End, Repeat);
  } else {
    // Unwind linear execution part.
    // Split and record the range by different inline context. For example:
    // [0x01] ... main:1          # Target
    // [0x02] ... main:2
    // [0x03] ... main:3 @ foo:1
    // [0x04] ... main:3 @ foo:2
    // [0x05] ... main:3 @ foo:3
    // [0x06] ... main:4
    // [0x07] ... main:5          # End
    // It will be recorded:
    // [main:*]         : [0x06, 0x07], [0x01, 0x02]
    // [main:3 @ foo:*] : [0x03, 0x05]
    while (IP.Address > Target) {
      uint64_t PrevIP = IP.Address;
      IP.backward();
      // Break into segments for implicit call/return due to inlining
      bool SameInlinee = Binary->inlineContextEqual(PrevIP, IP.Address);
      if (!SameInlinee) {
        State.switchToFrame(PrevIP);
        State.CurrentLeafFrame->recordRangeCount(PrevIP, End, Repeat);
        End = IP.Address;
      }
    }
    assert(IP.Address == Target && "The last one must be the target address.");
    // Record the remaining range, [0x01, 0x02] in the example
    State.switchToFrame(IP.Address);
    State.CurrentLeafFrame->recordRangeCount(IP.Address, End, Repeat);
  }
}

void VirtualUnwinder::unwindReturn(UnwindState &State) {
  // Add extra frame as we unwind through the return
  const LBREntry &LBR = State.getCurrentLBR();
  uint64_t CallAddr = Binary->getCallAddrFromFrameAddr(LBR.Target);
  State.switchToFrame(CallAddr);
  State.pushFrame(LBR.Source);
  State.InstPtr.update(LBR.Source);
}

void VirtualUnwinder::unwindBranch(UnwindState &State) {
  // TODO: Tolerate tail call for now, as we may see tail call from libraries.
  // This is only for intra function branches, excluding tail calls.
  uint64_t Source = State.getCurrentLBRSource();
  State.switchToFrame(Source);
  State.InstPtr.update(Source);
}

std::shared_ptr<StringBasedCtxKey> FrameStack::getContextKey() {
  std::shared_ptr<StringBasedCtxKey> KeyStr =
      std::make_shared<StringBasedCtxKey>();
  KeyStr->Context = Binary->getExpandedContext(Stack, KeyStr->WasLeafInlined);
  return KeyStr;
}

std::shared_ptr<AddrBasedCtxKey> AddressStack::getContextKey() {
  std::shared_ptr<AddrBasedCtxKey> KeyStr = std::make_shared<AddrBasedCtxKey>();
  KeyStr->Context = Stack;
  CSProfileGenerator::compressRecursionContext<uint64_t>(KeyStr->Context);
  // MaxContextDepth(--csprof-max-context-depth) is used to trim both symbolized
  // and unsymbolized profile context. Sometimes we want to at least preserve
  // the inlinings for the leaf frame(the profiled binary inlining),
  // --csprof-max-context-depth may not be flexible enough, in this case,
  // --csprof-max-unsymbolized-context-depth is used to limit the context for
  // unsymbolized profile. If both are set, use the minimum of them.
  int Depth = CSProfileGenerator::MaxContextDepth != -1
                  ? CSProfileGenerator::MaxContextDepth
                  : KeyStr->Context.size();
  Depth = CSProfMaxUnsymbolizedCtxDepth != -1
              ? std::min(static_cast<int>(CSProfMaxUnsymbolizedCtxDepth), Depth)
              : Depth;
  CSProfileGenerator::trimContext<uint64_t>(KeyStr->Context, Depth);
  return KeyStr;
}

template <typename T>
void VirtualUnwinder::collectSamplesFromFrame(UnwindState::ProfiledFrame *Cur,
                                              T &Stack) {
  if (Cur->RangeSamples.empty() && Cur->BranchSamples.empty())
    return;

  std::shared_ptr<ContextKey> Key = Stack.getContextKey();
  if (Key == nullptr)
    return;
  auto Ret = CtxCounterMap->try_emplace(Hashable<ContextKey>(Key));
  SampleCounter &SCounter = Ret.first->second;
  for (auto &I : Cur->RangeSamples)
    SCounter.recordRangeCount(std::get<0>(I), std::get<1>(I), std::get<2>(I));

  for (auto &I : Cur->BranchSamples)
    SCounter.recordBranchCount(std::get<0>(I), std::get<1>(I), std::get<2>(I));
}

template <typename T>
void VirtualUnwinder::collectSamplesFromFrameTrie(
    UnwindState::ProfiledFrame *Cur, T &Stack) {
  if (!Cur->isDummyRoot()) {
    // Truncate the context for external frame since this isn't a real call
    // context the compiler will see.
    if (Cur->isExternalFrame() || !Stack.pushFrame(Cur)) {
      // Process truncated context
      // Start a new traversal ignoring its bottom context
      T EmptyStack(Binary);
      collectSamplesFromFrame(Cur, EmptyStack);
      for (const auto &Item : Cur->Children) {
        collectSamplesFromFrameTrie(Item.second.get(), EmptyStack);
      }

      // Keep note of untracked call site and deduplicate them
      // for warning later.
      if (!Cur->isLeafFrame())
        UntrackedCallsites.insert(Cur->Address);

      return;
    }
  }

  collectSamplesFromFrame(Cur, Stack);
  // Process children frame
  for (const auto &Item : Cur->Children) {
    collectSamplesFromFrameTrie(Item.second.get(), Stack);
  }
  // Recover the call stack
  Stack.popFrame();
}

void VirtualUnwinder::collectSamplesFromFrameTrie(
    UnwindState::ProfiledFrame *Cur) {
  if (Binary->usePseudoProbes()) {
    AddressStack Stack(Binary);
    collectSamplesFromFrameTrie<AddressStack>(Cur, Stack);
  } else {
    FrameStack Stack(Binary);
    collectSamplesFromFrameTrie<FrameStack>(Cur, Stack);
  }
}

void VirtualUnwinder::recordBranchCount(const LBREntry &Branch,
                                        UnwindState &State, uint64_t Repeat) {
  if (Branch.Target == ExternalAddr)
    return;

  // Record external-to-internal pattern on the trie root, it later can be
  // used for generating head samples.
  if (Branch.Source == ExternalAddr) {
    State.getDummyRootPtr()->recordBranchCount(Branch.Source, Branch.Target,
                                               Repeat);
    return;
  }

  if (Binary->usePseudoProbes()) {
    // Same as recordRangeCount, We don't need to top frame probe since we will
    // extract it from branch's source address
    State.getParentFrame()->recordBranchCount(Branch.Source, Branch.Target,
                                              Repeat);
  } else {
    State.CurrentLeafFrame->recordBranchCount(Branch.Source, Branch.Target,
                                              Repeat);
  }
}

bool VirtualUnwinder::unwind(const PerfSample *Sample, uint64_t Repeat) {
  // Capture initial state as starting point for unwinding.
  UnwindState State(Sample, Binary);

  NumTotalBranches += State.LBRStack.size();
  // Now process the LBR samples in parrallel with stack sample
  // Note that we do not reverse the LBR entry order so we can
  // unwind the sample stack as we walk through LBR entries.
  while (State.hasNextLBR()) {
    State.checkStateConsistency();

    // Do not attempt linear unwind for the leaf range as it's incomplete.
    if (!State.IsLastLBR()) {
      // Unwind implicit calls/returns from inlining, along the linear path,
      // break into smaller sub section each with its own calling context.
      unwindLinear(State, Repeat);
    }

    // Save the LBR branch before it gets unwound.
    const LBREntry &Branch = State.getCurrentLBR();
    if (isCallState(State)) {
      // Unwind calls - we know we encountered call if LBR overlaps with
      // transition between leaf the 2nd frame. Note that for calls that
      // were not in the original stack sample, we should have added the
      // extra frame when processing the return paired with this call.
      unwindCall(State);
    } else if (isReturnState(State)) {
      // Unwind returns - check whether the IP is indeed at a return
      // instruction
      unwindReturn(State);
    } else if (isValidState(State)) {
      // Unwind branches
      unwindBranch(State);
    } else {
      // Skip unwinding the rest of LBR trace. Reset the stack and update the
      // state so that the rest of the trace can still be processed as if they
      // do not have stack samples.
      State.clearCallStack();
      State.InstPtr.update(State.getCurrentLBRSource());
      State.pushFrame(State.InstPtr.Address);
    }

    State.advanceLBR();
    // Record `branch` with calling context after unwinding.
    recordBranchCount(Branch, State, Repeat);
  }
  // As samples are aggregated on trie, record them into counter map
  collectSamplesFromFrameTrie(State.getDummyRootPtr());

  return true;
}

std::unique_ptr<PerfReaderBase>
PerfReaderBase::create(ProfiledBinary *Binary, InputFile &Input,
                       std::optional<int32_t> PIDFilter) {
  std::unique_ptr<PerfReaderBase> PerfReader;

  if (Input.Format == InputFormat::UnsymbolizedProfile) {
    PerfReader.reset(
        new UnsymbolizedProfileReader(Binary, Input.InputFilePath));
    return PerfReader;
  }

  // For perf data input, we need to convert them into perf script first.
  // If this is a kernel perf file, there is no need for retrieving PIDs.
  if (Input.Format == InputFormat::PerfData)
    Input = PerfScriptReader::convertPerfDataToTrace(Binary, Binary->isKernel(),
                                                     Input, PIDFilter);

  assert((Input.Format == InputFormat::PerfScript) &&
         "Should be a perfscript!");

  Input.Content = PerfScriptReader::checkPerfScriptType(
      Binary, Input.InputFilePath, Input.UseProcessMappings, PIDFilter);
  if (Input.Content == PerfContent::LBRStack) {
    PerfReader.reset(new HybridPerfReader(Binary, Input.InputFilePath,
                                          PIDFilter, Input.UseProcessMappings));
  } else if (Input.Content == PerfContent::LBR) {
    PerfReader.reset(new LBRPerfReader(Binary, Input.InputFilePath, PIDFilter,
                                       Input.UseProcessMappings));
  } else {
    exitWithError("Unsupported perfscript!");
  }

  return PerfReader;
}

Error PerfReaderBase::parseDataAccessPerfTraces(
    StringRef DataAccessPerfTraceFile, std::optional<int32_t> PIDFilter) {
  // A perf_record_sample line is like
  // . 1282514022939813 0x87b0 [0x60]: PERF_RECORD_SAMPLE(IP, 0x4002):
  // 3446532/3446532: 0x2608a2 period: 233 addr: 0x3b3fb0
  constexpr static StringRef DataAccessSamplePattern =
      "PERF_RECORD_SAMPLE\\([A-Za-z]+, 0x[0-9a-fA-F]+\\): "
      "([0-9]+)\\/[0-9]+: 0x([0-9a-fA-F]+) period: [0-9]+ addr: "
      "0x([0-9a-fA-F]+)";

  llvm::Regex LogRegex(DataAccessSamplePattern);

  auto BufferOrErr = MemoryBuffer::getFile(DataAccessPerfTraceFile);
  std::error_code EC = BufferOrErr.getError();
  if (EC)
    return make_error<StringError>("Failed to open perf trace file: " +
                                       DataAccessPerfTraceFile,
                                   inconvertibleErrorCode());

  assert(!SampleCounters.empty() && "Sample counters should not be empty!");
  SampleCounter &Counter = SampleCounters.begin()->second;
  line_iterator LineIt(*BufferOrErr.get(), true);

  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;

    MMapEvent MMap;
    if (Line.contains("PERF_RECORD_MMAP2")) {
      if (PerfScriptReader::extractMMapEventForBinary(Binary, Line, MMap)) {
        if (!MMap.MemProtectionFlag.contains("x")) {
          if (Error E = Binary->addMMapNonTextEvent(MMap)) {
            return E;
          }
        }
      }
      continue;
    }

    SmallVector<StringRef> Fields;
    if (LogRegex.match(Line, &Fields)) {
      int32_t PID = 0;
      if (Fields[1].getAsInteger(10, PID))
        return make_error<StringError>(
            "Failed to parse PID from perf trace line: " + Line,
            inconvertibleErrorCode());

      if (PIDFilter.has_value() && *PIDFilter != PID) {
        continue;
      }

      uint64_t DataAddress = 0;
      if (Fields[3].getAsInteger(16, DataAddress))
        return make_error<StringError>(
            "Failed to parse data address from perf trace line: " + Line,
            inconvertibleErrorCode());
      // Out of all the memory access events, the vtable accesses are used to
      // construct type profiles. We assume that this is under the Itanium
      // C++ ABI so we can use `_ZTV` prefix to identify vtable.
      StringRef DataSymbol = Binary->symbolizeDataAddress(
          Binary->CanonicalizeNonTextAddress(DataAddress));
      if (DataSymbol.starts_with("_ZTV")) {
        uint64_t IP = 0;
        Fields[2].getAsInteger(16, IP);
        Counter.recordDataAccessCount(Binary->canonicalizeVirtualAddress(IP),
                                      DataSymbol, 1);
      }
    }
  }
  return Error::success();
}

enum class TaskKind { Fork, Comm, Exec, Exit };

struct TaskEvent {
  TaskKind Kind;
  // PID owns the address space; TID identifies a thread using that space.
  int32_t PID;
  int32_t TID;
  int32_t ParentPID = 0;
  int32_t ParentTID = 0;
  // Perf's timestamp-zero task snapshot describes tasks already present when
  // recording began. Its fork records do not imply a new inherited mapping.
  bool IsSynthetic;
};

static TaskEvent parseTaskEvent(StringRef Line) {
  size_t Marker = Line.find("PERF_RECORD_");
  // The timestamp is in the perf prefix, but task ownership comes from the
  // record payload. The prefix can describe the task that emitted the event.
  StringRef Prefix = Line.take_front(Marker).rtrim();
  auto [BeforeTime, Time] = Prefix.rsplit(' ');
  double Timestamp = 0;
  if (BeforeTime.empty() || !Time.consume_back(":") ||
      Time.getAsDouble(Timestamp) || !std::isfinite(Timestamp))
    exitWithError("Invalid perf script task event");

  StringRef Record = Line.drop_front(Marker).rtrim();
  SmallVector<StringRef, 6> Fields;
  auto ParsePID = [](StringRef Field) {
    int32_t PID = 0;
    if (Field.getAsInteger(10, PID) || PID < 0)
      exitWithError("Invalid perf script task event");
    return PID;
  };
  TaskEvent Event{TaskKind::Comm, 0, 0, 0, 0, Timestamp == 0};
  // FORK carries child and parent IDs. EXIT uses the same printed shape,
  // but only the exiting PID/TID pair changes our ownership state.
  if (Regex("^PERF_RECORD_(FORK|EXIT)\\((-?[0-9]+):([0-9]+)\\):"
            "\\((-?[0-9]+):([0-9]+)\\)$")
          .match(Record, &Fields)) {
    Event.Kind = Fields[1] == "FORK" ? TaskKind::Fork : TaskKind::Exit;
    Event.PID = ParsePID(Fields[2]);
    Event.TID = ParsePID(Fields[3]);
    if (Event.Kind == TaskKind::Fork) {
      Event.ParentPID = ParsePID(Fields[4]);
      Event.ParentTID = ParsePID(Fields[5]);
    }
  } else if (Regex("^PERF_RECORD_COMM( exec)?: .*:(-?[0-9]+)/([0-9]+)$")
                 .match(Record, &Fields)) {
    // A name change keeps the mappings; COMM with the exec flag replaces the
    // process image, even when its PID and name happen to stay the same.
    Event.Kind = Fields[1].empty() ? TaskKind::Comm : TaskKind::Exec;
    Event.PID = ParsePID(Fields[2]);
    Event.TID = ParsePID(Fields[3]);
  } else {
    exitWithError("Invalid perf script task event");
  }
  return Event;
}

InputFile
PerfScriptReader::convertPerfDataToTrace(ProfiledBinary *Binary, bool SkipPID,
                                         InputFile &File,
                                         std::optional<int32_t> PIDFilter) {
  StringRef PerfData = File.InputFilePath;
  bool UseProcessMappings = File.UseProcessMappings;
  // Run perf script to retrieve PIDs matching binary we're interested in.
  auto PerfExecutable = sys::Process::FindInEnvPath("PATH", "perf");
  if (!PerfExecutable) {
    exitWithError("Perf not found.");
  }
  std::string PerfExecutablePath = *PerfExecutable;
  SmallString<128> PerfTraceFile;
  sys::fs::createUniquePath("perf-script-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%.tmp",
                            PerfTraceFile, /*MakeAbsolute=*/true);
  std::string ErrorFile = std::string(PerfTraceFile) + ".err";
  std::optional<StringRef> Redirects[] = {std::nullopt,             // Stdin
                                          StringRef(PerfTraceFile), // Stdout
                                          StringRef(ErrorFile)};    // Stderr
  PerfScriptReader::TempFileCleanups.emplace_back(PerfTraceFile);
  PerfScriptReader::TempFileCleanups.emplace_back(ErrorFile);

  auto RunPerfScript = [&](ArrayRef<StringRef> Args) {
    // ExecuteAndWait does not truncate redirected output files on Unix. Remove
    // both files so a shorter invocation cannot retain output from the
    // previous perf script invocation.
    for (StringRef Path : {StringRef(PerfTraceFile), StringRef(ErrorFile)}) {
      if (std::error_code EC = sys::fs::remove(Path))
        exitWithError(EC, Path);
    }

    std::string ExecutionError;
    bool ExecutionFailed = false;
    int ExitCode =
        sys::ExecuteAndWait(PerfExecutablePath, Args, std::nullopt, Redirects,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                            &ExecutionError, &ExecutionFailed);
    if (!ExecutionFailed && ExitCode == 0)
      return;

    std::string Message;
    raw_string_ostream OS(Message);
    if (ExecutionFailed || ExitCode == -1)
      OS << "Failed to execute perf script";
    else if (ExitCode == -2)
      OS << "Perf script terminated abnormally";
    else
      OS << "Perf script failed with exit code " << ExitCode;
    if (!ExecutionError.empty())
      OS << ": " << ExecutionError;

    if (auto ErrorBuffer = MemoryBuffer::getFile(ErrorFile)) {
      StringRef Stderr = ErrorBuffer.get()->getBuffer().trim();
      if (!Stderr.empty())
        OS << "\n" << Stderr;
    }
    exitWithError(OS.str());
  };

  std::string PIDs;
  if (!SkipPID) {
    // Discover sample PIDs. Mapped mode follows task records because children
    // can inherit the target without emitting their own mmap.
    SmallVector<StringRef, 8> Args = {PerfExecutablePath, "script",
                                      "--show-mmap-events"};
    if (UseProcessMappings)
      Args.push_back("--show-task-events");
    Args.append(
        {"-F", UseProcessMappings ? "pid,time" : "comm,pid", "-i", PerfData});
    RunPerfScript(Args);

    // Collect the PIDs
    TraceStream TraceIt(PerfTraceFile);
    DenseSet<int32_t> PIDSet;
    // ActivePIDs decides which later forks may inherit the target. EverPIDs
    // retains earlier owners after exit/exec, since their recorded samples
    // must still be extracted. This pass selects candidates, not addresses.
    std::unordered_set<int32_t> ActivePIDs;
    // A recording may begin after threads were created. Their observed subset
    // cannot prove process death unless a birth/exec established completeness.
    std::unordered_map<int32_t, std::unordered_set<int32_t>> ActiveThreads;
    std::unordered_set<int32_t> CompleteThreadPIDs;
    std::set<int32_t> EverPIDs;
    while (!TraceIt.isAtEoF()) {
      MMapEvent MMap;
      StringRef Line = TraceIt.getCurrentLine();
      // Mapped output prefixes metadata with a numeric PID. The legacy mmap
      // shortcut skips digit-leading lines, so inspect the marker here.
      bool IsMMap = UseProcessMappings ? Line.contains("PERF_RECORD_MMAP")
                                       : isMMapEvent(Line);
      if (IsMMap) {
        if (extractMMapEventForBinary(Binary, Line, MMap, UseProcessMappings)) {
          if (UseProcessMappings) {
            ActivePIDs.insert(MMap.PID);
            EverPIDs.insert(MMap.PID);
          } else {
            auto It = PIDSet.insert(MMap.PID);
            if (It.second && (!PIDFilter || MMap.PID == *PIDFilter)) {
              if (!PIDs.empty()) {
                PIDs.append(",");
              }
              PIDs.append(utostr(MMap.PID));
            }
          }
        }
        // An mmap from any binary also tells us its emitting thread is live.
        if (UseProcessMappings && MMap.PID >= 0 &&
            ActiveThreads[static_cast<int32_t>(MMap.PID)]
                .insert(MMap.TID)
                .second)
          CompleteThreadPIDs.erase(MMap.PID);
      } else if (UseProcessMappings && Line.contains("PERF_RECORD_")) {
        TaskEvent Event = parseTaskEvent(Line);
        if (Event.Kind == TaskKind::Fork && !Event.IsSynthetic) {
          // The parent may first appear in this fork record. Remember it so
          // a later child-thread exit cannot retire a still-live process.
          if (ActiveThreads[Event.ParentPID].insert(Event.ParentTID).second)
            CompleteThreadPIDs.erase(Event.ParentPID);
          if (Event.PID == Event.ParentPID) {
            // New thread: the process's existing address space is shared.
            ActiveThreads[Event.PID].insert(Event.TID);
          } else if (ActivePIDs.count(Event.ParentPID)) {
            ActivePIDs.insert(Event.PID);
            ActiveThreads[Event.PID] = {Event.TID};
            EverPIDs.insert(Event.PID);
          } else {
            // A reused child PID loses its previous target eligibility.
            ActivePIDs.erase(Event.PID);
            ActiveThreads[Event.PID] = {Event.TID};
          }
          if (Event.PID != Event.ParentPID)
            CompleteThreadPIDs.insert(Event.PID);
        } else if (Event.Kind == TaskKind::Comm) {
          if (ActiveThreads[Event.PID].insert(Event.TID).second)
            CompleteThreadPIDs.erase(Event.PID);
        } else if (Event.Kind == TaskKind::Exec) {
          // Stop inheritance from the old image. A later target mmap will
          // make the new image eligible again.
          ActivePIDs.erase(Event.PID);
          ActiveThreads[Event.PID] = {Event.TID};
          if (Event.IsSynthetic)
            CompleteThreadPIDs.erase(Event.PID);
          else
            CompleteThreadPIDs.insert(Event.PID);
        } else if (Event.Kind == TaskKind::Exit) {
          // An EXIT record describes one thread, not necessarily the process.
          auto It = ActiveThreads.find(Event.PID);
          if (It == ActiveThreads.end() || !It->second.erase(Event.TID)) {
            CompleteThreadPIDs.erase(Event.PID);
          } else if (It->second.empty()) {
            ActiveThreads.erase(It);
            if (CompleteThreadPIDs.erase(Event.PID))
              ActivePIDs.erase(Event.PID);
          }
        }
      }
      TraceIt.advance();
    }

    if (UseProcessMappings) {
      // Apply --pid only after following ancestry: the chosen process may
      // obtain all its mappings from an unselected parent.
      for (int32_t PID : EverPIDs) {
        if (PIDFilter && PID != *PIDFilter)
          continue;
        if (!PIDs.empty())
          PIDs.append(",");
        PIDs.append(utostr(PID));
      }
    }

    if (PIDs.empty()) {
      exitWithError("No relevant mmap event is found in perf data.");
    }
  }

  // Run perf script again to retrieve events for PIDs collected above
  SmallVector<StringRef, 8> ScriptSampleArgs;
  ScriptSampleArgs.push_back(PerfExecutablePath);
  ScriptSampleArgs.push_back("script");
  ScriptSampleArgs.push_back("--show-mmap-events");
  // perf's PID selection filters samples while these options retain metadata
  // needed to replay inherited mappings and later address-space changes.
  if (UseProcessMappings && !Binary->isKernel())
    ScriptSampleArgs.push_back("--show-task-events");
  ScriptSampleArgs.push_back("-F");
  ScriptSampleArgs.push_back(UseProcessMappings ? "pid,time,ip,brstack"
                                                : "ip,brstack");
  // Keep stack lines as bare addresses so the mapped parser can distinguish
  // them from the next PID/timestamp header, even after an incomplete stack.
  if (UseProcessMappings)
    ScriptSampleArgs.append({"-F", "-sym"});
  ScriptSampleArgs.push_back("-i");
  ScriptSampleArgs.push_back(PerfData);
  if (!PIDs.empty()) {
    ScriptSampleArgs.push_back("--pid");
    ScriptSampleArgs.push_back(PIDs);
  }
  // The unwinder expects physical frames: the sampled IP followed by saved
  // return addresses. Perf's inline expansion repeats IPs for logical inline
  // calls, which can truncate the stack when interpreted as return addresses.
  // llvm-profgen reconstructs inline contexts from the profiled binary itself.
  ScriptSampleArgs.push_back("--no-inline");
  RunPerfScript(ScriptSampleArgs);

  return {std::string(PerfTraceFile), InputFormat::PerfScript,
          PerfContent::UnknownContent, UseProcessMappings};
}

static StringRef filename(StringRef Path, bool UseBackSlash) {
  llvm::sys::path::Style PathStyle =
      UseBackSlash ? llvm::sys::path::Style::windows_backslash
                   : llvm::sys::path::Style::native;
  StringRef FileName = llvm::sys::path::filename(Path, PathStyle);

  // In case this file use \r\n as newline.
  if (UseBackSlash && FileName.back() == '\r')
    return FileName.drop_back();

  return FileName;
}

void PerfScriptReader::updateBinaryAddress(const MMapEvent &Event) {
  // Drop the event which doesn't belong to user-provided binary
  StringRef BinaryName = filename(Event.BinaryPath, Binary->isCOFF());
  bool IsKernel = Binary->isKernel();
  if (!IsKernel && Binary->getName() != BinaryName)
    return;
  if (IsKernel && !Binary->isKernelImageName(BinaryName))
    return;

  // Drop the event if process does not match pid filter
  if (PIDFilter && Event.PID != *PIDFilter)
    return;

  Binary->addMMapRange(Event.Address, Event.Size);

  // Check if the FileOffset falls within the [Event.Offset, Event.Offset +
  // Event.Size) range.
  auto MMapContainsFileOffset = [&](uint64_t FileOffset) {
    return Event.Offset <= FileOffset &&
           (FileOffset - Event.Offset) < Event.Size;
  };

  if (IsKernel || MMapContainsFileOffset(Binary->getTextSegmentOffset())) {
    // For ELF, subtract the file offset to get the runtime address
    // corresponding to file offset zero. Kernel mmap events report the text
    // address as Event.Offset, so use the text segment offset from the ELF
    // instead.
    const uint64_t RuntimeBaseAddress =
        Binary->isCOFF()
            ? Event.Address
            : Event.Address -
                  (IsKernel ? Binary->getTextSegmentOffset() : Event.Offset);
    // A binary image could be unloaded and then reloaded at different
    // place, so update binary load address.
    // Only update for the first executable segment and assume all other
    // segments are loaded at consecutive memory addresses, which is the case on
    // X64.
    Binary->setBaseAddress(RuntimeBaseAddress);
    Binary->setIsLoadedByMMap(true);
  } else {
    // Verify segments are loaded consecutively.
    const auto &Offsets = Binary->getTextSegmentOffsets();
    auto IsContiguousMMapForSegment = [&](auto SegmentIt, uint64_t FileOffset,
                                          uint64_t RuntimeAddress) {
      auto I = std::distance(Offsets.begin(), SegmentIt);
      const auto &PreferredAddrs = Binary->getPreferredTextSegmentAddresses();
      return PreferredAddrs[I] + (FileOffset - *SegmentIt) ==
             Binary->canonicalizeVirtualAddress(RuntimeAddress);
    };

    auto It = llvm::lower_bound(Offsets, Event.Offset);
    if (It != Offsets.end() && MMapContainsFileOffset(*It)) {
      // The event is for loading a separate executable segment.
      uint64_t RuntimeSegmentAddress = Event.Address + (*It - Event.Offset);
      if (!IsContiguousMMapForSegment(It, *It, RuntimeSegmentAddress))
        exitWithError("Executable segments not loaded consecutively");
    } else {
      if (It == Offsets.begin())
        exitWithError("File offset not found");
      else {
        // Find the segment the event falls in. A large segment could be loaded
        // via multiple mmap calls with consecutive memory addresses.
        --It;
        assert(*It < Event.Offset);
        if (!IsContiguousMMapForSegment(It, Event.Offset, Event.Address))
          exitWithError("Segment not loaded by consecutive mmaps");
      }
    }
  }
}

void PerfScriptReader::updateExecutableMapping(const MMapEvent &Event,
                                               bool IsTarget) {
  // An mmap is also evidence that its emitting thread is alive. The map is
  // owned by PID even when a worker TID, not the leader, emitted the record.
  if (!Binary->isKernel() && Event.PID >= 0 &&
      ThreadsByPID[static_cast<int32_t>(Event.PID)].insert(Event.TID).second)
    CompleteThreadPIDs.erase(Event.PID);
  if (Event.Size == 0 || (!Binary->isKernel() && Event.PID < 0))
    return;
  if (Binary->isKernel() && Event.PID != -1)
    return;

  // Kernel mmaps use PID -1, but samples carry the interrupted task's PID.
  // Store one shared kernel history; userspace histories remain per process.
  int32_t PID = static_cast<int32_t>(Event.PID);
  SmallVector<ExecutableMapping, 2> NewMappings;
  if (IsTarget &&
      (Binary->isKernel() || Event.MemProtectionFlag.contains('x'))) {
    if (Binary->isKernel() || Binary->isCOFF()) {
      // Kernel mmap offsets describe a text address, not an ELF file offset.
      // COFF mappings also use an image base, so neither uses the ELF join.
      uint64_t Address = Binary->getPreferredTextSegmentAddresses().front();
      NewMappings.push_back({Event.Address, Event.Size, Address, true});
    } else {
      // Join the mmap's file interval to each executable ELF segment. One mmap
      // may span several segments, or begin part-way through one segment.
      for (auto [Offset, Size, PreferredAddress] :
           llvm::zip_equal(Binary->getTextSegmentOffsets(),
                           Binary->getTextSegmentFileSizes(),
                           Binary->getPreferredTextSegmentAddresses())) {
        // Test for overlap using distances, avoiding overflowing end offsets.
        if (Size == 0 ||
            (Offset >= Event.Offset ? Offset - Event.Offset >= Event.Size
                                    : Event.Offset - Offset >= Size))
          continue;
        uint64_t FileOffset = std::max(Offset, Event.Offset);
        // The same file byte has one offset from the runtime mmap start and
        // another from the segment's preferred address in the binary.
        uint64_t MMapDelta = FileOffset - Event.Offset;
        uint64_t SegmentDelta = FileOffset - Offset;
        auto Address = checkedAddUnsigned(PreferredAddress, SegmentDelta);
        if (!Address)
          exitWithError("Invalid perf script mmap address");
        // mmap parsing already checked the runtime interval. Limit this entry
        // to its intersection with the file-backed executable segment.
        ExecutableMapping Mapping{
            Event.Address + MMapDelta,
            std::min(Event.Size - MMapDelta, Size - SegmentDelta), *Address,
            true};
        // Overlapping segments must translate their shared bytes identically.
        // Disjoint segments may legitimately have different load biases.
        for (const ExecutableMapping &Other : NewMappings) {
          uint64_t Start =
              std::max(Mapping.RuntimeAddress, Other.RuntimeAddress);
          if (Start - Mapping.RuntimeAddress < Mapping.Size &&
              Start - Other.RuntimeAddress < Other.Size &&
              Mapping.CanonicalAddress - Mapping.RuntimeAddress !=
                  Other.CanonicalAddress - Other.RuntimeAddress)
            exitWithError("Ambiguous perf script mmap load bias");
        }
        NewMappings.push_back(Mapping);
      }
    }
  }
  IsTarget = !NewMappings.empty();

  auto It = MappingsByPID.find(PID);
  // A non-target or non-executable mmap matters when it replaces target code.
  // Ignore it only when it cannot hide any previously recorded target range.
  if (!IsTarget &&
      (It == MappingsByPID.end() ||
       llvm::none_of(It->second, [&](const ExecutableMapping &Mapping) {
         if (!Mapping.IsTarget)
           return false;
         return Event.Address >= Mapping.RuntimeAddress
                    ? Event.Address - Mapping.RuntimeAddress < Mapping.Size
                    : Mapping.RuntimeAddress - Event.Address < Event.Size;
       })))
    return;

  auto &Mappings = MappingsByPID[PID];
  // First mask the entire replacement mmap, then add its executable subranges.
  // Reverse lookup sees the new code first and the mask in any remaining gaps.
  // Older entries stay valid outside this mmap, including partial remappings.
  Mappings.push_back({Event.Address, Event.Size, 0, false});
  for (const ExecutableMapping &Mapping : NewMappings) {
    Mappings.push_back(Mapping);
    // Warning ranges use the same binary addresses as the relocated samples.
    Binary->addMMapRange(Mapping.CanonicalAddress, Mapping.Size);
  }
  if (IsTarget)
    Binary->setIsLoadedByMMap(true);
  if (IsTarget && (Binary->isKernel() || !PIDFilter || PID == *PIDFilter))
    // A recording can end after the last mapped process exits. Retain evidence
    // that it had usable mappings when deciding whether the input is valid.
    HadUsableMapping = true;
}

std::optional<uint64_t>
PerfScriptReader::resolveMappedAddress(uint64_t Address, int32_t PID) const {
  auto It = MappingsByPID.find(Binary->isKernel() ? -1 : PID);
  if (It == MappingsByPID.end())
    return std::nullopt;
  // The latest event covering this byte wins. A replacement by another binary
  // stops lookup before it can expose an older target map underneath.
  for (const ExecutableMapping &Mapping : llvm::reverse(It->second)) {
    if (Address < Mapping.RuntimeAddress)
      continue;
    uint64_t Delta = Address - Mapping.RuntimeAddress;
    if (Delta < Mapping.Size) {
      // CanonicalAddress is the binary address at this runtime interval's
      // start; retain the sample's displacement within that interval.
      if (Mapping.IsTarget)
        return checkedAddUnsigned(Mapping.CanonicalAddress, Delta);
      return std::nullopt;
    }
  }
  return std::nullopt;
}

static std::string getContextKeyStr(ContextKey *K,
                                    const ProfiledBinary *Binary) {
  if (const auto *CtxKey = dyn_cast<StringBasedCtxKey>(K)) {
    return SampleContext::getContextString(CtxKey->Context);
  } else if (const auto *CtxKey = dyn_cast<AddrBasedCtxKey>(K)) {
    std::ostringstream OContextStr;
    for (uint32_t I = 0; I < CtxKey->Context.size(); I++) {
      if (OContextStr.str().size())
        OContextStr << " @ ";
      uint64_t Address = CtxKey->Context[I];
      if (UseOffset) {
        if (UseLoadableSegmentAsBase)
          Address -= Binary->getFirstLoadableAddress();
        else
          Address -= Binary->getPreferredBaseAddress();
      }
      OContextStr << "0x"
                  << utohexstr(Address,
                               /*LowerCase=*/true);
    }
    return OContextStr.str();
  } else {
    llvm_unreachable("unexpected key type");
  }
}

void HybridPerfReader::unwindSamples() {
  NamedRegionTimer T("unwind", "Unwind samples", TimerGroupName, TimerGroupDesc,
                     TimeProfGen);
  VirtualUnwinder Unwinder(&SampleCounters, Binary);
  for (const auto &Item : AggregatedSamples) {
    const PerfSample *Sample = Item.first.getPtr();
    Unwinder.unwind(Sample, Item.second);
  }

  // Warn about untracked frames due to missing probes.
  if (ShowDetailedWarning) {
    for (auto Address : Unwinder.getUntrackedCallsites())
      WithColor::warning() << "Profile context truncated due to missing probe "
                           << "for call instruction at "
                           << format("0x%" PRIx64, Address) << "\n";
  }

  emitWarningSummary(Unwinder.getUntrackedCallsites().size(),
                     SampleCounters.size(),
                     "of profiled contexts are truncated due to missing probe "
                     "for call instruction.");

  emitWarningSummary(
      Unwinder.NumMismatchedExtCallBranch, Unwinder.NumTotalBranches,
      "of branches'source is a call instruction but doesn't match call frame "
      "stack, likely due to unwinding error of external frame.");

  emitWarningSummary(Unwinder.NumPairedExtAddr * 2, Unwinder.NumTotalBranches,
                     "of branches containing paired external address.");

  emitWarningSummary(Unwinder.NumUnpairedExtAddr, Unwinder.NumTotalBranches,
                     "of branches containing external address but doesn't have "
                     "another external address to pair, likely due to "
                     "interrupt jmp or broken perf script.");

  emitWarningSummary(
      Unwinder.NumMismatchedProEpiBranch, Unwinder.NumTotalBranches,
      "of branches'source is a call instruction but doesn't match call frame "
      "stack, likely due to frame in prolog/epilog.");

  emitWarningSummary(Unwinder.NumMissingExternalFrame,
                     Unwinder.NumExtCallBranch,
                     "of artificial call branches but doesn't have an external "
                     "frame to match.");

  emitWarningSummary(NumBogusTrace, NumTotalHybridSample,
                     "of hybrid samples had a callchain leaf that disagreed "
                     "with the newest LBR target (bogus trace).");
  if (NumBogusTrace * 100 > NumTotalHybridSample)
    WithColor::warning() << "Bogus trace rate exceeds 1%: the profile has high "
                            "sample skid and may not be suitable for "
                            "optimization.\n";
}

/// Parse a hex address from \p Str.
static bool parseAddress(StringRef Str, uint64_t &Addr, bool HasPrefix,
                         bool AllowSymbols = true) {
  // Legacy input permits symbol/DSO suffixes. Mapped frames must be whole
  // address lines so a PID/time header cannot be consumed as another frame.
  if (AllowSymbols)
    Str = Str.take_while([](char C) { return !isspace(C); });
  else
    Str = Str.rtrim();
  if (Str.consume_front("0x") != HasPrefix)
    return true;
  return Str.getAsInteger(16, Addr);
}

static bool skipCallchainFrames(TraceStream &TraceIt) {
  uint64_t FrameAddr = 0;
  bool HasCallchain = false;
  while (!TraceIt.isAtEoF() &&
         !parseAddress(TraceIt.getCurrentLine().ltrim(), FrameAddr, false,
                       /*AllowSymbols=*/false)) {
    HasCallchain = true;
    TraceIt.advance();
  }
  return HasCallchain;
}

static int32_t parseSamplePID(StringRef &Line) {
  // Consume only the header prefix, leaving the IP/branches for flat samples
  // or an empty payload when perf prints the callchain on following lines.
  auto [PIDField, Rest] = Line.trim().split(' ');
  int32_t PID = 0;
  if (PIDField.getAsInteger(10, PID) || PID < 0)
    exitWithError("Invalid perf script sample PID");
  // The timestamp distinguishes the next sample header from a callchain frame.
  auto [Time, Payload] = Rest.ltrim().split(' ');
  double Timestamp = 0;
  if (!Time.consume_back(":") || Time.getAsDouble(Timestamp) ||
      !std::isfinite(Timestamp))
    exitWithError("Invalid perf script sample timestamp");
  Line = Payload.ltrim();
  return PID;
}

bool PerfScriptReader::extractLBRStack(TraceStream &TraceIt,
                                       SmallVectorImpl<LBREntry> &LBRStack,
                                       std::optional<int32_t> PID) {
  // The raw format of LBR stack is like:
  // 0x4005c8/0x4005dc/P/-/-/0 0x40062f/0x4005b0/P/-/-/0 ...
  //                           ... 0x4005c8/0x4005dc/P/-/-/0
  // It's in FIFO order and separated by whitespace.
  StringRef Line = TraceIt.getCurrentLine();
  if (UseProcessMappings && !PID) {
    // The hybrid reader passes the PID after consuming its header. The flat
    // reader starts at the header and may need to skip an unused callchain.
    PID = parseSamplePID(Line);
    if (Line.empty()) {
      TraceIt.advance();
      skipCallchainFrames(TraceIt);
      if (TraceIt.isAtEoF() ||
          !TraceIt.getCurrentLine().ltrim().starts_with("0x"))
        return false;
      Line = TraceIt.getCurrentLine();
    }
  }
  if (parseLBRLine(Line, LBRStack, PID))
    WithColor::warning() << "Invalid address in LBR record at line "
                         << TraceIt.getLineNumber() << ": "
                         << TraceIt.getCurrentLine() << "\n";
  TraceIt.advance();
  return !LBRStack.empty();
}

bool PerfScriptReader::parseLBRLine(StringRef Line,
                                    SmallVectorImpl<LBREntry> &LBRStack,
                                    std::optional<int32_t> PID) const {
  SmallVector<StringRef, 32> Records;
  Line.rtrim().split(Records, " ", -1, false);

  // Skip the leading instruction pointer.
  size_t Index = 0;
  uint64_t LeadingAddr;
  if (!Records.empty() && !Records[0].contains('/')) {
    if (parseAddress(Records[0], LeadingAddr, false))
      return true;
    Index = 1;
  }

  // Now extract LBR samples - note that we do not reverse the
  // LBR entry order so we can unwind the sample stack as we walk
  // through LBR entries.
  while (Index < Records.size()) {
    auto &Token = Records[Index++];
    if (Token.size() == 0)
      continue;

    SmallVector<StringRef, 8> Addresses;
    Token.split(Addresses, "/");
    uint64_t Src;
    uint64_t Dst;

    // Stop at broken LBR records.
    if (Addresses.size() < 2 || parseAddress(Addresses[0], Src, true) ||
        parseAddress(Addresses[1], Dst, true))
      return true;

    // Mapped input resolves both ends through the sampled process before
    // checking binary instructions. An unmapped end remains external, keeping
    // the unwinder's handling of calls into or out of other binaries.
    std::optional<uint64_t> ResolvedSrc =
        PID ? resolveMappedAddress(Src, *PID)
            : Binary->canonicalizeVirtualAddress(Src);
    std::optional<uint64_t> ResolvedDst =
        PID ? resolveMappedAddress(Dst, *PID)
            : Binary->canonicalizeVirtualAddress(Dst);
    bool SrcIsInternal = ResolvedSrc && Binary->addressIsCode(*ResolvedSrc);
    bool DstIsInternal = ResolvedDst && Binary->addressIsCode(*ResolvedDst);
    Src = SrcIsInternal ? *ResolvedSrc : ExternalAddr;
    Dst = DstIsInternal ? *ResolvedDst : ExternalAddr;
    // Filter external-to-external case to reduce LBR trace size.
    if (!SrcIsInternal && !DstIsInternal)
      continue;

    LBRStack.emplace_back(LBREntry(Src, Dst));
  }
  return false;
}

bool PerfScriptReader::extractCallstack(TraceStream &TraceIt,
                                        SmallVectorImpl<uint64_t> &CallStack,
                                        std::optional<int32_t> PID) {
  // The raw format of call stack is like:
  //            4005dc      # leaf frame
  //	          400634
  //	          400684      # root frame
  // It's in bottom-up order with each frame in one line.

  // Extract stack frames from sample
  while (!TraceIt.isAtEoF() && !isLBRSample(TraceIt.getCurrentLine(), true)) {
    StringRef FrameStr = TraceIt.getCurrentLine().ltrim();
    uint64_t FrameAddr = 0;
    if (parseAddress(FrameStr, FrameAddr, false, /*AllowSymbols=*/!PID)) {
      // With PID headers, leave non-frame lines for the outer parser.
      if (!PID)
        TraceIt.advance();
      return false;
    }
    TraceIt.advance();

    // Relocate before decoding the saved return PC into a call instruction.
    // The instruction decoder and debug information use binary addresses.
    std::optional<uint64_t> ResolvedFrameAddr =
        PID ? resolveMappedAddress(FrameAddr, *PID)
            : Binary->canonicalizeVirtualAddress(FrameAddr);
    // Currently intermixed frame from different binaries is not supported.
    if (!ResolvedFrameAddr || !Binary->addressIsCode(*ResolvedFrameAddr)) {
      if (CallStack.empty())
        NumLeafExternalFrame++;
      // Push a special value(ExternalAddr) for the external frames so that
      // unwinder can still work on this with artificial Call/Return branch.
      // After unwinding, the context will be truncated for external frame.
      // Also deduplicate the consecutive external addresses.
      if (CallStack.empty() || CallStack.back() != ExternalAddr)
        CallStack.emplace_back(ExternalAddr);
      continue;
    }
    FrameAddr = *ResolvedFrameAddr;

    // We need to translate return address to call address for non-leaf frames.
    if (!CallStack.empty()) {
      auto CallAddr = Binary->getCallAddrFromFrameAddr(FrameAddr);
      if (!CallAddr) {
        // Stop at an invalid return address caused by bad unwinding. This could
        // happen to frame-pointer-based unwinding and the callee functions that
        // do not have the frame pointer chain set up.
        InvalidReturnAddresses.insert(FrameAddr);
        break;
      }
      FrameAddr = CallAddr;
    }

    CallStack.emplace_back(FrameAddr);
  }

  // Strip out the bottom external addr.
  if (CallStack.size() > 1 && CallStack.back() == ExternalAddr)
    CallStack.pop_back();

  // Skip other unrelated line, find the next valid LBR line
  // Note that even for empty call stack, we should skip the address at the
  // bottom, otherwise the following pass may generate a truncated callstack
  while (!TraceIt.isAtEoF() && !isLBRSample(TraceIt.getCurrentLine(), true)) {
    // Leave the next sample header and mapping events for the outer parser.
    uint64_t FrameAddr = 0;
    if (PID && parseAddress(TraceIt.getCurrentLine().ltrim(), FrameAddr, false,
                            /*AllowSymbols=*/false))
      return false;
    TraceIt.advance();
  }
  if (PID && TraceIt.isAtEoF())
    return false;
  // Filter out broken stack sample. We may not have complete frame info
  // if sample end up in prolog/epilog, the result is dangling context not
  // connected to entry point. This should be relatively rare thus not much
  // impact on overall profile quality. However we do want to filter them
  // out to reduce the number of different calling contexts. One instance
  // of such case - when sample landed in prolog/epilog, somehow stack
  // walking will be broken in an unexpected way that higher frames will be
  // missing.
  return !CallStack.empty() &&
         !Binary->addressInPrologEpilog(CallStack.front());
}

void PerfScriptReader::warnIfMissingMMap() {
  // Mapped mode cannot fall back to a shared preferred load base; it checks
  // for usable per-process metadata after the full trace has been read.
  if (UseProcessMappings)
    return;
  if (!Binary->getMissingMMapWarned() && !Binary->getIsLoadedByMMap()) {
    WithColor::warning() << "No relevant mmap event is matched for "
                         << Binary->getName()
                         << ", will use preferred address ("
                         << format("0x%" PRIx64,
                                   Binary->getPreferredBaseAddress())
                         << ") as the base loading address!\n";
    // Avoid redundant warning, only warn at the first unmatched sample.
    Binary->setMissingMMapWarned(true);
  }
}

// The unwinder requires that LBR tip belong to the leaf frame.
// External addresses are not checked.
static bool isValidTrace(ProfiledBinary *Binary, uint64_t StackLeaf,
                         uint64_t LBRLeaf) {
  if (StackLeaf == ExternalAddr || LBRLeaf == ExternalAddr)
    return true;
  return Binary->findFuncRange(LBRLeaf) == Binary->findFuncRange(StackLeaf);
}

void HybridPerfReader::parseSample(TraceStream &TraceIt, uint64_t Count) {
  // The raw hybird sample started with call stack in FILO order and followed
  // intermediately by LBR sample
  // e.g.
  // 	          4005dc    # call stack leaf
  //	          400634
  //	          400684    # call stack root
  // 0x4005c8/0x4005dc/P/-/-/0   0x40062f/0x4005b0/P/-/-/0 ...
  //          ... 0x4005c8/0x4005dc/P/-/-/0    # LBR Entries
  //
  std::optional<int32_t> PID;
  if (UseProcessMappings) {
    // Perf puts PID/time on their own line for a resolved callchain. An IP
    // still on that header means this sample has no stack to seed the unwinder.
    StringRef Header = TraceIt.getCurrentLine();
    PID = parseSamplePID(Header);
    bool HasCallchain = Header.empty();
    TraceIt.advance();
    if (!HasCallchain)
      return;
  }
  std::shared_ptr<PerfSample> Sample = std::make_shared<PerfSample>();
#ifndef NDEBUG
  Sample->Linenum = TraceIt.getLineNumber();
#endif
  // Parsing call stack and populate into PerfSample.CallStack
  if (!extractCallstack(TraceIt, Sample->CallStack, PID)) {
    // Skip the next LBR line matched current call stack
    if (!TraceIt.isAtEoF() && isLBRSample(TraceIt.getCurrentLine(), true))
      TraceIt.advance();
    return;
  }

  warnIfMissingMMap();

  if (!TraceIt.isAtEoF() && isLBRSample(TraceIt.getCurrentLine(), true)) {
    // Parsing LBR stack and populate into PerfSample.LBRStack
    if (extractLBRStack(TraceIt, Sample->LBRStack, PID)) {
      if (IgnoreStackSamples) {
        Sample->CallStack.clear();
      } else {
        NumTotalHybridSample++;
        // Drop samples whose callchain and LBR disagree before the
        // canonicalization below hides the disagreement.
        uint64_t StackLeaf = Sample->CallStack.front();
        uint64_t LBRLeaf = Sample->LBRStack[0].Target;
        if (!isValidTrace(Binary, StackLeaf, LBRLeaf)) {
          NumBogusTrace++;
          if (ShowDetailedWarning)
            WithColor::warning()
                << "Bogus trace: stack tip = " << format_hex(StackLeaf, 10)
                << ", LBR tip = " << format_hex(LBRLeaf, 10) << "\n";
          return;
        }
        // Canonicalize stack leaf to avoid 'random' IP from leaf frame skew LBR
        // ranges
        Sample->CallStack.front() = Sample->LBRStack[0].Target;
      }
      // Record samples by aggregation
      AggregatedSamples[Hashable<PerfSample>(Sample)] += Count;
    }
  } else {
    // LBR sample is encoded in single line after stack sample
    exitWithError("'Hybrid perf sample is corrupted, No LBR sample line");
  }
}

void PerfScriptReader::writeUnsymbolizedProfile(StringRef Filename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC, llvm::sys::fs::OF_TextWithCRLF);
  if (EC)
    exitWithError(EC, Filename);
  writeUnsymbolizedProfile(OS);
}

// Use ordered map to make the output deterministic
using OrderedCounterForPrint = std::map<std::string, SampleCounter *>;

void PerfScriptReader::writeUnsymbolizedProfile(raw_fd_ostream &OS) {
  OrderedCounterForPrint OrderedCounters;
  for (auto &CI : SampleCounters) {
    OrderedCounters[getContextKeyStr(CI.first.getPtr(), Binary)] = &CI.second;
  }

  auto SCounterPrinter = [&](RangeSample &Counter, StringRef Separator,
                             uint32_t Indent) {
    OS.indent(Indent);
    OS << Counter.size() << "\n";
    for (auto &I : Counter) {
      uint64_t Start = I.first.first;
      uint64_t End = I.first.second;

      if (UseOffset) {
        if (UseLoadableSegmentAsBase) {
          Start -= Binary->getFirstLoadableAddress();
          End -= Binary->getFirstLoadableAddress();
        } else {
          Start -= Binary->getPreferredBaseAddress();
          End -= Binary->getPreferredBaseAddress();
        }
      }

      OS.indent(Indent);
      OS << Twine::utohexstr(Start) << Separator << Twine::utohexstr(End) << ":"
         << I.second << "\n";
    }
  };

  for (auto &CI : OrderedCounters) {
    uint32_t Indent = 0;
    if (ProfileIsCS) {
      // Context string key
      OS << "[" << CI.first << "]\n";
      Indent = 2;
    }

    SampleCounter &Counter = *CI.second;
    SCounterPrinter(Counter.RangeCounter, "-", Indent);
    SCounterPrinter(Counter.BranchCounter, "->", Indent);
  }
}

// Format of input:
// number of entries in RangeCounter
// from_1-to_1:count_1
// from_2-to_2:count_2
// ......
// from_n-to_n:count_n
// number of entries in BranchCounter
// src_1->dst_1:count_1
// src_2->dst_2:count_2
// ......
// src_n->dst_n:count_n
void UnsymbolizedProfileReader::readSampleCounters(TraceStream &TraceIt,
                                                   SampleCounter &SCounters) {
  auto exitWithErrorForTraceLine = [](TraceStream &TraceIt) {
    std::string Msg = TraceIt.isAtEoF()
                          ? "Invalid raw profile!"
                          : "Invalid raw profile at line " +
                                Twine(TraceIt.getLineNumber()).str() + ": " +
                                TraceIt.getCurrentLine().str();
    exitWithError(Msg);
  };
  auto ReadNumber = [&](uint64_t &Num) {
    if (TraceIt.isAtEoF())
      exitWithErrorForTraceLine(TraceIt);
    if (TraceIt.getCurrentLine().ltrim().getAsInteger(10, Num))
      exitWithErrorForTraceLine(TraceIt);
    TraceIt.advance();
  };

  auto ReadCounter = [&](RangeSample &Counter, StringRef Separator) {
    uint64_t Num = 0;
    ReadNumber(Num);
    while (Num--) {
      if (TraceIt.isAtEoF())
        exitWithErrorForTraceLine(TraceIt);
      StringRef Line = TraceIt.getCurrentLine().ltrim();

      uint64_t Count = 0;
      auto LineSplit = Line.split(":");
      if (LineSplit.second.empty() || LineSplit.second.getAsInteger(10, Count))
        exitWithErrorForTraceLine(TraceIt);

      uint64_t Source = 0;
      uint64_t Target = 0;
      auto Range = LineSplit.first.split(Separator);
      if (Range.second.empty() || Range.first.getAsInteger(16, Source) ||
          Range.second.getAsInteger(16, Target))
        exitWithErrorForTraceLine(TraceIt);

      if (UseOffset) {
        if (UseLoadableSegmentAsBase) {
          Source += Binary->getFirstLoadableAddress();
          Target += Binary->getFirstLoadableAddress();
        } else {
          Source += Binary->getPreferredBaseAddress();
          Target += Binary->getPreferredBaseAddress();
        }
      }

      Counter[{Source, Target}] += Count;
      TraceIt.advance();
    }
  };

  ReadCounter(SCounters.RangeCounter, "-");
  ReadCounter(SCounters.BranchCounter, "->");
}

void UnsymbolizedProfileReader::readUnsymbolizedProfile(StringRef FileName) {
  TraceStream TraceIt(FileName);
  while (!TraceIt.isAtEoF()) {
    std::shared_ptr<StringBasedCtxKey> Key =
        std::make_shared<StringBasedCtxKey>();
    StringRef Line = TraceIt.getCurrentLine();
    // Read context stack for CS profile.
    if (Line.starts_with("[")) {
      ProfileIsCS = true;
      auto I = ContextStrSet.insert(Line);
      SampleContext::createCtxVectorFromStr(I.first->getKey(), Key->Context);
      TraceIt.advance();
    }
    auto Ret = SampleCounters.try_emplace(Hashable<ContextKey>(Key));
    readSampleCounters(TraceIt, Ret.first->second);
  }
}

void UnsymbolizedProfileReader::parsePerfTraces() {
  readUnsymbolizedProfile(PerfTraceFile);
}

void PerfScriptReader::computeCounterFromLBR(const PerfSample *Sample,
                                             uint64_t Repeat) {
  SampleCounter &Counter = SampleCounters.begin()->second;
  uint64_t EndAddress = 0;
  for (const LBREntry &LBR : Sample->LBRStack) {
    uint64_t SourceAddress = LBR.Source;
    uint64_t TargetAddress = LBR.Target;

    // Record the branch if its SourceAddress is external. It can be the case an
    // external source call an internal function, later this branch will be used
    // to generate the function's head sample.
    if (Binary->addressIsCode(TargetAddress)) {
      Counter.recordBranchCount(SourceAddress, TargetAddress, Repeat);
    }

    // If this not the first LBR, update the range count between TO of current
    // LBR and FROM of next LBR.
    uint64_t StartAddress = TargetAddress;
    if (Binary->addressIsCode(StartAddress) &&
        Binary->addressIsCode(EndAddress) &&
        isValidFallThroughRange(StartAddress, EndAddress, Binary))
      Counter.recordRangeCount(StartAddress, EndAddress, Repeat);
    EndAddress = SourceAddress;
  }
}

void LBRPerfReader::parseSample(TraceStream &TraceIt, uint64_t Count) {
  std::shared_ptr<PerfSample> Sample = std::make_shared<PerfSample>();
  // Parsing LBR stack and populate into PerfSample.LBRStack
  if (extractLBRStack(TraceIt, Sample->LBRStack)) {
    warnIfMissingMMap();
    // Record LBR only samples by aggregation
    AggregatedSamples[Hashable<PerfSample>(Sample)] += Count;
  }
}

void PerfScriptReader::generateUnsymbolizedProfile() {
  // There is no context for LBR only sample, so initialize one entry with
  // fake "empty" context key.
  assert(SampleCounters.empty() &&
         "Sample counter map should be empty before raw profile generation");
  std::shared_ptr<StringBasedCtxKey> Key =
      std::make_shared<StringBasedCtxKey>();
  SampleCounters.try_emplace(Hashable<ContextKey>(Key));
  for (const auto &Item : AggregatedSamples) {
    const PerfSample *Sample = Item.first.getPtr();
    computeCounterFromLBR(Sample, Item.second);
  }
}

uint64_t PerfScriptReader::parseAggregatedCount(TraceStream &TraceIt) {
  // The aggregated count is optional, so do not skip the line and return 1 if
  // it's unmatched
  uint64_t Count = 1;
  if (!TraceIt.getCurrentLine().getAsInteger(10, Count))
    TraceIt.advance();
  return Count;
}

void PerfScriptReader::parseSample(TraceStream &TraceIt) {
  // Filter samples before updating statistics. Metadata is replayed for every
  // PID because a selected child may inherit an unselected parent's maps.
  if (UseProcessMappings && PIDFilter) {
    StringRef Line = TraceIt.getCurrentLine();
    if (parseSamplePID(Line) != *PIDFilter) {
      TraceIt.advance();
      if (Line.empty()) {
        // Consume the rest of this sample as a unit, preserving the next
        // header or task event if the excluded sample has no branch line.
        skipCallchainFrames(TraceIt);
        if (!TraceIt.isAtEoF() &&
            TraceIt.getCurrentLine().ltrim().starts_with("0x"))
          TraceIt.advance();
      }
      return;
    }
  }
  NumTotalSample++;
  uint64_t Count = UseProcessMappings ? 1 : parseAggregatedCount(TraceIt);
  assert(Count >= 1 && "Aggregated count should be >= 1!");
  parseSample(TraceIt, Count);
}

static bool extractMMapEvent(StringRef Line, MMapEvent &MMap,
                             bool RequireProcessMappingInfo) {
  // Parse a MMap2 line like:
  //  PERF_RECORD_MMAP2 2113428/2113428: [0x7fd4efb57000(0x204000) @ 0
  //  08:04 19532229 3585508847]: r-xp /usr/lib64/libdl-2.17.so
  constexpr static const char *const MMap2Pattern =
      "PERF_RECORD_MMAP2 (-?[0-9]+)/([0-9]+): "
      "\\[(0x[a-f0-9]+|0)\\((0x[a-f0-9]+|0)\\) @ "
      "(0x[a-f0-9]+|0) .*\\]: ([-a-z]+) (.*)";
  // Parse a MMap line like
  // PERF_RECORD_MMAP -1/0: [0xffffffff81e00000(0x3e8fa000) @ \
  //  0xffffffff81e00000]: x [kernel.kallsyms]_text
  constexpr static const char *const MMapPattern =
      "PERF_RECORD_MMAP (-?[0-9]+)/([0-9]+): "
      "\\[(0x[a-f0-9]+|0)\\((0x[a-f0-9]+|0)\\) @ "
      "(0x[a-f0-9]+|0)\\]: ([-a-z]+) (.*)";
  // Both patterns use these capture-group indices.
  enum EventIndex {
    WHOLE_LINE = 0,
    PID = 1,
    TID = 2,
    MMAPPED_ADDRESS = 3,
    MMAPPED_SIZE = 4,
    PAGE_OFFSET = 5,
    MEM_PROTECTION_FLAG = 6,
    BINARY_PATH = 7,
  };

  bool R = false;
  SmallVector<StringRef, 8> Fields;
  if (Line.contains("PERF_RECORD_MMAP2 ")) {
    Regex RegMmap2(MMap2Pattern);
    R = RegMmap2.match(Line, &Fields);
  } else if (Line.contains("PERF_RECORD_MMAP ")) {
    Regex RegMmap(MMapPattern);
    R = RegMmap.match(Line, &Fields);
  } else if (!RequireProcessMappingInfo)
    llvm_unreachable("unexpected MMAP event entry");

  if (!R) {
    if (RequireProcessMappingInfo)
      exitWithError("Invalid perf script mmap event");
    return false;
  }
  // Preserve the legacy address/size spelling outside mapped mode.
  if (!RequireProcessMappingInfo &&
      (Fields[MMAPPED_ADDRESS] == "0" || Fields[MMAPPED_SIZE] == "0"))
    return false;
  if (RequireProcessMappingInfo) {
    // Size-1 permits a valid mapping whose final byte is UINT64_MAX.
    int32_t MMapPID = 0;
    if (Fields[PID].getAsInteger(10, MMapPID) ||
        Fields[TID].getAsInteger(10, MMap.TID) ||
        Fields[MMAPPED_ADDRESS].getAsInteger(0, MMap.Address) ||
        Fields[MMAPPED_SIZE].getAsInteger(0, MMap.Size) ||
        (MMap.Size != 0 &&
         MMap.Size - 1 > std::numeric_limits<uint64_t>::max() - MMap.Address) ||
        Fields[PAGE_OFFSET].getAsInteger(0, MMap.Offset) ||
        Fields[BINARY_PATH].empty())
      exitWithError("Invalid perf script mmap event");
    MMap.PID = MMapPID;
  } else {
    long long MMapPID = 0;
    getAsSignedInteger(Fields[PID], 10, MMapPID);
    MMap.PID = MMapPID;
    Fields[MMAPPED_ADDRESS].getAsInteger(0, MMap.Address);
    Fields[MMAPPED_SIZE].getAsInteger(0, MMap.Size);
    Fields[PAGE_OFFSET].getAsInteger(0, MMap.Offset);
  }
  MMap.MemProtectionFlag = Fields[MEM_PROTECTION_FLAG];
  MMap.BinaryPath = Fields[BINARY_PATH];
  return true;
}

static bool mmapMatchesBinary(const ProfiledBinary &Binary,
                              const MMapEvent &MMap) {
  StringRef BinaryName = filename(MMap.BinaryPath, Binary.isCOFF());
  if (Binary.isKernel())
    return Binary.isKernelImageName(BinaryName);
  return Binary.getName() == BinaryName;
}

bool PerfScriptReader::extractMMapEventForBinary(
    ProfiledBinary *Binary, StringRef Line, MMapEvent &MMap,
    bool RequireProcessMappingInfo) {
  // Mapped mode needs records for other binaries too: a later mmap can cover
  // a target address even when it has a different filename or permissions.
  if (!RequireProcessMappingInfo && !Binary->isKernel() &&
      !Line.contains(Binary->getName()) && !ShowMmapEvents)
    return false;
  if (!extractMMapEvent(Line, MMap, RequireProcessMappingInfo)) {
    std::string WarningMsg = "Cannot parse mmap event: " + Line.str() + " \n";
    WithColor::warning() << WarningMsg;
    return false;
  }
  if (ShowMmapEvents) {
    outs() << "Mmap: Binary " << MMap.BinaryPath << " loaded at "
           << format("0x%" PRIx64 ":", MMap.Address) << " \n";
  }

  return mmapMatchesBinary(*Binary, MMap);
}

void PerfScriptReader::parseMMapEvent(TraceStream &TraceIt) {
  MMapEvent MMap;
  bool IsTarget = extractMMapEventForBinary(Binary, TraceIt.getCurrentLine(),
                                            MMap, UseProcessMappings);
  if (UseProcessMappings)
    updateExecutableMapping(MMap, IsTarget);
  else if (IsTarget)
    updateBinaryAddress(MMap);
  TraceIt.advance();
}

void PerfScriptReader::parseEventOrSample(TraceStream &TraceIt) {
  if (UseProcessMappings && TraceIt.getCurrentLine().trim().empty()) {
    TraceIt.advance();
    return;
  }
  StringRef Line = TraceIt.getCurrentLine();
  if (UseProcessMappings ? Line.contains("PERF_RECORD_MMAP")
                         : isMMapEvent(Line))
    parseMMapEvent(TraceIt);
  else if (UseProcessMappings && Line.contains("PERF_RECORD_")) {
    // Replay metadata in input order before resolving subsequent samples.
    // The kernel address space is shared and outlives individual tasks.
    TaskEvent Event = parseTaskEvent(Line);
    if (!Binary->isKernel()) {
      if (Event.Kind == TaskKind::Fork && !Event.IsSynthetic) {
        // FORK also proves the parent thread is live, even when recording
        // began too late to include a separate record announcing that thread.
        if (ThreadsByPID[Event.ParentPID].insert(Event.ParentTID).second)
          CompleteThreadPIDs.erase(Event.ParentPID);
        if (Event.PID == Event.ParentPID) {
          // Thread creation shares the existing process map.
          ThreadsByPID[Event.PID].insert(Event.TID);
        } else {
          // Process creation snapshots the parent's current map history.
          // Copying keeps later remaps or execs in either process independent.
          ThreadsByPID[Event.PID] = {Event.TID};
          CompleteThreadPIDs.insert(Event.PID);
          auto Parent = MappingsByPID.find(Event.ParentPID);
          if (Parent == MappingsByPID.end()) {
            // A reused child PID must not retain maps from its old lifetime.
            MappingsByPID.erase(Event.PID);
          } else {
            MappingsByPID[Event.PID] = Parent->second;
            if (!PIDFilter || Event.PID == *PIDFilter)
              HadUsableMapping = true;
          }
        }
      } else if (Event.Kind == TaskKind::Comm) {
        // A name change reveals a live thread but does not replace its image.
        if (ThreadsByPID[Event.PID].insert(Event.TID).second)
          CompleteThreadPIDs.erase(Event.PID);
      } else if (Event.Kind == TaskKind::Exec) {
        // Exec keeps the PID, replaces its image and removes sibling threads.
        // New mmap/task records will describe the replacement process state.
        MappingsByPID.erase(Event.PID);
        ThreadsByPID[Event.PID] = {Event.TID};
        if (Event.IsSynthetic)
          CompleteThreadPIDs.erase(Event.PID);
        else
          CompleteThreadPIDs.insert(Event.PID);
      } else if (Event.Kind == TaskKind::Exit) {
        // Only a complete inventory proves this is the final thread. Retain
        // mappings when recording began too late to observe every thread.
        auto It = ThreadsByPID.find(Event.PID);
        if (It == ThreadsByPID.end() || !It->second.erase(Event.TID)) {
          CompleteThreadPIDs.erase(Event.PID);
        } else if (It->second.empty()) {
          ThreadsByPID.erase(It);
          if (CompleteThreadPIDs.erase(Event.PID))
            MappingsByPID.erase(Event.PID);
        }
      }
    }
    TraceIt.advance();
  } else
    parseSample(TraceIt);
}

void PerfScriptReader::parseAndAggregateTrace() {
  NamedRegionTimer T("parseTrace", "Parse and aggregate trace", TimerGroupName,
                     TimerGroupDesc, TimeProfGen);
  // Trace line iterator
  TraceStream TraceIt(PerfTraceFile);
  while (!TraceIt.isAtEoF())
    parseEventOrSample(TraceIt);
}

// A LBR sample is like:
// 40062f 0x5c6313f/0x5c63170/P/-/-/0  0x5c630e7/0x5c63130/P/-/-/0 ...
// A heuristic for fast detection by checking whether a
// leading "  0x" and the '/' exist.
bool PerfScriptReader::isLBRSample(StringRef Line, bool CheckLineStart) {
  // Skip the leading instruction pointer
  SmallVector<StringRef, 32> Records;
  if (!CheckLineStart)
    Line = Line.trim();
  // Line might start with IP or only contain brstack. Check first two records
  // and fail if no record exists.
  Line.split(Records, " ", 2, CheckLineStart);
  for (StringRef Record : Records)
    if (Record.starts_with("0x") && Record.contains('/'))
      return true;
  return false;
}

bool PerfScriptReader::isMMapEvent(StringRef Line) {
  // Short cut to avoid string find is possible.
  if (Line.empty() || Line.size() < 50)
    return false;

  if (std::isdigit(Line[0]))
    return false;

  // PERF_RECORD_MMAP2 or PERF_RECORD_MMAP does not appear at the beginning of
  // the line for ` perf script  --show-mmap-events  -i ...`
  return Line.contains("PERF_RECORD_MMAP");
}

// The raw hybird sample is like
// e.g.
// 	          4005dc    # call stack leaf
//	          400634
//	          400684    # call stack root
// 0x4005c8/0x4005dc/P/-/-/0   0x40062f/0x4005b0/P/-/-/0 ...
//          ... 0x4005c8/0x4005dc/P/-/-/0    # LBR Entries
// Determine the perfscript contains hybrid samples(call stack + LBRs) by
// checking whether there is a non-empty call stack immediately followed by
// a LBR sample
PerfContent PerfScriptReader::checkPerfScriptType(
    ProfiledBinary *Binary, StringRef FileName, bool UseProcessMappings,
    std::optional<int32_t> PIDFilter) {
  TraceStream TraceIt(FileName);
  if (UseProcessMappings) {
    // Discover the reader type using the same mapping/task replay as the real
    // parse. A callchain from another PID or binary must not select CS mode
    // for a target whose usable samples contain only branch stacks.
    PerfScriptReader Reader(Binary, FileName, PIDFilter, true);
    bool HasLBR = false;
    while (!TraceIt.isAtEoF()) {
      StringRef Line = TraceIt.getCurrentLine().trim();
      if (Line.contains("PERF_RECORD_MMAP")) {
        MMapEvent MMap;
        extractMMapEvent(TraceIt.getCurrentLine(), MMap,
                         /*RequireProcessMappingInfo=*/true);
        Reader.updateExecutableMapping(MMap, mmapMatchesBinary(*Binary, MMap));
        TraceIt.advance();
        continue;
      }
      if (Line.contains("PERF_RECORD_")) {
        Reader.parseEventOrSample(TraceIt);
        continue;
      }
      if (!Line.empty()) {
        int32_t PID = parseSamplePID(Line);
        if (Line.empty()) {
          TraceIt.advance();
          bool HasCallchain = skipCallchainFrames(TraceIt);
          if (!TraceIt.isAtEoF() &&
              isLBRSample(TraceIt.getCurrentLine(), false)) {
            HasLBR = true;
            SmallVector<LBREntry, 16> LBRStack;
            if (!PIDFilter || PID == *PIDFilter)
              Reader.parseLBRLine(TraceIt.getCurrentLine(), LBRStack, PID);
            TraceIt.advance();
            // Require a selected sample with a branch touching target code,
            // not merely a syntactically valid stack from some other image.
            if (!LBRStack.empty() && HasCallchain)
              return PerfContent::LBRStack;
          }
          continue;
        }
        HasLBR |= isLBRSample(Line, false);
      }
      TraceIt.advance();
    }
    if (HasLBR)
      return PerfContent::LBR;
    exitWithError("Invalid perf script input!");
  }
  uint64_t FrameAddr = 0;
  while (!TraceIt.isAtEoF()) {
    // Skip the aggregated count
    if (!TraceIt.getCurrentLine().getAsInteger(10, FrameAddr))
      TraceIt.advance();

    // Detect sample with call stack
    int32_t Count = 0;
    while (!TraceIt.isAtEoF() &&
           !isLBRSample(TraceIt.getCurrentLine(), false) &&
           !parseAddress(TraceIt.getCurrentLine().ltrim(), FrameAddr, false)) {
      Count++;
      TraceIt.advance();
    }
    if (!TraceIt.isAtEoF()) {
      if (isLBRSample(TraceIt.getCurrentLine(), false)) {
        if (Count > 0)
          return PerfContent::LBRStack;
        else
          return PerfContent::LBR;
      }
      TraceIt.advance();
    }
  }

  exitWithError("Invalid perf script input!");
  return PerfContent::UnknownContent;
}

void HybridPerfReader::generateUnsymbolizedProfile() {
  ProfileIsCS = !IgnoreStackSamples;
  if (ProfileIsCS)
    unwindSamples();
  else
    PerfScriptReader::generateUnsymbolizedProfile();
}

void PerfScriptReader::warnTruncatedStack() {
  if (ShowDetailedWarning) {
    for (auto Address : InvalidReturnAddresses) {
      WithColor::warning()
          << "Truncated stack sample due to invalid return address at "
          << format("0x%" PRIx64, Address)
          << ", likely caused by frame pointer omission\n";
    }
  }
  emitWarningSummary(
      InvalidReturnAddresses.size(), AggregatedSamples.size(),
      "of truncated stack samples due to invalid return address, "
      "likely caused by frame pointer omission.");
}

void PerfScriptReader::warnInvalidRange() {
  DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> Ranges;

  for (const auto &Item : AggregatedSamples) {
    const PerfSample *Sample = Item.first.getPtr();
    uint64_t Count = Item.second;
    uint64_t EndAddress = 0;
    for (const LBREntry &LBR : Sample->LBRStack) {
      uint64_t SourceAddress = LBR.Source;
      uint64_t StartAddress = LBR.Target;
      if (EndAddress != 0)
        Ranges[{StartAddress, EndAddress}] += Count;
      EndAddress = SourceAddress;
    }
  }

  if (Ranges.empty()) {
    WithColor::warning() << "No samples in perf script!\n";
    return;
  }

  auto WarnInvalidRange = [&](uint64_t StartAddress, uint64_t EndAddress,
                              StringRef Msg) {
    if (!ShowDetailedWarning)
      return;
    WithColor::warning() << "[" << format("%8" PRIx64, StartAddress) << ","
                         << format("%8" PRIx64, EndAddress) << "]: " << Msg
                         << "\n";
  };

  const char *EndNotBoundaryMsg = "Range is not on instruction boundary, "
                                  "likely due to profile and binary mismatch.";
  const char *DanglingRangeMsg = "Range does not belong to any functions, "
                                 "likely from PLT, .init or .fini section.";
  const char *RangeCrossFuncMsg =
      "Fall through range should not cross function boundaries, likely due to "
      "profile and binary mismatch.";
  const char *BogusRangeMsg = "Range start is after or too far from range end.";

  uint64_t TotalRangeNum = 0;
  uint64_t InstNotBoundary = 0;
  uint64_t UnmatchedRange = 0;
  uint64_t RecoveredRange = 0;
  uint64_t RangeCrossFunc = 0;
  uint64_t BogusRange = 0;

  for (auto &I : Ranges) {
    uint64_t StartAddress = I.first.first;
    uint64_t EndAddress = I.first.second;
    TotalRangeNum += I.second;

    if (!Binary->addressIsCode(StartAddress) &&
        !Binary->addressIsCode(EndAddress))
      continue;

    if (!Binary->addressIsCode(StartAddress) ||
        !Binary->addressIsTransfer(EndAddress)) {
      InstNotBoundary += I.second;
      WarnInvalidRange(StartAddress, EndAddress, EndNotBoundaryMsg);
    }

    auto *FRange = Binary->findFuncRange(StartAddress);
    if (!FRange) {
      UnmatchedRange += I.second;
      WarnInvalidRange(StartAddress, EndAddress, DanglingRangeMsg);
      continue;
    }

    if (FRange->Func->NameStatus != DwarfNameStatus::Matched)
      RecoveredRange += I.second;

    if (EndAddress >= FRange->EndAddress) {
      RangeCrossFunc += I.second;
      WarnInvalidRange(StartAddress, EndAddress, RangeCrossFuncMsg);
    }

    if (Binary->addressIsCode(StartAddress) &&
        Binary->addressIsCode(EndAddress) &&
        !isValidFallThroughRange(StartAddress, EndAddress, Binary)) {
      BogusRange += I.second;
      WarnInvalidRange(StartAddress, EndAddress, BogusRangeMsg);
    }
  }

  emitWarningSummary(
      InstNotBoundary, TotalRangeNum,
      "of samples are from ranges that are not on instruction boundary.");
  emitWarningSummary(
      UnmatchedRange, TotalRangeNum,
      "of samples are from ranges that do not belong to any functions.");
  emitWarningSummary(RecoveredRange, TotalRangeNum,
                     "of samples are from ranges that belong to functions "
                     "recovered from symbol table.");
  emitWarningSummary(
      RangeCrossFunc, TotalRangeNum,
      "of samples are from ranges that do cross function boundaries.");
  emitWarningSummary(
      BogusRange, TotalRangeNum,
      "of samples are from ranges that have range start after or too far from "
      "range end acrossing the unconditinal jmp.");
}

void PerfScriptReader::warnIfBranchTargetMismatch() {
  // Collect unique branch source and target addresses from LBR samples,
  // then check what percentage don't match known instructions in the binary.

  uint64_t MismatchedBranches = 0;
  uint64_t MismatchedIndirectTargets = 0;
  uint64_t MismatchedTargets = 0;
  uint64_t TotalSamples = 0;

  for (const auto &Item : AggregatedSamples) {
    const PerfSample *Sample = Item.first.getPtr();
    for (const LBREntry &LBR : Sample->LBRStack) {
      uint64_t Source = LBR.Source;
      uint64_t Target = LBR.Target;
      if (Source == ExternalAddr || Target == ExternalAddr)
        continue;
      TotalSamples++;

      // Validate Branch sources are Call/Branch/Indirect Branch
      if (!Binary->addressIsTransfer(Source))
        MismatchedBranches++;

      // Validate Indirect Branch targets landed in code. This may over estimate
      // the vaid targets only because there's no good way to determine jump
      // table targets
      if (Binary->addressIsIndirectBranch(Source)) {
        if (!Binary->addressIsCode(Target))
          MismatchedIndirectTargets++;
      } else if (!Binary->addressIsBranchTarget(Target) &&
                 !Binary->findFuncRangeForStartAddr(Target))
        MismatchedTargets++;
    }
  }

  emitWarningSummary(MismatchedBranches, TotalSamples,
                     "of branch samples do not match the binary.");
  emitWarningSummary(MismatchedTargets, TotalSamples,
                     "of branch targets do not match the binary.");
  emitWarningSummary(MismatchedIndirectTargets, TotalSamples,
                     "of indirect branch targets do not match the binary.");
}

void PerfScriptReader::parsePerfTraces() {
  // Parse perf traces and do aggregation.
  parseAndAggregateTrace();
  // An empty result can be valid after filtering, but missing mapping metadata
  // cannot establish which binary instructions the samples referred to.
  if (UseProcessMappings && !HadUsableMapping)
    exitWithError("Invalid perf script input: no usable mmap");
  if (Binary->isKernel() && !Binary->getIsLoadedByMMap()) {
    exitWithError(
        "Kernel is requested, but no kernel is found in mmap events.");
  }

  emitWarningSummary(NumLeafExternalFrame, NumTotalSample,
                     "of samples have leaf external frame in call stack.");
  emitWarningSummary(NumLeadingOutgoingLBR, NumTotalSample,
                     "of samples have leading external LBR.");

  // Generate unsymbolized profile.
  warnTruncatedStack();
  warnInvalidRange();
  warnIfBranchTargetMismatch();
  generateUnsymbolizedProfile();
  AggregatedSamples.clear();

  if (SkipSymbolization)
    writeUnsymbolizedProfile(OutputFilename);
}

SmallVector<CleanupInstaller, 2> PerfScriptReader::TempFileCleanups;

void ETMReader::recordProcessedRange(uint64_t Start, uint64_t End,
                                     uint64_t Count) {
  assert(!Counters.empty() && "Counters should not be empty!");
  auto &Counter = Counters.begin()->second;
  Counter.recordRangeCount(Start, End, Count);
}

class ETMCallback : public ETMDecoder::Callback {
  ETMReader *Reader;

public:
  ETMCallback(ETMReader *R) : Reader(R) {}
  void processInstructionRange(uint64_t Start, uint64_t End) override {
    Reader->recordProcessedRange(Start, End, 1);
  }
};

void ETMReader::parseETMTraces() {
  auto BufferOrErr = MemoryBuffer::getFile(TraceFile);
  if (std::error_code EC = BufferOrErr.getError())
    exitWithError("Could not open ETM trace file: " + EC.message());

  ArrayRef<uint8_t> Data(
      reinterpret_cast<const uint8_t *>((*BufferOrErr)->getBufferStart()),
      (*BufferOrErr)->getBufferSize());

  // There is no context for ETM instruction traces.
  // Initialize the SampleCounters map with a single empty context key
  // to aggregate all instruction hits into a global bucket.
  auto Key = std::make_shared<StringBasedCtxKey>();
  Counters.try_emplace(Hashable<ContextKey>(Key));

  // The protocol utilizes a 0x80 byte as an initial synchronization header.
  // Perform a manual search for this sync point to discard any leading
  // padding or truncated packets before decoding begins.
  size_t StartIdx = 0;
  while (StartIdx < Data.size() && Data[StartIdx] != 0x80)
    StartIdx++;
  if (StartIdx >= Data.size())
    exitWithError("No synchronization header (0x80) found in the bitstream.");
  ArrayRef<uint8_t> TraceSlice = Data.slice(StartIdx);

  auto DecoderOrErr = ETMDecoder::create(
      Binary->getBinary(), Binary->getTriple(), static_cast<uint8_t>(TraceID));

  if (!DecoderOrErr)
    exitWithError(toString(DecoderOrErr.takeError()));
  auto Decoder = std::move(*DecoderOrErr);

  ETMCallback CB(this);
  if (Error E = Decoder->processTrace(TraceSlice, CB))
    exitWithError(toString(std::move(E)));
}

} // end namespace sampleprof
} // end namespace llvm
