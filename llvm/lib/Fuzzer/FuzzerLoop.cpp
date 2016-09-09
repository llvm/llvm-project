//===- FuzzerLoop.cpp - Fuzzer's main loop --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Fuzzer's main loop.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <algorithm>
#include <cstring>
#include <memory>

#if defined(__has_include)
#if __has_include(<sanitizer / coverage_interface.h>)
#include <sanitizer/coverage_interface.h>
#endif
#if __has_include(<sanitizer / lsan_interface.h>)
#include <sanitizer/lsan_interface.h>
#endif
#endif

#define NO_SANITIZE_MEMORY
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#undef NO_SANITIZE_MEMORY
#define NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#endif
#endif

namespace fuzzer {
static const size_t kMaxUnitSizeToPrint = 256;

thread_local bool Fuzzer::IsMyThread;

static void MissingExternalApiFunction(const char *FnName) {
  Printf("ERROR: %s is not defined. Exiting.\n"
         "Did you use -fsanitize-coverage=... to build your code?\n",
         FnName);
  exit(1);
}

#define CHECK_EXTERNAL_FUNCTION(fn)                                            \
  do {                                                                         \
    if (!(EF->fn))                                                             \
      MissingExternalApiFunction(#fn);                                         \
  } while (false)

// Only one Fuzzer per process.
static Fuzzer *F;

void Fuzzer::ResetEdgeCoverage() {
  CHECK_EXTERNAL_FUNCTION(__sanitizer_reset_coverage);
  EF->__sanitizer_reset_coverage();
}

void Fuzzer::ResetCounters() {
  if (Options.UseCounters) {
    EF->__sanitizer_update_counter_bitset_and_clear_counters(0);
  }
  if (EF->__sanitizer_get_coverage_pc_buffer_pos)
    PcBufferPos = EF->__sanitizer_get_coverage_pc_buffer_pos();
}

void Fuzzer::PrepareCounters(Fuzzer::Coverage *C) {
  if (Options.UseCounters) {
    size_t NumCounters = EF->__sanitizer_get_number_of_counters();
    C->CounterBitmap.resize(NumCounters);
  }
}

// Records data to a maximum coverage tracker. Returns true if additional
// coverage was discovered.
bool Fuzzer::RecordMaxCoverage(Fuzzer::Coverage *C) {
  bool Res = false;

  uint64_t NewBlockCoverage = EF->__sanitizer_get_total_unique_coverage();
  if (NewBlockCoverage > C->BlockCoverage) {
    Res = true;
    C->BlockCoverage = NewBlockCoverage;
  }

  if (Options.UseIndirCalls &&
      EF->__sanitizer_get_total_unique_caller_callee_pairs) {
    uint64_t NewCallerCalleeCoverage =
        EF->__sanitizer_get_total_unique_caller_callee_pairs();
    if (NewCallerCalleeCoverage > C->CallerCalleeCoverage) {
      Res = true;
      C->CallerCalleeCoverage = NewCallerCalleeCoverage;
    }
  }

  if (Options.UseCounters) {
    uint64_t CounterDelta =
        EF->__sanitizer_update_counter_bitset_and_clear_counters(
            C->CounterBitmap.data());
    if (CounterDelta > 0) {
      Res = true;
      C->CounterBitmapBits += CounterDelta;
    }
  }

  size_t NewPCMapBits = PCMapMergeFromCurrent(C->PCMap);
  if (NewPCMapBits > C->PCMapBits) {
    Res = true;
    C->PCMapBits = NewPCMapBits;
  }

  size_t NewVPMapBits = VPMapMergeFromCurrent(C->VPMap);
  if (NewVPMapBits > C->VPMapBits) {
    Res = true;
    C->VPMapBits = NewVPMapBits;
  }

  if (EF->__sanitizer_get_coverage_pc_buffer_pos) {
    uint64_t NewPcBufferPos = EF->__sanitizer_get_coverage_pc_buffer_pos();
    if (NewPcBufferPos > PcBufferPos) {
      Res = true;
      PcBufferPos = NewPcBufferPos;
    }

    if (PcBufferLen && NewPcBufferPos >= PcBufferLen) {
      Printf("ERROR: PC buffer overflow\n");
      _Exit(1);
    }
  }

  return Res;
}

// Leak detection is expensive, so we first check if there were more mallocs
// than frees (using the sanitizer malloc hooks) and only then try to call lsan.
struct MallocFreeTracer {
  void Start() {
    Mallocs = 0;
    Frees = 0;
  }
  // Returns true if there were more mallocs than frees.
  bool Stop() { return Mallocs > Frees; }
  std::atomic<size_t> Mallocs;
  std::atomic<size_t> Frees;
};

static MallocFreeTracer AllocTracer;

void MallocHook(const volatile void *ptr, size_t size) {
  AllocTracer.Mallocs++;
}
void FreeHook(const volatile void *ptr) {
  AllocTracer.Frees++;
}

Fuzzer::Fuzzer(UserCallback CB, MutationDispatcher &MD, FuzzingOptions Options)
    : CB(CB), MD(MD), Options(Options) {
  SetDeathCallback();
  InitializeTraceState();
  assert(!F);
  F = this;
  ResetCoverage();
  IsMyThread = true;
  if (Options.DetectLeaks && EF->__sanitizer_install_malloc_and_free_hooks)
    EF->__sanitizer_install_malloc_and_free_hooks(MallocHook, FreeHook);

  if (Options.PrintNewCovPcs) {
    PcBufferLen = 1 << 24;
    PcBuffer = new uintptr_t[PcBufferLen];
    EF->__sanitizer_set_coverage_pc_buffer(PcBuffer, PcBufferLen);
  }
}

Fuzzer::~Fuzzer() { }

void Fuzzer::LazyAllocateCurrentUnitData() {
  if (CurrentUnitData || Options.MaxLen == 0) return;
  CurrentUnitData = new uint8_t[Options.MaxLen];
}

void Fuzzer::SetDeathCallback() {
  CHECK_EXTERNAL_FUNCTION(__sanitizer_set_death_callback);
  EF->__sanitizer_set_death_callback(StaticDeathCallback);
}

void Fuzzer::StaticDeathCallback() {
  assert(F);
  F->DeathCallback();
}

void Fuzzer::DumpCurrentUnit(const char *Prefix) {
  if (!CurrentUnitData) return;  // Happens when running individual inputs.
  MD.PrintMutationSequence();
  Printf("; base unit: %s\n", Sha1ToString(BaseSha1).c_str());
  size_t UnitSize = CurrentUnitSize;
  if (UnitSize <= kMaxUnitSizeToPrint) {
    PrintHexArray(CurrentUnitData, UnitSize, "\n");
    PrintASCII(CurrentUnitData, UnitSize, "\n");
  }
  WriteUnitToFileWithPrefix({CurrentUnitData, CurrentUnitData + UnitSize},
                            Prefix);
}

NO_SANITIZE_MEMORY
void Fuzzer::DeathCallback() {
  DumpCurrentUnit("crash-");
  PrintFinalStats();
}

void Fuzzer::StaticAlarmCallback() {
  assert(F);
  F->AlarmCallback();
}

void Fuzzer::StaticCrashSignalCallback() {
  assert(F);
  F->CrashCallback();
}

void Fuzzer::StaticInterruptCallback() {
  assert(F);
  F->InterruptCallback();
}

void Fuzzer::CrashCallback() {
  Printf("==%d== ERROR: libFuzzer: deadly signal\n", GetPid());
  if (EF->__sanitizer_print_stack_trace)
    EF->__sanitizer_print_stack_trace();
  Printf("NOTE: libFuzzer has rudimentary signal handlers.\n"
         "      Combine libFuzzer with AddressSanitizer or similar for better "
         "crash reports.\n");
  Printf("SUMMARY: libFuzzer: deadly signal\n");
  DumpCurrentUnit("crash-");
  PrintFinalStats();
  exit(Options.ErrorExitCode);
}

void Fuzzer::InterruptCallback() {
  Printf("==%d== libFuzzer: run interrupted; exiting\n", GetPid());
  PrintFinalStats();
  _Exit(0);  // Stop right now, don't perform any at-exit actions.
}

NO_SANITIZE_MEMORY
void Fuzzer::AlarmCallback() {
  assert(Options.UnitTimeoutSec > 0);
  if (!InFuzzingThread()) return;
  if (!CurrentUnitSize)
    return; // We have not started running units yet.
  size_t Seconds =
      duration_cast<seconds>(system_clock::now() - UnitStartTime).count();
  if (Seconds == 0)
    return;
  if (Options.Verbosity >= 2)
    Printf("AlarmCallback %zd\n", Seconds);
  if (Seconds >= (size_t)Options.UnitTimeoutSec) {
    Printf("ALARM: working on the last Unit for %zd seconds\n", Seconds);
    Printf("       and the timeout value is %d (use -timeout=N to change)\n",
           Options.UnitTimeoutSec);
    DumpCurrentUnit("timeout-");
    Printf("==%d== ERROR: libFuzzer: timeout after %d seconds\n", GetPid(),
           Seconds);
    if (EF->__sanitizer_print_stack_trace)
      EF->__sanitizer_print_stack_trace();
    Printf("SUMMARY: libFuzzer: timeout\n");
    PrintFinalStats();
    _Exit(Options.TimeoutExitCode); // Stop right now.
  }
}

void Fuzzer::RssLimitCallback() {
  Printf(
      "==%d== ERROR: libFuzzer: out-of-memory (used: %zdMb; limit: %zdMb)\n",
      GetPid(), GetPeakRSSMb(), Options.RssLimitMb);
  Printf("   To change the out-of-memory limit use -rss_limit_mb=<N>\n\n");
  if (EF->__sanitizer_print_memory_profile)
    EF->__sanitizer_print_memory_profile(50);
  DumpCurrentUnit("oom-");
  Printf("SUMMARY: libFuzzer: out-of-memory\n");
  PrintFinalStats();
  _Exit(Options.ErrorExitCode); // Stop right now.
}

void Fuzzer::PrintStats(const char *Where, const char *End) {
  size_t ExecPerSec = execPerSec();
  if (Options.OutputCSV) {
    static bool csvHeaderPrinted = false;
    if (!csvHeaderPrinted) {
      csvHeaderPrinted = true;
      Printf("runs,block_cov,bits,cc_cov,corpus,execs_per_sec,tbms,reason\n");
    }
    Printf("%zd,%zd,%zd,%zd,%zd,%zd,%s\n", TotalNumberOfRuns,
           MaxCoverage.BlockCoverage, MaxCoverage.CounterBitmapBits,
           MaxCoverage.CallerCalleeCoverage, Corpus.size(), ExecPerSec, Where);
  }

  if (!Options.Verbosity)
    return;
  Printf("#%zd\t%s", TotalNumberOfRuns, Where);
  if (MaxCoverage.BlockCoverage)
    Printf(" cov: %zd", MaxCoverage.BlockCoverage);
  if (MaxCoverage.PCMapBits)
    Printf(" path: %zd", MaxCoverage.PCMapBits);
  if (MaxCoverage.VPMapBits)
    Printf(" vp: %zd", MaxCoverage.VPMapBits);
  if (auto TB = MaxCoverage.CounterBitmapBits)
    Printf(" bits: %zd", TB);
  if (MaxCoverage.CallerCalleeCoverage)
    Printf(" indir: %zd", MaxCoverage.CallerCalleeCoverage);
  Printf(" units: %zd exec/s: %zd", Corpus.size(), ExecPerSec);
  Printf("%s", End);
}

void Fuzzer::PrintFinalStats() {
  if (!Options.PrintFinalStats) return;
  size_t ExecPerSec = execPerSec();
  Printf("stat::number_of_executed_units: %zd\n", TotalNumberOfRuns);
  Printf("stat::average_exec_per_sec:     %zd\n", ExecPerSec);
  Printf("stat::new_units_added:          %zd\n", NumberOfNewUnitsAdded);
  Printf("stat::slowest_unit_time_sec:    %zd\n", TimeOfLongestUnitInSeconds);
  Printf("stat::peak_rss_mb:              %zd\n", GetPeakRSSMb());
}

size_t Fuzzer::MaxUnitSizeInCorpus() const {
  size_t Res = 0;
  for (auto &X : Corpus)
    Res = std::max(Res, X.size());
  return Res;
}

void Fuzzer::SetMaxLen(size_t MaxLen) {
  assert(Options.MaxLen == 0); // Can only reset MaxLen from 0 to non-0.
  assert(MaxLen);
  Options.MaxLen = MaxLen;
  Printf("INFO: -max_len is not provided, using %zd\n", Options.MaxLen);
}


void Fuzzer::RereadOutputCorpus(size_t MaxSize) {
  if (Options.OutputCorpus.empty())
    return;
  std::vector<Unit> AdditionalCorpus;
  ReadDirToVectorOfUnits(Options.OutputCorpus.c_str(), &AdditionalCorpus,
                         &EpochOfLastReadOfOutputCorpus, MaxSize);
  if (Corpus.empty()) {
    Corpus = AdditionalCorpus;
    return;
  }
  if (!Options.Reload)
    return;
  if (Options.Verbosity >= 2)
    Printf("Reload: read %zd new units.\n", AdditionalCorpus.size());
  for (auto &X : AdditionalCorpus) {
    if (X.size() > MaxSize)
      X.resize(MaxSize);
    if (UnitHashesAddedToCorpus.insert(Hash(X)).second) {
      if (RunOne(X)) {
        Corpus.push_back(X);
        UpdateCorpusDistribution();
        PrintStats("RELOAD");
      }
    }
  }
}

void Fuzzer::ShuffleCorpus(UnitVector *V) {
  std::random_shuffle(V->begin(), V->end(), MD.GetRand());
  if (Options.PreferSmall)
    std::stable_sort(V->begin(), V->end(), [](const Unit &A, const Unit &B) {
      return A.size() < B.size();
    });
}

// Tries random prefixes of corpus items.
void Fuzzer::TruncateUnits(std::vector<Unit> *NewCorpus) {
  std::vector<double> Fractions = {0.25, 0.5, 0.75, 1.0};

  size_t TruncInputs = 0;
  for (double Fraction : Fractions) {
    for (const auto &U : Corpus) {
      uint64_t S = MD.GetRand()(U.size() * Fraction);
      if (!S || !RunOne(U.data(), S))
        continue;
      TruncInputs++;
      Unit U1(U.begin(), U.begin() + S);
      NewCorpus->push_back(U1);
    }
  }
  if (TruncInputs)
    Printf("\tINFO   TRUNC %zd units added to in-memory corpus\n", TruncInputs);
}

void Fuzzer::ShuffleAndMinimize() {
  PrintStats("READ  ");
  std::vector<Unit> NewCorpus;
  if (Options.ShuffleAtStartUp)
    ShuffleCorpus(&Corpus);

  if (Options.TruncateUnits) {
    ResetCoverage();
    TruncateUnits(&NewCorpus);
    ResetCoverage();
  }

  for (const auto &U : Corpus) {
    bool NewCoverage = RunOne(U);
    if (!Options.PruneCorpus || NewCoverage) {
      NewCorpus.push_back(U);
      if (Options.Verbosity >= 2)
        Printf("NEW0: %zd L %zd\n", MaxCoverage.BlockCoverage, U.size());
    }
    TryDetectingAMemoryLeak(U.data(), U.size(),
                            /*DuringInitialCorpusExecution*/ true);
  }
  Corpus = NewCorpus;
  UpdateCorpusDistribution();
  for (auto &X : Corpus)
    UnitHashesAddedToCorpus.insert(Hash(X));
  PrintStats("INITED");
  if (Corpus.empty()) {
    Printf("ERROR: no interesting inputs were found. "
           "Is the code instrumented for coverage? Exiting.\n");
    exit(1);
  }
}

bool Fuzzer::UpdateMaxCoverage() {
  PrevPcBufferPos = PcBufferPos;
  bool Res = RecordMaxCoverage(&MaxCoverage);

  return Res;
}

bool Fuzzer::RunOne(const uint8_t *Data, size_t Size) {
  TotalNumberOfRuns++;

  ExecuteCallback(Data, Size);
  bool Res = UpdateMaxCoverage();

  auto UnitStopTime = system_clock::now();
  auto TimeOfUnit =
      duration_cast<seconds>(UnitStopTime - UnitStartTime).count();
  if (!(TotalNumberOfRuns & (TotalNumberOfRuns - 1)) &&
      secondsSinceProcessStartUp() >= 2)
    PrintStats("pulse ");
  if (TimeOfUnit > TimeOfLongestUnitInSeconds &&
      TimeOfUnit >= Options.ReportSlowUnits) {
    TimeOfLongestUnitInSeconds = TimeOfUnit;
    Printf("Slowest unit: %zd s:\n", TimeOfLongestUnitInSeconds);
    WriteUnitToFileWithPrefix({Data, Data + Size}, "slow-unit-");
  }
  return Res;
}

void Fuzzer::RunOneAndUpdateCorpus(const uint8_t *Data, size_t Size) {
  if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
    return;
  if (RunOne(Data, Size))
    ReportNewCoverage({Data, Data + Size});
}

size_t Fuzzer::GetCurrentUnitInFuzzingThead(const uint8_t **Data) const {
  assert(InFuzzingThread());
  *Data = CurrentUnitData;
  return CurrentUnitSize;
}

void Fuzzer::ExecuteCallback(const uint8_t *Data, size_t Size) {
  assert(InFuzzingThread());
  LazyAllocateCurrentUnitData();
  UnitStartTime = system_clock::now();
  // We copy the contents of Unit into a separate heap buffer
  // so that we reliably find buffer overflows in it.
  uint8_t *DataCopy = new uint8_t[Size];
  memcpy(DataCopy, Data, Size);
  if (CurrentUnitData && CurrentUnitData != Data)
    memcpy(CurrentUnitData, Data, Size);
  AssignTaintLabels(DataCopy, Size);
  CurrentUnitSize = Size;
  AllocTracer.Start();
  ResetCounters();  // Reset coverage right before the callback.
  int Res = CB(DataCopy, Size);
  (void)Res;
  HasMoreMallocsThanFrees = AllocTracer.Stop();
  CurrentUnitSize = 0;
  assert(Res == 0);
  delete[] DataCopy;
}

std::string Fuzzer::Coverage::DebugString() const {
  std::string Result =
      std::string("Coverage{") + "BlockCoverage=" +
      std::to_string(BlockCoverage) + " CallerCalleeCoverage=" +
      std::to_string(CallerCalleeCoverage) + " CounterBitmapBits=" +
      std::to_string(CounterBitmapBits) + " PCMapBits=" +
      std::to_string(PCMapBits) + " VPMapBits " +
      std::to_string(VPMapBits) + "}";
  return Result;
}

void Fuzzer::WriteToOutputCorpus(const Unit &U) {
  if (Options.OnlyASCII)
    assert(IsASCII(U));
  if (Options.OutputCorpus.empty())
    return;
  std::string Path = DirPlusFile(Options.OutputCorpus, Hash(U));
  WriteToFile(U, Path);
  if (Options.Verbosity >= 2)
    Printf("Written to %s\n", Path.c_str());
}

void Fuzzer::WriteUnitToFileWithPrefix(const Unit &U, const char *Prefix) {
  if (!Options.SaveArtifacts)
    return;
  std::string Path = Options.ArtifactPrefix + Prefix + Hash(U);
  if (!Options.ExactArtifactPath.empty())
    Path = Options.ExactArtifactPath; // Overrides ArtifactPrefix.
  WriteToFile(U, Path);
  Printf("artifact_prefix='%s'; Test unit written to %s\n",
         Options.ArtifactPrefix.c_str(), Path.c_str());
  if (U.size() <= kMaxUnitSizeToPrint)
    Printf("Base64: %s\n", Base64(U).c_str());
}

void Fuzzer::SaveCorpus() {
  if (Options.OutputCorpus.empty())
    return;
  for (const auto &U : Corpus)
    WriteToFile(U, DirPlusFile(Options.OutputCorpus, Hash(U)));
  if (Options.Verbosity)
    Printf("Written corpus of %zd files to %s\n", Corpus.size(),
           Options.OutputCorpus.c_str());
}

void Fuzzer::PrintStatusForNewUnit(const Unit &U) {
  if (!Options.PrintNEW)
    return;
  PrintStats("NEW   ", "");
  if (Options.Verbosity) {
    Printf(" L: %zd ", U.size());
    MD.PrintMutationSequence();
    Printf("\n");
  }
}

void Fuzzer::PrintNewPCs() {
  if (Options.PrintNewCovPcs && PrevPcBufferPos != PcBufferPos) {
    int NumPrinted = 0;
    for (size_t I = PrevPcBufferPos; I < PcBufferPos; ++I) {
      if (NumPrinted++ > 30) break;  // Don't print too many new PCs.
      if (EF->__sanitizer_symbolize_pc) {
        char PcDescr[1024];
        EF->__sanitizer_symbolize_pc(reinterpret_cast<void*>(PcBuffer[I]),
                                     "%p %F %L", PcDescr, sizeof(PcDescr));
        PcDescr[sizeof(PcDescr) - 1] = 0;  // Just in case.
        Printf("\tNEW_PC: %s\n", PcDescr);
      } else {
        Printf("\tNEW_PC: %p\n", PcBuffer[I]);
      }
    }
  }
}

void Fuzzer::ReportNewCoverage(const Unit &U) {
  Corpus.push_back(U);
  UpdateCorpusDistribution();
  UnitHashesAddedToCorpus.insert(Hash(U));
  MD.RecordSuccessfulMutationSequence();
  PrintStatusForNewUnit(U);
  WriteToOutputCorpus(U);
  NumberOfNewUnitsAdded++;
  PrintNewPCs();
}

// Finds minimal number of units in 'Extra' that add coverage to 'Initial'.
// We do it by actually executing the units, sometimes more than once,
// because we may be using different coverage-like signals and the only
// common thing between them is that we can say "this unit found new stuff".
UnitVector Fuzzer::FindExtraUnits(const UnitVector &Initial,
                                  const UnitVector &Extra) {
  UnitVector Res = Extra;
  size_t OldSize = Res.size();
  for (int Iter = 0; Iter < 10; Iter++) {
    ShuffleCorpus(&Res);
    ResetCoverage();

    for (auto &U : Initial)
      RunOne(U);

    Corpus.clear();
    for (auto &U : Res)
      if (RunOne(U))
        Corpus.push_back(U);

    char Stat[7] = "MIN   ";
    Stat[3] = '0' + Iter;
    PrintStats(Stat);

    size_t NewSize = Corpus.size();
    assert(NewSize <= OldSize);
    Res.swap(Corpus);

    if (NewSize + 5 >= OldSize)
      break;
    OldSize = NewSize;
  }
  return Res;
}

void Fuzzer::Merge(const std::vector<std::string> &Corpora) {
  if (Corpora.size() <= 1) {
    Printf("Merge requires two or more corpus dirs\n");
    return;
  }
  std::vector<std::string> ExtraCorpora(Corpora.begin() + 1, Corpora.end());

  assert(Options.MaxLen > 0);
  UnitVector Initial, Extra;
  ReadDirToVectorOfUnits(Corpora[0].c_str(), &Initial, nullptr, Options.MaxLen);
  for (auto &C : ExtraCorpora)
    ReadDirToVectorOfUnits(C.c_str(), &Extra, nullptr, Options.MaxLen);

  if (!Initial.empty()) {
    Printf("=== Minimizing the initial corpus of %zd units\n", Initial.size());
    Initial = FindExtraUnits({}, Initial);
  }

  Printf("=== Merging extra %zd units\n", Extra.size());
  auto Res = FindExtraUnits(Initial, Extra);

  for (auto &U: Res)
    WriteToOutputCorpus(U);

  Printf("=== Merge: written %zd units\n", Res.size());
}

// Tries detecting a memory leak on the particular input that we have just
// executed before calling this function.
void Fuzzer::TryDetectingAMemoryLeak(const uint8_t *Data, size_t Size,
                                     bool DuringInitialCorpusExecution) {
  if (!HasMoreMallocsThanFrees) return;  // mallocs==frees, a leak is unlikely.
  if (!Options.DetectLeaks) return;
  if (!&(EF->__lsan_enable) || !&(EF->__lsan_disable) ||
      !(EF->__lsan_do_recoverable_leak_check))
    return;  // No lsan.
  // Run the target once again, but with lsan disabled so that if there is
  // a real leak we do not report it twice.
  EF->__lsan_disable();
  RunOne(Data, Size);
  EF->__lsan_enable();
  if (!HasMoreMallocsThanFrees) return;  // a leak is unlikely.
  if (NumberOfLeakDetectionAttempts++ > 1000) {
    Options.DetectLeaks = false;
    Printf("INFO: libFuzzer disabled leak detection after every mutation.\n"
           "      Most likely the target function accumulates allocated\n"
           "      memory in a global state w/o actually leaking it.\n"
           "      If LeakSanitizer is enabled in this process it will still\n"
           "      run on the process shutdown.\n");
    return;
  }
  // Now perform the actual lsan pass. This is expensive and we must ensure
  // we don't call it too often.
  if (EF->__lsan_do_recoverable_leak_check()) { // Leak is found, report it.
    if (DuringInitialCorpusExecution)
      Printf("\nINFO: a leak has been found in the initial corpus.\n\n");
    Printf("INFO: to ignore leaks on libFuzzer side use -detect_leaks=0.\n\n");
    CurrentUnitSize = Size;
    DumpCurrentUnit("leak-");
    PrintFinalStats();
    _Exit(Options.ErrorExitCode);  // not exit() to disable lsan further on.
  }
}

void Fuzzer::MutateAndTestOne() {
  LazyAllocateCurrentUnitData();
  MD.StartMutationSequence();

  auto &U = ChooseUnitToMutate();
  ComputeSHA1(U.data(), U.size(), BaseSha1);  // Remember where we started.
  assert(CurrentUnitData);
  size_t Size = U.size();
  assert(Size <= Options.MaxLen && "Oversized Unit");
  memcpy(CurrentUnitData, U.data(), Size);

  for (int i = 0; i < Options.MutateDepth; i++) {
    size_t NewSize = 0;
    NewSize = MD.Mutate(CurrentUnitData, Size, Options.MaxLen);
    assert(NewSize > 0 && "Mutator returned empty unit");
    assert(NewSize <= Options.MaxLen &&
           "Mutator return overisized unit");
    Size = NewSize;
    if (i == 0)
      StartTraceRecording();
    RunOneAndUpdateCorpus(CurrentUnitData, Size);
    StopTraceRecording();
    TryDetectingAMemoryLeak(CurrentUnitData, Size,
                            /*DuringInitialCorpusExecution*/ false);
  }
}

// Returns an index of random unit from the corpus to mutate.
// Hypothesis: units added to the corpus last are more likely to be interesting.
// This function gives more weight to the more recent units.
size_t Fuzzer::ChooseUnitIdxToMutate() {
  size_t Idx =
      static_cast<size_t>(CorpusDistribution(MD.GetRand().Get_mt19937()));
  assert(Idx < Corpus.size());
  return Idx;
}

void Fuzzer::ResetCoverage() {
  ResetEdgeCoverage();
  MaxCoverage.Reset();
  PrepareCounters(&MaxCoverage);
}

// Experimental search heuristic: drilling.
// - Read, shuffle, execute and minimize the corpus.
// - Choose one random unit.
// - Reset the coverage.
// - Start fuzzing as if the chosen unit was the only element of the corpus.
// - When done, reset the coverage again.
// - Merge the newly created corpus into the original one.
void Fuzzer::Drill() {
  // The corpus is already read, shuffled, and minimized.
  assert(!Corpus.empty());
  Options.PrintNEW = false; // Don't print NEW status lines when drilling.

  Unit U = ChooseUnitToMutate();

  ResetCoverage();

  std::vector<Unit> SavedCorpus;
  SavedCorpus.swap(Corpus);
  Corpus.push_back(U);
  UpdateCorpusDistribution();
  assert(Corpus.size() == 1);
  RunOne(U);
  PrintStats("DRILL ");
  std::string SavedOutputCorpusPath; // Don't write new units while drilling.
  SavedOutputCorpusPath.swap(Options.OutputCorpus);
  Loop();

  ResetCoverage();

  PrintStats("REINIT");
  SavedOutputCorpusPath.swap(Options.OutputCorpus);
  for (auto &U : SavedCorpus)
    RunOne(U);
  PrintStats("MERGE ");
  Options.PrintNEW = true;
  size_t NumMerged = 0;
  for (auto &U : Corpus) {
    if (RunOne(U)) {
      PrintStatusForNewUnit(U);
      NumMerged++;
      WriteToOutputCorpus(U);
    }
  }
  PrintStats("MERGED");
  if (NumMerged && Options.Verbosity)
    Printf("Drilling discovered %zd new units\n", NumMerged);
}

void Fuzzer::Loop() {
  system_clock::time_point LastCorpusReload = system_clock::now();
  if (Options.DoCrossOver)
    MD.SetCorpus(&Corpus);
  while (true) {
    auto Now = system_clock::now();
    if (duration_cast<seconds>(Now - LastCorpusReload).count()) {
      RereadOutputCorpus(Options.MaxLen);
      LastCorpusReload = Now;
    }
    if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
      break;
    if (Options.MaxTotalTimeSec > 0 &&
        secondsSinceProcessStartUp() >
            static_cast<size_t>(Options.MaxTotalTimeSec))
      break;
    // Perform several mutations and runs.
    MutateAndTestOne();
  }

  PrintStats("DONE  ", "\n");
  MD.PrintRecommendedDictionary();
}

void Fuzzer::UpdateCorpusDistribution() {
  size_t N = Corpus.size();
  std::vector<double> Intervals(N + 1);
  std::vector<double> Weights(N);
  std::iota(Intervals.begin(), Intervals.end(), 0);
  std::iota(Weights.begin(), Weights.end(), 1);
  CorpusDistribution = std::piecewise_constant_distribution<double>(
      Intervals.begin(), Intervals.end(), Weights.begin());
}

} // namespace fuzzer

extern "C" {

size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  assert(fuzzer::F);
  return fuzzer::F->GetMD().DefaultMutate(Data, Size, MaxSize);
}
}  // extern "C"
