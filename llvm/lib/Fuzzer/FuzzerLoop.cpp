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

extern "C" {
// Re-declare some of the sanitizer functions as "weak" so that
// libFuzzer can be linked w/o the sanitizers and sanitizer-coverage
// (in which case it will complain at start-up time).
__attribute__((weak)) void __sanitizer_print_stack_trace();
__attribute__((weak)) void __sanitizer_reset_coverage();
__attribute__((weak)) size_t __sanitizer_get_total_unique_caller_callee_pairs();
__attribute__((weak)) size_t __sanitizer_get_total_unique_coverage();
__attribute__((weak)) void
__sanitizer_set_death_callback(void (*callback)(void));
__attribute__((weak)) size_t __sanitizer_get_number_of_counters();
__attribute__((weak)) uintptr_t
__sanitizer_update_counter_bitset_and_clear_counters(uint8_t *bitset);
__attribute__((weak)) uintptr_t
__sanitizer_get_coverage_pc_buffer(uintptr_t **data);

__attribute__((weak)) size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                                     size_t MaxSize,
                                                     unsigned int Seed);
__attribute__((weak)) void __sanitizer_malloc_hook(void *ptr, size_t size);
__attribute__((weak)) void __sanitizer_free_hook(void *ptr);
__attribute__((weak)) void __lsan_enable();
__attribute__((weak)) void __lsan_disable();
__attribute__((weak)) int __lsan_do_recoverable_leak_check();
}

namespace fuzzer {
static const size_t kMaxUnitSizeToPrint = 256;

static void MissingWeakApiFunction(const char *FnName) {
  Printf("ERROR: %s is not defined. Exiting.\n"
         "Did you use -fsanitize-coverage=... to build your code?\n",
         FnName);
  exit(1);
}

#define CHECK_WEAK_API_FUNCTION(fn)                                            \
  do {                                                                         \
    if (!fn)                                                                   \
      MissingWeakApiFunction(#fn);                                             \
  } while (false)

// Only one Fuzzer per process.
static Fuzzer *F;

size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  assert(F);
  return F->GetMD().Mutate(Data, Size, MaxSize);
}

Fuzzer::Fuzzer(UserCallback CB, MutationDispatcher &MD, FuzzingOptions Options)
    : CB(CB), MD(MD), Options(Options) {
  SetDeathCallback();
  InitializeTraceState();
  assert(!F);
  F = this;
}

void Fuzzer::SetDeathCallback() {
  CHECK_WEAK_API_FUNCTION(__sanitizer_set_death_callback);
  __sanitizer_set_death_callback(StaticDeathCallback);
}

void Fuzzer::StaticDeathCallback() {
  assert(F);
  F->DeathCallback();
}

void Fuzzer::DumpCurrentUnit(const char *Prefix) {
  if (CurrentUnitSize <= kMaxUnitSizeToPrint) {
    PrintHexArray(CurrentUnitData, CurrentUnitSize, "\n");
    PrintASCII(CurrentUnitData, CurrentUnitSize, "\n");
  }
  WriteUnitToFileWithPrefix(
      {CurrentUnitData, CurrentUnitData + CurrentUnitSize}, Prefix);
}

NO_SANITIZE_MEMORY
void Fuzzer::DeathCallback() {
  if (!CurrentUnitSize) return;
  Printf("DEATH:\n");
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
  if (__sanitizer_print_stack_trace)
    __sanitizer_print_stack_trace();
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
    if (__sanitizer_print_stack_trace)
      __sanitizer_print_stack_trace();
    Printf("SUMMARY: libFuzzer: timeout\n");
    PrintFinalStats();
    _Exit(Options.TimeoutExitCode); // Stop right now.
  }
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
           LastRecordedBlockCoverage, TotalBits(),
           LastRecordedCallerCalleeCoverage, Corpus.size(), ExecPerSec,
           Where);
  }

  if (!Options.Verbosity)
    return;
  Printf("#%zd\t%s", TotalNumberOfRuns, Where);
  if (LastRecordedBlockCoverage)
    Printf(" cov: %zd", LastRecordedBlockCoverage);
  if (LastRecordedPcMapSize)
    Printf(" path: %zd", LastRecordedPcMapSize);
  if (auto TB = TotalBits())
    Printf(" bits: %zd", TB);
  if (LastRecordedCallerCalleeCoverage)
    Printf(" indir: %zd", LastRecordedCallerCalleeCoverage);
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

void Fuzzer::ShuffleAndMinimize() {
  PrintStats("READ  ");
  std::vector<Unit> NewCorpus;
  if (Options.ShuffleAtStartUp)
    ShuffleCorpus(&Corpus);

  for (const auto &U : Corpus) {
    if (RunOne(U)) {
      NewCorpus.push_back(U);
      if (Options.Verbosity >= 2)
        Printf("NEW0: %zd L %zd\n", LastRecordedBlockCoverage, U.size());
    }
  }
  Corpus = NewCorpus;
  UpdateCorpusDistribution();
  for (auto &X : Corpus)
    UnitHashesAddedToCorpus.insert(Hash(X));
  PrintStats("INITED");
  CheckForMemoryLeaks();
}

bool Fuzzer::RunOne(const uint8_t *Data, size_t Size) {
  TotalNumberOfRuns++;

  PrepareCoverageBeforeRun();
  ExecuteCallback(Data, Size);
  bool Res = CheckCoverageAfterRun();

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

void Fuzzer::RunOneAndUpdateCorpus(uint8_t *Data, size_t Size) {
  if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
    return;
  if (Options.OnlyASCII)
    ToASCII(Data, Size);
  if (RunOne(Data, Size))
    ReportNewCoverage({Data, Data + Size});
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
  size_t Mallocs;
  size_t Frees;
};

static thread_local MallocFreeTracer AllocTracer;

extern "C" {
void __sanitizer_malloc_hook(void *ptr, size_t size) { AllocTracer.Mallocs++; }
void __sanitizer_free_hook(void *ptr) { AllocTracer.Frees++; }
}  // extern "C"

void Fuzzer::ExecuteCallback(const uint8_t *Data, size_t Size) {
  UnitStartTime = system_clock::now();
  // We copy the contents of Unit into a separate heap buffer
  // so that we reliably find buffer overflows in it.
  std::unique_ptr<uint8_t[]> DataCopy(new uint8_t[Size]);
  memcpy(DataCopy.get(), Data, Size);
  AssignTaintLabels(DataCopy.get(), Size);
  CurrentUnitData = DataCopy.get();
  CurrentUnitSize = Size;
  AllocTracer.Start();
  int Res = CB(DataCopy.get(), Size);
  (void)Res;
  HasMoreMallocsThanFrees = AllocTracer.Stop();
  CurrentUnitSize = 0;
  CurrentUnitData = nullptr;
  assert(Res == 0);
}

size_t Fuzzer::RecordBlockCoverage() {
  CHECK_WEAK_API_FUNCTION(__sanitizer_get_total_unique_coverage);
  uintptr_t PrevCoverage = LastRecordedBlockCoverage;
  LastRecordedBlockCoverage = __sanitizer_get_total_unique_coverage();

  if (PrevCoverage == LastRecordedBlockCoverage || !Options.PrintNewCovPcs)
    return LastRecordedBlockCoverage;

  uintptr_t PrevBufferLen = LastCoveragePcBufferLen;
  uintptr_t *CoverageBuf;
  LastCoveragePcBufferLen = __sanitizer_get_coverage_pc_buffer(&CoverageBuf);
  assert(CoverageBuf);
  for (size_t i = PrevBufferLen; i < LastCoveragePcBufferLen; ++i) {
    Printf("%p\n", CoverageBuf[i]);
  }

  return LastRecordedBlockCoverage;
}

size_t Fuzzer::RecordCallerCalleeCoverage() {
  if (!Options.UseIndirCalls)
    return 0;
  if (!__sanitizer_get_total_unique_caller_callee_pairs)
    return 0;
  return LastRecordedCallerCalleeCoverage =
             __sanitizer_get_total_unique_caller_callee_pairs();
}

void Fuzzer::PrepareCoverageBeforeRun() {
  if (Options.UseCounters) {
    size_t NumCounters = __sanitizer_get_number_of_counters();
    CounterBitmap.resize(NumCounters);
    __sanitizer_update_counter_bitset_and_clear_counters(0);
  }
  RecordBlockCoverage();
  RecordCallerCalleeCoverage();
}

bool Fuzzer::CheckCoverageAfterRun() {
  size_t OldCoverage = LastRecordedBlockCoverage;
  size_t NewCoverage = RecordBlockCoverage();
  size_t OldCallerCalleeCoverage = LastRecordedCallerCalleeCoverage;
  size_t NewCallerCalleeCoverage = RecordCallerCalleeCoverage();
  size_t NumNewBits = 0;
  size_t OldPcMapSize = LastRecordedPcMapSize;
  PcMapMergeCurrentToCombined();
  size_t NewPcMapSize = PcMapCombinedSize();
  LastRecordedPcMapSize = NewPcMapSize;
  if (NewPcMapSize > OldPcMapSize)
    return true;
  if (Options.UseCounters)
    NumNewBits = __sanitizer_update_counter_bitset_and_clear_counters(
        CounterBitmap.data());
  return NewCoverage > OldCoverage ||
         NewCallerCalleeCoverage > OldCallerCalleeCoverage || NumNewBits;
}

void Fuzzer::WriteToOutputCorpus(const Unit &U) {
  if (Options.OutputCorpus.empty())
    return;
  std::string Path = DirPlusFile(Options.OutputCorpus, Hash(U));
  WriteToFile(U, Path);
  if (Options.Verbosity >= 2)
    Printf("Written to %s\n", Path.c_str());
  assert(!Options.OnlyASCII || IsASCII(U));
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

void Fuzzer::ReportNewCoverage(const Unit &U) {
  Corpus.push_back(U);
  UpdateCorpusDistribution();
  UnitHashesAddedToCorpus.insert(Hash(U));
  MD.RecordSuccessfulMutationSequence();
  PrintStatusForNewUnit(U);
  WriteToOutputCorpus(U);
  NumberOfNewUnitsAdded++;
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
    Res.swap(Corpus);

    if (NewSize == OldSize)
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

// Tries to call lsan, and if there are leaks exits. We call this right after
// the initial corpus was read because if there are leaky inputs in the corpus
// further fuzzing will likely hit OOMs.
void Fuzzer::CheckForMemoryLeaks() {
  if (!Options.DetectLeaks) return;
  if (!__lsan_do_recoverable_leak_check)
    return;
  if (__lsan_do_recoverable_leak_check()) {
    Printf("==%d== ERROR: libFuzzer: initial corpus triggers memory leaks.\n"
           "Exiting now. Use -detect_leaks=0 to disable leak detection here.\n"
           "LeakSanitizer will still check for leaks at the process exit.\n",
           GetPid());
    PrintFinalStats();
    _Exit(Options.ErrorExitCode);
  }
}

// Tries detecting a memory leak on the particular input that we have just
// executed before calling this function.
void Fuzzer::TryDetectingAMemoryLeak(uint8_t *Data, size_t Size) {
  if (!HasMoreMallocsThanFrees) return;  // mallocs==frees, a leak is unlikely.
  if (!Options.DetectLeaks) return;
  if (!&__lsan_enable || !&__lsan_disable || !__lsan_do_recoverable_leak_check)
    return;  // No lsan.
  // Run the target once again, but with lsan disabled so that if there is
  // a real leak we do not report it twice.
  __lsan_disable();
  RunOneAndUpdateCorpus(Data, Size);
  __lsan_enable();
  if (!HasMoreMallocsThanFrees) return;  // a leak is unlikely.
  // Now perform the actual lsan pass. This is expensive and we must ensure
  // we don't call it too often.
  if (__lsan_do_recoverable_leak_check()) {  // Leak is found, report it.
    CurrentUnitData = Data;
    CurrentUnitSize = Size;
    DumpCurrentUnit("leak-");
    PrintFinalStats();
    _Exit(Options.ErrorExitCode);  // not exit() to disable lsan further on.
  }
}

void Fuzzer::MutateAndTestOne() {
  MD.StartMutationSequence();

  auto &U = ChooseUnitToMutate();
  MutateInPlaceHere.resize(Options.MaxLen);
  size_t Size = U.size();
  assert(Size <= Options.MaxLen && "Oversized Unit");
  memcpy(MutateInPlaceHere.data(), U.data(), Size);

  for (int i = 0; i < Options.MutateDepth; i++) {
    size_t NewSize = 0;
    if (LLVMFuzzerCustomMutator)
      NewSize = LLVMFuzzerCustomMutator(MutateInPlaceHere.data(), Size,
                                        Options.MaxLen, MD.GetRand().Rand());
    else
      NewSize = MD.Mutate(MutateInPlaceHere.data(), Size, Options.MaxLen);
    assert(NewSize > 0 && "Mutator returned empty unit");
    assert(NewSize <= Options.MaxLen &&
           "Mutator return overisized unit");
    Size = NewSize;
    if (i == 0)
      StartTraceRecording();
    RunOneAndUpdateCorpus(MutateInPlaceHere.data(), Size);
    StopTraceRecording();
    TryDetectingAMemoryLeak(MutateInPlaceHere.data(), Size);
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
  CHECK_WEAK_API_FUNCTION(__sanitizer_reset_coverage);
  __sanitizer_reset_coverage();
  CounterBitmap.clear();
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
