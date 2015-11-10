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
#include <sanitizer/coverage_interface.h>
#include <algorithm>

extern "C" {
// Re-declare some of the sanitizer functions as "weak" so that
// libFuzzer can be linked w/o the sanitizers and sanitizer-coveragte
// (in which case it will complain at start-up time).
__attribute__((weak)) void __sanitizer_print_stack_trace();
__attribute__((weak)) size_t __sanitizer_get_total_unique_caller_callee_pairs();
__attribute__((weak)) size_t __sanitizer_get_total_unique_coverage();
__attribute__((weak))
void __sanitizer_set_death_callback(void (*callback)(void));
__attribute__((weak)) size_t __sanitizer_get_number_of_counters();
__attribute__((weak))
uintptr_t __sanitizer_update_counter_bitset_and_clear_counters(uint8_t *bitset);
}

namespace fuzzer {
static const size_t kMaxUnitSizeToPrint = 256;

static void MissingWeakApiFunction(const char *FnName) {
  Printf("ERROR: %s is not defined. Exiting.\n"
         "Did you use -fsanitize-coverage=... to build your code?\n", FnName);
  exit(1);
}

#define CHECK_WEAK_API_FUNCTION(fn)                                            \
  do {                                                                         \
    if (!fn)                                                                   \
      MissingWeakApiFunction(#fn);                                             \
  } while (false)

// Only one Fuzzer per process.
static Fuzzer *F;

Fuzzer::Fuzzer(UserSuppliedFuzzer &USF, FuzzingOptions Options)
    : USF(USF), Options(Options) {
  SetDeathCallback();
  InitializeTraceState();
  assert(!F);
  F = this;
}

void Fuzzer::SetDeathCallback() {
  CHECK_WEAK_API_FUNCTION(__sanitizer_set_death_callback);
  __sanitizer_set_death_callback(StaticDeathCallback);
}

void Fuzzer::PrintUnitInASCII(const Unit &U, const char *PrintAfter) {
  PrintASCII(U, PrintAfter);
}

void Fuzzer::StaticDeathCallback() {
  assert(F);
  F->DeathCallback();
}

void Fuzzer::DeathCallback() {
  Printf("DEATH:\n");
  if (CurrentUnit.size() <= kMaxUnitSizeToPrint) {
    Print(CurrentUnit, "\n");
    PrintUnitInASCII(CurrentUnit, "\n");
  }
  WriteUnitToFileWithPrefix(CurrentUnit, "crash-");
}

void Fuzzer::StaticAlarmCallback() {
  assert(F);
  F->AlarmCallback();
}

void Fuzzer::AlarmCallback() {
  assert(Options.UnitTimeoutSec > 0);
  size_t Seconds =
      duration_cast<seconds>(system_clock::now() - UnitStartTime).count();
  if (Seconds == 0) return;
  if (Options.Verbosity >= 2)
    Printf("AlarmCallback %zd\n", Seconds);
  if (Seconds >= (size_t)Options.UnitTimeoutSec) {
    Printf("ALARM: working on the last Unit for %zd seconds\n", Seconds);
    Printf("       and the timeout value is %d (use -timeout=N to change)\n",
           Options.UnitTimeoutSec);
    if (CurrentUnit.size() <= kMaxUnitSizeToPrint) {
      Print(CurrentUnit, "\n");
      PrintUnitInASCII(CurrentUnit, "\n");
    }
    WriteUnitToFileWithPrefix(CurrentUnit, "timeout-");
    Printf("==%d== ERROR: libFuzzer: timeout after %d seconds\n", GetPid(),
           Seconds);
    if (__sanitizer_print_stack_trace)
      __sanitizer_print_stack_trace();
    Printf("SUMMARY: libFuzzer: timeout\n");
    exit(1);
  }
}

void Fuzzer::PrintStats(const char *Where, const char *End) {
  if (!Options.Verbosity) return;
  size_t Seconds = secondsSinceProcessStartUp();
  size_t ExecPerSec = (Seconds ? TotalNumberOfRuns / Seconds : 0);
  Printf("#%zd\t%s", TotalNumberOfRuns, Where);
  if (LastRecordedBlockCoverage)
    Printf(" cov: %zd", LastRecordedBlockCoverage);
  if (auto TB = TotalBits())
    Printf(" bits: %zd", TB);
  if (LastRecordedCallerCalleeCoverage)
    Printf(" indir: %zd", LastRecordedCallerCalleeCoverage);
  Printf(" units: %zd exec/s: %zd", Corpus.size(), ExecPerSec);
  if (TotalNumberOfExecutedTraceBasedMutations)
    Printf(" tbm: %zd", TotalNumberOfExecutedTraceBasedMutations);
  Printf("%s", End);
}

void Fuzzer::RereadOutputCorpus() {
  if (Options.OutputCorpus.empty()) return;
  std::vector<Unit> AdditionalCorpus;
  ReadDirToVectorOfUnits(Options.OutputCorpus.c_str(), &AdditionalCorpus,
                         &EpochOfLastReadOfOutputCorpus);
  if (Corpus.empty()) {
    Corpus = AdditionalCorpus;
    return;
  }
  if (!Options.Reload) return;
  if (Options.Verbosity >= 2)
    Printf("Reload: read %zd new units.\n",  AdditionalCorpus.size());
  for (auto &X : AdditionalCorpus) {
    if (X.size() > (size_t)Options.MaxLen)
      X.resize(Options.MaxLen);
    if (UnitHashesAddedToCorpus.insert(Hash(X)).second) {
      CurrentUnit.clear();
      CurrentUnit.insert(CurrentUnit.begin(), X.begin(), X.end());
      if (RunOne(CurrentUnit)) {
        Corpus.push_back(X);
        if (Options.Verbosity >= 1)
          PrintStats("RELOAD");
      }
    }
  }
}

void Fuzzer::ShuffleAndMinimize() {
  bool PreferSmall = (Options.PreferSmallDuringInitialShuffle == 1 ||
                      (Options.PreferSmallDuringInitialShuffle == -1 &&
                       USF.GetRand().RandBool()));
  if (Options.Verbosity)
    Printf("PreferSmall: %d\n", PreferSmall);
  PrintStats("READ  ");
  std::vector<Unit> NewCorpus;
  if (Options.ShuffleAtStartUp) {
    std::random_shuffle(Corpus.begin(), Corpus.end(), USF.GetRand());
    if (PreferSmall)
      std::stable_sort(
          Corpus.begin(), Corpus.end(),
          [](const Unit &A, const Unit &B) { return A.size() < B.size(); });
  }
  Unit &U = CurrentUnit;
  for (const auto &C : Corpus) {
    for (size_t First = 0; First < 1; First++) {
      U.clear();
      size_t Last = std::min(First + Options.MaxLen, C.size());
      U.insert(U.begin(), C.begin() + First, C.begin() + Last);
      if (Options.OnlyASCII)
        ToASCII(U);
      if (RunOne(U)) {
        NewCorpus.push_back(U);
        if (Options.Verbosity >= 2)
          Printf("NEW0: %zd L %zd\n", LastRecordedBlockCoverage, U.size());
      }
    }
  }
  Corpus = NewCorpus;
  for (auto &X : Corpus)
    UnitHashesAddedToCorpus.insert(Hash(X));
  PrintStats("INITED");
}

bool Fuzzer::RunOne(const Unit &U) {
  UnitStartTime = system_clock::now();
  TotalNumberOfRuns++;

  PrepareCoverageBeforeRun();
  ExecuteCallback(U);
  bool Res = CheckCoverageAfterRun();

  auto UnitStopTime = system_clock::now();
  auto TimeOfUnit =
      duration_cast<seconds>(UnitStopTime - UnitStartTime).count();
  if (!(TotalNumberOfRuns & (TotalNumberOfRuns - 1))
      && secondsSinceProcessStartUp() >= 2
      && Options.Verbosity)
    PrintStats("pulse ");
  if (TimeOfUnit > TimeOfLongestUnitInSeconds &&
      TimeOfUnit >= Options.ReportSlowUnits) {
    TimeOfLongestUnitInSeconds = TimeOfUnit;
    Printf("Slowest unit: %zd s:\n", TimeOfLongestUnitInSeconds);
    WriteUnitToFileWithPrefix(U, "slow-unit-");
  }
  return Res;
}

void Fuzzer::RunOneAndUpdateCorpus(Unit &U) {
  if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
    return;
  if (Options.OnlyASCII)
    ToASCII(U);
  if (RunOne(U))
    ReportNewCoverage(U);
}

void Fuzzer::ExecuteCallback(const Unit &U) {
  int Res = USF.TargetFunction(U.data(), U.size());
  (void)Res;
  assert(Res == 0);
}

size_t Fuzzer::RecordBlockCoverage() {
  CHECK_WEAK_API_FUNCTION(__sanitizer_get_total_unique_coverage);
  return LastRecordedBlockCoverage = __sanitizer_get_total_unique_coverage();
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
  if (Options.UseCounters)
    NumNewBits = __sanitizer_update_counter_bitset_and_clear_counters(
        CounterBitmap.data());
  return NewCoverage > OldCoverage ||
         NewCallerCalleeCoverage > OldCallerCalleeCoverage || NumNewBits;
}

void Fuzzer::WriteToOutputCorpus(const Unit &U) {
  if (Options.OutputCorpus.empty()) return;
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
  WriteToFile(U, Path);
  Printf("artifact_prefix='%s'; Test unit written to %s\n",
         Options.ArtifactPrefix.c_str(), Path.c_str());
  if (U.size() <= kMaxUnitSizeToPrint) {
    Printf("Base64: ");
    PrintFileAsBase64(Path);
  }
}

void Fuzzer::SaveCorpus() {
  if (Options.OutputCorpus.empty()) return;
  for (const auto &U : Corpus)
    WriteToFile(U, DirPlusFile(Options.OutputCorpus, Hash(U)));
  if (Options.Verbosity)
    Printf("Written corpus of %zd files to %s\n", Corpus.size(),
           Options.OutputCorpus.c_str());
}

void Fuzzer::ReportNewCoverage(const Unit &U) {
  Corpus.push_back(U);
  UnitHashesAddedToCorpus.insert(Hash(U));
  PrintStats("NEW   ", "");
  if (Options.Verbosity) {
    Printf(" L: %zd", U.size());
    if (U.size() < 30) {
      Printf(" ");
      PrintUnitInASCII(U, "\t");
      Print(U);
    }
    Printf("\n");
  }
  WriteToOutputCorpus(U);
  if (Options.ExitOnFirst)
    exit(0);
}

void Fuzzer::Merge(const std::vector<std::string> &Corpora) {
  if (Corpora.size() <= 1) {
    Printf("Merge requires two or more corpus dirs\n");
    return;
  }
  auto InitialCorpusDir = Corpora[0];
  ReadDir(InitialCorpusDir, nullptr);
  Printf("Merge: running the initial corpus '%s' of %d units\n",
         InitialCorpusDir.c_str(), Corpus.size());
  for (auto &U : Corpus)
    RunOne(U);

  std::vector<std::string> ExtraCorpora(Corpora.begin() + 1, Corpora.end());

  size_t NumTried = 0;
  size_t NumMerged = 0;
  for (auto &C : ExtraCorpora) {
    Corpus.clear();
    ReadDir(C, nullptr);
    Printf("Merge: merging the extra corpus '%s' of %zd units\n", C.c_str(),
           Corpus.size());
    for (auto &U : Corpus) {
      NumTried++;
      if (RunOne(U)) {
        WriteToOutputCorpus(U);
        NumMerged++;
      }
    }
  }
  Printf("Merge: written %zd out of %zd units\n", NumMerged, NumTried);
}

void Fuzzer::MutateAndTestOne(Unit *U) {
  for (int i = 0; i < Options.MutateDepth; i++) {
    StartTraceRecording();
    size_t Size = U->size();
    U->resize(Options.MaxLen);
    size_t NewSize = USF.Mutate(U->data(), Size, U->size());
    assert(NewSize > 0 && "Mutator returned empty unit");
    assert(NewSize <= (size_t)Options.MaxLen &&
           "Mutator return overisized unit");
    U->resize(NewSize);
    RunOneAndUpdateCorpus(*U);
    size_t NumTraceBasedMutations = StopTraceRecording();
    size_t TBMWidth =
        std::min((size_t)Options.TBMWidth, NumTraceBasedMutations);
    size_t TBMDepth =
        std::min((size_t)Options.TBMDepth, NumTraceBasedMutations);
    Unit BackUp = *U;
    for (size_t w = 0; w < TBMWidth; w++) {
      *U = BackUp;
      for (size_t d = 0; d < TBMDepth; d++) {
        TotalNumberOfExecutedTraceBasedMutations++;
        ApplyTraceBasedMutation(USF.GetRand()(NumTraceBasedMutations), U);
        RunOneAndUpdateCorpus(*U);
      }
    }
  }
}

// Returns an index of random unit from the corpus to mutate.
// Hypothesis: units added to the corpus last are more likely to be interesting.
// This function gives more wieght to the more recent units.
size_t Fuzzer::ChooseUnitToMutate() {
    size_t N = Corpus.size();
    size_t Total = (N + 1) * N / 2;
    size_t R = USF.GetRand()(Total);
    size_t IdxBeg = 0, IdxEnd = N;
    // Binary search.
    while (IdxEnd - IdxBeg >= 2) {
      size_t Idx = IdxBeg + (IdxEnd - IdxBeg) / 2;
      if (R > (Idx + 1) * Idx / 2)
        IdxBeg = Idx;
      else
        IdxEnd = Idx;
    }
    assert(IdxBeg < N);
    return IdxBeg;
}

void Fuzzer::Loop() {
  for (auto &U: Options.Dictionary)
    USF.GetMD().AddWordToDictionary(U.data(), U.size());

  while (true) {
    size_t J1 = ChooseUnitToMutate();;
    SyncCorpus();
    RereadOutputCorpus();
    if (TotalNumberOfRuns >= Options.MaxNumberOfRuns)
      return;
    if (Options.MaxTotalTimeSec > 0 &&
        secondsSinceProcessStartUp() >
        static_cast<size_t>(Options.MaxTotalTimeSec))
      return;
    CurrentUnit = Corpus[J1];
    // Optionally, cross with another unit.
    if (Options.DoCrossOver && USF.GetRand().RandBool()) {
      size_t J2 = ChooseUnitToMutate();
      if (!Corpus[J1].empty() && !Corpus[J2].empty()) {
        assert(!Corpus[J2].empty());
        CurrentUnit.resize(Options.MaxLen);
        size_t NewSize = USF.CrossOver(
            Corpus[J1].data(), Corpus[J1].size(), Corpus[J2].data(),
            Corpus[J2].size(), CurrentUnit.data(), CurrentUnit.size());
        assert(NewSize > 0 && "CrossOver returned empty unit");
        assert(NewSize <= (size_t)Options.MaxLen &&
               "CrossOver returned overisized unit");
        CurrentUnit.resize(NewSize);
      }
    }
    // Perform several mutations and runs.
    MutateAndTestOne(&CurrentUnit);
  }
}

void Fuzzer::SyncCorpus() {
  if (Options.SyncCommand.empty() || Options.OutputCorpus.empty()) return;
  auto Now = system_clock::now();
  if (duration_cast<seconds>(Now - LastExternalSync).count() <
      Options.SyncTimeout)
    return;
  LastExternalSync = Now;
  ExecuteCommand(Options.SyncCommand + " " + Options.OutputCorpus);
}

}  // namespace fuzzer
