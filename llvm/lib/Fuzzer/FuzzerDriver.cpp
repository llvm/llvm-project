//===- FuzzerDriver.cpp - FuzzerDriver function and flags -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// FuzzerDriver and flag parsing.
//===----------------------------------------------------------------------===//

#include "FuzzerInterface.h"
#include "FuzzerInternal.h"

#include <cstring>
#include <chrono>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

namespace fuzzer {

// Program arguments.
struct FlagDescription {
  const char *Name;
  const char *Description;
  int   Default;
  int   *IntFlag;
  const char **StrFlag;
};

struct {
#define FUZZER_FLAG_INT(Name, Default, Description) int Name;
#define FUZZER_FLAG_STRING(Name, Description) const char *Name;
#include "FuzzerFlags.def"
#undef FUZZER_FLAG_INT
#undef FUZZER_FLAG_STRING
} Flags;

static const FlagDescription FlagDescriptions [] {
#define FUZZER_FLAG_INT(Name, Default, Description)                            \
  { #Name, Description, Default, &Flags.Name, nullptr},
#define FUZZER_FLAG_STRING(Name, Description)                                  \
  { #Name, Description, 0, nullptr, &Flags.Name },
#include "FuzzerFlags.def"
#undef FUZZER_FLAG_INT
#undef FUZZER_FLAG_STRING
};

static const size_t kNumFlags =
    sizeof(FlagDescriptions) / sizeof(FlagDescriptions[0]);

static std::vector<std::string> *Inputs;
static std::string *ProgName;

static void PrintHelp() {
  Printf("Usage: %s [-flag1=val1 [-flag2=val2 ...] ] [dir1 [dir2 ...] ]\n",
         ProgName->c_str());
  Printf("\nFlags: (strictly in form -flag=value)\n");
  size_t MaxFlagLen = 0;
  for (size_t F = 0; F < kNumFlags; F++)
    MaxFlagLen = std::max(strlen(FlagDescriptions[F].Name), MaxFlagLen);

  for (size_t F = 0; F < kNumFlags; F++) {
    const auto &D = FlagDescriptions[F];
    Printf(" %s", D.Name);
    for (size_t i = 0, n = MaxFlagLen - strlen(D.Name); i < n; i++)
      Printf(" ");
    Printf("\t");
    Printf("%d\t%s\n", D.Default, D.Description);
  }
  Printf("\nFlags starting with '--' will be ignored and "
            "will be passed verbatim to subprocesses.\n");
}

static const char *FlagValue(const char *Param, const char *Name) {
  size_t Len = strlen(Name);
  if (Param[0] == '-' && strstr(Param + 1, Name) == Param + 1 &&
      Param[Len + 1] == '=')
      return &Param[Len + 2];
  return nullptr;
}

static bool ParseOneFlag(const char *Param) {
  if (Param[0] != '-') return false;
  if (Param[1] == '-') {
    static bool PrintedWarning = false;
    if (!PrintedWarning) {
      PrintedWarning = true;
      Printf("WARNING: libFuzzer ignores flags that start with '--'\n");
    }
    return true;
  }
  for (size_t F = 0; F < kNumFlags; F++) {
    const char *Name = FlagDescriptions[F].Name;
    const char *Str = FlagValue(Param, Name);
    if (Str)  {
      if (FlagDescriptions[F].IntFlag) {
        int Val = std::stol(Str);
        *FlagDescriptions[F].IntFlag = Val;
        if (Flags.verbosity >= 2)
          Printf("Flag: %s %d\n", Name, Val);;
        return true;
      } else if (FlagDescriptions[F].StrFlag) {
        *FlagDescriptions[F].StrFlag = Str;
        if (Flags.verbosity >= 2)
          Printf("Flag: %s %s\n", Name, Str);
        return true;
      }
    }
  }
  PrintHelp();
  exit(1);
}

// We don't use any library to minimize dependencies.
static void ParseFlags(const std::vector<std::string> &Args) {
  for (size_t F = 0; F < kNumFlags; F++) {
    if (FlagDescriptions[F].IntFlag)
      *FlagDescriptions[F].IntFlag = FlagDescriptions[F].Default;
    if (FlagDescriptions[F].StrFlag)
      *FlagDescriptions[F].StrFlag = nullptr;
  }
  Inputs = new std::vector<std::string>;
  for (size_t A = 1; A < Args.size(); A++) {
    if (ParseOneFlag(Args[A].c_str())) continue;
    Inputs->push_back(Args[A]);
  }
}

static std::mutex Mu;

static void PulseThread() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(600));
    std::lock_guard<std::mutex> Lock(Mu);
    Printf("pulse...\n");
  }
}

static void WorkerThread(const std::string &Cmd, std::atomic<int> *Counter,
                        int NumJobs, std::atomic<bool> *HasErrors) {
  while (true) {
    int C = (*Counter)++;
    if (C >= NumJobs) break;
    std::string Log = "fuzz-" + std::to_string(C) + ".log";
    std::string ToRun = Cmd + " > " + Log + " 2>&1\n";
    if (Flags.verbosity)
      Printf("%s", ToRun.c_str());
    int ExitCode = ExecuteCommand(ToRun.c_str());
    if (ExitCode != 0)
      *HasErrors = true;
    std::lock_guard<std::mutex> Lock(Mu);
    Printf("================== Job %d exited with exit code %d ============\n",
           C, ExitCode);
    fuzzer::CopyFileToErr(Log);
  }
}

static int RunInMultipleProcesses(const std::vector<std::string> &Args,
                                  int NumWorkers, int NumJobs) {
  std::atomic<int> Counter(0);
  std::atomic<bool> HasErrors(false);
  std::string Cmd;
  for (auto &S : Args) {
    if (FlagValue(S.c_str(), "jobs") || FlagValue(S.c_str(), "workers"))
      continue;
    Cmd += S + " ";
  }
  std::vector<std::thread> V;
  std::thread Pulse(PulseThread);
  Pulse.detach();
  for (int i = 0; i < NumWorkers; i++)
    V.push_back(std::thread(WorkerThread, Cmd, &Counter, NumJobs, &HasErrors));
  for (auto &T : V)
    T.join();
  return HasErrors ? 1 : 0;
}

int RunOneTest(Fuzzer *F, const char *InputFilePath) {
  Unit U = FileToVector(InputFilePath);
  Unit PreciseSizedU(U);
  assert(PreciseSizedU.size() == PreciseSizedU.capacity());
  F->ExecuteCallback(PreciseSizedU);
  return 0;
}

int FuzzerDriver(int argc, char **argv, UserCallback Callback) {
  FuzzerRandomLibc Rand(0);
  SimpleUserSuppliedFuzzer SUSF(&Rand, Callback);
  return FuzzerDriver(argc, argv, SUSF);
}

int FuzzerDriver(int argc, char **argv, UserSuppliedFuzzer &USF) {
  std::vector<std::string> Args(argv, argv + argc);
  return FuzzerDriver(Args, USF);
}

int FuzzerDriver(const std::vector<std::string> &Args, UserCallback Callback) {
  FuzzerRandomLibc Rand(0);
  SimpleUserSuppliedFuzzer SUSF(&Rand, Callback);
  return FuzzerDriver(Args, SUSF);
}

int FuzzerDriver(const std::vector<std::string> &Args,
                 UserSuppliedFuzzer &USF) {
  using namespace fuzzer;
  assert(!Args.empty());
  ProgName = new std::string(Args[0]);
  ParseFlags(Args);
  if (Flags.help) {
    PrintHelp();
    return 0;
  }

  if (Flags.jobs > 0 && Flags.workers == 0) {
    Flags.workers = std::min(NumberOfCpuCores() / 2, Flags.jobs);
    if (Flags.workers > 1)
      Printf("Running %d workers\n", Flags.workers);
  }

  if (Flags.workers > 0 && Flags.jobs > 0)
    return RunInMultipleProcesses(Args, Flags.workers, Flags.jobs);

  Fuzzer::FuzzingOptions Options;
  Options.Verbosity = Flags.verbosity;
  Options.MaxLen = Flags.max_len;
  Options.UnitTimeoutSec = Flags.timeout;
  Options.MaxTotalTimeSec = Flags.max_total_time;
  Options.DoCrossOver = Flags.cross_over;
  Options.MutateDepth = Flags.mutate_depth;
  Options.ExitOnFirst = Flags.exit_on_first;
  Options.UseCounters = Flags.use_counters;
  Options.UseIndirCalls = Flags.use_indir_calls;
  Options.UseTraces = Flags.use_traces;
  Options.ShuffleAtStartUp = Flags.shuffle;
  Options.PreferSmallDuringInitialShuffle =
      Flags.prefer_small_during_initial_shuffle;
  Options.Reload = Flags.reload;
  Options.OnlyASCII = Flags.only_ascii;
  Options.TBMDepth = Flags.tbm_depth;
  Options.TBMWidth = Flags.tbm_width;
  Options.OutputCSV = Flags.output_csv;
  if (Flags.runs >= 0)
    Options.MaxNumberOfRuns = Flags.runs;
  if (!Inputs->empty())
    Options.OutputCorpus = (*Inputs)[0];
  if (Flags.sync_command)
    Options.SyncCommand = Flags.sync_command;
  Options.SyncTimeout = Flags.sync_timeout;
  Options.ReportSlowUnits = Flags.report_slow_units;
  if (Flags.artifact_prefix)
    Options.ArtifactPrefix = Flags.artifact_prefix;
  if (Flags.exact_artifact_path)
    Options.ExactArtifactPath = Flags.exact_artifact_path;
  std::vector<Unit> Dictionary;
  if (Flags.dict)
    if (!ParseDictionaryFile(FileToString(Flags.dict), &Dictionary))
      return 1;
  if (Flags.verbosity > 0 && !Dictionary.empty())
    Printf("Dictionary: %zd entries\n", Dictionary.size());
  Options.SaveArtifacts = !Flags.test_single_input;

  Fuzzer F(USF, Options);

  for (auto &U: Dictionary)
    USF.GetMD().AddWordToDictionary(U.data(), U.size());

  // Timer
  if (Flags.timeout > 0)
    SetTimer(Flags.timeout / 2 + 1);

  if (Flags.test_single_input) {
    RunOneTest(&F, Flags.test_single_input);
    exit(0);
  }

  if (Flags.merge) {
    F.Merge(*Inputs);
    exit(0);
  }

  unsigned Seed = Flags.seed;
  // Initialize Seed.
  if (Seed == 0)
    Seed = time(0) * 10000 + getpid();
  if (Flags.verbosity)
    Printf("Seed: %u\n", Seed);
  USF.GetRand().ResetSeed(Seed);

  F.RereadOutputCorpus();
  for (auto &inp : *Inputs)
    if (inp != Options.OutputCorpus)
      F.ReadDir(inp, nullptr);

  if (F.CorpusSize() == 0)
    F.AddToCorpus(Unit());  // Can't fuzz empty corpus, so add an empty input.
  F.ShuffleAndMinimize();
  if (Flags.save_minimized_corpus)
    F.SaveCorpus();
  else if (Flags.drill)
    F.Drill();
  else
    F.Loop();

  if (Flags.verbosity)
    Printf("Done %d runs in %zd second(s)\n", F.getTotalNumberOfRuns(),
           F.secondsSinceProcessStartUp());

  exit(0);  // Don't let F destroy itself.
}

}  // namespace fuzzer
