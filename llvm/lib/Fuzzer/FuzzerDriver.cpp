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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

// This function should be present in the libFuzzer so that the client
// binary can test for its existence.
extern "C" __attribute__((used)) void __libfuzzer_is_present() {}

namespace fuzzer {

// Program arguments.
struct FlagDescription {
  const char *Name;
  const char *Description;
  int   Default;
  int   *IntFlag;
  const char **StrFlag;
  unsigned int *UIntFlag;
};

struct {
#define FUZZER_DEPRECATED_FLAG(Name)
#define FUZZER_FLAG_INT(Name, Default, Description) int Name;
#define FUZZER_FLAG_UNSIGNED(Name, Default, Description) unsigned int Name;
#define FUZZER_FLAG_STRING(Name, Description) const char *Name;
#include "FuzzerFlags.def"
#undef FUZZER_DEPRECATED_FLAG
#undef FUZZER_FLAG_INT
#undef FUZZER_FLAG_UNSIGNED
#undef FUZZER_FLAG_STRING
} Flags;

static const FlagDescription FlagDescriptions [] {
#define FUZZER_DEPRECATED_FLAG(Name)                                           \
  {#Name, "Deprecated; don't use", 0, nullptr, nullptr, nullptr},
#define FUZZER_FLAG_INT(Name, Default, Description)                            \
  {#Name, Description, Default, &Flags.Name, nullptr, nullptr},
#define FUZZER_FLAG_UNSIGNED(Name, Default, Description)                       \
  {#Name,   Description, static_cast<int>(Default),                            \
   nullptr, nullptr, &Flags.Name},
#define FUZZER_FLAG_STRING(Name, Description)                                  \
  {#Name, Description, 0, nullptr, &Flags.Name, nullptr},
#include "FuzzerFlags.def"
#undef FUZZER_DEPRECATED_FLAG
#undef FUZZER_FLAG_INT
#undef FUZZER_FLAG_UNSIGNED
#undef FUZZER_FLAG_STRING
};

static const size_t kNumFlags =
    sizeof(FlagDescriptions) / sizeof(FlagDescriptions[0]);

static std::vector<std::string> *Inputs;
static std::string *ProgName;

static void PrintHelp() {
  Printf("Usage:\n");
  auto Prog = ProgName->c_str();
  Printf("\nTo run fuzzing pass 0 or more directories.\n");
  Printf("%s [-flag1=val1 [-flag2=val2 ...] ] [dir1 [dir2 ...] ]\n", Prog);

  Printf("\nTo run individual tests without fuzzing pass 1 or more files:\n");
  Printf("%s [-flag1=val1 [-flag2=val2 ...] ] file1 [file2 ...]\n", Prog);

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

// Avoid calling stol as it triggers a bug in clang/glibc build.
static long MyStol(const char *Str) {
  long Res = 0;
  long Sign = 1;
  if (*Str == '-') {
    Str++;
    Sign = -1;
  }
  for (size_t i = 0; Str[i]; i++) {
    char Ch = Str[i];
    if (Ch < '0' || Ch > '9')
      return Res;
    Res = Res * 10 + (Ch - '0');
  }
  return Res * Sign;
}

static bool ParseOneFlag(const char *Param) {
  if (Param[0] != '-') return false;
  if (Param[1] == '-') {
    static bool PrintedWarning = false;
    if (!PrintedWarning) {
      PrintedWarning = true;
      Printf("INFO: libFuzzer ignores flags that start with '--'\n");
    }
    for (size_t F = 0; F < kNumFlags; F++)
      if (FlagValue(Param + 1, FlagDescriptions[F].Name))
        Printf("WARNING: did you mean '%s' (single dash)?\n", Param + 1);
    return true;
  }
  for (size_t F = 0; F < kNumFlags; F++) {
    const char *Name = FlagDescriptions[F].Name;
    const char *Str = FlagValue(Param, Name);
    if (Str)  {
      if (FlagDescriptions[F].IntFlag) {
        int Val = MyStol(Str);
        *FlagDescriptions[F].IntFlag = Val;
        if (Flags.verbosity >= 2)
          Printf("Flag: %s %d\n", Name, Val);;
        return true;
      } else if (FlagDescriptions[F].UIntFlag) {
        unsigned int Val = std::stoul(Str);
        *FlagDescriptions[F].UIntFlag = Val;
        if (Flags.verbosity >= 2)
          Printf("Flag: %s %u\n", Name, Val);
        return true;
      } else if (FlagDescriptions[F].StrFlag) {
        *FlagDescriptions[F].StrFlag = Str;
        if (Flags.verbosity >= 2)
          Printf("Flag: %s %s\n", Name, Str);
        return true;
      } else {  // Deprecated flag.
        Printf("Flag: %s: deprecated, don't use\n", Name);
        return true;
      }
    }
  }
  Printf("\n\nWARNING: unrecognized flag '%s'; "
         "use -help=1 to list all flags\n\n", Param);
  return true;
}

// We don't use any library to minimize dependencies.
static void ParseFlags(const std::vector<std::string> &Args) {
  for (size_t F = 0; F < kNumFlags; F++) {
    if (FlagDescriptions[F].IntFlag)
      *FlagDescriptions[F].IntFlag = FlagDescriptions[F].Default;
    if (FlagDescriptions[F].UIntFlag)
      *FlagDescriptions[F].UIntFlag =
          static_cast<unsigned int>(FlagDescriptions[F].Default);
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
    SleepSeconds(600);
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
    int ExitCode = ExecuteCommand(ToRun);
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

static void RssThread(Fuzzer *F, size_t RssLimitMb) {
  while (true) {
    SleepSeconds(1);
    size_t Peak = GetPeakRSSMb();
    if (Peak > RssLimitMb)
      F->RssLimitCallback();
  }
}

static void StartRssThread(Fuzzer *F, size_t RssLimitMb) {
  if (!RssLimitMb) return;
  std::thread T(RssThread, F, RssLimitMb);
  T.detach();
}

int RunOneTest(Fuzzer *F, const char *InputFilePath) {
  Unit U = FileToVector(InputFilePath);
  Unit PreciseSizedU(U);
  assert(PreciseSizedU.size() == PreciseSizedU.capacity());
  F->RunOne(PreciseSizedU.data(), PreciseSizedU.size());
  return 0;
}

static bool AllInputsAreFiles() {
  if (Inputs->empty()) return false;
  for (auto &Path : *Inputs)
    if (!IsFile(Path))
      return false;
  return true;
}

int FuzzerDriver(int *argc, char ***argv, UserCallback Callback) {
  using namespace fuzzer;
  assert(argc && argv && "Argument pointers cannot be nullptr");
  EF = new ExternalFunctions();
  if (EF->LLVMFuzzerInitialize)
    EF->LLVMFuzzerInitialize(argc, argv);
  const std::vector<std::string> Args(*argv, *argv + *argc);
  assert(!Args.empty());
  ProgName = new std::string(Args[0]);
  ParseFlags(Args);
  if (Flags.help) {
    PrintHelp();
    return 0;
  }

  if (Flags.close_fd_mask & 2)
    DupAndCloseStderr();
  if (Flags.close_fd_mask & 1)
    CloseStdout();

  if (Flags.jobs > 0 && Flags.workers == 0) {
    Flags.workers = std::min(NumberOfCpuCores() / 2, Flags.jobs);
    if (Flags.workers > 1)
      Printf("Running %d workers\n", Flags.workers);
  }

  if (Flags.workers > 0 && Flags.jobs > 0)
    return RunInMultipleProcesses(Args, Flags.workers, Flags.jobs);

  const size_t kMaxSaneLen = 1 << 20;
  const size_t kMinDefaultLen = 64;
  FuzzingOptions Options;
  Options.Verbosity = Flags.verbosity;
  Options.MaxLen = Flags.max_len;
  Options.UnitTimeoutSec = Flags.timeout;
  Options.TimeoutExitCode = Flags.timeout_exitcode;
  Options.MaxTotalTimeSec = Flags.max_total_time;
  Options.DoCrossOver = Flags.cross_over;
  Options.MutateDepth = Flags.mutate_depth;
  Options.UseCounters = Flags.use_counters;
  Options.UseIndirCalls = Flags.use_indir_calls;
  Options.UseTraces = Flags.use_traces;
  Options.UseMemcmp = Flags.use_memcmp;
  Options.ShuffleAtStartUp = Flags.shuffle;
  Options.PreferSmall = Flags.prefer_small;
  Options.Reload = Flags.reload;
  Options.OnlyASCII = Flags.only_ascii;
  Options.OutputCSV = Flags.output_csv;
  Options.DetectLeaks = Flags.detect_leaks;
  Options.RssLimitMb = Flags.rss_limit_mb;
  if (Flags.runs >= 0)
    Options.MaxNumberOfRuns = Flags.runs;
  if (!Inputs->empty())
    Options.OutputCorpus = (*Inputs)[0];
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
  bool DoPlainRun = AllInputsAreFiles();
  Options.SaveArtifacts = !DoPlainRun;
  Options.PrintNewCovPcs = Flags.print_new_cov_pcs;
  Options.PrintFinalStats = Flags.print_final_stats;
  Options.TruncateUnits = Flags.truncate_units;
  Options.PruneCorpus = Flags.prune_corpus;

  unsigned Seed = Flags.seed;
  // Initialize Seed.
  if (Seed == 0)
    Seed = (std::chrono::system_clock::now().time_since_epoch().count() << 10) +
           getpid();
  if (Flags.verbosity)
    Printf("INFO: Seed: %u\n", Seed);

  Random Rand(Seed);
  MutationDispatcher MD(Rand, Options);
  Fuzzer F(Callback, MD, Options);

  for (auto &U: Dictionary)
    if (U.size() <= Word::GetMaxSize())
      MD.AddWordToManualDictionary(Word(U.data(), U.size()));

  StartRssThread(&F, Flags.rss_limit_mb);

  // Timer
  if (Flags.timeout > 0)
    SetTimer(Flags.timeout / 2 + 1);
  if (Flags.handle_segv) SetSigSegvHandler();
  if (Flags.handle_bus) SetSigBusHandler();
  if (Flags.handle_abrt) SetSigAbrtHandler();
  if (Flags.handle_ill) SetSigIllHandler();
  if (Flags.handle_fpe) SetSigFpeHandler();
  if (Flags.handle_int) SetSigIntHandler();
  if (Flags.handle_term) SetSigTermHandler();

  if (DoPlainRun) {
    Options.SaveArtifacts = false;
    int Runs = std::max(1, Flags.runs);
    Printf("%s: Running %zd inputs %d time(s) each.\n", ProgName->c_str(),
           Inputs->size(), Runs);
    for (auto &Path : *Inputs) {
      auto StartTime = system_clock::now();
      Printf("Running: %s\n", Path.c_str());
      for (int Iter = 0; Iter < Runs; Iter++)
        RunOneTest(&F, Path.c_str());
      auto StopTime = system_clock::now();
      auto MS = duration_cast<milliseconds>(StopTime - StartTime).count();
      Printf("Executed %s in %zd ms\n", Path.c_str(), (long)MS);
    }
    F.PrintFinalStats();
    exit(0);
  }


  if (Flags.merge) {
    if (Options.MaxLen == 0)
      F.SetMaxLen(kMaxSaneLen);
    F.Merge(*Inputs);
    exit(0);
  }

  size_t TemporaryMaxLen = Options.MaxLen ? Options.MaxLen : kMaxSaneLen;

  F.RereadOutputCorpus(TemporaryMaxLen);
  for (auto &inp : *Inputs)
    if (inp != Options.OutputCorpus)
      F.ReadDir(inp, nullptr, TemporaryMaxLen);

  if (Options.MaxLen == 0)
    F.SetMaxLen(
        std::min(std::max(kMinDefaultLen, F.MaxUnitSizeInCorpus()), kMaxSaneLen));

  if (F.CorpusSize() == 0) {
    F.AddToCorpus(Unit());  // Can't fuzz empty corpus, so add an empty input.
    if (Options.Verbosity)
      Printf("INFO: A corpus is not provided, starting from an empty corpus\n");
  }
  F.ShuffleAndMinimize();
  if (Flags.drill)
    F.Drill();
  else
    F.Loop();

  if (Flags.verbosity)
    Printf("Done %d runs in %zd second(s)\n", F.getTotalNumberOfRuns(),
           F.secondsSinceProcessStartUp());
  F.PrintFinalStats();

  exit(0);  // Don't let F destroy itself.
}

// Storage for global ExternalFunctions object.
ExternalFunctions *EF = nullptr;

}  // namespace fuzzer
