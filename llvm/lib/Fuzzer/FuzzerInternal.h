//===- FuzzerInternal.h - Internal header for the Fuzzer --------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the main class fuzzer::Fuzzer and most functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_INTERNAL_H
#define LLVM_FUZZER_INTERNAL_H

#include <cassert>
#include <climits>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_set>

#include "FuzzerInterface.h"

namespace fuzzer {
using namespace std::chrono;

std::string FileToString(const std::string &Path);
Unit FileToVector(const std::string &Path);
void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V,
                            long *Epoch);
void WriteToFile(const Unit &U, const std::string &Path);
void CopyFileToErr(const std::string &Path);
// Returns "Dir/FileName" or equivalent for the current OS.
std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName);

void Printf(const char *Fmt, ...);
void PrintHexArray(const Unit &U, const char *PrintAfter = "");
void PrintHexArray(const uint8_t *Data, size_t Size,
                   const char *PrintAfter = "");
void PrintASCII(const uint8_t *Data, size_t Size, const char *PrintAfter = "");
void PrintASCII(const Unit &U, const char *PrintAfter = "");
std::string Hash(const Unit &U);
void SetTimer(int Seconds);
std::string Base64(const Unit &U);
int ExecuteCommand(const std::string &Command);

// Private copy of SHA1 implementation.
static const int kSHA1NumBytes = 20;
// Computes SHA1 hash of 'Len' bytes in 'Data', writes kSHA1NumBytes to 'Out'.
void ComputeSHA1(const uint8_t *Data, size_t Len, uint8_t *Out);

// Changes U to contain only ASCII (isprint+isspace) characters.
// Returns true iff U has been changed.
bool ToASCII(Unit &U);
bool IsASCII(const Unit &U);

int NumberOfCpuCores();
int GetPid();

// Dictionary.

// Parses one dictionary entry.
// If successfull, write the enty to Unit and returns true,
// otherwise returns false.
bool ParseOneDictionaryEntry(const std::string &Str, Unit *U);
// Parses the dictionary file, fills Units, returns true iff all lines
// were parsed succesfully.
bool ParseDictionaryFile(const std::string &Text, std::vector<Unit> *Units);

class Fuzzer {
 public:
  struct FuzzingOptions {
    int Verbosity = 1;
    int MaxLen = 0;
    int UnitTimeoutSec = 300;
    int MaxTotalTimeSec = 0;
    bool DoCrossOver = true;
    int  MutateDepth = 5;
    bool ExitOnFirst = false;
    bool UseCounters = false;
    bool UseIndirCalls = true;
    bool UseTraces = false;
    bool UseMemcmp = true;
    bool UseFullCoverageSet  = false;
    bool Reload = true;
    bool ShuffleAtStartUp = true;
    int PreferSmallDuringInitialShuffle = -1;
    size_t MaxNumberOfRuns = ULONG_MAX;
    int SyncTimeout = 600;
    int ReportSlowUnits = 10;
    bool OnlyASCII = false;
    std::string OutputCorpus;
    std::string SyncCommand;
    std::string ArtifactPrefix = "./";
    std::string ExactArtifactPath;
    bool SaveArtifacts = true;
    bool PrintNEW = true;  // Print a status line when new units are found;
    bool OutputCSV = false;
    bool PrintNewCovPcs = false;
  };
  Fuzzer(UserSuppliedFuzzer &USF, FuzzingOptions Options);
  void AddToCorpus(const Unit &U) { Corpus.push_back(U); }
  size_t ChooseUnitIdxToMutate();
  const Unit &ChooseUnitToMutate() { return Corpus[ChooseUnitIdxToMutate()]; };
  void Loop();
  void Drill();
  void ShuffleAndMinimize();
  void InitializeTraceState();
  void AssignTaintLabels(uint8_t *Data, size_t Size);
  size_t CorpusSize() const { return Corpus.size(); }
  void ReadDir(const std::string &Path, long *Epoch) {
    Printf("Loading corpus: %s\n", Path.c_str());
    ReadDirToVectorOfUnits(Path.c_str(), &Corpus, Epoch);
  }
  void RereadOutputCorpus();
  // Save the current corpus to OutputCorpus.
  void SaveCorpus();

  size_t secondsSinceProcessStartUp() {
    return duration_cast<seconds>(system_clock::now() - ProcessStartTime)
        .count();
  }

  size_t getTotalNumberOfRuns() { return TotalNumberOfRuns; }

  static void StaticAlarmCallback();

  void ExecuteCallback(const Unit &U);

  // Merge Corpora[1:] into Corpora[0].
  void Merge(const std::vector<std::string> &Corpora);

 private:
  void AlarmCallback();
  void MutateAndTestOne();
  void ReportNewCoverage(const Unit &U);
  bool RunOne(const Unit &U);
  void RunOneAndUpdateCorpus(Unit &U);
  void WriteToOutputCorpus(const Unit &U);
  void WriteUnitToFileWithPrefix(const Unit &U, const char *Prefix);
  void PrintStats(const char *Where, const char *End = "\n");
  void PrintStatusForNewUnit(const Unit &U);

  void SyncCorpus();

  size_t RecordBlockCoverage();
  size_t RecordCallerCalleeCoverage();
  void PrepareCoverageBeforeRun();
  bool CheckCoverageAfterRun();


  // Trace-based fuzzing: we run a unit with some kind of tracing
  // enabled and record potentially useful mutations. Then
  // We apply these mutations one by one to the unit and run it again.

  // Start tracing; forget all previously proposed mutations.
  void StartTraceRecording();
  // Stop tracing.
  void StopTraceRecording();

  void SetDeathCallback();
  static void StaticDeathCallback();
  void DeathCallback();

  uint8_t *CurrentUnitData;
  size_t CurrentUnitSize;

  size_t TotalNumberOfRuns = 0;
  size_t TotalNumberOfExecutedTraceBasedMutations = 0;

  std::vector<Unit> Corpus;
  std::unordered_set<std::string> UnitHashesAddedToCorpus;

  // For UseCounters
  std::vector<uint8_t> CounterBitmap;
  size_t TotalBits() {  // Slow. Call it only for printing stats.
    size_t Res = 0;
    for (auto x : CounterBitmap) Res += __builtin_popcount(x);
    return Res;
  }

  UserSuppliedFuzzer &USF;
  FuzzingOptions Options;
  system_clock::time_point ProcessStartTime = system_clock::now();
  system_clock::time_point LastExternalSync = system_clock::now();
  system_clock::time_point UnitStartTime;
  long TimeOfLongestUnitInSeconds = 0;
  long EpochOfLastReadOfOutputCorpus = 0;
  size_t LastRecordedBlockCoverage = 0;
  size_t LastRecordedCallerCalleeCoverage = 0;
  size_t LastCoveragePcBufferLen = 0;
};

class SimpleUserSuppliedFuzzer: public UserSuppliedFuzzer {
 public:
  SimpleUserSuppliedFuzzer(FuzzerRandomBase *Rand, UserCallback Callback)
      : UserSuppliedFuzzer(Rand), Callback(Callback) {}

  virtual int TargetFunction(const uint8_t *Data, size_t Size) override {
    return Callback(Data, Size);
  }

 private:
  UserCallback Callback = nullptr;
};

};  // namespace fuzzer

#endif // LLVM_FUZZER_INTERNAL_H
