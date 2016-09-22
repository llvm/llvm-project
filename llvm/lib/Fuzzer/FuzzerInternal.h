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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <string.h>

#include "FuzzerDefs.h"
#include "FuzzerExtFunctions.h"
#include "FuzzerInterface.h"
#include "FuzzerOptions.h"
#include "FuzzerValueBitMap.h"

namespace fuzzer {

using namespace std::chrono;

// See FuzzerTraceState.cpp
void EnableValueProfile();
size_t VPMapMergeFromCurrent(ValueBitMap &M);

class Fuzzer {
public:

  // Aggregates all available coverage measurements.
  struct Coverage {
    Coverage() { Reset(); }

    void Reset() {
      BlockCoverage = 0;
      CallerCalleeCoverage = 0;
      CounterBitmapBits = 0;
      CounterBitmap.clear();
      VPMap.Reset();
      TPCMap.Reset();
      VPMapBits = 0;
    }

    std::string DebugString() const;

    size_t BlockCoverage;
    size_t CallerCalleeCoverage;
    // Precalculated number of bits in CounterBitmap.
    size_t CounterBitmapBits;
    std::vector<uint8_t> CounterBitmap;
    ValueBitMap TPCMap;
    ValueBitMap VPMap;
    size_t VPMapBits;
  };

  Fuzzer(UserCallback CB, InputCorpus &Corpus, MutationDispatcher &MD,
         FuzzingOptions Options);
  ~Fuzzer();
  void Loop();
  void ShuffleAndMinimize(UnitVector *V);
  void InitializeTraceState();
  void AssignTaintLabels(uint8_t *Data, size_t Size);
  void RereadOutputCorpus(size_t MaxSize);

  size_t secondsSinceProcessStartUp() {
    return duration_cast<seconds>(system_clock::now() - ProcessStartTime)
        .count();
  }
  size_t execPerSec() {
    size_t Seconds = secondsSinceProcessStartUp();
    return Seconds ? TotalNumberOfRuns / Seconds : 0;
  }

  size_t getTotalNumberOfRuns() { return TotalNumberOfRuns; }

  static void StaticAlarmCallback();
  static void StaticCrashSignalCallback();
  static void StaticInterruptCallback();

  void ExecuteCallback(const uint8_t *Data, size_t Size);
  bool RunOne(const uint8_t *Data, size_t Size);

  // Merge Corpora[1:] into Corpora[0].
  void Merge(const std::vector<std::string> &Corpora);
  // Returns a subset of 'Extra' that adds coverage to 'Initial'.
  UnitVector FindExtraUnits(const UnitVector &Initial, const UnitVector &Extra);
  MutationDispatcher &GetMD() { return MD; }
  void PrintFinalStats();
  void SetMaxLen(size_t MaxLen);
  void RssLimitCallback();

  // Public for tests.
  void ResetCoverage();

  bool InFuzzingThread() const { return IsMyThread; }
  size_t GetCurrentUnitInFuzzingThead(const uint8_t **Data) const;

private:
  void AlarmCallback();
  void CrashCallback();
  void InterruptCallback();
  void MutateAndTestOne();
  void ReportNewCoverage(InputInfo *II, const Unit &U);
  void PrintNewPCs();
  void PrintOneNewPC(uintptr_t PC);
  bool RunOne(const Unit &U) { return RunOne(U.data(), U.size()); }
  void WriteToOutputCorpus(const Unit &U);
  void WriteUnitToFileWithPrefix(const Unit &U, const char *Prefix);
  void PrintStats(const char *Where, const char *End = "\n");
  void PrintStatusForNewUnit(const Unit &U);
  void ShuffleCorpus(UnitVector *V);
  void TryDetectingAMemoryLeak(const uint8_t *Data, size_t Size,
                               bool DuringInitialCorpusExecution);

  bool UpdateMaxCoverage();

  // Trace-based fuzzing: we run a unit with some kind of tracing
  // enabled and record potentially useful mutations. Then
  // We apply these mutations one by one to the unit and run it again.

  // Start tracing; forget all previously proposed mutations.
  void StartTraceRecording();
  // Stop tracing.
  void StopTraceRecording();

  void SetDeathCallback();
  static void StaticDeathCallback();
  void DumpCurrentUnit(const char *Prefix);
  void DeathCallback();

  void ResetEdgeCoverage();
  void ResetCounters();
  void PrepareCounters(Fuzzer::Coverage *C);
  bool RecordMaxCoverage(Fuzzer::Coverage *C);

  void LazyAllocateCurrentUnitData();
  uint8_t *CurrentUnitData = nullptr;
  std::atomic<size_t> CurrentUnitSize;
  uint8_t BaseSha1[kSHA1NumBytes];  // Checksum of the base unit.

  size_t TotalNumberOfRuns = 0;
  size_t NumberOfNewUnitsAdded = 0;

  bool HasMoreMallocsThanFrees = false;
  size_t NumberOfLeakDetectionAttempts = 0;

  UserCallback CB;
  InputCorpus &Corpus;
  MutationDispatcher &MD;
  FuzzingOptions Options;

  system_clock::time_point ProcessStartTime = system_clock::now();
  system_clock::time_point UnitStartTime;
  long TimeOfLongestUnitInSeconds = 0;
  long EpochOfLastReadOfOutputCorpus = 0;

  // Maximum recorded coverage.
  Coverage MaxCoverage;

  // For -print_pcs
  uintptr_t* PcBuffer = nullptr;
  size_t PcBufferLen = 0;
  size_t PcBufferPos = 0, PrevPcBufferPos = 0;

  // Need to know our own thread.
  static thread_local bool IsMyThread;

  bool InMergeMode = false;
};

}; // namespace fuzzer

#endif // LLVM_FUZZER_INTERNAL_H
