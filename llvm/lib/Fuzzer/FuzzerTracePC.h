//===- FuzzerTracePC.h - Internal header for the Fuzzer ---------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// fuzzer::TracePC
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_TRACE_PC
#define LLVM_FUZZER_TRACE_PC

#include "FuzzerDefs.h"
#include "FuzzerDictionary.h"
#include "FuzzerValueBitMap.h"

#include <set>

namespace fuzzer {

// TableOfRecentCompares (TORC) remembers the most recently performed
// comparisons of type T.
// We record the arguments of CMP instructions in this table unconditionally
// because it seems cheaper this way than to compute some expensive
// conditions inside __sanitizer_cov_trace_cmp*.
// After the unit has been executed we may decide to use the contents of
// this table to populate a Dictionary.
template<class T, size_t kSizeT>
struct TableOfRecentCompares {
  static const size_t kSize = kSizeT;
  struct Pair {
    T A, B;
  };
  ATTRIBUTE_NO_SANITIZE_ALL
  void Insert(size_t Idx, T Arg1, T Arg2) {
    Idx = Idx % kSize;
    Table[Idx].A = Arg1;
    Table[Idx].B = Arg2;
  }

  Pair Get(size_t I) { return Table[I % kSize]; }

  Pair Table[kSize];
};

class TracePC {
 public:
  static const size_t kNumPCs = 1 << 21;

  void HandleTrace(uint32_t *guard, uintptr_t PC);
  void HandleInit(uint32_t *start, uint32_t *stop);
  void HandleCallerCallee(uintptr_t Caller, uintptr_t Callee);
  template <class T> void HandleCmp(uintptr_t PC, T Arg1, T Arg2);
  size_t GetTotalPCCoverage();
  void SetUseCounters(bool UC) { UseCounters = UC; }
  void SetUseValueProfile(bool VP) { UseValueProfile = VP; }
  void SetPrintNewPCs(bool P) { DoPrintNewPCs = P; }
  template <class Callback> size_t CollectFeatures(Callback CB) const;

  void ResetMaps() {
    ValueProfileMap.Reset();
    memset(Counters(), 0, GetNumPCs());
  }

  void UpdateFeatureSet(size_t CurrentElementIdx, size_t CurrentElementSize);
  void PrintFeatureSet();

  void PrintModuleInfo();

  void PrintCoverage();
  void DumpCoverage();

  void AddValueForMemcmp(void *caller_pc, const void *s1, const void *s2,
                         size_t n, bool StopAtZero);

  TableOfRecentCompares<uint32_t, 32> TORC4;
  TableOfRecentCompares<uint64_t, 32> TORC8;
  TableOfRecentCompares<Word, 32> TORCW;

  void PrintNewPCs();
  void InitializePrintNewPCs();
  size_t GetNumPCs() const { return Min(kNumPCs, NumGuards + 1); }
  uintptr_t GetPC(size_t Idx) {
    assert(Idx < GetNumPCs());
    return PCs()[Idx];
  }

private:
  bool UseCounters = false;
  bool UseValueProfile = false;
  bool DoPrintNewPCs = false;

  struct Module {
    uint32_t *Start, *Stop;
  };

  Module Modules[4096];
  size_t NumModules;  // linker-initialized.
  size_t NumGuards;  // linker-initialized.

  uint8_t *Counters() const;
  uintptr_t *PCs() const;

  std::set<uintptr_t> *PrintedPCs;

  ValueBitMap ValueProfileMap;
};

template <class Callback>
size_t TracePC::CollectFeatures(Callback CB) const {
  size_t Res = 0;
  const size_t Step = 8;
  uint8_t *Counters = this->Counters();
  assert(reinterpret_cast<uintptr_t>(Counters) % Step == 0);
  size_t N = GetNumPCs();
  N = (N + Step - 1) & ~(Step - 1);  // Round up.
  for (size_t Idx = 0; Idx < N; Idx += Step) {
    uint64_t Bundle = *reinterpret_cast<uint64_t*>(&Counters[Idx]);
    if (!Bundle) continue;
    for (size_t i = Idx; i < Idx + Step; i++) {
      uint8_t Counter = (Bundle >> ((i - Idx) * 8)) & 0xff;
      if (!Counter) continue;
      unsigned Bit = 0;
      /**/ if (Counter >= 128) Bit = 7;
      else if (Counter >= 32) Bit = 6;
      else if (Counter >= 16) Bit = 5;
      else if (Counter >= 8) Bit = 4;
      else if (Counter >= 4) Bit = 3;
      else if (Counter >= 3) Bit = 2;
      else if (Counter >= 2) Bit = 1;
      size_t Feature = (i * 8 + Bit);
      if (CB(Feature))
        Res++;
    }
  }
  if (UseValueProfile)
    ValueProfileMap.ForEach([&](size_t Idx) {
      if (CB(N * 8 + Idx))
        Res++;
    });
  return Res;
}

extern TracePC TPC;

}  // namespace fuzzer

#endif  // LLVM_FUZZER_TRACE_PC
