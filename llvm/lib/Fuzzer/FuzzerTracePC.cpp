//===- FuzzerTracePC.cpp - PC tracing--------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Trace PCs.
// This module implements __sanitizer_cov_trace_pc_guard[_init],
// the callback required for -fsanitize-coverage=trace-pc-guard instrumentation.
//
//===----------------------------------------------------------------------===//

#include "FuzzerDefs.h"
#include "FuzzerTracePC.h"
#include "FuzzerValueBitMap.h"

namespace fuzzer {

TracePC TPC;

void TracePC::HandleTrace(uint32_t *Guard, uintptr_t PC) {
  uint32_t Idx = *Guard;
  if (!Idx) return;
  uint8_t *CounterPtr = &Counters[Idx % kNumCounters];
  uint8_t Counter = *CounterPtr;
  if (Counter == 0) {
    if (!PCs[Idx]) {
      AddNewPCID(Idx);
      TotalPCCoverage++;
      PCs[Idx] = PC;
    }
  }
  if (UseCounters) {
    if (Counter < 128)
      *CounterPtr = Counter + 1;
    else
      *Guard = 0;
  } else {
    *CounterPtr = 1;
    *Guard = 0;
  }
}

void TracePC::HandleInit(uint32_t *Start, uint32_t *Stop) {
  if (Start == Stop || *Start) return;
  assert(NumModules < sizeof(Modules) / sizeof(Modules[0]));
  for (uint32_t *P = Start; P < Stop; P++)
    *P = ++NumGuards;
  Modules[NumModules].Start = Start;
  Modules[NumModules].Stop = Stop;
  NumModules++;
}

void TracePC::PrintModuleInfo() {
  Printf("INFO: Loaded %zd modules (%zd guards): ", NumModules, NumGuards);
  for (size_t i = 0; i < NumModules; i++)
    Printf("[%p, %p), ", Modules[i].Start, Modules[i].Stop);
  Printf("\n");
}

void TracePC::ResetGuards() {
  uint32_t N = 0;
  for (size_t M = 0; M < NumModules; M++)
    for (uint32_t *X = Modules[M].Start; X < Modules[M].Stop; X++)
      *X = ++N;
  assert(N == NumGuards);
}

bool TracePC::FinalizeTrace(size_t InputSize) {
  bool Res = false;
  if (TotalPCCoverage) {
    const size_t Step = 8;
    assert(reinterpret_cast<uintptr_t>(Counters) % Step == 0);
    size_t N = Min(kNumCounters, NumGuards + 1);
    N = (N + Step - 1) & ~(Step - 1);  // Round up.
    for (size_t Idx = 0; Idx < N; Idx += Step) {
      uint64_t Bundle = *reinterpret_cast<uint64_t*>(&Counters[Idx]);
      if (!Bundle) continue;
      for (size_t i = Idx; i < Idx + Step; i++) {
        uint8_t Counter = (Bundle >> (i * 8)) & 0xff;
        if (!Counter) continue;
        Counters[i] = 0;
        unsigned Bit = 0;
        /**/ if (Counter >= 128) Bit = 7;
        else if (Counter >= 32) Bit = 6;
        else if (Counter >= 16) Bit = 5;
        else if (Counter >= 8) Bit = 4;
        else if (Counter >= 4) Bit = 3;
        else if (Counter >= 3) Bit = 2;
        else if (Counter >= 2) Bit = 1;
        size_t Feature = i * 8 + Bit;
        CounterMap.AddValue(Feature);
        uint32_t *SizePtr = &InputSizesPerFeature[Feature % kFeatureSetSize];
        if (!*SizePtr || *SizePtr > InputSize) {
          *SizePtr = InputSize;
          Res = true;
        }
      }
    }
  }
  return Res;
}

void TracePC::HandleCallerCallee(uintptr_t Caller, uintptr_t Callee) {
  const uintptr_t kBits = 12;
  const uintptr_t kMask = (1 << kBits) - 1;
  CounterMap.AddValue((Caller & kMask) | ((Callee & kMask) << kBits));
}

void TracePC::PrintCoverage() {
  Printf("COVERAGE:\n");
  for (size_t i = 0; i < Min(NumGuards + 1, kNumPCs); i++) {
    if (PCs[i])
      PrintPC("COVERED: %p %F %L\n", "COVERED: %p\n", PCs[i]);
  }
}

} // namespace fuzzer

extern "C" {
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint32_t *Guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(Guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *Start, uint32_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_indir(uintptr_t Callee) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleCallerCallee(PC, Callee);
}
}
