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

#include "FuzzerInternal.h"

namespace fuzzer {

TracePC TPC;
const size_t TracePC::kNumCounters;

void TracePC::HandleTrace(uintptr_t *Guard, uintptr_t PC) {
  uintptr_t Idx = *Guard;
  if (!Idx) return;
  if (UseCounters) {
    uint8_t Counter = Counters[Idx % kNumCounters];
    if (Counter == 0) {
      if (TotalCoverageMap.AddValue(Idx)) {
        TotalCoverage++;
        AddNewPC(PC);
      }
    }
    if (Counter < 128)
      Counters[Idx % kNumCounters] = Counter + 1;
    else
      *Guard = 0;
  } else {
    *Guard = 0;
    TotalCoverage++;
    AddNewPC(PC);
  }
}

void TracePC::HandleInit(uintptr_t *Start, uintptr_t *Stop) {
  if (Start == Stop || *Start) return;
  assert(NumModules < sizeof(Modules) / sizeof(Modules[0]));
  for (uintptr_t *P = Start; P < Stop; P++)
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
  uintptr_t N = 0;
  for (size_t M = 0; M < NumModules; M++)
    for (uintptr_t *X = Modules[M].Start; X < Modules[M].Stop; X++)
      *X = ++N;
  assert(N == NumGuards);
}

void TracePC::FinalizeTrace() {
  if (UseCounters && TotalCoverage) {
    for (size_t Idx = 1, N = std::min(kNumCounters, NumGuards); Idx < N;
         Idx++) {
      uint8_t Counter = Counters[Idx];
      if (!Counter) continue;
      Counters[Idx] = 0;
      unsigned Bit = 0;
      /**/ if (Counter >= 128) Bit = 7;
      else if (Counter >= 32) Bit = 6;
      else if (Counter >= 16) Bit = 5;
      else if (Counter >= 8) Bit = 4;
      else if (Counter >= 4) Bit = 3;
      else if (Counter >= 3) Bit = 2;
      else if (Counter >= 2) Bit = 1;
      CounterMap.AddValue(Idx * 8 + Bit);
    }
  }
}

size_t TracePC::UpdateCounterMap(ValueBitMap *Map) {
  if (!TotalCoverage) return 0;
  size_t NewTotalCounterBits = Map->MergeFrom(CounterMap);
  size_t Delta = NewTotalCounterBits - TotalCounterBits;
  TotalCounterBits = NewTotalCounterBits;
  return Delta;
}

void TracePC::HandleCallerCallee(uintptr_t Caller, uintptr_t Callee) {
  const uintptr_t kBits = 12;
  const uintptr_t kMask = (1 << kBits) - 1;
  CounterMap.AddValue((Caller & kMask) | ((Callee & kMask) << kBits));
}

} // namespace fuzzer

extern "C" {
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uintptr_t *Guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(Guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uintptr_t *Start, uintptr_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_indir(uintptr_t Callee) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleCallerCallee(PC, Callee);
}
}
