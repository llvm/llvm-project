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

void TracePC::HandleTrace(uint64_t *Guard, uintptr_t PC) {
  const uint64_t kBit63 = 1ULL << 63;
  uint64_t Value = *Guard;
  if (Value & kBit63) return;
  // Printf("   >> %16zx %p\n", Value, Guard);
  if (UseCounters) {
    uint64_t Counter = Value & 0xff;
    if (Counter == 0) {
      size_t Idx = Value >> 32;
      if (TotalCoverageMap.AddValue(Idx)) {
        TotalCoverage++;
        AddNewPC(PC);
      }
    }
    if (Counter < 255)
      Value++;
    else
      Value |= kBit63;
  } else {
    Value |= kBit63;
    TotalCoverage++;
    AddNewPC(PC);
  }
  // Printf("   << %16zx\n", Value);
  *Guard = Value;
}

void TracePC::HandleInit(uint64_t *Start, uint64_t *Stop) {
  if (Start == Stop || *Start) return;
  assert(NumModules < sizeof(Modules) / sizeof(Modules[0]));
  for (uint64_t *P = Start; P < Stop; P++)
    *P = (++NumGuards) << 32;
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
  for (size_t M = 0; M < NumModules; M++)
    for (uint64_t *X = Modules[M].Start; X < Modules[M].Stop; X++)
      *X = (*X >> 32) << 32;
}

void TracePC::FinalizeTrace() {
  if (UseCounters && TotalCoverage) {
    for (size_t M = 0; M < NumModules; M++) {
      for (uint64_t *X = Modules[M].Start; X < Modules[M].Stop; X++) {
        uint64_t Value = *X & 0xff;
        uint64_t Idx = *X >> 32;
        if (Value >= 1) {
          unsigned Bit = 0;
          /**/ if (Value >= 128) Bit = 7;
          else if (Value >= 32) Bit = 6;
          else if (Value >= 16) Bit = 5;
          else if (Value >= 8) Bit = 4;
          else if (Value >= 4) Bit = 3;
          else if (Value >= 3) Bit = 2;
          else if (Value >= 2) Bit = 1;
          CounterMap.AddValue(Idx * 8 + Bit);
        }
        *X = Idx << 32;
      }
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
void __sanitizer_cov_trace_pc_guard(uint64_t *Guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(Guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint64_t *Start, uint64_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_indir(uintptr_t Callee) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleCallerCallee(PC, Callee);
}
}
