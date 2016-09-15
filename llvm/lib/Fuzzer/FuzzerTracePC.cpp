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

void TracePC::HandleTrace(uint8_t *Guard, uintptr_t PC) {
  if (UseCounters) {
    uintptr_t GV = *Guard;
    if (GV == 0)
      TotalCoverage++;
    if (GV < 255)
      GV++;
    *Guard = GV;
  } else {
    *Guard = 0xff;
    TotalCoverage++;
  }
}

void TracePC::HandleInit(uint8_t *Start, uint8_t *Stop) {
  // TODO: this handles only one DSO/binary.
  this->Start = Start;
  this->Stop = Stop;
}

void TracePC::FinalizeTrace() {
  if (UseCounters && TotalCoverage) {
    for (uint8_t *X = Start; X < Stop; X++) {
      uint8_t Value = *X;
      size_t Idx = X - Start;
      if (Value >= 2) {
        unsigned Bit = 31 - __builtin_clz(Value);
        assert(Bit < 8);
        CounterMap.AddValue(Idx * 8 + Bit);
      }
      *X = 1;
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

} // namespace fuzzer

extern "C" {
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint8_t *Guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(Guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint8_t *Start, uint8_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}
}
