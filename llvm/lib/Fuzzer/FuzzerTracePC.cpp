//===- FuzzerTracePC.cpp - PC tracing--------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Trace PCs.
// This module implements __sanitizer_cov_trace_pc, a callback required
// for -fsanitize-coverage=trace-pc instrumentation.
//
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"

namespace fuzzer {

static size_t PreviouslyComputedPCHash;
static ValueBitMap CurrentPCMap;

// Merges CurrentPCMap into M, returns the number of new bits.
size_t PCMapMergeFromCurrent(ValueBitMap &M) {
  if (!PreviouslyComputedPCHash)
    return 0;
  PreviouslyComputedPCHash = 0;
  return M.MergeFrom(CurrentPCMap);
}

static void HandlePC(uint32_t PC) {
  // We take 12 bits of PC and mix it with the previous PCs.
  uintptr_t Next = (PreviouslyComputedPCHash << 5) ^ (PC & 4095);
  CurrentPCMap.AddValue(Next);
  PreviouslyComputedPCHash = Next;
}

} // namespace fuzzer

extern "C" {
void __sanitizer_cov_trace_pc() {
  fuzzer::HandlePC(static_cast<uint32_t>(
      reinterpret_cast<uintptr_t>(__builtin_return_address(0))));
}

void __sanitizer_cov_trace_pc_indir(int *) {
  // Stub to allow linking with code built with
  // -fsanitize=indirect-calls,trace-pc.
  // This isn't used currently.
}
}
