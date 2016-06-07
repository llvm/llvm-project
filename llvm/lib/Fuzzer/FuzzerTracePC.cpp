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

void PcCoverageMap::Reset() { memset(Map, 0, sizeof(Map)); }

void PcCoverageMap::Update(uintptr_t Addr) {
  uintptr_t Idx = Addr % kMapSizeInBits;
  uintptr_t WordIdx = Idx / kBitsInWord;
  uintptr_t BitIdx = Idx % kBitsInWord;
  Map[WordIdx] |= 1UL << BitIdx;
}

size_t PcCoverageMap::MergeFrom(const PcCoverageMap &Other) {
  uintptr_t Res = 0;
  for (size_t i = 0; i < kMapSizeInWords; i++)
    Res += __builtin_popcountl(Map[i] |= Other.Map[i]);
  return Res;
}

static PcCoverageMap CurrentMap;
static thread_local uintptr_t Prev;

void PcMapResetCurrent() {
  if (Prev) {
    Prev = 0;
    CurrentMap.Reset();
  }
}

size_t PcMapMergeInto(PcCoverageMap *Map) {
  if (!Prev)
    return 0;
  return Map->MergeFrom(CurrentMap);
}

static void HandlePC(uint32_t PC) {
  // We take 12 bits of PC and mix it with the previous PCs.
  uintptr_t Next = (Prev << 5) ^ (PC & 4095);
  CurrentMap.Update(Next);
  Prev = Next;
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
