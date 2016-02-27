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
static const size_t kMapSizeInBits        = 65371; // Prime.
static const size_t kMapSizeInBitsAligned = 65536;  // 2^16
static const size_t kBitsInWord =(sizeof(uintptr_t) * 8);
static const size_t kMapSizeInWords = kMapSizeInBitsAligned / kBitsInWord;
static uintptr_t CurrentMap[kMapSizeInWords] __attribute__((aligned(512)));
static uintptr_t CombinedMap[kMapSizeInWords] __attribute__((aligned(512)));
static size_t CombinedMapSize;
static thread_local uintptr_t Prev;

void PcMapResetCurrent() {
  if (Prev) {
    Prev = 0;
    memset(CurrentMap, 0, sizeof(CurrentMap));
  }
}

void PcMapMergeCurrentToCombined() {
  if (!Prev) return;
  uintptr_t Res = 0;
  for (size_t i = 0; i < kMapSizeInWords; i++)
    Res += __builtin_popcountl(CombinedMap[i] |= CurrentMap[i]);
  CombinedMapSize = Res;
}

size_t PcMapCombinedSize() { return CombinedMapSize; }

static void HandlePC(uint32_t PC) {
  // We take 12 bits of PC and mix it with the previous PCs.
  uintptr_t Next = (Prev << 5) ^ (PC & 4095);
  uintptr_t Idx = Next % kMapSizeInBits;
  uintptr_t WordIdx = Idx / kBitsInWord;
  uintptr_t BitIdx  = Idx % kBitsInWord;
  CurrentMap[WordIdx] |= 1UL << BitIdx;
  Prev = Next;
}

} // namespace fuzzer

extern "C" void __sanitizer_cov_trace_pc() {
  fuzzer::HandlePC(static_cast<uint32_t>(
      reinterpret_cast<uintptr_t>(__builtin_return_address(0))));
}
