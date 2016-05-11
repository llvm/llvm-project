//===- FuzzerTracePC.h - INTERNAL - Path tracer. --------*- C++ -* ===//
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
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_TRACE_PC_H
#define LLVM_FUZZER_TRACE_PC_H

namespace fuzzer {
struct PcCoverageMap {
  static const size_t kMapSizeInBits = 65371;        // Prime.
  static const size_t kMapSizeInBitsAligned = 65536; // 2^16
  static const size_t kBitsInWord = (sizeof(uintptr_t) * 8);
  static const size_t kMapSizeInWords = kMapSizeInBitsAligned / kBitsInWord;

  void Reset();
  inline void Update(uintptr_t Addr);
  size_t MergeFrom(const PcCoverageMap &Other);

  uintptr_t Map[kMapSizeInWords] __attribute__((aligned(512)));
};

// Clears the current PC Map.
void PcMapResetCurrent();
// Merges the current PC Map into the combined one, and clears the former.
size_t PcMapMergeInto(PcCoverageMap *Map);
}

#endif
