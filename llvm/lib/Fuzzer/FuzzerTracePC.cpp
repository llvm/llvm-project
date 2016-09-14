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

void TracePC::HandleTrace(uint8_t *guard, uintptr_t PC) {
  *guard = 0xff;
  TotalCoverage++;
}
void TracePC::HandleInit(uint8_t *start, uint8_t *stop) {
  Printf("INFO: guards: [%p,%p)\n", start, stop);
}
size_t TracePC::GetTotalCoverage() { return TotalCoverage; }

} // namespace fuzzer

extern "C" {
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint8_t *guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint8_t *start, uint8_t *stop) {
}
}
