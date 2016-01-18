//===- FuzzerDFSan.h - Internal header for the Fuzzer -----------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// DFSan interface.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_DFSAN_H
#define LLVM_FUZZER_DFSAN_H

#ifndef LLVM_FUZZER_SUPPORTS_DFSAN
# if defined(__linux__)
#  define LLVM_FUZZER_SUPPORTS_DFSAN 1
# else
#  define LLVM_FUZZER_SUPPORTS_DFSAN 0
# endif  // __linux__
#endif  // LLVM_FUZZER_SUPPORTS_DFSAN

#include <sanitizer/dfsan_interface.h>
#if LLVM_FUZZER_SUPPORTS_DFSAN

extern "C" {
__attribute__((weak))
dfsan_label dfsan_create_label(const char *desc, void *userdata);
__attribute__((weak))
void dfsan_set_label(dfsan_label label, void *addr, size_t size);
__attribute__((weak))
void dfsan_add_label(dfsan_label label, void *addr, size_t size);
__attribute__((weak))
const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label);
__attribute__((weak))
dfsan_label dfsan_read_label(const void *addr, size_t size);
}  // extern "C"

namespace fuzzer {
static bool ReallyHaveDFSan() {
  return &dfsan_create_label != nullptr;
}
}  // namespace fuzzer
#else
namespace fuzzer {
static bool ReallyHaveDFSan() { return false; }
}  // namespace fuzzer

#endif

#endif // LLVM_FUZZER_DFSAN_H
