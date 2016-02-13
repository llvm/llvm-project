//===- FuzzerInterface.h - Interface header for the Fuzzer ------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the interface between the Fuzzer and the library being tested.
//===----------------------------------------------------------------------===//

// WARNING: keep the interface free of STL or any other header-based C++ lib,
// to avoid bad interactions between the code used in the fuzzer and
// the code used in the target function.

#ifndef LLVM_FUZZER_INTERFACE_H
#define LLVM_FUZZER_INTERFACE_H

#include <cstddef>
#include <cstdint>

// Plain C interface. Should be sufficient for most uses.
extern "C" {
// The target function, mandatory.
// Must return 0.
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
// The initialization function, optional.
int LLVMFuzzerInitialize(int *argc, char ***argv);
// Custom mutator, optional.
// Mutates raw data in [Data, Data+Size] inplace.
// Returns the new size, which is not greater than MaxSize.
// Given the same Seed produces the same mutation.
size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize,
                               unsigned int Seed);

}  // extern "C"

namespace fuzzer {

/// Returns an int 0. Values other than zero are reserved for future.
typedef int (*UserCallback)(const uint8_t *Data, size_t Size);
/** Simple C-like interface with a single user-supplied callback.

Usage:

#\code
#include "FuzzerInterface.h"

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  DoStuffWithData(Data, Size);
  return 0;
}

// Optional.
// Define this only if you need to read/modify argc/argv at startup
// and you are using libFuzzer's main().
// Must return 0.
int LLVMFuzzerInitialize(int *argc, char ***argv) {
  ReadAndMaybeModify(argc, argv);
  return 0;
}

// Implement your own main() or use the one from FuzzerMain.cpp.
// *NOT* recommended for most cases.
int main(int argc, char **argv) {
  InitializeMeIfNeeded();
  return fuzzer::FuzzerDriver(argc, argv, LLVMFuzzerTestOneInput);
}
#\endcode
*/
int FuzzerDriver(int argc, char **argv, UserCallback Callback);

// Mutates raw data in [Data, Data+Size] inplace.
// Returns the new size, which is not greater than MaxSize.
// Can be used inside the user-supplied LLVMFuzzerTestOneInput.
size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_INTERFACE_H
