//===- FuzzerMain.cpp - main() function and flags -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// main() and flags.
//===----------------------------------------------------------------------===//

#include "FuzzerInterface.h"
#include "FuzzerInternal.h"

extern "C" {
// This function should be defined by the user.
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
// This function may optionally be defined by the user.
__attribute__((weak)) int LLVMFuzzerInitialize(int *argc, char ***argv);
}  // extern "C"

int main(int argc, char **argv) {
  if (LLVMFuzzerInitialize)
    LLVMFuzzerInitialize(&argc, &argv);
  return fuzzer::FuzzerDriver(argc, argv, LLVMFuzzerTestOneInput);
}
