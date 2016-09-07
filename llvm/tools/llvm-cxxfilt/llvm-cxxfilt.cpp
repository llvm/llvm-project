//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"

#include <stdio.h>
#include <stdlib.h>

using namespace llvm;

int main(int argc, char **argv) {
  for (int I = 1; I < argc; ++I) {
    const char *Mangled = argv[I];
    int Status;
    char *Demangled = itaniumDemangle(Mangled, nullptr, nullptr, &Status);
    if (Demangled)
      printf("%s\n", Demangled);
    else
      printf("%s\n", Mangled);
    free(Demangled);
  }
  return 0;
}
