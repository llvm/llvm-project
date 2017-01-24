//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>

using namespace llvm;

static void demangle(llvm::raw_ostream &OS, const std::string &Mangled) {
  int Status;
  char *Demangled = nullptr;
  if ((Mangled.size() >= 2 && Mangled.compare(0, 2, "_Z")) ||
      (Mangled.size() >= 4 && Mangled.compare(0, 4, "___Z")))
    Demangled = itaniumDemangle(Mangled.c_str(), nullptr, nullptr, &Status);
  OS << (Demangled ? Demangled : Mangled) << '\n';
  free(Demangled);
}

int main(int argc, char **argv) {
  if (argc == 1)
    for (std::string Mangled; std::getline(std::cin, Mangled);)
      demangle(llvm::outs(), Mangled);
  else
    for (int I = 1; I < argc; ++I)
      demangle(llvm::outs(), argv[I]);

  return EXIT_SUCCESS;
}
