//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>

using namespace llvm;

static cl::opt<bool>
    Types("types",
          cl::desc("attempt to demangle types as well as function names"),
          cl::init(false));
static cl::alias TypesShort("t", cl::desc("alias for --types"),
                            cl::aliasopt(Types));

static cl::list<std::string>
Decorated(cl::Positional, cl::desc("<mangled>"), cl::ZeroOrMore);

static void demangle(llvm::raw_ostream &OS, const std::string &Mangled) {
  int Status;
  char *Demangled = nullptr;
  if (Types || ((Mangled.size() >= 2 && Mangled.compare(0, 2, "_Z")) ||
                (Mangled.size() >= 4 && Mangled.compare(0, 4, "___Z"))))
    Demangled = itaniumDemangle(Mangled.c_str(), nullptr, nullptr, &Status);
  OS << (Demangled ? Demangled : Mangled) << '\n';
  free(Demangled);
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");

  if (Decorated.empty())
    for (std::string Mangled; std::getline(std::cin, Mangled);)
      demangle(llvm::outs(), Mangled);
  else
    for (const auto &Symbol : Decorated)
      demangle(llvm::outs(), Symbol);

  return EXIT_SUCCESS;
}
