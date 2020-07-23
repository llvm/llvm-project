//===-- llvm-libtool-darwin.cpp - a tool for creating libraries -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for creating static and dynamic libraries for Darwin.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

cl::OptionCategory LibtoolCategory("llvm-libtool-darwin Options");

static cl::opt<std::string> OutputFile("o", cl::desc("Specify output filename"),
                                       cl::value_desc("filename"), cl::Required,
                                       cl::cat(LibtoolCategory));

static cl::list<std::string> InputFiles(cl::Positional,
                                        cl::desc("<input files>"),
                                        cl::OneOrMore,
                                        cl::cat(LibtoolCategory));

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  cl::HideUnrelatedOptions({&LibtoolCategory, &ColorCategory});
  cl::ParseCommandLineOptions(Argc, Argv, "llvm-libtool-darwin\n");
}
