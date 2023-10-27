//===-- llvm-readtapi.cpp - tapi file reader and manipulator -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the command-line driver for llvm-readtapi.
//
//===----------------------------------------------------------------------===//
#include "DiffEngine.h"
#include "llvm/Object/TapiUniversal.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace llvm;
using namespace MachO;
using namespace object;

namespace {
cl::OptionCategory TapiCat("llvm-readtapi options");
cl::OptionCategory CompareCat("llvm-readtapi --compare options");

cl::opt<std::string> InputFileName(cl::Positional, cl::desc("<tapi file>"),
                                   cl::Required, cl::cat(TapiCat));
cl::opt<std::string> CompareInputFileName(cl::Positional,
                                          cl::desc("<comparison file>"),
                                          cl::Required, cl::cat(CompareCat));
enum OutputKind {
  Compare,
};

cl::opt<OutputKind>
    Output(cl::desc("choose command action:"),
           cl::values(clEnumValN(Compare, "compare",
                                 "compare tapi file for library differences")),
           cl::init(OutputKind::Compare), cl::cat(TapiCat));
} // anonymous namespace

Expected<std::unique_ptr<Binary>> convertFileToBinary(std::string &Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (BufferOrErr.getError())
    return errorCodeToError(BufferOrErr.getError());
  return createBinary(BufferOrErr.get()->getMemBufferRef());
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  cl::HideUnrelatedOptions(TapiCat);
  cl::ParseCommandLineOptions(Argc, Argv,
                              "TAPI File Reader and Manipulator Tool");

  if (Output == OutputKind::Compare) {
    if (InputFileName.empty() || CompareInputFileName.empty()) {
      cl::PrintHelpMessage();
      return EXIT_FAILURE;
    }

    ExitOnError ExitOnErr("error: '" + InputFileName + "' ",
                          /*DefaultErrorExitCode=*/2);
    auto BinLHS = ExitOnErr(convertFileToBinary(InputFileName));

    TapiUniversal *FileLHS = dyn_cast<TapiUniversal>(BinLHS.get());
    if (!FileLHS) {
      ExitOnErr(createStringError(std::errc::executable_format_error,
                                  "unsupported file format"));
    }

    ExitOnErr.setBanner("error: '" + CompareInputFileName + "' ");
    auto BinRHS = ExitOnErr(convertFileToBinary(CompareInputFileName));

    TapiUniversal *FileRHS = dyn_cast<TapiUniversal>(BinRHS.get());
    if (!FileRHS) {
      ExitOnErr(createStringError(std::errc::executable_format_error,
                                  "unsupported file format"));
    }

    raw_ostream &OS = outs();
    return DiffEngine(FileLHS, FileRHS).compareFiles(OS);
  }

  return 0;
}
