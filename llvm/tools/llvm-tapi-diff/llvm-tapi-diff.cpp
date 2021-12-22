//===-- llvm-tapi-diff.cpp - tbd comparator command-line driver --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the command-line driver for the llvm-tapi difference
// engine.
//
//===----------------------------------------------------------------------===//
#include "DiffEngine.h"
#include "llvm/Object/TapiUniversal.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace llvm;
using namespace MachO;
using namespace object;

namespace {
cl::OptionCategory NMCat("llvm-tapi-diff Options");
cl::opt<std::string> InputFileNameLHS(cl::Positional, cl::desc("<first file>"),
                                      cl::cat(NMCat));
cl::opt<std::string> InputFileNameRHS(cl::Positional, cl::desc("<second file>"),
                                      cl::cat(NMCat));
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
  cl::HideUnrelatedOptions(NMCat);
  cl::ParseCommandLineOptions(Argc, Argv, "Text-based Stubs Comparison Tool");
  if (InputFileNameLHS.empty() || InputFileNameRHS.empty()) {
    cl::PrintHelpMessage();
    return EXIT_FAILURE;
  }

  ExitOnError ExitOnErr("error: '" + InputFileNameLHS + "' ",
                        /*DefaultErrorExitCode=*/2);
  auto BinLHS = ExitOnErr(convertFileToBinary(InputFileNameLHS));

  TapiUniversal *FileLHS = dyn_cast<TapiUniversal>(BinLHS.get());
  if (!FileLHS) {
    ExitOnErr(createStringError(std::errc::executable_format_error,
                                "unsupported file format"));
  }

  ExitOnErr.setBanner("error: '" + InputFileNameRHS + "' ");
  auto BinRHS = ExitOnErr(convertFileToBinary(InputFileNameRHS));

  TapiUniversal *FileRHS = dyn_cast<TapiUniversal>(BinRHS.get());
  if (!FileRHS) {
    ExitOnErr(createStringError(std::errc::executable_format_error,
                                "unsupported file format"));
  }

  raw_ostream &OS = outs();

  return DiffEngine(FileLHS, FileRHS).compareFiles(OS);
}
