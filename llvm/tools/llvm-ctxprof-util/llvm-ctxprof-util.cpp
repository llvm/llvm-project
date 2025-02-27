//===--- llvm-ctxprof-util - utilities for ctxprof --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Utilities for manipulating contextual profiles
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalValue.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::SubCommand FromYAML("fromYAML", "Convert from yaml");
static cl::SubCommand ToYAML("toYAML", "Convert to yaml");

static cl::opt<std::string> InputFilename(
    "input", cl::value_desc("input"), cl::init("-"),
    cl::desc(
        "Input file. The format is an array of contexts.\n"
        "Each context is a dictionary with the following keys:\n"
        "'Guid', mandatory. The value is a 64-bit integer.\n"
        "'Counters', mandatory. An array of 32-bit ints. These are the "
        "counter values.\n"
        "'Contexts', optional. An array containing arrays of contexts. The "
        "context array at a position 'i' is the set of callees at that "
        "callsite index. Use an empty array to indicate no callees."),
    cl::sub(FromYAML), cl::sub(ToYAML));

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::init("-"),
                                           cl::desc("Output file"),
                                           cl::sub(FromYAML), cl::sub(ToYAML));

namespace {
// Save the bitstream profile from the JSON representation.
Error convertFromYaml() {
  auto BufOrError =
      MemoryBuffer::getFileOrSTDIN(InputFilename, /*IsText=*/true);
  if (!BufOrError)
    return createFileError(InputFilename, BufOrError.getError());

  std::error_code EC;
  // Using a fd_ostream instead of a fd_stream. The latter would be more
  // efficient as the bitstream writer supports incremental flush to it, but the
  // json scenario is for test, and file size scalability doesn't really concern
  // us.
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC)
    return createStringError(EC, "failed to open output");

  return llvm::createCtxProfFromYAML(BufOrError.get()->getBuffer(), Out);
}

Error convertToYaml() {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!BufOrError)
    return createFileError(InputFilename, BufOrError.getError());

  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC)
    return createStringError(EC, "failed to open output");
  PGOCtxProfileReader Reader(BufOrError.get()->getBuffer());
  auto Prof = Reader.loadContexts();
  if (!Prof)
    return Prof.takeError();
  llvm::convertCtxProfToYaml(Out, *Prof);
  Out << "\n";
  return Error::success();
}
} // namespace

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLVM Contextual Profile Utils\n");
  ExitOnError ExitOnErr("llvm-ctxprof-util: ");
  auto HandleErr = [&](Error E) -> int {
    if (E) {
      handleAllErrors(std::move(E), [&](const ErrorInfoBase &E) {
        E.log(errs());
        errs() << "\n";
      });
      return 1;
    }
    return 0;
  };

  if (FromYAML)
    return HandleErr(convertFromYaml());

  if (ToYAML)
    return HandleErr(convertToYaml());

  cl::PrintHelpMessage();
  return 1;
}
