//===--- PGOCtxProfJSONReader.h - JSON format  ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// JSON format for the contextual profile for testing.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ProfileData/CtxInstrContextNode.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::SubCommand FromJSON("fromJSON", "Convert from json");

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
    cl::sub(FromJSON));

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::init("-"),
                                           cl::desc("Output file"),
                                           cl::sub(FromJSON));

// Save the bitstream profile from the JSON representation.
Error convertFromJSON() {
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

  return llvm::createCtxProfFromJSON(BufOrError.get()->getBuffer(), Out);
}

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "LLVM Contextual Profile Utils\n");
  ExitOnError ExitOnErr("llvm-ctxprof-util: ");
  if (FromJSON) {
    if (auto E = convertFromJSON()) {
      handleAllErrors(std::move(E), [&](const ErrorInfoBase &E) {
        E.log(errs());
        errs() << "\n";
      });
      return 1;
    }
    return 0;
  }
  cl::PrintHelpMessage();
  return 1;
}
