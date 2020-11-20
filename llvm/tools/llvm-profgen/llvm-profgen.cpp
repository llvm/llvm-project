//===- llvm-profgen.cpp - LLVM SPGO profile generation tool ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-profgen generates SPGO profiles from perf script ouput.
//
//===----------------------------------------------------------------------===//

#include "ErrorHandling.h"
#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

static cl::list<std::string> PerfTraceFilenames(
    "perfscript", cl::value_desc("perfscript"), cl::OneOrMore,
    llvm::cl::MiscFlags::CommaSeparated,
    cl::desc("Path of perf-script trace created by Linux perf tool with "
             "`script` command(the raw perf.data should be profiled with -b)"));

static cl::list<std::string>
    BinaryFilenames("binary", cl::value_desc("binary"), cl::OneOrMore,
                    llvm::cl::MiscFlags::CommaSeparated,
                    cl::desc("Path of profiled binary files"));

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::Required,
                                           cl::desc("Output profile file"));

using namespace llvm;
using namespace sampleprof;

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm SPGO profile generator\n");

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  // Load binaries and parse perf events and samples
  PerfReader Reader(BinaryFilenames);
  Reader.parsePerfTraces(PerfTraceFilenames);

  return EXIT_SUCCESS;
}
