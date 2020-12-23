//===- llvm-profgen.cpp - LLVM SPGO profile generation tool -----*- C++ -*-===//
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
#include "ProfileGenerator.h"
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

using namespace llvm;
using namespace sampleprof;

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "llvm SPGO profile generator\n");

  // Load binaries and parse perf events and samples
  PerfReader Reader(BinaryFilenames);
  Reader.parsePerfTraces(PerfTraceFilenames);

  std::unique_ptr<ProfileGenerator> Generator = ProfileGenerator::create(
      Reader.getBinarySampleCounters(), Reader.getPerfScriptType());
  Generator->generateProfile();
  Generator->write();

  return EXIT_SUCCESS;
}
