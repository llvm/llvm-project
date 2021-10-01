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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

static cl::OptionCategory ProfGenCategory("ProfGen Options");

cl::opt<std::string> PerfTraceFilename(
    "perfscript", cl::value_desc("perfscript"), cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated,
    cl::desc("Path of perf-script trace created by Linux perf tool with "
             "`script` command(the raw perf.data should be profiled with -b)"),
    cl::cat(ProfGenCategory));

static cl::opt<std::string> PerfDataFilename(
    "perfdata", cl::value_desc("perfdata"), cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated,
    cl::desc("Path of raw perf data created by Linux perf tool (it should be "
             "profiled with -b)"),
    cl::cat(ProfGenCategory));

static cl::opt<std::string> BinaryPath(
    "binary", cl::value_desc("binary"), cl::Required,
    cl::desc("Path of profiled binary, only one binary is supported."),
    cl::cat(ProfGenCategory));

extern cl::opt<bool> ShowDisassemblyOnly;
extern cl::opt<bool> ShowSourceLocations;
extern cl::opt<bool> SkipSymbolization;

using namespace llvm;
using namespace sampleprof;

// Validate the command line input.
static void validateCommandLine() {
  // Validate input profile is provided only once
  if (PerfDataFilename.getNumOccurrences() &&
      PerfTraceFilename.getNumOccurrences()) {
    std::string Msg = "`-perfdata` and `-perfscript` cannot be used together.";
    exitWithError(Msg);
  }

  // Validate input profile is provided
  if (!PerfDataFilename.getNumOccurrences() &&
      !PerfTraceFilename.getNumOccurrences()) {
    std::string Msg =
        "Use `-perfdata` or `-perfscript` to provide input perf profile.";
    exitWithError(Msg);
  }

  // Allow the invalid perfscript if we only use to show binary disassembly.
  if (!ShowDisassemblyOnly) {
    if (PerfTraceFilename.getNumOccurrences() &&
        !llvm::sys::fs::exists(PerfTraceFilename)) {
      std::string Msg =
          "Input perf script(" + PerfTraceFilename + ") doesn't exist.";
      exitWithError(Msg);
    }

    if (PerfDataFilename.getNumOccurrences() &&
        !llvm::sys::fs::exists(PerfDataFilename)) {
      std::string Msg =
          "Input perf data(" + PerfDataFilename + ") doesn't exist.";
      exitWithError(Msg);
    }
  }

  if (!llvm::sys::fs::exists(BinaryPath)) {
    std::string Msg = "Input binary(" + BinaryPath + ") doesn't exist.";
    exitWithError(Msg);
  }

  if (CSProfileGenerator::MaxCompressionSize < -1) {
    exitWithError("Value of --compress-recursion should >= -1");
  }
  if (ShowSourceLocations && !ShowDisassemblyOnly) {
    exitWithError("--show-source-locations should work together with "
                  "--show-disassembly-only!");
  }
}

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::HideUnrelatedOptions({&ProfGenCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "llvm SPGO profile generator\n");
  validateCommandLine();

  // Load symbols and disassemble the code of a given binary.
  std::unique_ptr<ProfiledBinary> Binary =
      std::make_unique<ProfiledBinary>(BinaryPath);
  if (ShowDisassemblyOnly)
    return EXIT_SUCCESS;

  // Parse perf events and samples
  StringRef PerfInputFile;
  bool IsPerfData = PerfDataFilename.getNumOccurrences();
  if (IsPerfData)
    PerfInputFile = PerfDataFilename;
  else
    PerfInputFile = PerfTraceFilename;
  std::unique_ptr<PerfReaderBase> Reader =
      PerfReaderBase::create(Binary.get(), PerfInputFile, IsPerfData);
  Reader->parsePerfTraces();

  if (SkipSymbolization)
    return EXIT_SUCCESS;

  std::unique_ptr<ProfileGeneratorBase> Generator =
      ProfileGeneratorBase::create(Binary.get(), Reader->getSampleCounters(),
                                   Reader->profileIsCS());
  Generator->generateProfile();
  Generator->write();

  return EXIT_SUCCESS;
}
