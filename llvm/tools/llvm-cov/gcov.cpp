//===- gcov.cpp - GCOV compatible LLVM coverage tool ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/GCOV.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include <system_error>
using namespace llvm;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "GcovOpts.inc"
#undef OPTION
};

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "GcovOpts.inc"

class GcovOptTable : public opt::OptTable {
public:
  GcovOptTable() : OptTable(optionTables()) {
    setGroupedShortOptions(true);
    setDashDashParsing(true);
  }
};
} // namespace

static void reportCoverage(StringRef SourceFile, StringRef ObjectDir,
                           StringRef InputGCNO, StringRef InputGCDA,
                           bool DumpGCOV, const GCOV::Options &Options) {
  SmallString<128> CoverageFileStem(ObjectDir);
  if (CoverageFileStem.empty()) {
    // If no directory was specified with -o, look next to the source file.
    CoverageFileStem = sys::path::parent_path(SourceFile);
    sys::path::append(CoverageFileStem, sys::path::stem(SourceFile));
  } else if (sys::fs::is_directory(ObjectDir))
    // A directory name was given. Use it and the source file name.
    sys::path::append(CoverageFileStem, sys::path::stem(SourceFile));
  else
    // A file was given. Ignore the source file and look next to this file.
    sys::path::replace_extension(CoverageFileStem, "");

  std::string GCNO = InputGCNO.empty() ? std::string(CoverageFileStem) + ".gcno"
                                       : InputGCNO.str();
  std::string GCDA = InputGCDA.empty() ? std::string(CoverageFileStem) + ".gcda"
                                       : InputGCDA.str();
  GCOVFile GF;

  // Open .gcda and .gcda without requiring a NUL terminator. The concurrent
  // modification may nullify the NUL terminator condition.
  ErrorOr<std::unique_ptr<MemoryBuffer>> GCNO_Buff =
      MemoryBuffer::getFileOrSTDIN(GCNO, /*IsText=*/false,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = GCNO_Buff.getError()) {
    errs() << GCNO << ": " << EC.message() << "\n";
    return;
  }
  GCOVBuffer GCNO_GB(GCNO_Buff.get().get());
  if (!GF.readGCNO(GCNO_GB)) {
    errs() << "Invalid .gcno File!\n";
    return;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> GCDA_Buff =
      MemoryBuffer::getFileOrSTDIN(GCDA, /*IsText=*/false,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = GCDA_Buff.getError()) {
    if (EC != errc::no_such_file_or_directory) {
      errs() << GCDA << ": " << EC.message() << "\n";
      return;
    }
    // Clear the filename to make it clear we didn't read anything.
    GCDA = "-";
  } else {
    GCOVBuffer gcda_buf(GCDA_Buff.get().get());
    if (!gcda_buf.readGCDAFormat())
      errs() << GCDA << ":not a gcov data file\n";
    else if (!GF.readGCDA(gcda_buf))
      errs() << "Invalid .gcda File!\n";
  }

  if (DumpGCOV)
    GF.print(errs());

  gcovOneInput(Options, SourceFile, GCNO, GCDA, GF);
}

int gcovMain(int argc, const char *argv[]) {
  StringRef ToolName = sys::path::filename(argv[0]);
  auto Error = [&](const Twine &Msg) {
    WithColor::error(errs(), ToolName) << Msg << '\n';
    exit(1);
  };
  BumpPtrAllocator A;
  StringSaver Saver(A);
  GcovOptTable Tbl;
  opt::InputArgList Args =
      Tbl.parseArgs(argc, const_cast<char **>(argv), OPT_UNKNOWN, Saver, Error);
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(), (ToolName + " [options] SOURCEFILE").str().c_str(),
                  "LLVM code coverage tool");
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    cl::PrintVersionMessage();
    return 0;
  }
  std::vector<std::string> SourceFiles = Args.getAllArgValues(OPT_INPUT);
  if (SourceFiles.empty())
    Error("no source file specified");

  GCOV::Options Options(
      Args.hasArg(OPT_all_blocks), Args.hasArg(OPT_branch_probabilities),
      Args.hasArg(OPT_branch_counts), Args.hasArg(OPT_function_summaries),
      Args.hasArg(OPT_preserve_paths), Args.hasArg(OPT_unconditional_branches),
      Args.hasArg(OPT_intermediate_format), Args.hasArg(OPT_long_file_names),
      Args.hasArg(OPT_demangled_names), Args.hasArg(OPT_no_output),
      Args.hasArg(OPT_relative_only), Args.hasArg(OPT_print_stdout),
      Args.hasArg(OPT_hash_filenames),
      Args.getLastArgValue(OPT_source_prefix_EQ).str());

  StringRef ObjectDir = Args.getLastArgValue(OPT_object_directory_EQ);
  StringRef InputGCNO = Args.getLastArgValue(OPT_gcno_EQ);
  StringRef InputGCDA = Args.getLastArgValue(OPT_gcda_EQ);
  bool DumpGCOV = Args.hasArg(OPT_dump_gcov);
  for (const std::string &SourceFile : SourceFiles)
    reportCoverage(SourceFile, ObjectDir, InputGCNO, InputGCDA, DumpGCOV,
                   Options);
  return 0;
}
