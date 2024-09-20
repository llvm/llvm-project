//===-- llvm-debuginfod-find.cpp - Simple CLI for libdebuginfod-client ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the llvm-debuginfod-find tool. This tool
/// queries the debuginfod servers in the DEBUGINFOD_URLS environment
/// variable (delimited by space (" ")) for the executable,
/// debuginfo, or specified source file of the binary matching the
/// given build-id.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Debuginfod/BuildIDFetcher.h"
#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LLVMDriver.h"

using namespace llvm;

// Command-line option boilerplate.
namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class DebuginfodFindOptTable : public opt::GenericOptTable {
public:
  DebuginfodFindOptTable() : GenericOptTable(InfoTable) {}
};

} // end anonymous namespace

static std::string InputBuildID;
static bool FetchExecutable;
static bool FetchDebuginfo;
static std::string FetchSource;
static bool DumpToStdout;
static std::vector<std::string> DebugFileDirectory;

static void parseArgs(int argc, char **argv) {
  DebuginfodFindOptTable Tbl;
  llvm::StringRef ToolName = argv[0];
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver{A};
  opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        llvm::errs() << Msg << '\n';
        std::exit(1);
      });

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(llvm::outs(),
                  "llvm-debuginfod-find [options] <input build_id>",
                  ToolName.str().c_str());
    std::exit(0);
  }

  InputBuildID = Args.getLastArgValue(OPT_INPUT);

  FetchExecutable = Args.hasArg(OPT_fetch_executable);
  FetchDebuginfo = Args.hasArg(OPT_fetch_debuginfo);
  DumpToStdout = Args.hasArg(OPT_dump_to_stdout);
  FetchSource = Args.getLastArgValue(OPT_fetch_source, "");
  DebugFileDirectory = Args.getAllArgValues(OPT_debug_file_directory);
}

[[noreturn]] static void helpExit() {
  errs() << "Must specify exactly one of --executable, "
            "--source=/path/to/file, or --debuginfo.\n";
  exit(1);
}

/*
cl::OptionCategory DebuginfodFindCategory("llvm-debuginfod-find Options");

cl::opt<std::string> InputBuildID(cl::Positional, cl::Required,
                                  cl::desc("<input build_id>"), cl::init("-"),
                                  cl::cat(DebuginfodFindCategory));

static cl::opt<bool>
    FetchExecutable("executable", cl::init(false),
                    cl::desc("If set, fetch a binary file associated with this "
                             "build id, containing the executable sections."),
                    cl::cat(DebuginfodFindCategory));

static cl::opt<bool>
    FetchDebuginfo("debuginfo", cl::init(false),
                   cl::desc("If set, fetch a binary file associated with this "
                            "build id, containing the debuginfo sections."),
                   cl::cat(DebuginfodFindCategory));

static cl::opt<std::string> FetchSource(
    "source", cl::init(""),
    cl::desc("Fetch a source file associated with this build id, which is at "
             "this relative path relative to the compilation directory."),
    cl::cat(DebuginfodFindCategory));

static cl::opt<bool>
    DumpToStdout("dump", cl::init(false),
                 cl::desc("If set, dumps the contents of the fetched artifact "
                          "to standard output. Otherwise, dumps the absolute "
                          "path to the cached artifact on disk."),
                 cl::cat(DebuginfodFindCategory));

static cl::list<std::string> DebugFileDirectory(
    "debug-file-directory",
    cl::desc("Path to directory where to look for debug files."),
    cl::cat(DebuginfodFindCategory));

*/

ExitOnError ExitOnDebuginfodFindError;

static std::string fetchDebugInfo(object::BuildIDRef BuildID);

int llvm_debuginfod_find_main(int argc, char **argv,
                              const llvm::ToolContext &) {
  // InitLLVM X(argc, argv);
  HTTPClient::initialize();
  parseArgs(argc, argv);

  if (FetchExecutable + FetchDebuginfo + (FetchSource != "") != 1)
    helpExit();

  std::string IDString;
  if (!tryGetFromHex(InputBuildID, IDString)) {
    errs() << "Build ID " << InputBuildID << " is not a hex string.\n";
    exit(1);
  }
  object::BuildID ID(IDString.begin(), IDString.end());

  std::string Path;
  if (FetchSource != "")
    Path =
        ExitOnDebuginfodFindError(getCachedOrDownloadSource(ID, FetchSource));
  else if (FetchExecutable)
    Path = ExitOnDebuginfodFindError(getCachedOrDownloadExecutable(ID));
  else if (FetchDebuginfo)
    Path = fetchDebugInfo(ID);
  else
    llvm_unreachable("We have already checked that exactly one of the above "
                     "conditions is true.");

  if (DumpToStdout) {
    // Print the contents of the artifact.
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(
        Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    ExitOnDebuginfodFindError(errorCodeToError(Buf.getError()));
    outs() << Buf.get()->getBuffer();
  } else
    // Print the path to the cached artifact file.
    outs() << Path << "\n";

  return 0;
}

// Find a debug file in local build ID directories and via debuginfod.
std::string fetchDebugInfo(object::BuildIDRef BuildID) {
  if (std::optional<std::string> Path =
          DebuginfodFetcher(DebugFileDirectory).fetch(BuildID))
    return *Path;
  errs() << "Build ID " << llvm::toHex(BuildID, /*Lowercase=*/true)
         << " could not be found.\n";
  exit(1);
}
