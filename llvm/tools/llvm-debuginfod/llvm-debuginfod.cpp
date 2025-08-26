//===-- llvm-debuginfod.cpp - federating debuginfod server ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the llvm-debuginfod tool, which serves the debuginfod
/// protocol over HTTP. The tool periodically scans zero or more filesystem
/// directories for ELF binaries to serve, and federates requests for unknown
/// build IDs to the debuginfod servers set in the DEBUGINFOD_URLS environment
/// variable.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;

// Command-line option boilerplate.
namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class DebuginfodOptTable : public opt::GenericOptTable {
public:
  DebuginfodOptTable()
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};
} // end anonymous namespace

// Options
static unsigned Port;
static std::string HostInterface;
static int ScanInterval;
static double MinInterval;
static size_t MaxConcurrency;
static bool VerboseLogging;
static std::vector<std::string> ScanPaths;

ExitOnError ExitOnErr;

template <typename T>
static void parseIntArg(const opt::InputArgList &Args, int ID, T &Value,
                        T Default) {
  if (const opt::Arg *A = Args.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!llvm::to_integer(V, Value, 0)) {
      errs() << A->getSpelling() + ": expected an integer, but got '" + V + "'";
      exit(1);
    }
  } else {
    Value = Default;
  }
}

static void parseArgs(int argc, char **argv) {
  DebuginfodOptTable Tbl;
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
                  "llvm-debuginfod [options] <Directories to scan>",
                  ToolName.str().c_str());
    std::exit(0);
  }

  VerboseLogging = Args.hasArg(OPT_verbose_logging);
  ScanPaths = Args.getAllArgValues(OPT_INPUT);

  parseIntArg(Args, OPT_port, Port, 0u);
  parseIntArg(Args, OPT_scan_interval, ScanInterval, 300);
  parseIntArg(Args, OPT_max_concurrency, MaxConcurrency, size_t(0));

  if (const opt::Arg *A = Args.getLastArg(OPT_min_interval)) {
    StringRef V(A->getValue());
    if (!llvm::to_float(V, MinInterval)) {
      errs() << A->getSpelling() + ": expected a number, but got '" + V + "'";
      exit(1);
    }
  } else {
    MinInterval = 10.0;
  }

  HostInterface = Args.getLastArgValue(OPT_host_interface, "0.0.0.0");
}

int llvm_debuginfod_main(int argc, char **argv, const llvm::ToolContext &) {
  HTTPClient::initialize();
  parseArgs(argc, argv);

  SmallVector<StringRef, 1> Paths;
  llvm::append_range(Paths, ScanPaths);

  DefaultThreadPool Pool(hardware_concurrency(MaxConcurrency));
  DebuginfodLog Log;
  DebuginfodCollection Collection(Paths, Log, Pool, MinInterval);
  DebuginfodServer Server =
      ExitOnErr(DebuginfodServer::create(Log, Collection));
  if (!Port)
    Port = ExitOnErr(Server.Server.bind(HostInterface.c_str()));
  else
    ExitOnErr(Server.Server.bind(Port, HostInterface.c_str()));

  Log.push("Listening on port " + Twine(Port).str());

  Pool.async([&]() { ExitOnErr(Server.Server.listen()); });
  Pool.async([&]() {
    while (true) {
      DebuginfodLogEntry Entry = Log.pop();
      if (VerboseLogging) {
        outs() << Entry.Message << "\n";
        outs().flush();
      }
    }
  });
  if (Paths.size())
    ExitOnErr(Collection.updateForever(std::chrono::seconds(ScanInterval)));
  Pool.wait();
  llvm_unreachable("The ThreadPool should never finish running its tasks.");
}
