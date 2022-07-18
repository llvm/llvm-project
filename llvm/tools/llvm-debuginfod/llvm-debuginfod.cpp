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

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ThreadPool.h"

using namespace llvm;

cl::OptionCategory DebuginfodCategory("llvm-debuginfod Options");

static cl::list<std::string> ScanPaths(cl::Positional,
                                       cl::desc("<Directories to scan>"),
                                       cl::cat(DebuginfodCategory));

static cl::opt<unsigned>
    Port("p", cl::init(0),
         cl::desc("Port to listen on. Set to 0 to bind to any available port."),
         cl::cat(DebuginfodCategory));

static cl::opt<std::string>
    HostInterface("i", cl::init("0.0.0.0"),
                  cl::desc("Host interface to bind to."),
                  cl::cat(DebuginfodCategory));

static cl::opt<int>
    ScanInterval("t", cl::init(300),
                 cl::desc("Number of seconds to wait between subsequent "
                          "automated scans of the filesystem."),
                 cl::cat(DebuginfodCategory));

static cl::opt<double> MinInterval(
    "m", cl::init(10),
    cl::desc(
        "Minimum number of seconds to wait before an on-demand update can be "
        "triggered by a request for a buildid which is not in the collection."),
    cl::cat(DebuginfodCategory));

static cl::opt<size_t>
    MaxConcurrency("c", cl::init(0),
                   cl::desc("Maximum number of files to scan concurrently. If "
                            "0, use the hardware concurrency."),
                   cl::cat(DebuginfodCategory));

static cl::opt<bool> VerboseLogging("v", cl::init(false),
                                    cl::desc("Enable verbose logging."),
                                    cl::cat(DebuginfodCategory));

ExitOnError ExitOnErr;

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  HTTPClient::initialize();
  cl::HideUnrelatedOptions({&DebuginfodCategory});
  cl::ParseCommandLineOptions(argc, argv);

  SmallVector<StringRef, 1> Paths;
  for (const std::string &Path : ScanPaths)
    Paths.push_back(Path);

  ThreadPool Pool(hardware_concurrency(MaxConcurrency));
  DebuginfodLog Log;
  DebuginfodCollection Collection(Paths, Log, Pool, MinInterval);
  DebuginfodServer Server(Log, Collection);

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
