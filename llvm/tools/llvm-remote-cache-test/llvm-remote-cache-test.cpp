//===-- llvm-remote-cache-test.cpp - Remote Cache Service -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A server for of the remote cache service protocol, for testing purposes.
//
//===----------------------------------------------------------------------===//

#include "RemoteCacheProvider.h"
#include "RemoteCacheServer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <thread>

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

using namespace llvm;
using namespace remote_cache_test;

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  cl::OptionCategory OptCategory("llvm-remote-cache-test options");
  cl::extrahelp MoreHelp(
      "\n"
      "Implements the remote caching gRPC protocol with an on-disk LLVMCAS as "
      "the caching backend.\n"
      "Supports two modes of operation:\n"
      "  * Start and run continuously as a server, or\n"
      "  * Start and run the given command, after setting the "
      "LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH environment variable.\n"
      "    It exits with the same exit code as the command."
      "\n");

  cl::opt<std::string> CachePath("cache-path", cl::desc("Cache data path"),
                                 cl::Required, cl::value_desc("path"),
                                 cl::cat(OptCategory));
  // This is here only to improve the help message (for "USAGE:" line).
  cl::list<std::string> Inputs(cl::Positional, cl::desc("[-- command ...]"));

  auto Sep = std::find_if(Argv + 1, Argv + Argc, [](const char *Arg) {
    return StringRef(Arg) == "--";
  });
  bool ServerMode = Sep == Argv + Argc;

  cl::HideUnrelatedOptions(OptCategory);
  cl::ParseCommandLineOptions(Sep - Argv, Argv, "llvm-remote-cache-test");

  SmallString<128> SocketPath{CachePath};
  sys::path::append(SocketPath, "sock");

  static ExitOnError ExitOnErr("llvm-remote-cache-test: ");

  if (ServerMode) {
    outs() << "Server listening on " << SocketPath << '\n';
    RemoteCacheServer Server = createServer(
        SocketPath, ExitOnErr(createLLVMCASCacheProvider(CachePath)));
    Server.Run();
    return 0;
  }

  SmallVector<const char *, 256> CmdArgs(Sep + 1, Argv + Argc);
  if (CmdArgs.empty()) {
    // No command arguments.
    cl::PrintHelpMessage();
    return 1;
  }

  ErrorOr<std::string> ExecPathOrErr = sys::findProgramByName(CmdArgs.front());
  if (!ExecPathOrErr) {
    errs() << "error: cannot find executable " << CmdArgs.front() << '\n';
    return 1;
  }
  std::string ExecPath = std::move(*ExecPathOrErr);
  CmdArgs[0] = ExecPath.c_str();

  SmallVector<StringRef, 32> RefArgs;
  RefArgs.reserve(CmdArgs.size());
  for (const char *Arg : CmdArgs) {
    RefArgs.push_back(Arg);
  }

  RemoteCacheServer Server = createServer(
      SocketPath, ExitOnErr(createLLVMCASCacheProvider(CachePath)));
  std::thread ServerThread([&Server]() { Server.Run(); });

  setenv("LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH", SocketPath.c_str(), true);

  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(RefArgs.front(), RefArgs, /*Env*/ None,
                                   /*Redirects*/ {}, /*SecondsToWait*/ 0,
                                   /*MemoryLimit*/ 0, &ErrMsg);
  if (!ErrMsg.empty()) {
    errs() << "error: failed executing command: " << ErrMsg << '\n';
  }

  Server.Shutdown();
  ServerThread.join();

  return Result;
}
