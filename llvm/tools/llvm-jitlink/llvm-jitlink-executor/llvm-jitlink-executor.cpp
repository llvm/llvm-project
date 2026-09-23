//===- llvm-jitlink-executor.cpp - Out-of-proc executor for llvm-jitlink -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple out-of-process executor for llvm-jitlink.
//
//===----------------------------------------------------------------------===//

#include "../ConnectionUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX, LLVM_ENABLE_THREADS
#include "llvm/ExecutionEngine/Orc/Shared/ConnectionSpec.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/DefaultHostBootstrapValues.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleRemoteEPCServer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/UnwindInfoManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <sstream>

using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

LLVM_ATTRIBUTE_USED void linkComponents() {
  errs() << (void *)&llvm_orc_registerEHFrameSectionAllocAction
         << (void *)&llvm_orc_deregisterEHFrameSectionAllocAction
         << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}

void printErrorAndExit(Twine ErrMsg) {
#ifndef NDEBUG
  const char *DebugOption = "[debug] ";
#else
  const char *DebugOption = "";
#endif

  errs() << "error: " << ErrMsg.str() << "\n\n"
         << "Usage:\n"
         << "  llvm-jitlink-executor " << DebugOption
         << "[test-jitloadergdb] socket:adopt=<sockfd> [args...]\n"
         << "  llvm-jitlink-executor " << DebugOption
         << "[test-jitloadergdb] tcp:connect=<host>:<port> [args...]\n"
         << "  llvm-jitlink-executor " << DebugOption
         << "[test-jitloadergdb] tcp:listen=<host>:<port> [args...]\n";
  exit(1);
}

#if LLVM_ENABLE_THREADS

// JITLink debug support plugins put information about JITed code in this GDB
// JIT Interface global from OrcTargetProcess.
extern "C" LLVM_ABI struct jit_descriptor __jit_debug_descriptor;

static void *findLastDebugDescriptorEntryPtr() {
  struct jit_code_entry *Last = __jit_debug_descriptor.first_entry;
  while (Last && Last->next_entry)
    Last = Last->next_entry;
  return Last;
}

#endif

Expected<std::unique_ptr<SimpleRemoteEPCServer>> createServerWithFD(int FD) {
#if LLVM_ENABLE_THREADS
  return SimpleRemoteEPCServer::Create<FDSimpleRemoteEPCTransport>(
      [](SimpleRemoteEPCServer::Setup &S) -> Error {
        S.setDispatcher(
            std::make_unique<SimpleRemoteEPCServer::ThreadDispatcher>());
        S.bootstrapSymbols() = SimpleRemoteEPCServer::defaultBootstrapSymbols();
        addDefaultBootstrapValuesForHostProcess(S.bootstrapMap(),
                                                S.bootstrapSymbols());
#ifdef __APPLE__
        if (UnwindInfoManager::TryEnable())
          UnwindInfoManager::addBootstrapSymbols(S.bootstrapSymbols());
#endif // __APPLE__
        S.services().push_back(
            std::make_unique<rt_bootstrap::SimpleExecutorMemoryManager>());
        S.services().push_back(
            std::make_unique<
                rt_bootstrap::ExecutorSharedMemoryMapperService>());
        return Error::success();
      },
      FD, FD);
#else
  llvm_unreachable("Not available on LLVM_ENABLE_THREADS=Off builds");
#endif // !LLVM_ENABLE_THREADS
}

Expected<std::unique_ptr<SimpleRemoteEPCServer>>
connectWithSocket(const ConnectionSpec &CS) {
  if (CS.getAction() != "adopt")
    return make_error<StringError>(
        "In " + CS.str() +
            ", the socket transport supports only the \"adopt\" action",
        inconvertibleErrorCode());

  int FD;
  if (CS.getDescriptor().getAsInteger(10, FD))
    return make_error<StringError>(
        "In " + CS.str() + ", " + CS.getDescriptor() + " is not an integer",
        inconvertibleErrorCode());

  return createServerWithFD(FD);
}

Expected<std::unique_ptr<SimpleRemoteEPCServer>>
connectWithTCPConnect(const ConnectionSpec &CS) {
#ifndef LLVM_ON_UNIX
  return make_error<StringError>("TCP connection not supported",
                                 inconvertibleErrorCode());
#else
  auto [Host, PortStr] = CS.getDescriptor().split(':');
  if (Host.empty() || PortStr.empty())
    return make_error<StringError>("In " + CS.str() +
                                       ", expected <host>:<port>",
                                   inconvertibleErrorCode());

  auto SockFD = connectTCPSocket(Host, PortStr);
  if (!SockFD)
    return make_error<StringError>("In " + CS.str() + ", " +
                                       toString(SockFD.takeError()),
                                   inconvertibleErrorCode());

  return createServerWithFD(*SockFD);
#endif // LLVM_ON_UNIX
}

Expected<std::unique_ptr<SimpleRemoteEPCServer>>
connectWithTCPListen(const ConnectionSpec &CS) {
#ifndef LLVM_ON_UNIX
  return make_error<StringError>("TCP connection not supported",
                                 inconvertibleErrorCode());
#else
  auto [Host, PortStr] = CS.getDescriptor().split(':');

  auto ListenFD = listenTCPSocket(Host, PortStr);
  if (!ListenFD)
    return make_error<StringError>("In " + CS.str() + ", " +
                                       toString(ListenFD.takeError()),
                                   inconvertibleErrorCode());

  auto SockFD = acceptTCPConnection(*ListenFD);
  if (!SockFD)
    return make_error<StringError>("In " + CS.str() + ", " +
                                       toString(SockFD.takeError()),
                                   inconvertibleErrorCode());

  return createServerWithFD(*SockFD);
#endif // LLVM_ON_UNIX
}

Expected<std::unique_ptr<SimpleRemoteEPCServer>>
connectWithTCP(const ConnectionSpec &CS) {
  if (CS.getAction() == "connect")
    return connectWithTCPConnect(CS);
  if (CS.getAction() == "listen")
    return connectWithTCPListen(CS);

  return make_error<StringError>("In " + CS.str() + ", unrecognized action \"" +
                                     CS.getAction() + "\"",
                                 inconvertibleErrorCode());
}

Expected<std::unique_ptr<SimpleRemoteEPCServer>>
createServer(const ConnectionSpec &CS) {
  if (CS.getTransport() == "socket")
    return connectWithSocket(CS);
  if (CS.getTransport() == "tcp")
    return connectWithTCP(CS);

  return make_error<StringError>("In " + CS.str() +
                                     ", unrecognized transport \"" +
                                     CS.getTransport() + "\"",
                                 inconvertibleErrorCode());
}

int main(int argc, char *argv[]) {
#if LLVM_ENABLE_THREADS

  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  unsigned FirstProgramArg = 1;

  if (argc < 2)
    printErrorAndExit("insufficient arguments");

  StringRef NextArg = argv[FirstProgramArg++];
#ifndef NDEBUG
  if (NextArg == "debug") {
    DebugFlag = true;
    NextArg = argv[FirstProgramArg++];
  }
#endif

  std::vector<StringRef> TestOutputFlags;
  while (NextArg.starts_with("test-")) {
    TestOutputFlags.push_back(NextArg);
    NextArg = argv[FirstProgramArg++];
  }

  if (llvm::is_contained(TestOutputFlags, "test-jitloadergdb"))
    fprintf(stderr, "__jit_debug_descriptor.last_entry = 0x%016" PRIx64 "\n",
            pointerToJITTargetAddress(findLastDebugDescriptorEntryPtr()));

  auto ConnSpec = ExitOnErr(ConnectionSpec::parse(NextArg));
  auto Server = ExitOnErr(createServer(ConnSpec));

  ExitOnErr(Server->waitForDisconnect());

  if (llvm::is_contained(TestOutputFlags, "test-jitloadergdb"))
    fprintf(stderr, "__jit_debug_descriptor.last_entry = 0x%016" PRIx64 "\n",
            pointerToJITTargetAddress(findLastDebugDescriptorEntryPtr()));

  return 0;

#else
  errs() << argv[0]
         << " error: this tool requires threads, but LLVM was "
            "built with LLVM_ENABLE_THREADS=Off\n";
  return 1;
#endif
}
