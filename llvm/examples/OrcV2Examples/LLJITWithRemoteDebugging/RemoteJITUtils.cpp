//===-- RemoteJITUtils.cpp - Utilities for remote-JITing --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemoteJITUtils.h"

#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/OrcRPCExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/RPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

using namespace llvm;
using namespace llvm::orc;

namespace llvm {
namespace orc {

class RemoteExecutorProcessControl
    : public OrcRPCExecutorProcessControlBase<
          shared::MultiThreadedRPCEndpoint<JITLinkExecutor::RPCChannel>> {
public:
  using RPCChannel = JITLinkExecutor::RPCChannel;
  using RPCEndpoint = shared::MultiThreadedRPCEndpoint<RPCChannel>;

private:
  using ThisT = RemoteExecutorProcessControl;
  using BaseT = OrcRPCExecutorProcessControlBase<RPCEndpoint>;
  using MemoryAccess = OrcRPCEPCMemoryAccess<ThisT>;
  using MemoryManager = OrcRPCEPCJITLinkMemoryManager<ThisT>;

public:
  using BaseT::initializeORCRPCEPCBase;

  RemoteExecutorProcessControl(ExecutionSession &ES,
                               std::unique_ptr<RPCChannel> Channel,
                               std::unique_ptr<RPCEndpoint> Endpoint);

  void initializeMemoryManagement();
  Error disconnect() override;

private:
  std::unique_ptr<RPCChannel> Channel;
  std::unique_ptr<RPCEndpoint> Endpoint;
  std::unique_ptr<MemoryAccess> OwnedMemAccess;
  std::unique_ptr<MemoryManager> OwnedMemMgr;
  std::atomic<bool> Finished{false};
  std::thread ListenerThread;
};

RemoteExecutorProcessControl::RemoteExecutorProcessControl(
    ExecutionSession &ES, std::unique_ptr<RPCChannel> Channel,
    std::unique_ptr<RPCEndpoint> Endpoint)
    : BaseT(ES.getSymbolStringPool(), *Endpoint,
            [&ES](Error Err) { ES.reportError(std::move(Err)); }),
      Channel(std::move(Channel)), Endpoint(std::move(Endpoint)) {

  ListenerThread = std::thread([&]() {
    while (!Finished) {
      if (auto Err = this->Endpoint->handleOne()) {
        reportError(std::move(Err));
        return;
      }
    }
  });
}

void RemoteExecutorProcessControl::initializeMemoryManagement() {
  OwnedMemAccess = std::make_unique<MemoryAccess>(*this);
  OwnedMemMgr = std::make_unique<MemoryManager>(*this);

  // Base class needs non-owning access.
  MemAccess = OwnedMemAccess.get();
  MemMgr = OwnedMemMgr.get();
}

Error RemoteExecutorProcessControl::disconnect() {
  std::promise<MSVCPError> P;
  auto F = P.get_future();
  auto Err = closeConnection([&](Error Err) -> Error {
    P.set_value(std::move(Err));
    Finished = true;
    return Error::success();
  });
  ListenerThread.join();
  return joinErrors(std::move(Err), F.get());
}

} // namespace orc
} // namespace llvm

JITLinkExecutor::JITLinkExecutor() = default;
JITLinkExecutor::~JITLinkExecutor() = default;

Expected<std::unique_ptr<ObjectLayer>>
JITLinkExecutor::operator()(ExecutionSession &ES, const Triple &TT) {
  return std::make_unique<ObjectLinkingLayer>(ES, EPC->getMemMgr());
}

Error JITLinkExecutor::addDebugSupport(ObjectLayer &ObjLayer) {
  auto Registrar = createJITLoaderGDBRegistrar(*EPC);
  if (!Registrar)
    return Registrar.takeError();

  cast<ObjectLinkingLayer>(&ObjLayer)->addPlugin(
      std::make_unique<DebugObjectManagerPlugin>(ObjLayer.getExecutionSession(),
                                                 std::move(*Registrar)));

  return Error::success();
}

Expected<std::unique_ptr<DefinitionGenerator>>
JITLinkExecutor::loadDylib(StringRef RemotePath) {
  if (auto Handle = EPC->loadDylib(RemotePath.data()))
    return std::make_unique<EPCDynamicLibrarySearchGenerator>(*EPC, *Handle);
  else
    return Handle.takeError();
}

Expected<int> JITLinkExecutor::runAsMain(JITEvaluatedSymbol MainSym,
                                         ArrayRef<std::string> Args) {
  return EPC->runAsMain(MainSym.getAddress(), Args);
}

Error JITLinkExecutor::disconnect() { return EPC->disconnect(); }

static std::string defaultPath(const char *HostArgv0, StringRef ExecutorName) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void *)(intptr_t)defaultPath;
  SmallString<256> FullName(sys::fs::getMainExecutable(HostArgv0, P));
  sys::path::remove_filename(FullName);
  sys::path::append(FullName, ExecutorName);
  return FullName.str().str();
}

Expected<std::unique_ptr<ChildProcessJITLinkExecutor>>
JITLinkExecutor::FindLocal(const char *HostArgv) {
  std::string BestGuess = defaultPath(HostArgv, "llvm-jitlink-executor");
  auto Executor = CreateLocal(BestGuess);
  if (!Executor) {
    consumeError(Executor.takeError());
    return make_error<StringError>(
        formatv("Unable to find usable executor: {0}", BestGuess),
        inconvertibleErrorCode());
  }
  return Executor;
}

Expected<std::unique_ptr<ChildProcessJITLinkExecutor>>
JITLinkExecutor::CreateLocal(std::string ExecutablePath) {
  if (!sys::fs::can_execute(ExecutablePath))
    return make_error<StringError>(
        formatv("Specified executor invalid: {0}", ExecutablePath),
        inconvertibleErrorCode());
  return std::unique_ptr<ChildProcessJITLinkExecutor>(
      new ChildProcessJITLinkExecutor(std::move(ExecutablePath)));
}

TCPSocketJITLinkExecutor::TCPSocketJITLinkExecutor(
    std::unique_ptr<RemoteExecutorProcessControl> EPC) {
  this->EPC = std::move(EPC);
}

#ifndef LLVM_ON_UNIX

// FIXME: Add support for Windows.
Error ChildProcessJITLinkExecutor::launch(ExecutionSession &ES) {
  return make_error<StringError>(
      "Remote JITing not yet supported on non-unix platforms",
      inconvertibleErrorCode());
}

// FIXME: Add support for Windows.
Expected<std::unique_ptr<TCPSocketJITLinkExecutor>>
JITLinkExecutor::ConnectTCPSocket(StringRef NetworkAddress,
                                  ExecutionSession &ES) {
  return make_error<StringError>(
      "Remote JITing not yet supported on non-unix platforms",
      inconvertibleErrorCode());
}

#else

Error ChildProcessJITLinkExecutor::launch(ExecutionSession &ES) {
  constexpr int ReadEnd = 0;
  constexpr int WriteEnd = 1;

  // Pipe FDs.
  int ToExecutor[2];
  int FromExecutor[2];

  // Create pipes to/from the executor..
  if (pipe(ToExecutor) != 0 || pipe(FromExecutor) != 0)
    return make_error<StringError>("Unable to create pipe for executor",
                                   inconvertibleErrorCode());

  ProcessID = fork();
  if (ProcessID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(ToExecutor[WriteEnd]);
    close(FromExecutor[ReadEnd]);

    // Execute the child process.
    std::unique_ptr<char[]> ExecPath, FDSpecifier;
    {
      ExecPath = std::make_unique<char[]>(ExecutablePath.size() + 1);
      strcpy(ExecPath.get(), ExecutablePath.data());

      std::string FDSpecifierStr("filedescs=");
      FDSpecifierStr += utostr(ToExecutor[ReadEnd]);
      FDSpecifierStr += ',';
      FDSpecifierStr += utostr(FromExecutor[WriteEnd]);
      FDSpecifier = std::make_unique<char[]>(FDSpecifierStr.size() + 1);
      strcpy(FDSpecifier.get(), FDSpecifierStr.c_str());
    }

    char *const Args[] = {ExecPath.get(), FDSpecifier.get(), nullptr};
    int RC = execvp(ExecPath.get(), Args);
    if (RC != 0)
      return make_error<StringError>(
          "Unable to launch out-of-process executor '" + ExecutablePath + "'\n",
          inconvertibleErrorCode());

    llvm_unreachable("Fork won't return in success case");
  }
  // else we're the parent...

  // Close the child ends of the pipes
  close(ToExecutor[ReadEnd]);
  close(FromExecutor[WriteEnd]);

  auto Channel =
      std::make_unique<RPCChannel>(FromExecutor[ReadEnd], ToExecutor[WriteEnd]);
  auto Endpoint = std::make_unique<RemoteExecutorProcessControl::RPCEndpoint>(
      *Channel, true);

  EPC = std::make_unique<RemoteExecutorProcessControl>(ES, std::move(Channel),
                                                       std::move(Endpoint));

  if (auto Err = EPC->initializeORCRPCEPCBase())
    return joinErrors(std::move(Err), EPC->disconnect());

  EPC->initializeMemoryManagement();

  shared::registerStringError<RPCChannel>();
  return Error::success();
}

static Expected<int> connectTCPSocketImpl(std::string Host,
                                          std::string PortStr) {
  addrinfo *AI;
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_NUMERICSERV;

  if (int EC = getaddrinfo(Host.c_str(), PortStr.c_str(), &Hints, &AI))
    return make_error<StringError>(
        formatv("address resolution failed ({0})", gai_strerror(EC)),
        inconvertibleErrorCode());

  // Cycle through the returned addrinfo structures and connect to the first
  // reachable endpoint.
  int SockFD;
  addrinfo *Server;
  for (Server = AI; Server != nullptr; Server = Server->ai_next) {
    // If socket fails, maybe it's because the address family is not supported.
    // Skip to the next addrinfo structure.
    if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0)
      continue;

    // If connect works, we exit the loop with a working socket.
    if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
      break;

    close(SockFD);
  }
  freeaddrinfo(AI);

  // Did we reach the end of the loop without connecting to a valid endpoint?
  if (Server == nullptr)
    return make_error<StringError>("invalid hostname",
                                   inconvertibleErrorCode());

  return SockFD;
}

Expected<std::unique_ptr<TCPSocketJITLinkExecutor>>
JITLinkExecutor::ConnectTCPSocket(StringRef NetworkAddress,
                                  ExecutionSession &ES) {
  auto CreateErr = [NetworkAddress](StringRef Details) {
    return make_error<StringError>(
        formatv("Failed to connect TCP socket '{0}': {1}", NetworkAddress,
                Details),
        inconvertibleErrorCode());
  };

  StringRef Host, PortStr;
  std::tie(Host, PortStr) = NetworkAddress.split(':');
  if (Host.empty())
    return CreateErr("host name cannot be empty");
  if (PortStr.empty())
    return CreateErr("port cannot be empty");
  int Port = 0;
  if (PortStr.getAsInteger(10, Port))
    return CreateErr("port number is not a valid integer");

  Expected<int> SockFD = connectTCPSocketImpl(Host.str(), PortStr.str());
  if (!SockFD)
    return CreateErr(toString(SockFD.takeError()));

  auto Channel = std::make_unique<RPCChannel>(*SockFD, *SockFD);
  auto Endpoint = std::make_unique<RemoteExecutorProcessControl::RPCEndpoint>(
      *Channel, true);

  auto EPC = std::make_unique<RemoteExecutorProcessControl>(
      ES, std::move(Channel), std::move(Endpoint));

  if (auto Err = EPC->initializeORCRPCEPCBase())
    return joinErrors(std::move(Err), EPC->disconnect());

  EPC->initializeMemoryManagement();
  shared::registerStringError<RPCChannel>();

  return std::unique_ptr<TCPSocketJITLinkExecutor>(
      new TCPSocketJITLinkExecutor(std::move(EPC)));
}

#endif
