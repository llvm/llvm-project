//===---- SimpleRemoteEPCServer.h - EPC over abstract channel ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EPC over simple abstract channel.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_SIMPLEREMOTEEPCSERVER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_SIMPLEREMOTEEPCSERVER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>

namespace llvm {
namespace orc {

/// A simple EPC server implementation.
class SimpleRemoteEPCServer : public SimpleRemoteEPCTransportClient {
public:
  using ReportErrorFunction = unique_function<void(Error)>;

  class Dispatcher {
  public:
    virtual ~Dispatcher();
    virtual void dispatch(unique_function<void()> Work) = 0;
    virtual void shutdown() = 0;
  };

#if LLVM_ENABLE_THREADS
  class ThreadDispatcher : public Dispatcher {
  public:
    void dispatch(unique_function<void()> Work) override;
    void shutdown() override;

  private:
    std::mutex DispatchMutex;
    bool Running = true;
    size_t Outstanding = 0;
    std::condition_variable OutstandingCV;
  };
#endif

  static StringMap<ExecutorAddress> defaultBootstrapSymbols();

  template <typename TransportT, typename... TransportTCtorArgTs>
  static Expected<std::unique_ptr<SimpleRemoteEPCServer>>
  Create(std::unique_ptr<Dispatcher> D,
         StringMap<ExecutorAddress> BootstrapSymbols,
         TransportTCtorArgTs &&...TransportTCtorArgs) {
    auto SREPCServer = std::make_unique<SimpleRemoteEPCServer>();
    SREPCServer->D = std::move(D);
    SREPCServer->ReportError = [](Error Err) {
      logAllUnhandledErrors(std::move(Err), errs(), "SimpleRemoteEPCServer ");
    };
    auto T = TransportT::Create(
        *SREPCServer, std::forward<TransportTCtorArgTs>(TransportTCtorArgs)...);
    if (!T)
      return T.takeError();
    SREPCServer->T = std::move(*T);
    if (auto Err = SREPCServer->sendSetupMessage(std::move(BootstrapSymbols)))
      return std::move(Err);
    return std::move(SREPCServer);
  }

  /// Set an error reporter for this server.
  void setErrorReporter(ReportErrorFunction ReportError) {
    this->ReportError = std::move(ReportError);
  }

  /// Call to handle an incoming message.
  ///
  /// Returns 'Disconnect' if the message is a 'detach' message from the remote
  /// otherwise returns 'Continue'. If the server has moved to an error state,
  /// returns an error, which should be reported and treated as a 'Disconnect'.
  Expected<HandleMessageAction>
  handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                ExecutorAddress TagAddr,
                SimpleRemoteEPCArgBytesVector ArgBytes) override;

  Error waitForDisconnect();

  void handleDisconnect(Error Err) override;

private:
  Error sendSetupMessage(StringMap<ExecutorAddress> BootstrapSymbols);

  Error handleResult(uint64_t SeqNo, ExecutorAddress TagAddr,
                     SimpleRemoteEPCArgBytesVector ArgBytes);
  void handleCallWrapper(uint64_t RemoteSeqNo, ExecutorAddress TagAddr,
                         SimpleRemoteEPCArgBytesVector ArgBytes);

  static shared::detail::CWrapperFunctionResult
  loadDylibWrapper(const char *ArgData, size_t ArgSize);

  static shared::detail::CWrapperFunctionResult
  lookupSymbolsWrapper(const char *ArgData, size_t ArgSize);

  Expected<tpctypes::DylibHandle> loadDylib(const std::string &Path,
                                            uint64_t Mode);

  Expected<std::vector<std::vector<ExecutorAddress>>>
  lookupSymbols(const std::vector<RemoteSymbolLookup> &L);

  shared::WrapperFunctionResult
  doJITDispatch(const void *FnTag, const char *ArgData, size_t ArgSize);

  static shared::detail::CWrapperFunctionResult
  jitDispatchEntry(void *DispatchCtx, const void *FnTag, const char *ArgData,
                   size_t ArgSize);

  uint64_t getNextSeqNo() { return NextSeqNo++; }
  void releaseSeqNo(uint64_t) {}

  using PendingJITDispatchResultsMap =
      DenseMap<uint64_t, std::promise<shared::WrapperFunctionResult> *>;

  std::mutex ServerStateMutex;
  std::condition_variable ShutdownCV;
  enum { ServerRunning, ServerShuttingDown, ServerShutDown } RunState;
  Error ShutdownErr = Error::success();
  std::unique_ptr<SimpleRemoteEPCTransport> T;
  std::unique_ptr<Dispatcher> D;
  ReportErrorFunction ReportError;

  uint64_t NextSeqNo = 0;
  PendingJITDispatchResultsMap PendingJITDispatchResults;
  std::vector<sys::DynamicLibrary> Dylibs;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_SIMPLEREMOTEEPCSERVER_H
