//===---- SimpleRemoteEPC.h - Simple remote executor control ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple remote executor process control.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEEPC_H
#define LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEEPC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

namespace llvm {
namespace orc {

class SimpleRemoteEPC : public ExecutorProcessControl,
                        public SimpleRemoteEPCTransportClient {
public:
  /// Create a SimpleRemoteEPC using the given transport type and args.
  template <typename TransportT, typename... TransportTCtorArgTs>
  static Expected<std::unique_ptr<SimpleRemoteEPC>>
  Create(TransportTCtorArgTs &&...TransportTCtorArgs) {
    std::unique_ptr<SimpleRemoteEPC> SREPC(
        new SimpleRemoteEPC(std::make_shared<SymbolStringPool>()));

    // Prepare for setup packet.
    std::promise<MSVCPExpected<SimpleRemoteEPCExecutorInfo>> EIP;
    auto EIF = EIP.get_future();
    SREPC->prepareToReceiveSetupMessage(EIP);
    auto T = TransportT::Create(
        *SREPC, std::forward<TransportTCtorArgTs>(TransportTCtorArgs)...);
    if (!T)
      return T.takeError();
    auto EI = EIF.get();
    if (!EI) {
      (*T)->disconnect();
      return EI.takeError();
    }
    if (auto Err = SREPC->setup(std::move(*T), std::move(*EI)))
      return joinErrors(std::move(Err), SREPC->disconnect());
    return std::move(SREPC);
  }

  SimpleRemoteEPC(const SimpleRemoteEPC &) = delete;
  SimpleRemoteEPC &operator=(const SimpleRemoteEPC &) = delete;
  SimpleRemoteEPC(SimpleRemoteEPC &&) = delete;
  SimpleRemoteEPC &operator=(SimpleRemoteEPC &) = delete;
  ~SimpleRemoteEPC();

  /// Called at the end of the construction process to set up the instance.
  ///
  /// Override to set up custom memory manager and/or memory access objects.
  /// This method must be called at the *end* of the subclass's
  /// implementation.
  virtual Error setup(std::unique_ptr<SimpleRemoteEPCTransport> T,
                      const SimpleRemoteEPCExecutorInfo &EI);

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override;

  Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                              ArrayRef<std::string> Args) override;

  void callWrapperAsync(SendResultFunction OnComplete,
                        JITTargetAddress WrapperFnAddr,
                        ArrayRef<char> ArgBuffer) override;

  Error disconnect() override;

  Expected<HandleMessageAction>
  handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                ExecutorAddress TagAddr,
                SimpleRemoteEPCArgBytesVector ArgBytes) override;

  void handleDisconnect(Error Err) override;

protected:
  void setMemoryManager(std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);
  void setMemoryAccess(std::unique_ptr<MemoryAccess> MemAccess);

private:
  SimpleRemoteEPC(std::shared_ptr<SymbolStringPool> SSP)
      : ExecutorProcessControl(std::move(SSP)) {}

  Error setupDefaultMemoryManager(const SimpleRemoteEPCExecutorInfo &EI);
  Error setupDefaultMemoryAccess(const SimpleRemoteEPCExecutorInfo &EI);

  Error handleSetup(uint64_t SeqNo, ExecutorAddress TagAddr,
                    SimpleRemoteEPCArgBytesVector ArgBytes);
  void prepareToReceiveSetupMessage(
      std::promise<MSVCPExpected<SimpleRemoteEPCExecutorInfo>> &ExecInfoP);

  Error handleResult(uint64_t SeqNo, ExecutorAddress TagAddr,
                     SimpleRemoteEPCArgBytesVector ArgBytes);
  void handleCallWrapper(uint64_t RemoteSeqNo, ExecutorAddress TagAddr,
                         SimpleRemoteEPCArgBytesVector ArgBytes);

  uint64_t getNextSeqNo() { return NextSeqNo++; }
  void releaseSeqNo(uint64_t SeqNo) {}

  using PendingCallWrapperResultsMap = DenseMap<uint64_t, SendResultFunction>;

  std::atomic_bool Disconnected{false};
  std::mutex SimpleRemoteEPCMutex;
  std::unique_ptr<SimpleRemoteEPCTransport> T;
  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
  std::unique_ptr<MemoryAccess> OwnedMemAccess;

  ExecutorAddress LoadDylibAddr;
  ExecutorAddress LookupSymbolsAddr;
  ExecutorAddress RunAsMainAddr;

  uint64_t NextSeqNo = 0;
  PendingCallWrapperResultsMap PendingCallWrapperResults;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SIMPLEREMOTEEPC_H
