//===---- InProcessEPC.h - In-process EPC for new ORC runtime ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExecutorProcessControl implementation for in-process JITs that use the new
// ORC runtime (orc-rt). Interfaces with orc_rt::InProcessControllerAccess via
// direct function calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INPROCESSEPC_H
#define LLVM_EXECUTIONENGINE_ORC_INPROCESSEPC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/InProcessMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/Compiler.h"

#include <memory>
#include <mutex>

namespace llvm::orc {

/// An ExecutorProcessControl implementation for in-process JITs that use the
/// new ORC runtime (llvm-project/orc-rt).
///
/// This class communicates with the runtime's InProcessControllerAccess via
/// direct function calls through a virtual connection object.
class LLVM_ABI InProcessEPC : public ExecutorProcessControl {
public:
  /// Pseudo-connection C struct. Used to facilitate calls between InProcessEPC
  /// and InProcessControllerAccess without relying on anything but C ABI.
  /// Must be kept in-sync with the corresponding struct in
  /// orc_rt::InProcessControllerAccess.
  struct Connection {
    void (*Retain)(Connection *C) = nullptr;
    void (*Release)(Connection *C) = nullptr;
    void (*Disconnect)(Connection *C) = nullptr;
    int (*EnterMessageScope)(Connection *C) = nullptr;
    void (*LeaveMessageScope)(Connection *C) = nullptr;

    /// Accessors to be set by the InProcessEPC instance.
    void *IPEPC = nullptr;
    void (*CallJITDispatch)(void *IPEPC, uint64_t CallId, void *HandlerTag,
                            shared::CWrapperFunctionBuffer ArgBytes) = nullptr;
    void (*ReturnWrapperResult)(void *IPEPC, uint64_t CallId,
                                shared::CWrapperFunctionBuffer ResultBytes) =
        nullptr;

    /// Accessors to be set by the InProcessControllerAccess instance.
    void *IPCA = nullptr;
    void (*CallWrapper)(void *IPCA, uint64_t CallId, void *Fn,
                        shared::CWrapperFunctionBuffer ArgBytes) = nullptr;
    void (*ReturnJITDispatchResult)(
        void *IPCA, uint64_t CallId,
        shared::CWrapperFunctionBuffer ResultBytes) = nullptr;
  };

  /// Provides access to bootstrap info.
  /// Must be kept in-sync with the corresponding struct in
  /// orc_rt::InProcessControllerAccess.
  struct BootstrapInfoAccess {
    uint64_t (*GetPageSize)(void *BIA) = nullptr;
    const char *(*GetTargetTriple)(void *BIA) = nullptr;

    int (*GetNextValue)(void *BIA, const char **Name, const char **ValueBytes,
                        uint64_t *ValueSize) = nullptr;
    int (*GetNextSymbol)(void *BIA, const char **Name,
                         uint64_t *Addr) = nullptr;
  };

  /// Create a new InProcessEPC.
  ///
  /// If no symbol string pool is given then one will be created.
  /// If no task dispatcher is given an InPlaceTaskDispatcher will be used.
  static Expected<std::unique_ptr<InProcessEPC>>
  Create(Connection *C, BootstrapInfoAccess *BIA,
         std::shared_ptr<SymbolStringPool> SSP = nullptr,
         std::unique_ptr<TaskDispatcher> D = nullptr);

  ~InProcessEPC();

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override;

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override;

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override;

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override;

  Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
  createDefaultMemoryManager() override;

  Expected<std::unique_ptr<DylibManager>> createDefaultDylibMgr() override;

  Expected<std::unique_ptr<MemoryAccess>> createDefaultMemoryAccess() override;

  Error disconnect() override;

private:
  InProcessEPC(Connection *C, std::shared_ptr<SymbolStringPool> SSP,
               std::unique_ptr<TaskDispatcher> D)
      : ExecutorProcessControl(std::move(SSP), std::move(D)), C(C) {
    C->Retain(C);
  }

  uint64_t registerPendingCallWrapperResult(IncomingWFRHandler H);
  void doDisconnect();

  // Incoming JIT-dispatch call from the ORC runtime.
  void callJITDispatch(uint64_t CallId, void *HandlerTag,
                       shared::CWrapperFunctionBuffer ArgBytes);
  static void callJITDispatchEntry(void *IPEPC, uint64_t CallId,
                                   void *HandlerTag,
                                   shared::CWrapperFunctionBuffer ArgBytes);

  // Incoming wrapper function result from the ORC runtime.
  void returnWrapperResult(uint64_t CallId,
                           shared::CWrapperFunctionBuffer ResultBytes);
  static void
  returnWrapperResultEntry(void *IPEPC, uint64_t CallId,
                           shared::CWrapperFunctionBuffer ResultBytes);

  Connection *C;

  std::mutex M;
  uint64_t NextCallId = 0;
  DenseMap<uint64_t, IncomingWFRHandler> PendingCallWrapperResults;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_INPROCESSEPC_H
