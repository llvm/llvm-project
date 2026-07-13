//===---------------- InProcessControllerAccess.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Makes direct calls from / to the controller, which must exist in the same
// process.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INPROCESSCONTROLLERACCESS_H
#define ORC_RT_INPROCESSCONTROLLERACCESS_H

#include "orc-rt-c/WrapperFunction.h"
#include "orc-rt/Error.h"
#include "orc-rt/Session.h"
#include "orc-rt/move_only_function.h"

#include <mutex>
#include <unordered_map>

namespace orc_rt {

/// Provides direct access from/to an ExecutorProcessControl object in the same
/// process.
class InProcessControllerAccess : public Session::ControllerAccess {
public:
  /// Pseudo-connection C struct. Used to facilitate calls between InProcessEPC
  /// and InProcessControllerAccess without relying on anything but C ABI.
  /// Must be kept in-sync with the corresponding struct in InProcessEPC.
  struct Connection {
    void (*Retain)(Connection *C) = nullptr;
    void (*Release)(Connection *C) = nullptr;
    void (*Disconnect)(Connection *C) = nullptr;
    int (*EnterMessageScope)(Connection *C) = nullptr;
    void (*LeaveMessageScope)(Connection *C) = nullptr;

    /// Accessors to be set by the InProcessEPC instance.
    void *IPEPC = nullptr;
    void (*CallJITDispatch)(void *IPEPC, uint64_t CallId, void *HandlerTag,
                            orc_rt_WrapperFunctionBuffer ArgBytes) = nullptr;
    void (*ReturnWrapperResult)(void *IPEPC, uint64_t CallId,
                                orc_rt_WrapperFunctionBuffer ResultBytes) =
        nullptr;

    /// Accessors to be set by the InProcessControllerAccess instance.
    void *IPCA = nullptr;
    void (*CallWrapper)(void *IPCA, uint64_t CallId, void *Fn,
                        orc_rt_WrapperFunctionBuffer ArgBytes) = nullptr;
    void (*ReturnJITDispatchResult)(void *IPCA, uint64_t CallId,
                                    orc_rt_WrapperFunctionBuffer ResultBytes) =
        nullptr;
  };

  struct ConnectionImpl;

  /// Provides access to bootstrap info.
  /// Must be kept in-sync with the corresponding struct in InProcessEPC.
  struct BootstrapInfoAccess {
    uint64_t (*GetPageSize)(void *BIA) = nullptr;
    const char *(*GetTargetTriple)(void *BIA) = nullptr;

    int (*GetNextValue)(void *BIA, const char **Name, const char **ValueBytes,
                        uint64_t *ValueSize) = nullptr;
    int (*GetNextSymbol)(void *BIA, const char **Name,
                         uint64_t *Addr) = nullptr;
  };

  struct BootstrapInfoAccessImpl;

  /// OnConnect callback type.
  ///
  /// An instance of this type will be stored in the InProcessControllerAccess
  /// object and called from InProcessControllerAccess::connect. The IPCA&
  /// reference argument remains valid for the lifetime of the calling IPCA
  /// instance. The Connection *C argument is ref-counted; the IPCA holds the
  /// initial reference, keeping C alive for at least the duration of the call.
  /// Implementations that need to use C after OnConnect returns must call
  /// C->Retain(C) (paired with C->Release(C) when finished) to take their own
  /// reference. The BI and BCA arguments are valid only for the duration of
  /// the call and must not be captured by the callback.
  ///
  /// It is expected that clients will use these arguments to construct an
  /// llvm::orc::InProcessEPC object via that class's Create method, in which
  /// case llvm::orc::InProcessEPC::Create will manage the retain and release of
  /// Connection *C.
  using OnConnectFn = move_only_function<Error(InProcessControllerAccess &IPCA,
                                               BootstrapInfo &BI, Connection *C,
                                               BootstrapInfoAccess *BCA)>;

  /// Create an InProcessControllerAccess instance.
  InProcessControllerAccess(Session &S, OnConnectFn OnConnect)
      : Session::ControllerAccess(S), OnConnect(std::move(OnConnect)) {}

  InProcessControllerAccess(const InProcessControllerAccess &) = delete;
  InProcessControllerAccess &
  operator=(const InProcessControllerAccess &) = delete;
  InProcessControllerAccess(InProcessControllerAccess &&) = delete;
  InProcessControllerAccess &operator=(InProcessControllerAccess &&) = delete;

  ~InProcessControllerAccess();

  void connect(BootstrapInfo BI) override;

  void disconnect() override;

  void callController(OnCallHandlerCompleteFn OnComplete, HandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override;
  void sendWrapperResult(uint64_t CallId,
                         WrapperFunctionBuffer ResultBytes) override;

private:
  uint64_t registerPendingHandler(OnCallHandlerCompleteFn OnComplete);
  void doDisconnect();

  void callWrapper(uint64_t CallId, void *Fn,
                   orc_rt_WrapperFunctionBuffer ArgBytes);
  static void callWrapperEntry(void *IPCA, uint64_t CallId, void *Fn,
                               orc_rt_WrapperFunctionBuffer ArgBytes);

  void returnJITDispatchResult(uint64_t CallId,
                               orc_rt_WrapperFunctionBuffer ResultBytes);
  static void
  returnJITDispatchResultEntry(void *IPCA, uint64_t CallId,
                               orc_rt_WrapperFunctionBuffer ResultBytes);

  OnConnectFn OnConnect;
  ConnectionImpl *C = nullptr;

  std::mutex M;
  uint64_t NextPendingCall = 0;

  using PendingCallsMap = std::unordered_map<uint64_t, OnCallHandlerCompleteFn>;
  PendingCallsMap PendingCalls;
};

} // namespace orc_rt

#endif // ORC_RT_INPROCESSCONTROLLERACCESS_H
