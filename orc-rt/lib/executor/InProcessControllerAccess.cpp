//===- InProcessControllerAccess.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the
// orc-rt/InProcessControllerAccess.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/InProcessControllerAccess.h"

#include <cassert>

namespace orc_rt {

struct InProcessControllerAccess::ConnectionImpl : public Connection {
public:
  ConnectionImpl(InProcessControllerAccess &Instance) {
    Retain = &retainEntry;
    Release = &releaseEntry;
    Disconnect = &disconnectEntry;
    EnterMessageScope = &enterMessageScopeEntry;
    LeaveMessageScope = &leaveMessageScopeEntry;
    IPCA = &Instance;
    CallWrapper = InProcessControllerAccess::callWrapperEntry;
    ReturnJITDispatchResult =
        InProcessControllerAccess::returnJITDispatchResultEntry;
  }

private:
  void retain() {
    std::scoped_lock<std::mutex> Lock(M);
    ++RefCount;
  }

  static void retainEntry(Connection *C) {
    static_cast<ConnectionImpl *>(C)->retain();
  }

  int release() {
    std::scoped_lock<std::mutex> Lock(M);
    --RefCount;
    return RefCount == 0;
  }

  static void releaseEntry(Connection *C) {
    if (static_cast<ConnectionImpl *>(C)->release())
      delete static_cast<ConnectionImpl *>(C);
  }

  void disconnect() {
    {
      std::unique_lock<std::mutex> Lock(M);
      if (!Connected)
        return;
      Connected = false;
      CV.wait(Lock, [this]() { return InFlightCalls == 0; });
    }
    static_cast<InProcessControllerAccess *>(IPCA)->doDisconnect();
  }

  static void disconnectEntry(Connection *C) {
    static_cast<ConnectionImpl *>(C)->disconnect();
  }

  int enterMessageScope() {
    std::scoped_lock<std::mutex> Lock(M);
    if (!Connected)
      return 0;
    ++InFlightCalls;
    return 1;
  }

  static int enterMessageScopeEntry(Connection *C) {
    return static_cast<ConnectionImpl *>(C)->enterMessageScope();
  }

  void leaveMessageScope() {
    bool NotifyCV = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      --InFlightCalls;
      if (InFlightCalls == 0 && !Connected)
        NotifyCV = true;
    }
    if (NotifyCV)
      CV.notify_one();
  }

  static void leaveMessageScopeEntry(Connection *C) {
    static_cast<ConnectionImpl *>(C)->leaveMessageScope();
  }

  std::mutex M;
  std::condition_variable CV;
  bool Connected = true;
  size_t InFlightCalls = 0;
  size_t RefCount = 1;
};

struct InProcessControllerAccess::BootstrapInfoAccessImpl
    : public BootstrapInfoAccess {
public:
  BootstrapInfoAccessImpl(BootstrapInfo &BI)
      : BI(BI), BVI(BI.values().begin()), BSI(BI.symbols().begin()) {
    GetPageSize = getPageSizeEntry;
    GetTargetTriple = getTargetTripleEntry;
    GetNextValue = getNextValueEntry;
    GetNextSymbol = getNextSymbolEntry;
  }

private:
  uint64_t getPageSize() const noexcept { return BI.processInfo().pageSize(); }

  static uint64_t getPageSizeEntry(void *BIA) noexcept {
    return static_cast<BootstrapInfoAccessImpl *>(BIA)->getPageSize();
  }

  const char *getTargetTriple() const noexcept {
    return BI.processInfo().targetTriple().c_str();
  }

  static const char *getTargetTripleEntry(void *BIA) {
    return static_cast<BootstrapInfoAccessImpl *>(BIA)->getTargetTriple();
  }

  int getNextValue(const char **Name, const char **Value, uint64_t *ValueSize) {
    if (BVI == BI.values().end())
      return 0;
    *Name = BVI->first.c_str();
    *Value = BVI->second.data();
    *ValueSize = BVI->second.size();
    ++BVI;
    return 1;
  }

  static int getNextValueEntry(void *BIA, const char **Name, const char **Value,
                               uint64_t *ValueSize) {
    return static_cast<BootstrapInfoAccessImpl *>(BIA)->getNextValue(
        Name, Value, ValueSize);
  }

  int getNextSymbol(const char **Name, uint64_t *Addr) {
    if (BSI == BI.symbols().end())
      return 0;
    *Name = BSI->first.c_str();
    *Addr = ExecutorAddr::fromPtr(BSI->second).getValue();
    ++BSI;
    return 1;
  }

  static int getNextSymbolEntry(void *BIA, const char **Name, uint64_t *Addr) {
    return static_cast<BootstrapInfoAccessImpl *>(BIA)->getNextSymbol(Name,
                                                                      Addr);
  }

  BootstrapInfo &BI;
  BootstrapInfo::ValueMap::iterator BVI;
  SimpleSymbolTable::iterator BSI;
};

InProcessControllerAccess::~InProcessControllerAccess() {
  // 'if (C)' to handle the case where the instance is destroyed without
  // connect ever being run. (TODO: should calling 'connect' be required?)
  if (C)
    C->Release(C);
}

void InProcessControllerAccess::connect(BootstrapInfo BI) {
  assert(!C && "connect called twice?");
  C = new ConnectionImpl(*this);
  BootstrapInfoAccessImpl BIA(BI);

  if (auto Err = OnConnect(*this, BI, C, &BIA)) {
    reportError(std::move(Err));
    // Call disconnect. There's a benign race here: if IPEPC also called
    // C->Disconnect(C) then it might have acquired responsibility for calling
    // InProcessControllerAccess::onDisconnect. In this case control may return
    // from disconnect below early, potentially causing us to return from
    // 'connect' before notifyDisconnect is called. This may lead to confusing
    // logs (since reportError will log error but connect will appear to
    // succeed), however onDisconnect will still be called eventually, and the
    // Session will detach as if the remote had initiated the action after a
    // successful connect.
    disconnect();
    return;
  }

  assert(C->IPEPC && "IPEPC not set by OnConnect");
  assert(C->CallJITDispatch && "CallJITDispatch not set by OnConnect");
  assert(C->ReturnWrapperResult && "ReturnWrapperResult not set by OnConnect");
}

void InProcessControllerAccess::disconnect() {
  assert(C && "disconnect called before connect");
  C->Disconnect(C);
}

void InProcessControllerAccess::callController(
    OnControllerCallReturnFn OnComplete, HandlerTag T,
    WrapperFunctionBuffer ArgBytes) {
  assert(C && "callController called before connect");
  if (C->EnterMessageScope(C)) {
    C->CallJITDispatch(C->IPEPC, registerPendingHandler(std::move(OnComplete)),
                       T, ArgBytes.release());
    C->LeaveMessageScope(C);
  } else
    OnComplete(
        WrapperFunctionBuffer::createOutOfBandError("connection closed"));
}

void InProcessControllerAccess::sendWrapperResult(
    uint64_t CallId, WrapperFunctionBuffer ResultBytes) {
  assert(C && "sendWrapperResult called before connect");
  if (C->EnterMessageScope(C)) {
    C->ReturnWrapperResult(C->IPEPC, CallId, ResultBytes.release());
    C->LeaveMessageScope(C);
  }
}

uint64_t InProcessControllerAccess::registerPendingHandler(
    OnControllerCallReturnFn OnComplete) {
  std::scoped_lock<std::mutex> Lock(M);
  PendingCalls[NextPendingCall] = std::move(OnComplete);
  return NextPendingCall++;
}

void InProcessControllerAccess::doDisconnect() {
  // Drain pending calls.
  PendingCallsMap ToDrain;
  {
    std::scoped_lock<std::mutex> Lock(M);
    ToDrain = std::move(PendingCalls);
  }
  for (auto &[_, H] : ToDrain)
    H(WrapperFunctionBuffer::createOutOfBandError("disconnected"));

  notifyDisconnected();
}

void InProcessControllerAccess::callWrapper(
    uint64_t CallId, void *Fn, orc_rt_WrapperFunctionBuffer ArgBytes) {
  handleWrapperCall(CallId, reinterpret_cast<orc_rt_WrapperFunction>(Fn),
                    WrapperFunctionBuffer(ArgBytes));
}

void InProcessControllerAccess::callWrapperEntry(
    void *IPCA, uint64_t CallId, void *Fn,
    orc_rt_WrapperFunctionBuffer ArgBytes) {
  assert(IPCA);
  static_cast<InProcessControllerAccess *>(IPCA)->callWrapper(CallId, Fn,
                                                              ArgBytes);
}

void InProcessControllerAccess::returnJITDispatchResult(
    uint64_t CallId, orc_rt_WrapperFunctionBuffer ResultBytes) {

  OnControllerCallReturnFn OnComplete;
  {
    std::scoped_lock<std::mutex> Lock(M);
    auto I = PendingCalls.find(CallId);
    if (I != PendingCalls.end()) {
      OnComplete = std::move(I->second);
      PendingCalls.erase(I);
    }
  }

  if (!OnComplete)
    return reportError(make_error<StringError>("Invalid call id"));

  OnComplete(WrapperFunctionBuffer(ResultBytes));
}

void InProcessControllerAccess::returnJITDispatchResultEntry(
    void *IPCA, uint64_t CallId, orc_rt_WrapperFunctionBuffer ResultBytes) {
  assert(IPCA);
  static_cast<InProcessControllerAccess *>(IPCA)->returnJITDispatchResult(
      CallId, ResultBytes);
}

} // namespace orc_rt
