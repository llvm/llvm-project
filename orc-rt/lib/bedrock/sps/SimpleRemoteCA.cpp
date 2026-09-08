//===- SimpleRemoteCA.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SimpleRemote-protocol ControllerAccess base class.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"

#include "orc-rt/support/Compiler.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"

#include <cassert>
#include <string>

namespace orc_rt {

void SimpleRemoteCA::disconnect() {
  {
    std::scoped_lock<std::mutex> Lock(M);
    // Not Accepting means teardown is already under way or finished, or the
    // connection never opened. All are no-ops: the Session tolerates a
    // disconnect racing a controller-initiated one.
    if (ConnState != State::Accepting)
      return;
    ConnState = State::TearingDown;
  }

  // The transport sends the hang-up itself, so that it lands after the queue it
  // is about to discard rather than behind a result that raced this.
  beginTeardown();
}

void SimpleRemoteCA::callController(OnControllerCallReturn OnComplete,
                                    orc_rt_ControllerHandlerTag T,
                                    WrapperFunctionBuffer ArgBytes) {
  // Sent outside tryRegisterCall's critical section, so framing and packaging
  // stay off it. Safe because the Session holds a shared_ptr<ControllerAccess>
  // across this call, so a teardown racing the send cannot destroy *this.
  if (auto SeqNo = tryRegisterCall(OnComplete))
    return sendMessage(Opcode::Call, *SeqNo, T, std::move(ArgBytes));

  // The connection is gone, so no result can arrive. The caller is still on the
  // stack, so fail the handler there.
  failControllerCallInline(std::move(OnComplete));
}

void SimpleRemoteCA::sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                                       uint64_t CallId) {
  // No state check: a result has no pending call on this side, so a transport
  // that has gone away can drop it with nothing left unsettled.
  sendMessage(Opcode::Result, CallId, nullptr, std::move(ResultBytes));
}

void SimpleRemoteCA::beginAccepting(const BootstrapInfo &BI) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    // A transport that enabled delivery before calling this can have dropped
    // out and completed teardown already. The Session has been notified, so
    // there is nothing left to accept on and nothing to send.
    if (ConnState == State::Disconnected)
      return;
    assert(ConnState == State::NotConnected && "beginAccepting called twice");
    ConnState = State::Accepting;
  }

  // State first: a teardown landing in the window then closes an accepting
  // connection, where sending first would leave this to reopen one teardown had
  // just closed. A call landing there registers instead of failing spuriously.
  // Setup may reach a transport that has already gone, which drops it.
  sendMessage(Opcode::Setup, 0, nullptr, encodeSetupMessage(BI));
}

void SimpleRemoteCA::finishTeardown(Error Err) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    // Transports owe exactly one call, and both natural implementations give
    // it: a socket reactor on its way out, and XPC on the single invalidation
    // event it guarantees.
    assert(ConnState != State::Disconnected &&
           "finishTeardown called more than once");
    ConnState = State::Disconnected;
  }

  // The drain must precede the notification, while the keepalive group is still
  // open, or the handlers are dropped rather than dispatched.
  failAllPendingCalls();
  notifyDisconnected(std::move(Err));
}

const char *SimpleRemoteCA::getOpcodeName(Opcode Op) noexcept {
  switch (Op) {
  case Opcode::Setup:
    return "setup";
  case Opcode::Hangup:
    return "hang-up";
  case Opcode::Result:
    return "result";
  case Opcode::Call:
    return "call";
  }
  ORC_RT_UNREACHABLE("Unrecognized opcode");
}

WrapperFunctionBuffer
SimpleRemoteCA::encodeSetupMessage(const BootstrapInfo &BI) {
  using SPSSetup = SPSTuple<SPSString, uint64_t,
                            SPSSequence<SPSTuple<SPSString, SPSSequence<char>>>,
                            SPSSequence<SPSTuple<SPSString, SPSExecutorAddr>>>;

  // Force uint64_t PageSize.
  // FIXME: Remove once we allow size_t serialization.
  uint64_t PageSize = BI.processInfo().pageSize();
  auto Symbols = iterator_range(BI.symbols());
  auto BootstrapTuple =
      std::tie(BI.processInfo().targetTriple(), PageSize, BI.values(), Symbols);

  using SetupSerialize =
      SPSSerializationTraits<SPSSetup, decltype(BootstrapTuple)>;

  auto Payload =
      WrapperFunctionBuffer::allocate(SetupSerialize::size(BootstrapTuple));
  SPSOutputBuffer OB(Payload.data(), Payload.size());
  if (!SetupSerialize::serialize(OB, BootstrapTuple))
    ORC_RT_UNREACHABLE("serialization should not fail");

  return Payload;
}

WrapperFunctionBuffer SimpleRemoteCA::encodeHangupPayload(Error Err) {
  SPSSerializableError SE(std::move(Err));
  using SPSSerialize = SPSArgList<SPSError>;
  auto Payload = WrapperFunctionBuffer::allocate(SPSSerialize::size(SE));
  SPSOutputBuffer OB(Payload.data(), Payload.size());
  if (!SPSSerialize::serialize(OB, SE))
    ORC_RT_UNREACHABLE("serialization should not fail");
  return Payload;
}

Expected<SimpleRemoteCA::Action>
SimpleRemoteCA::handleMessage(uint64_t OpC, uint64_t SeqNo, ExecutorAddr Tag,
                              WrapperFunctionBuffer Payload) {
  if (OpC > static_cast<uint64_t>(Opcode::LastOpcode))
    return make_error<StringError>("Invalid opcode " + std::to_string(OpC));

  switch (static_cast<Opcode>(OpC)) {
  case Opcode::Setup:
    return make_error<StringError>("Unexpected Setup message");
  case Opcode::Hangup: {
    // A hang-up carries no sequence number or tag, and a payload holding the
    // reason the controller is going away.
    if (SeqNo != 0 || Tag)
      return make_error<StringError>("Malformed hang-up message");
    SPSSerializableError SE;
    SPSInputBuffer IB(Payload.data(), Payload.size());
    if (!SPSArgList<SPSError>::deserialize(IB, SE))
      return make_error<StringError>(
          "Malformed hang-up message: could not deserialize reason");
    // An orderly hang-up ends the session; one carrying a reason ends it with
    // that reason, which the reactor reports as the disconnection error.
    if (Error Err = SE.toError())
      return std::move(Err);
    return Action::End;
  }
  case Opcode::Result:
    // A result is not associated with a handler tag.
    if (Tag)
      return make_error<StringError>(
          "Result message should not carry a handler tag");
    if (auto Err = handleResult(SeqNo, std::move(Payload)))
      return std::move(Err);
    return Action::Continue;
  case Opcode::Call:
    handleWrapperCall(Tag.toPtr<orc_rt_WrapperFunction>(), std::move(Payload),
                      SeqNo);
    return Action::Continue;
  }
  ORC_RT_UNREACHABLE("Unrecognized opcode");
}

std::optional<uint64_t>
SimpleRemoteCA::tryRegisterCall(OnControllerCallReturn &OnComplete) {
  std::scoped_lock<std::mutex> Lock(M);
  if (ConnState != State::Accepting)
    return std::nullopt;
  PendingCalls.try_emplace(NextSeqNo, std::move(OnComplete));
  return NextSeqNo++;
}

void SimpleRemoteCA::failAllPendingCalls() {
  PendingCallsMap Failed;
  {
    std::scoped_lock<std::mutex> Lock(M);
    std::swap(Failed, PendingCalls);
  }

  for (auto &[SeqNo, OnComplete] : Failed)
    failPendingControllerCall(std::move(OnComplete));
}

Error SimpleRemoteCA::handleResult(uint64_t SeqNo,
                                   WrapperFunctionBuffer ResultBytes) {
  OnControllerCallReturn OnComplete;
  {
    std::scoped_lock<std::mutex> Lock(M);
    auto I = PendingCalls.find(SeqNo);
    if (I == PendingCalls.end())
      return make_error<StringError>("No pending call for sequence number " +
                                     std::to_string(SeqNo));
    OnComplete = std::move(I->second);
    PendingCalls.erase(I);
  }

  handleControllerCallResult(std::move(OnComplete), std::move(ResultBytes));
  return Error::success();
}

} // namespace orc_rt
