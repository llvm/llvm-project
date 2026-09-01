//===-------------------- SimpleRemoteCA.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A ControllerAccess base class implementing the SimpleRemote wire protocol.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H
#define ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H

#include "orc-rt/bedrock/BootstrapInfo.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/ExecutorAddress.h"
#include "orc-rt/support/WrapperFunction.h"

#include <mutex>
#include <optional>
#include <unordered_map>

namespace orc_rt {

/// ControllerAccess base for the SimpleRemote protocol.
///
/// Implements the protocol semantics -- setup and hang-up message encoding,
/// message dispatch and validation, pending-call tracking, connection state and
/// teardown sequencing -- while leaving wire framing and byte transport to
/// subclasses. Subclasses feed received messages to handleMessage and implement
/// two hooks, sendMessage and beginTeardown.
class SimpleRemoteCA : public Session::ControllerAccess {
public:
  void disconnect() final;

  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag T,
                      WrapperFunctionBuffer ArgBytes) final;

  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) final;

protected:
  /// SimpleRemote message opcodes.
  ///
  /// These values are on-the-wire values, shared with LLVM's
  /// SimpleRemoteEPCOpcode: do not renumber or reorder them.
  enum class Opcode { Setup, Hangup, Result, Call, LastOpcode = Call };

  /// Result of handleMessage: whether the session continues, or the controller
  /// has hung up and the session should end.
  enum class Action { Continue, End };

  static const char *getOpcodeName(Opcode Op) noexcept;

  SimpleRemoteCA(Session &S) : ControllerAccess(S) {}

  /// Serializes Err as the payload of a hang-up message.
  ///
  /// A hang-up always carries a serialized Error saying why the session is
  /// ending: a success value for an orderly disconnect, otherwise the reason.
  static WrapperFunctionBuffer encodeHangupPayload(Error Err);

  /// Starts accepting controller calls and sends the setup message. Subclasses
  /// call this from connect once the transport can carry messages.
  ///
  /// Safe to call whether or not the transport has started receiving. A
  /// transport that has already dropped out and finished teardown gets a no-op:
  /// the Session has been notified, and no setup is sent.
  void beginAccepting(const BootstrapInfo &BI);

  /// Completes teardown: drains pending calls, then notifies the Session
  /// exactly once. Subclasses call this when the transport is finished, however
  /// teardown began.
  ///
  /// Call it exactly once.
  ///
  /// Err is the disconnection mode: success for an orderly hang-up from either
  /// side, otherwise what went wrong. May synchronously destroy *this, so
  /// nothing may touch it afterwards.
  void finishTeardown(Error Err);

  /// Dispatches a received message. Transports call this once per message,
  /// after de-framing. OpC is the raw wire opcode: this validates it and the
  /// per-opcode header semantics. Transports are responsible only for verifying
  /// that the payload deserializes as the message requires.
  ///
  /// Every error returned here is terminal for the session: shut the transport
  /// down and pass the error to finishTeardown, rather than continuing to read.
  /// Action::End means the controller hung up cleanly -- shut down and pass
  /// Error::success().
  ///
  /// Calls may come from any thread, but must be serialized with one another
  /// and must all complete before finishTeardown: the Result path completes a
  /// pending call, which is only legal while the managed-code group is still
  /// open.
  Expected<Action> handleMessage(uint64_t OpC, uint64_t SeqNo, ExecutorAddr Tag,
                                 WrapperFunctionBuffer Payload);

  /// Hands a framed message to the transport. Called with no lock held, so
  /// framing or packaging happens off the critical section.
  ///
  /// Best-effort: a transport that has gone away should drop the message rather
  /// than report anything. A Call's handler is failed by the drain, and a
  /// Result has no pending call on this side, so nothing is left unsettled.
  virtual void sendMessage(Opcode Op, uint64_t SeqNo,
                           orc_rt_ControllerHandlerTag T,
                           WrapperFunctionBuffer Payload) = 0;

  /// Starts shutting the transport down. Must not block: teardown holds up
  /// Session detach. Call finishTeardown once the transport is finished.
  ///
  /// Called only for a local disconnect, which is orderly by definition, so
  /// make a best effort to send encodeHangupPayload(Error::success()) as the
  /// last message before shutting down. Only the transport can place it last.
  ///
  /// Teardown the transport initiates instead -- a hang-up from the controller,
  /// or an error out of handleMessage -- goes straight to finishTeardown,
  /// having sent its own reason first if it has one.
  virtual void beginTeardown() = 0;

private:
  /// Whether the connection will accept new controller calls, and how far
  /// teardown has progressed. Guarded by M.
  enum class State {
    NotConnected, ///< Before beginAccepting, or connect failed.
    Accepting,    ///< Calls may be registered.
    TearingDown,  ///< Latched closed; the transport is shutting down.
    Disconnected, ///< finishTeardown has run; the Session has been notified.
  };

  /// Registers OnComplete and returns the sequence number to send the call
  /// under, or nullopt if the connection is no longer accepting calls -- in
  /// which case OnComplete is untouched and the caller must fail it inline.
  ///
  /// The state check and the registration are one critical section, so a call
  /// racing teardown is either drained by finishTeardown or failed inline:
  /// never left pending with no result to come.
  std::optional<uint64_t> tryRegisterCall(OnControllerCallReturn &OnComplete);

  void failAllPendingCalls();
  Error handleResult(uint64_t SeqNo, WrapperFunctionBuffer ResultBytes);

  static WrapperFunctionBuffer encodeSetupMessage(const BootstrapInfo &BI);

  std::mutex M;
  State ConnState = State::NotConnected;

  // SeqNo zero is reserved for messages with no pending result (Setup, Hangup).
  uint64_t NextSeqNo = 1;

  using PendingCallsMap = std::unordered_map<uint64_t, OnControllerCallReturn>;
  PendingCallsMap PendingCalls;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H
