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

#ifndef ORC_RT_SIMPLEREMOTECA_H
#define ORC_RT_SIMPLEREMOTECA_H

#include "orc-rt/bedrock/BootstrapInfo.h"
#include "orc-rt/bedrock/Error.h"
#include "orc-rt/bedrock/ExecutorAddress.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/bedrock/WrapperFunction.h"

#include <mutex>
#include <unordered_map>

namespace orc_rt {

/// ControllerAccess base for the SimpleRemote protocol.
///
/// Implements the protocol semantics -- setup and hang-up message encoding,
/// message dispatch and validation, pending-call tracking, and completion of
/// controller calls -- while leaving all wire framing and byte transport,
/// including sending, to subclasses. Subclasses feed received messages to
/// handleMessage and build outgoing ones from the Opcode set,
/// encodeSetupMessage and encodeHangupPayload.
class SimpleRemoteCA : public Session::ControllerAccess {
protected:
  /// SimpleRemote message opcodes.
  ///
  /// These values are on-the-wire values, shared with LLVM's
  /// SimpleRemoteEPCOpcode: do not renumber or reorder them.
  enum class Opcode { Setup, Hangup, Result, Call, LastOpcode = Call };

  /// Result of handleMessage: whether the session continues, or the controller
  /// has hung up and the session should end.
  enum class Action { Continue, End };

  /// Returns a display name for Op, for logging.
  static const char *getOpcodeName(Opcode Op) noexcept;

  SimpleRemoteCA(Session &S) : ControllerAccess(S) {}

  /// Serializes a setup message from BI and returns its payload. Subclasses
  /// send this as the first message during connect -- an Opcode::Setup message
  /// with no sequence number or handler tag.
  static WrapperFunctionBuffer encodeSetupMessage(const BootstrapInfo &BI);

  /// Serializes Err as the payload of a hang-up message.
  ///
  /// A hang-up always carries a serialized Error saying why the session is
  /// ending: a success value for an orderly disconnect, otherwise the reason.
  /// handleMessage decodes an incoming hang-up through the matching code, so
  /// the two directions cannot drift apart.
  static WrapperFunctionBuffer encodeHangupPayload(Error Err);

  /// Dispatches a received message. Transports call this once per message,
  /// after de-framing. OpC is the raw wire opcode: this validates it and the
  /// per-opcode header semantics. Transports are responsible only for verifying
  /// that the payload deserializes as the message requires.
  ///
  /// Every error returned here is terminal for the session: report it via
  /// notifyDisconnected (as for Action::End, but with the error as the
  /// disconnection reason) rather than continuing to read.
  ///
  /// Calls may come from any thread, but must be serialized with one another
  /// and must all complete before failAllPendingCalls: the Result path
  /// completes a pending call, which is only legal while the managed-code group
  /// is still open. One racing teardown may hit that assertion, or touch a
  /// *this that notifyDisconnected has already destroyed.
  Expected<Action> handleMessage(uint64_t OpC, uint64_t SeqNo, ExecutorAddr Tag,
                                 WrapperFunctionBuffer Payload);

  /// Records OnComplete as awaiting a result and returns the sequence number to
  /// send the call under. The result is delivered when a matching Result
  /// message arrives (handleMessage), or the call is failed (see below).
  ///
  /// SimpleRemoteCA tracks no connection state, so it cannot reject a call
  /// registered after failAllPendingCalls has run: such a call is never
  /// drained, and its handler is silently dropped, leaving the caller awaiting
  /// a result that can no longer arrive. Subclasses must therefore make their
  /// disconnecting check and this registration atomic with respect to the drain
  /// -- e.g. both under the lock that publishes the disconnecting state (see
  /// Session::ControllerAccess::callController).
  uint64_t registerPendingCall(OnControllerCallReturn OnComplete);

  /// Fails every pending call with a disconnect error -- the disconnect drain.
  /// Call before notifyDisconnected, while the managed-code group is still open
  /// (see Session::ControllerAccess::disconnect). Dispatched under a token.
  void failAllPendingCalls();

private:
  Error handleResult(uint64_t SeqNo, WrapperFunctionBuffer ResultBytes);

  std::mutex M;

  // SeqNo zero is reserved for messages with no pending result (Setup, Hangup).
  uint64_t NextSeqNo = 1;

  using PendingCallsMap = std::unordered_map<uint64_t, OnControllerCallReturn>;
  PendingCallsMap PendingCalls;
};

} // namespace orc_rt

#endif // ORC_RT_SIMPLEREMOTECA_H
