//===-------------------- SimpleRemoteCA.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transport-independent half of the SimpleRemote protocol.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H
#define ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H

#include "orc-rt/bedrock/BootstrapInfo.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/ExecutorAddress.h"
#include "orc-rt/support/WrapperFunction.h"

#include <cstdint>
#include <unordered_map>

namespace orc_rt {

/// Base for ControllerAccess implementations that speak the SimpleRemote
/// protocol: payload encoding, message semantics, and the table of calls
/// awaiting a result. A subclass supplies the transport -- framing, I/O, and
/// the synchronization those need.
///
/// This class does not enforce synchronization or track connection state: that
/// is left to subclasses (who usually have those those things for the sake of
/// the transport). Subclasses must ensure that notifyDisconnect is called
/// exactly once, and that every registered handler is notified with either a
/// result or an error.
class SimpleRemoteCA : public Session::ControllerAccess {
protected:
  /// SimpleRemote message opcodes.
  ///
  /// On-the-wire values, shared with LLVM's SimpleRemoteEPCOpcode: do not
  /// renumber or reorder.
  enum class Opcode : uint64_t {
    Setup,
    Hangup,
    Result,
    Call,
    LastOpcode = Call
  };

  /// The name of Op, for logging.
  static const char *getOpcodeName(Opcode Op) noexcept;

  /// Whether the session continues, or the controller has hung up and it
  /// should end.
  enum class Action { Continue, End };

  using PendingCallsMap = std::unordered_map<uint64_t, OnControllerCallReturn>;

  SimpleRemoteCA(Session &S) : ControllerAccess(S) {}

  /// Serializes BI as the payload of a setup message.
  static WrapperFunctionBuffer encodeSetup(const BootstrapInfo &BI);

  /// Serializes Err as the payload of a hang-up message.
  ///
  /// A hang-up always carries a reason: success for an orderly disconnect,
  /// otherwise what went wrong.
  static WrapperFunctionBuffer encodeHangup(Error Err);

  /// Decodes a hang-up payload produced by encodeHangup, returning the reason
  /// it carries.
  ///
  /// A payload that will not decode -- including an empty one, which is never
  /// valid -- comes back as an error describing itself. Both outcomes end the
  /// session, so both are reported the same way.
  static Error decodeHangup(WrapperFunctionBuffer Payload);

  /// Registers OnComplete and returns the sequence number to send its call
  /// under.
  ///
  /// Unsynchronized. Register and queue the message in one critical section, so
  /// that a call is never left pending with nothing to answer it, nor sent with
  /// no handler to complete.
  uint64_t registerCall(OnControllerCallReturn OnComplete);

  /// Removes and returns the handler registered under SeqNo, or a null handler
  /// if there is none.
  ///
  /// Unsynchronized. Run the handler outside the lock: it runs managed code.
  OnControllerCallReturn takeCall(uint64_t SeqNo);

  /// Removes and returns every registered handler, for a transport to fail on
  /// its way out.
  ///
  /// Unsynchronized. Call before notifying the Session, while the managed-code
  /// group is still open, or the handlers are dropped rather than dispatched.
  PendingCallsMap takeAllCalls();

  /// Acts on one de-framed message. OpC is the raw wire opcode: this validates
  /// it along with the header semantics each opcode requires, so a transport
  /// need only deliver the fields and payload intact.
  ///
  /// Every error returned is terminal: stop reading and end the session with
  /// it. Action::End means the controller hung up cleanly.
  ///
  /// Call with no transport lock held: dispatching a Call, and completing a
  /// Result, run managed code that may re-enter this object -- through
  /// callController, and so into that same lock.
  ///
  /// Calls must be serialized with one another, and must all complete before
  /// the Session is notified, since a Result completes a pending call.
  Expected<Action> handleMessage(uint64_t OpC, uint64_t SeqNo, ExecutorAddr Tag,
                                 WrapperFunctionBuffer Payload);

  /// Removes the handler for SeqNo, or returns a null handler if there is none.
  ///
  /// handleMessage needs this to complete a Result, and a result arriving on
  /// the transport's reader can race a call being registered on another thread.
  /// Implement it as takeCall under whatever guards registerCall.
  virtual OnControllerCallReturn takePendingCall(uint64_t SeqNo) = 0;

private:
  /// Completes the pending call SeqNo with ResultBytes. Fails if no such call
  /// is outstanding, which means the peer answered a call that was never made.
  Error handleResult(uint64_t SeqNo, WrapperFunctionBuffer ResultBytes);

  // Guarded by the transport's lock. See the class comment.
  uint64_t NextSeqNo = 1;
  PendingCallsMap PendingCalls;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SPS_SIMPLEREMOTECA_H
