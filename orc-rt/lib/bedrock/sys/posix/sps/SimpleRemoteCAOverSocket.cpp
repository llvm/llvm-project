//===- SimpleRemoteCAOverSocket.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SimpleRemote protocol over a connected socket, on POSIX.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCAOverSocket.h"

#include "orc-rt-c/support/Logging.h"
#include "orc-rt-internal/support/sys/Errno.h"
#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"
#include "orc-rt/support/Compiler.h"
#include "orc-rt/support/span.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <deque>
#include <fcntl.h>
#include <optional>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace orc_rt {

namespace {

class SocketSimpleRemoteCA : public SimpleRemoteCA {
public:
  static Expected<std::shared_ptr<SocketSimpleRemoteCA>>
  Create(Session &S, SocketHandle Sock);

private:
  enum class State {
    NotConnected, ///< Before connect. Nothing may be registered or queued.
    Running,      ///< Calls may be registered, messages queued.
    Draining,     ///< Hang-up queued and latched: nothing may follow it.
    Closed,       ///< Reactor stopped and descriptors released.
  };

  /// A framed message waiting to go out. Owns the payload. Supports interrupted
  /// sends via pending/advance/complete.
  class OutgoingMessage {
  public:
    OutgoingMessage(Opcode Op, uint64_t SeqNo, uint64_t Tag,
                    WrapperFunctionBuffer Payload);

    /// The next bytes to send. Empty once the message has gone.
    span<const char> pending() const;

    void advance(size_t N) { Sent += N; }
    bool complete() const { return Sent == MsgHeader::Size + Payload.size(); }

  private:
    char Header[MsgHeader::Size];
    WrapperFunctionBuffer Payload;
    size_t Sent = 0;
  };

  /// Incoming message. Supports interrupted reads via pending/advance/complete.
  /// The header is filled first then decoded to determine the allocation size
  /// for the payload buffer.
  class IncomingMessage {
  public:
    /// Where the next bytes read should land. Empty once the current buffer is
    /// full: the header is then ready to decode, or the message is complete.
    span<char> pending();

    void advance(size_t N) { Filled += N; }

    /// True once the payload buffer is full and the message can be dispatched.
    bool complete() const {
      return Decoded && Filled == MsgHeader::Size + Payload.size();
    }

    /// Decodes the completed header and allocates space for the payload.
    /// Fails on a header that describes an impossible message.
    Error decodeHeader();

    const MsgHeader::Fields &fields() const { return F; }

    /// Take the payload. Resets this value for the next message.
    WrapperFunctionBuffer take();

  private:
    char Header[MsgHeader::Size];
    WrapperFunctionBuffer Payload;
    size_t Filled = 0;
    bool Decoded = false;
    MsgHeader::Fields F;
  };

  SocketSimpleRemoteCA(Session &S, SocketHandle Sock, SocketHandle WakeRead,
                       SocketHandle WakeWrite)
      : SimpleRemoteCA(S), Sock(std::move(Sock)), WakeRead(std::move(WakeRead)),
        WakeWrite(std::move(WakeWrite)) {}

  // Session::ControllerAccess.
  void connect(BootstrapInfo BI) override;
  void disconnect() override;
  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override;
  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) override;

  /// Wake the reactor thread.
  ///
  /// Makes a poll in progress return, or the next one return at once.
  ///
  /// Caller must hold M and must not be in the Closed state: the
  /// reactor releases the descriptors under the same lock, so holding
  /// it is what keeps this from writing to a closed one.
  void wakeReactorLocked();

  /// Reactor thread entry point. Runs the reactor loop, then performs cleanup:
  /// releasing descriptors, failing outstanding calls, and notifying the
  /// session.
  void runReactor();

  /// Message IO loop: polls, reads, dispatches and sends until the connection
  /// ends. Returns the reason it stopped: success if either side hung up.
  Error reactorLoop();

  /// Sends as much of the queue as the socket will take. Reactor thread only.
  Error drainSends();

  /// Receives what is available and hands each complete message to
  /// handleMessage. Incoming carries any part-assembled message across calls.
  /// Reactor thread only.
  ///
  /// Only a hang-up ends the session cleanly. Unexpected end of stream is an
  /// error.
  Expected<Action> readAndDispatch(IncomingMessage &Incoming);

  /// Takes the handler for SeqNo under M.
  OnControllerCallReturn takePendingCall(uint64_t SeqNo) override;

  SocketHandle Sock;
  SocketHandle WakeRead, WakeWrite;

  std::mutex M;
  State CurState = State::NotConnected;

  /// Framed messages awaiting send. Only the reactor pops, so it may hold a
  /// reference to the front across a send without the lock.
  ///
  /// TODO: Currently unbounded. We may want to add a bound on this.
  std::deque<OutgoingMessage> Queue;
};

Error makeError(const char *Op, int ErrNum) {
  return make_error<StringError>(std::string(Op) +
                                 " failed: " + sys::strError(ErrNum));
}

bool isWouldBlock(int ErrNum) {
  return ErrNum == EAGAIN || ErrNum == EWOULDBLOCK;
}

/// O_NONBLOCK rather than MSG_DONTWAIT on each send/recv: Darwin defines
/// MSG_DONTWAIT but does not honour it on sends.
Error setNonBlocking(int FD) {
  int Flags = fcntl(FD, F_GETFL, 0);
  if (Flags == -1)
    return makeError("fcntl(F_GETFL)", errno);
  if ((Flags & O_NONBLOCK) == 0 && fcntl(FD, F_SETFL, Flags | O_NONBLOCK) == -1)
    return makeError("fcntl(F_SETFL)", errno);
  return Error::success();
}

template <typename OpT> ssize_t retryOnEINTR(OpT Op) {
  for (;;) {
    ssize_t N = Op();
    if (N >= 0 || errno != EINTR)
      return N;
  }
}

} // namespace

SocketSimpleRemoteCA::OutgoingMessage::OutgoingMessage(
    Opcode Op, uint64_t SeqNo, uint64_t Tag, WrapperFunctionBuffer Payload)
    : Payload(std::move(Payload)) {
  assert(!this->Payload.getOutOfBandError() &&
         "Out-of-band errors have no byte representation: encode as a result "
         "kind before framing");
  MsgHeader::encode(Header, Op, SeqNo, Tag, this->Payload.size());
}

span<const char> SocketSimpleRemoteCA::OutgoingMessage::pending() const {
  if (Sent < MsgHeader::Size)
    return {Header + Sent, MsgHeader::Size - Sent};
  size_t InPayload = Sent - MsgHeader::Size;
  return {Payload.data() + InPayload, Payload.size() - InPayload};
}

span<char> SocketSimpleRemoteCA::IncomingMessage::pending() {
  if (!Decoded)
    return {Header + Filled, MsgHeader::Size - Filled};
  size_t InPayload = Filled - MsgHeader::Size;
  return {Payload.data() + InPayload, Payload.size() - InPayload};
}

Error SocketSimpleRemoteCA::IncomingMessage::decodeHeader() {
  F = MsgHeader::decode(Header);
  if (F.MsgSize < MsgHeader::Size)
    return make_error<StringError>("Message size smaller than its header");

  Payload = WrapperFunctionBuffer::allocate(F.MsgSize - MsgHeader::Size);
  Decoded = true;
  return Error::success();
}

WrapperFunctionBuffer SocketSimpleRemoteCA::IncomingMessage::take() {
  auto P = std::move(Payload);
  Payload = WrapperFunctionBuffer();
  Filled = 0;
  Decoded = false;
  return P;
}

Expected<std::shared_ptr<SocketSimpleRemoteCA>>
SocketSimpleRemoteCA::Create(Session &S, SocketHandle Sock) {
  // Sock is owned here, so every early return below closes it.

  // This class assumes a byte stream, so check that this is a SOCK_STREAM.
  int Type;
  socklen_t TypeLen = sizeof(Type);
  if (::getsockopt(Sock.get(), SOL_SOCKET, SO_TYPE, &Type, &TypeLen) != 0)
    return makeError("getsockopt(SO_TYPE)", errno);
  if (Type != SOCK_STREAM)
    return make_error<StringError>(
        "SimpleRemote over a socket requires a stream socket");

  if (auto Err = setNonBlocking(Sock.get()))
    return std::move(Err);

  int Pair[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, Pair) != 0)
    return makeError("socketpair", errno);
  SocketHandle WakeRead(Pair[0]), WakeWrite(Pair[1]);

  // Both ends non-blocking. A blocking write end would stall a sender inside M,
  // which the reactor needs before it can drain -- a deadlock; a blocking read
  // end would leave the drain loop waiting for a byte after emptying the queue.
  for (int W : {WakeRead.get(), WakeWrite.get()})
    if (auto Err = setNonBlocking(W))
      return std::move(Err);

  // Not make_shared: the constructor is private.
  return std::shared_ptr<SocketSimpleRemoteCA>(new SocketSimpleRemoteCA(
      S, std::move(Sock), std::move(WakeRead), std::move(WakeWrite)));
}

void SocketSimpleRemoteCA::wakeReactorLocked() {
  assert(CurState != State::Closed && "wake on a released descriptor");
  char C = 0;
  ssize_t N = retryOnEINTR(
      [&] { return ::send(WakeWrite.get(), &C, 1, MSG_NOSIGNAL); });
  int ErrNum = errno;
  if (N < 0 && !isWouldBlock(ErrNum)) {
    // send to wait socket failed. Log the reason in case this jams up the
    // reactor.
    ORC_RT_LOG(Info, ControllerAccess,
               "SimpleRemoteCA/socket wake-send error: " ORC_RT_LOG_PUB_S,
               sys::strError(ErrNum).c_str());
  }
}

void SocketSimpleRemoteCA::connect(BootstrapInfo BI) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(CurState == State::NotConnected && "connect called twice");
    CurState = State::Running;
    // Queued before the reactor exists, and the reactor is the only sender, so
    // setup is the first message on the wire.
    Queue.emplace_back(Opcode::Setup, 0, 0, encodeSetup(BI));
  }

  // The Session holds this alive until notifyDisconnected, which the reactor
  // reaches only as it exits, so the thread needs no reference of its own.
  //
  // FIXME: Report a spawn failure, and offer a pumped mode that borrows the
  // caller's thread. std::thread's constructor aborts rather than reporting
  // under -fno-exceptions, so connect cannot surface one today.
  std::thread([this] { runReactor(); }).detach();
}

void SocketSimpleRemoteCA::disconnect() {
  std::scoped_lock<std::mutex> Lock(M);
  // Anything but Running means teardown is under way or done, or the connection
  // never opened. The Session tolerates a disconnect racing a remote one.
  if (CurState != State::Running)
    return;

  // Queued and latched in one lock hold. Draining is what keeps anything from
  // landing behind the hang-up, so it must take effect with the same atomicity
  // as the queueing.
  Queue.emplace_back(Opcode::Hangup, 0, 0, encodeHangup(Error::success()));
  CurState = State::Draining;
  wakeReactorLocked();
}

void SocketSimpleRemoteCA::callController(OnControllerCallReturn OnComplete,
                                          orc_rt_ControllerHandlerTag T,
                                          WrapperFunctionBuffer ArgBytes) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (CurState == State::Running) {
      // Registered and queued in one lock hold, so a call is never left pending
      // with nothing to answer it, nor sent with no handler to complete.
      uint64_t SeqNo = registerCall(std::move(OnComplete));
      Queue.emplace_back(Opcode::Call, SeqNo,
                         ExecutorAddr::fromPtr(T).getValue(),
                         std::move(ArgBytes));
      wakeReactorLocked();
      return;
    }
  }

  // The connection is gone, so no result can arrive. The caller is still on the
  // stack, so fail the handler there.
  failControllerCallInline(std::move(OnComplete));
}

void SocketSimpleRemoteCA::sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                                             uint64_t CallId) {
  // Encoded before the lock: an out-of-band error has no byte representation,
  // so it travels as a distinct result kind with its message as the payload.
  auto [Kind, Payload] = encodeResult(std::move(ResultBytes));

  // No state check of its own: a result has no pending call on this side, so a
  // departed connection drops it with nothing left unsettled.
  std::scoped_lock<std::mutex> Lock(M);
  if (CurState != State::Running)
    return;
  Queue.emplace_back(Opcode::Result, CallId, static_cast<uint64_t>(Kind),
                     std::move(Payload));
  wakeReactorLocked();
}

void SocketSimpleRemoteCA::runReactor() {
  Error Err = reactorLoop();

  {
    std::scoped_lock<std::mutex> Lock(M);
    // Closed refuses every further send and wake.
    CurState = State::Closed;
    Sock.reset();
    WakeRead.reset();
    WakeWrite.reset();
  }

  // Before the notification, while the managed-code group is still open, or the
  // handlers are dropped rather than dispatched.
  PendingCallsMap Failed;
  {
    std::scoped_lock<std::mutex> Lock(M);
    Failed = takeAllCalls();
  }
  for (auto &[SeqNo, OnComplete] : Failed)
    failPendingControllerCall(std::move(OnComplete));

  // Last thing to touch this: it may run the destructor.
  notifyDisconnected(std::move(Err));
}

Error SocketSimpleRemoteCA::reactorLoop() {
  // Part-assembled incoming message.
  IncomingMessage Incoming;

  // Bounds the drain after a local disconnect so that a controller that has
  // stopped reading doesn't prevent the reactor from exiting.
  // The deadline is established on the first call. Returns the milliseconds
  // left to wait.
  auto DrainTimeRemaining =
      [DrainDeadline =
           std::optional<std::chrono::steady_clock::time_point>()]() mutable
      -> int {
    constexpr auto DrainTimeout = std::chrono::seconds(5);
    auto Now = std::chrono::steady_clock::now();
    if (!DrainDeadline)
      DrainDeadline = Now + DrainTimeout;
    auto Remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        *DrainDeadline - Now);
    return std::max(static_cast<int>(Remaining.count()), 0);
  };

  for (;;) {
    pollfd PollFDs[2];
    PollFDs[0].fd = Sock.get();
    PollFDs[0].events = POLLIN;
    PollFDs[0].revents = 0;
    PollFDs[1].fd = WakeRead.get();
    PollFDs[1].events = POLLIN;
    PollFDs[1].revents = 0;

    // Check for outgoing messages / draining-state.
    bool Draining = false;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Queue.empty())
        PollFDs[0].events |= POLLOUT;
      else if (CurState == State::Draining)
        return Error::success(); // The hang-up has gone out.
      Draining = CurState == State::Draining;
    }

    // Poll timeout.
    int Timeout = -1;

    // If draining, bound the poll by the timeout window or exit if the window
    // has closed.
    if (Draining) {
      Timeout = DrainTimeRemaining();
      if (Timeout == 0) {
        ORC_RT_LOG(Info, ControllerAccess,
                   "SimpleRemoteCA/socket drain timed out with messages still "
                   "queued; dropping them");
        return Error::success();
      }
    }

    // Check readiness (with timout if draining).
    while (::poll(PollFDs, 2, Timeout) < 0) {
      if (errno == EINTR)
        continue;
      return makeError("poll", errno);
    }

    // Drain the wake notification so that next poll blocks.
    if (PollFDs[1].revents & POLLIN) {
      char Buf[64];
      while (::recv(WakeRead.get(), Buf, sizeof(Buf), 0) > 0)
        ;
    }

    // Read before sending, so that a peer's parting hang-up reaches us rather
    // than the EPIPE from a send to a peer that has already gone.
    //
    // POLLHUP and POLLERR read too. A closed peer may have left buffered
    // messages, and recv returning zero is what separates an orderly close from
    // a truncated one; POLLERR says only that something is pending, so recv is
    // what reports what.
    auto Next = Action::Continue;
    if (PollFDs[0].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) {
      auto A = readAndDispatch(Incoming);
      if (!A)
        return A.takeError();
      Next = *A;
    }

    // If readAndDispatch got a hang-up then just do a best-effort send of the
    // remaining queue items, then return.
    if (Next == Action::End) {
      if (auto Err = drainSends()) {
        [[maybe_unused]] std::string Msg = toString(std::move(Err));
        ORC_RT_LOG(Info, ControllerAccess,
                   "SimpleRemoteCA/socket final-flush: " ORC_RT_LOG_PUB_S,
                   Msg.c_str());
      }
      return Error::success();
    }

    // Otherwise send any remaining queue items.
    if (PollFDs[0].revents & POLLOUT)
      if (auto Err = drainSends())
        return Err;
  }
}

Error SocketSimpleRemoteCA::drainSends() {
  for (;;) {
    OutgoingMessage *Msg = nullptr;
    {
      std::scoped_lock<std::mutex> Lock(M);
      if (Queue.empty())
        return Error::success();
      Msg = &Queue.front();
    }

    // Sent without the lock: only the reactor pops, and appending to a deque
    // never moves an existing element.
    auto Bytes = Msg->pending();
    ssize_t N = retryOnEINTR([&] {
      return ::send(Sock.get(), Bytes.data(), Bytes.size(), MSG_NOSIGNAL);
    });
    if (N < 0)
      return isWouldBlock(errno) ? Error::success() : makeError("send", errno);
    Msg->advance(N);

    if (Msg->complete()) {
      std::scoped_lock<std::mutex> Lock(M);
      Queue.pop_front();
    }
  }
}

Expected<SocketSimpleRemoteCA::Action>
SocketSimpleRemoteCA::readAndDispatch(IncomingMessage &Incoming) {
  for (;;) {
    // A non-blocking recv can stop anywhere, including mid-header, so this
    // resumes wherever the last left off.
    if (auto Bytes = Incoming.pending(); !Bytes.empty()) {
      ssize_t N = retryOnEINTR(
          [&] { return ::recv(Sock.get(), Bytes.data(), Bytes.size(), 0); });
      if (N < 0) {
        if (isWouldBlock(errno))
          return Action::Continue;
        return makeError("recv", errno);
      }
      if (N == 0)
        return make_error<StringError>(
            "Connection closed without a hang-up message");
      Incoming.advance(N);
      continue;
    }

    // Header full: decode it, then read the payload it describes. An empty
    // payload leaves nothing pending, so the next pass dispatches.
    if (!Incoming.complete()) {
      if (auto Err = Incoming.decodeHeader())
        return Err;
      continue;
    }

    // Taken before dispatching: a handler may send, re-entering this object.
    auto F = Incoming.fields();
    auto A = handleMessage(F.OpC, F.SeqNo, F.Tag, Incoming.take());
    if (!A || *A == Action::End)
      return A;
  }
}

SocketSimpleRemoteCA::OnControllerCallReturn
SocketSimpleRemoteCA::takePendingCall(uint64_t SeqNo) {
  std::scoped_lock<std::mutex> Lock(M);
  return takeCall(SeqNo);
}

Expected<std::shared_ptr<Session::ControllerAccess>>
createSimpleRemoteCAOverSocket(Session &S, SocketHandle Sock) {
  return SocketSimpleRemoteCA::Create(S, std::move(Sock));
}

} // namespace orc_rt
