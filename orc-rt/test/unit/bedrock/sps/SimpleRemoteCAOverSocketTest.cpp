//===- SimpleRemoteCAOverSocketTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for createSimpleRemoteCAOverSocket.
//
// Each test attaches a CA to one end of a socket pair and plays the controller
// on the other, so the reactor, the connection state machine and teardown are
// covered without a second process.
//
// The controller side frames messages with the CA's own encoder, reached
// through the friend fixture. That makes these tests about behaviour rather
// than byte layout; conformance to LLVM's SimpleRemoteEPC is established by the
// cross-process regression tests.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCAOverSocket.h"

#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"

#include "gtest/gtest.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
#include "ErrorMatchers.h"
#include "bedrock/SocketTestUtils.h"

#include "orc-rt-internal/support/Endian.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::HasSubstr;

namespace orc_rt {

/// Plays the controller against a SimpleRemote CA running over a socket.
///
/// Frames messages with the protocol's own encoders rather than reimplementing
/// them. Those are protected on SimpleRemoteCA, so a derived type re-exports
/// the pieces the controller side needs; the transport itself is not a type a
/// test can reach, which is the point of it being a factory.
class SimpleRemoteCAOverSocketTest : public ::testing::Test {
  struct CA : SimpleRemoteCA {
    using SimpleRemoteCA::decodeResult;
    using SimpleRemoteCA::encodeHangup;
    using SimpleRemoteCA::encodeResult;
    using MsgHeader = SimpleRemoteCA::MsgHeader;
    using Opcode = SimpleRemoteCA::Opcode;
    using ResultKind = SimpleRemoteCA::ResultKind;
  };

public:
  using MsgHeader = CA::MsgHeader;
  using Opcode = CA::Opcode;

  /// A connected socket pair, established before each test: Near is the end the
  /// CA adopts, Far the end the test drives as the controller.
  ///
  /// Declared before S so that the Session is torn down first, while Far is
  /// still open -- that is the order a test's own locals had.
  SocketHandle Near, Far;

  /// The Session under test. noErrors is not boilerplate: Session routes the
  /// disconnect reason to reportError when no on-disconnect handler was
  /// installed, so any test that ends abnormally without asking for the error
  /// aborts here rather than passing quietly. Tests that want the error install
  /// their own handler before attaching.
  Session S{mockExecutorProcessInfo(), inlineDispatch, noErrors};

  void SetUp() override {
    auto P = makeStreamSocketPair();
    ASSERT_TRUE(!!P) << toString(P.takeError());
    Near = std::move(P->first);
    Far = std::move(P->second);
  }

  /// Creates a CA over Near and attaches it, the way a connector would.
  Error attachOverSocket() {
    auto CA = createSimpleRemoteCAOverSocket(S, std::move(Near));
    if (!CA)
      return CA.takeError();
    S.attach(std::move(*CA), BootstrapInfo(S));
    return Error::success();
  }

  static constexpr size_t HeaderSize = MsgHeader::Size;

  /// A message as it appears on the wire, with the size field consumed.
  struct Frame {
    MsgHeader::Fields Fields;
    std::vector<char> Payload;

    Opcode opcode() const { return static_cast<Opcode>(Fields.OpC); }
    uint64_t seqNo() const { return Fields.SeqNo; }
    uint64_t tag() const { return Fields.Tag; }

    /// The payload as a view, for comparison against test data.
    std::string_view payload() const {
      return {Payload.data(), Payload.size()};
    }
  };

  /// A framed message, ready to write to the socket.
  static std::vector<char> frame(Opcode Op, uint64_t SeqNo, uint64_t Tag,
                                 std::string_view Payload) {
    std::vector<char> Buf(HeaderSize + Payload.size());
    MsgHeader::encode(Buf.data(), Op, SeqNo, Tag, Payload.size());
    if (!Payload.empty())
      memcpy(Buf.data() + HeaderSize, Payload.data(), Payload.size());
    return Buf;
  }

  static Error writeFrame(NativeSocketHandle Sock, Opcode Op, uint64_t SeqNo,
                          uint64_t Tag = 0, std::string_view Payload = {}) {
    auto Buf = frame(Op, SeqNo, Tag, Payload);
    return sendAll(Sock, Buf.data(), Buf.size());
  }

  static Expected<Frame> readFrame(NativeSocketHandle Sock) {
    char H[HeaderSize];
    auto N = recvAll(Sock, H, HeaderSize);
    if (!N)
      return N.takeError();
    if (*N != HeaderSize)
      return make_error<StringError>(
          "peer closed before a full header arrived");

    Frame F;
    F.Fields = MsgHeader::decode(H);
    if (F.Fields.MsgSize < HeaderSize)
      return make_error<StringError>("framed message size is below the header");

    if (size_t PayloadSize = F.Fields.MsgSize - HeaderSize) {
      F.Payload.resize(PayloadSize);
      auto M = recvAll(Sock, F.Payload.data(), PayloadSize);
      if (!M)
        return M.takeError();
      if (*M != PayloadSize)
        return make_error<StringError>("peer closed mid-payload");
    }
    return F;
  }

  /// The payload a controller sends to hang up, via the CA's own encoder.
  static std::vector<char> hangupPayload(Error Err) {
    auto P = CA::encodeHangup(std::move(Err));
    return std::vector<char>(P.data(), P.data() + P.size());
  }

  using ResultKind = CA::ResultKind;

  /// The wire tag naming a wrapper function in this process.
  static uint64_t wrapperTag(void *Fn) {
    return ExecutorAddr::fromPtr(Fn).getValue();
  }

  /// The tag value that marks a result message as carrying a result of Kind.
  static uint64_t resultTag(ResultKind Kind) {
    return static_cast<uint64_t>(Kind);
  }

  /// The tag and payload a controller sends to return ResultBytes, via the CA's
  /// own encoder.
  static std::pair<uint64_t, std::vector<char>>
  resultMessage(WrapperFunctionBuffer ResultBytes) {
    auto [Kind, P] = CA::encodeResult(std::move(ResultBytes));
    return {resultTag(Kind), std::vector<char>(P.data(), P.data() + P.size())};
  }

  /// Decodes a result message the CA sent, as the controller would.
  static WrapperFunctionBuffer decodeResult(ResultKind Kind,
                                            std::string_view Payload) {
    return CA::decodeResult(
        Kind, WrapperFunctionBuffer::copyFrom(Payload.data(), Payload.size()));
  }
};

} // namespace orc_rt

namespace {

std::string_view view(const std::vector<char> &V) {
  return {V.data(), V.size()};
}

// Wrapper that echoes its arguments back as the result.
void echoWrapper(orc_rt_SessionRef S, orc_rt_WrapperFunctionBuffer ArgBytes,
                 orc_rt_WrapperFunctionReturn Return, uint64_t CallId) {
  Return(S, ArgBytes, CallId);
}

// A wrapper that defers its result: it stashes everything needed to return, so
// a test can complete the call later. A plain function pointer has nowhere to
// put context, so the caller passes the address of its own DeferredCall as the
// call's payload.
//
// The wrapper runs on the reactor thread and signals completion once its fields
// are populated. The test can wait on this by calling waitForCall.
//
// A caller's DeferredCall has to outlive any chance of the wrapper running,
// which for a stack one means waiting on waitForCall before leaving the scope.
struct DeferredCall {
  orc_rt_SessionRef S = nullptr;
  orc_rt_WrapperFunctionBuffer ArgBytes{};
  orc_rt_WrapperFunctionReturn Return = nullptr;
  uint64_t CallId = 0;

  /// The payload that names this object to deferringWrapper.
  std::string_view asCallPayload() const {
    return {reinterpret_cast<const char *>(&Self), sizeof(Self)};
  }

  void publish(orc_rt_SessionRef S, orc_rt_WrapperFunctionBuffer ArgBytes,
               orc_rt_WrapperFunctionReturn Return, uint64_t CallId) {
    this->S = S;
    this->ArgBytes = ArgBytes;
    this->Return = Return;
    this->CallId = CallId;
    Published.set_value();
  }

  /// Blocks until the wrapper has run, then returns its return function. The
  /// other fields are safe to read once this has returned; going through here
  /// is what makes them so.
  orc_rt_WrapperFunctionReturn waitForCall() {
    PublishedF.get();
    return Return;
  }

  static void wrapper(orc_rt_SessionRef S,
                      orc_rt_WrapperFunctionBuffer ArgBytes,
                      orc_rt_WrapperFunctionReturn Return, uint64_t CallId) {
    WrapperFunctionBuffer Args(ArgBytes);
    assert(Args.size() == sizeof(DeferredCall *) &&
           "deferringWrapper expects a DeferredCall address as its payload");
    DeferredCall *D = nullptr;
    memcpy(&D, Args.data(), sizeof(D));

    // ArgBytes is handed on rather than disposed: it goes back out as the
    // result.
    D->publish(S, Args.release(), Return, CallId);
  }

private:
  /// Storage for the address, so asCallPayload has something stable to point
  /// at: a view over a temporary would dangle.
  DeferredCall *const Self = this;

  // Declared in this order: PublishedF's initializer reads Published.
  std::promise<void> Published;
  std::future<void> PublishedF = Published.get_future();
};

// A wrapper that fails the way the wrapper machinery itself fails: an
// out-of-band error says the call could not be made sense of, rather than
// carrying a result the wrapper chose to return.
void outOfBandErrorWrapper(orc_rt_SessionRef S,
                           orc_rt_WrapperFunctionBuffer ArgBytes,
                           orc_rt_WrapperFunctionReturn Return,
                           uint64_t CallId) {
  WrapperFunctionBuffer Args(ArgBytes); // Disposed on the way out.
  Return(S,
         WrapperFunctionBuffer::createOutOfBandError(
             "Could not deserialize wrapper function arg data")
             .release(),
         CallId);
}

} // namespace

TEST_F(SimpleRemoteCAOverSocketTest, RejectsANonStreamSocket) {
  // The framing reads a message in as many parts as the stream delivers it, so
  // a socket that preserves message boundaries would truncate one.
  auto H = makeNativeNonStreamSocket();
  ASSERT_TRUE(H.has_value()) << "could not create a socket for the test";

  EXPECT_THAT_EXPECTED(
      createSimpleRemoteCAOverSocket(S, SocketHandle(*H)),
      FailedWithMessage(HasSubstr("requires a stream socket")));
  EXPECT_FALSE(isNativeSocketOpen(*H))
      << "a rejected socket is still owned, and must be closed";
}

TEST_F(SimpleRemoteCAOverSocketTest, SetupIsSentOnConnect) {
  ASSERT_FALSE(!!attachOverSocket());

  auto F = readFrame(Far.get());
  ASSERT_TRUE(!!F) << toString(F.takeError());
  EXPECT_EQ(F->opcode(), Opcode::Setup);
  EXPECT_EQ(F->seqNo(), 0u) << "setup carries no sequence number";
  EXPECT_EQ(F->tag(), 0u) << "setup carries no handler tag";
  EXPECT_FALSE(F->Payload.empty()) << "setup carries the bootstrap payload";
}

TEST_F(SimpleRemoteCAOverSocketTest,
       ControllerCallIsFramedAndResultCompletesIt) {
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  std::future<std::string> Result;
  S.callController(
      [SetResult = waitFor(Result)](WrapperFunctionBuffer R) mutable {
        SetResult(std::string(R.data(), R.size()));
      },
      nullptr, WrapperFunctionBuffer::copyFrom("args", 4));

  auto Call = readFrame(Far.get());
  ASSERT_TRUE(!!Call) << toString(Call.takeError());
  EXPECT_EQ(Call->opcode(), Opcode::Call);
  EXPECT_NE(Call->seqNo(), 0u)
      << "a call awaiting a result needs a sequence no.";
  EXPECT_EQ(Call->payload(), "args");

  // Reply under the same sequence number; the handler completes.
  ASSERT_FALSE(
      !!writeFrame(Far.get(), Opcode::Result, Call->seqNo(), 0, "reply"));
  EXPECT_EQ(Result.get(), "reply");
}

TEST_F(SimpleRemoteCAOverSocketTest, ControllerInitiatedCallReturnsAResult) {
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // A Call names the wrapper by tag, and its sequence number is the call id.
  auto Tag = wrapperTag(reinterpret_cast<void *>(echoWrapper));
  ASSERT_FALSE(
      !!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/42, Tag, "world"));

  auto R = readFrame(Far.get());
  ASSERT_TRUE(!!R) << toString(R.takeError());
  EXPECT_EQ(R->opcode(), Opcode::Result);
  EXPECT_EQ(R->seqNo(), 42u) << "the result must carry the call id back";
  EXPECT_EQ(R->payload(), "world");
}

TEST_F(SimpleRemoteCAOverSocketTest, OutOfBandErrorResultIsFramedAsItsOwnKind) {
  // An out-of-band error is a pointer to a message, not a serialized value, so
  // it has no bytes to put in a payload. Before it had a result kind of its own
  // there was no way to send one at all: framing it aborted the reactor thread.
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  auto Tag = wrapperTag(reinterpret_cast<void *>(outOfBandErrorWrapper));
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/7, Tag, "junk"));

  auto R = readFrame(Far.get());
  ASSERT_TRUE(!!R) << toString(R.takeError());
  EXPECT_EQ(R->opcode(), Opcode::Result);
  EXPECT_EQ(R->seqNo(), 7u) << "the result must carry the call id back";
  EXPECT_EQ(R->tag(), resultTag(ResultKind::OutOfBandError));

  // The controller recovers the message the wrapper produced, rather than the
  // empty result it would otherwise report as a deserialization failure.
  auto Decoded = decodeResult(ResultKind::OutOfBandError, R->payload());
  ASSERT_NE(Decoded.getOutOfBandError(), nullptr);
  EXPECT_STREQ(Decoded.getOutOfBandError(),
               "Could not deserialize wrapper function arg data");
}

TEST_F(SimpleRemoteCAOverSocketTest,
       OutOfBandErrorFromControllerCompletesTheCall) {
  // The other direction: a controller-side wrapper fails the same way, and the
  // handler waiting on the result must see an out-of-band error rather than an
  // empty result.
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  std::future<std::string> Result;
  S.callController(
      [SetResult = waitFor(Result)](WrapperFunctionBuffer R) mutable {
        const char *Msg = R.getOutOfBandError();
        SetResult(Msg ? std::string(Msg) : std::string("<not out-of-band>"));
      },
      nullptr, WrapperFunctionBuffer::copyFrom("args", 4));

  auto Call = readFrame(Far.get());
  ASSERT_TRUE(!!Call) << toString(Call.takeError());

  auto [Tag, Payload] = resultMessage(
      WrapperFunctionBuffer::createOutOfBandError("controller lost its mind"));
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Result, Call->seqNo(), Tag,
                            view(Payload)));

  EXPECT_EQ(Result.get(), "controller lost its mind");
}

TEST_F(SimpleRemoteCAOverSocketTest, UnknownResultKindEndsTheSession) {
  // An unrecognized kind means the controller is speaking a dialect this build
  // does not know, so the payload cannot be interpreted: terminal, as an
  // unrecognized opcode is.
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  std::future<std::string> Result;
  S.callController(
      [SetResult = waitFor(Result)](WrapperFunctionBuffer R) mutable {
        const char *Msg = R.getOutOfBandError();
        SetResult(Msg ? std::string(Msg) : std::string("<not out-of-band>"));
      },
      nullptr, WrapperFunctionBuffer::copyFrom("args", 4));

  auto Call = readFrame(Far.get());
  ASSERT_TRUE(!!Call) << toString(Call.takeError());

  uint64_t UnknownKind = static_cast<uint64_t>(ResultKind::LastResultKind) + 1;
  ASSERT_FALSE(
      !!writeFrame(Far.get(), Opcode::Result, Call->seqNo(), UnknownKind, ""));

  auto Err = Disconnected.get();
  ASSERT_TRUE(!!Err) << "an uninterpretable result must not end cleanly";
  EXPECT_EQ(toString(std::move(Err)),
            "Malformed result message: invalid kind 2");

  // The call it could not answer is still settled on the way out, rather than
  // left waiting forever.
  EXPECT_NE(Result.get(), "<not out-of-band>");
}

TEST_F(SimpleRemoteCAOverSocketTest, EmptyPayloadsRoundTrip) {
  // A message whose size is exactly the header. Nothing is pending the moment
  // the header is decoded, which is the case that stops "header full" and
  // "header decoded" being the same state, in both directions: the echoed
  // result is empty too.
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  auto Tag = wrapperTag(reinterpret_cast<void *>(echoWrapper));
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/8, Tag));

  auto R = readFrame(Far.get());
  ASSERT_TRUE(!!R) << toString(R.takeError());
  EXPECT_EQ(R->opcode(), Opcode::Result);
  EXPECT_EQ(R->seqNo(), 8u);
  EXPECT_TRUE(R->Payload.empty());

  // Still framing correctly afterwards: an empty message must not desynchronise
  // the stream.
  ASSERT_FALSE(
      !!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/9, Tag, "after"));
  auto R2 = readFrame(Far.get());
  ASSERT_TRUE(!!R2) << toString(R2.takeError());
  EXPECT_EQ(R2->seqNo(), 9u);
  EXPECT_EQ(R2->payload(), "after");
}

TEST_F(SimpleRemoteCAOverSocketTest, PartialWritesAreReassembled) {
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // One byte at a time, so the reactor sees the message split in every possible
  // place -- including mid-header.
  auto Tag = wrapperTag(reinterpret_cast<void *>(echoWrapper));
  auto Buf = frame(Opcode::Call, /*SeqNo=*/7, Tag, "drip");
  for (char C : Buf)
    ASSERT_FALSE(!!sendAll(Far.get(), &C, 1));

  auto R = readFrame(Far.get());
  ASSERT_TRUE(!!R) << toString(R.takeError());
  EXPECT_EQ(R->opcode(), Opcode::Result);
  EXPECT_EQ(R->seqNo(), 7u);
  EXPECT_EQ(R->payload(), "drip");
}

TEST_F(SimpleRemoteCAOverSocketTest, NothingIsQueuedBehindTheHangup) {
  // The hang-up must be the last message on the wire, and sendWrapperResult
  // does no state check -- the base leaves that to the transport. So a result
  // that arrives once teardown has begun must be dropped rather than appended.
  //
  // The window is between beginTeardown queueing the hang-up and the reactor
  // finishing, which is narrow. It is held open here by never reading the far
  // end until the very end: the reactor parks in would-block with a part-sent
  // message, so nothing drains while the late result is submitted.
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // Echo back more than the socket buffer can hold, so the reactor stalls
  // part-way through sending the result.
  const std::string Big(StallingPayloadSize, 'x');
  auto EchoTag = wrapperTag(reinterpret_cast<void *>(echoWrapper));
  ASSERT_FALSE(
      !!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/1, EchoTag, Big));

  // A second call that will not have returned when teardown starts. The
  // wrapper is told where to park its state by being handed this object's
  // address as the call payload.
  DeferredCall Deferred;
  auto DeferTag = wrapperTag(reinterpret_cast<void *>(DeferredCall::wrapper));
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/2, DeferTag,
                            Deferred.asCallPayload()));

  // Wait for the wrapper to run without draining the socket.
  auto DeferredReturn = Deferred.waitForCall();

  // Queues the hang-up and latches the queue. Returns without waiting for the
  // reactor, which is still stalled.
  S.detach();

  // Too late: this must not reach the wire.
  DeferredReturn(Deferred.S, Deferred.ArgBytes, Deferred.CallId);

  // Now drain. The stalled result completes, then the hang-up, then EOF.
  auto R = readFrame(Far.get());
  ASSERT_TRUE(!!R) << toString(R.takeError());
  EXPECT_EQ(R->opcode(), Opcode::Result);
  EXPECT_EQ(R->Payload.size(), Big.size());

  auto H = readFrame(Far.get());
  ASSERT_TRUE(!!H) << toString(H.takeError());
  EXPECT_EQ(H->opcode(), Opcode::Hangup) << "the hang-up did not come last";

  char Byte = 0;
  auto N = recvAll(Far.get(), &Byte, 1);
  ASSERT_TRUE(!!N) << toString(N.takeError());
  EXPECT_EQ(*N, 0u) << "a message was queued behind the hang-up";
}

TEST_F(SimpleRemoteCAOverSocketTest, HangupFromControllerEndsTheSession) {
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // An orderly hang-up carries a success Error as its reason.
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Hangup, 0, 0,
                            view(hangupPayload(Error::success()))));

  EXPECT_FALSE(!!Disconnected.get()) << "an orderly hang-up is not an error";
}

TEST_F(SimpleRemoteCAOverSocketTest, PeerReasonSurvivesAStalledSendQueue) {
  // A hang-up reason from the controller must be what ends the session, even
  // when we have messages queued that can no longer be delivered. Sending first
  // would fail with EPIPE and report that instead, losing the reason.
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // Echo back more than the socket will hold, and never read it, so the reactor
  // is left with a part-sent message queued.
  const std::string Big(StallingPayloadSize, 'x');
  auto Tag = wrapperTag(reinterpret_cast<void *>(echoWrapper));
  ASSERT_FALSE(!!writeFrame(Far.get(), Opcode::Call, /*SeqNo=*/1, Tag, Big));

  // Hang up with a reason, then vanish. The reason is buffered on our side of
  // the socket and survives the close.
  ASSERT_FALSE(!!writeFrame(
      Far.get(), Opcode::Hangup, 0, 0,
      view(hangupPayload(make_error<StringError>("controller ran out of x")))));
  Far.reset();

  auto Err = Disconnected.get();
  ASSERT_TRUE(!!Err) << "a hang-up carrying a reason ends with that reason";
  EXPECT_EQ(toString(std::move(Err)), "controller ran out of x");
}

TEST_F(SimpleRemoteCAOverSocketTest, TruncatedMessageIsReportedAsAnError) {
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // Half a header, then gone: distinguishable from a close at a boundary.
  char Half[HeaderSize / 2] = {};
  ASSERT_FALSE(!!sendAll(Far.get(), Half, sizeof(Half)));
  Far.reset();

  auto Err = Disconnected.get();
  EXPECT_TRUE(!!Err) << "a truncated message must not look like a clean end";
  EXPECT_EQ(toString(std::move(Err)),
            "Connection closed without a hang-up message");
}

TEST_F(SimpleRemoteCAOverSocketTest, PeerCloseWithoutAHangupIsAnError) {
  // A peer that means to end the session says so with a hang-up. Vanishing
  // instead means it crashed or was killed, which must not be reported as a
  // clean shutdown -- a controller that dies has to fail the session.
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // Closed between messages, so nothing is truncated -- it is simply gone.
  Far.reset();

  auto Err = Disconnected.get();
  ASSERT_TRUE(!!Err) << "a silent close is not an orderly end";
  EXPECT_EQ(toString(std::move(Err)),
            "Connection closed without a hang-up message");
}

TEST_F(SimpleRemoteCAOverSocketTest, MessageSizeBelowHeaderIsRejected) {
  std::future<Error> Disconnected;
  S.setOnDisconnect(waitFor(Disconnected));
  ASSERT_FALSE(!!attachOverSocket());
  ASSERT_TRUE(!!readFrame(Far.get())) << "expected setup first";

  // A size that excludes its own header would make the payload length negative.
  char H[HeaderSize] = {};
  endian_write<uint64_t>(H, HeaderSize - 1, endian::little);
  ASSERT_FALSE(!!sendAll(Far.get(), H, sizeof(H)));

  auto Err = Disconnected.get();
  EXPECT_TRUE(!!Err);
  EXPECT_EQ(toString(std::move(Err)), "Message size smaller than its header");
}
