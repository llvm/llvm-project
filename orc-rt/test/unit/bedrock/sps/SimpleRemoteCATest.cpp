//===- SimpleRemoteCATest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's SimpleRemoteCA protocol base class.
//
// SimpleRemoteCA is transport-independent, so these tests drive its protocol
// operations directly through a capture-only test double and observe the
// results via a Session.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"

#include "gtest/gtest.h"

#include "CommonTestUtils.h"

#include "orc-rt/support/sps/SimplePackedSerialization.h"

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace orc_rt;

namespace {

// A SimpleRemoteCA with no real transport. It exposes the protected protocol
// operations for tests, records every message the base asks it to send, and
// completes teardown as soon as the base begins it -- the shortest thing a real
// transport could do.
class TestSimpleRemoteCA : public SimpleRemoteCA {
public:
  using SimpleRemoteCA::Action;
  using SimpleRemoteCA::encodeHangupPayload;
  using SimpleRemoteCA::finishTeardown;
  using SimpleRemoteCA::handleMessage;
  using SimpleRemoteCA::Opcode;

  // A message the base handed to the transport.
  struct Sent {
    Opcode Op;
    uint64_t SeqNo;
    orc_rt_ControllerHandlerTag Tag;
    WrapperFunctionBuffer Payload;
  };

  /// Messages is owned by the caller so that it outlives a CA that teardown
  /// destroys.
  TestSimpleRemoteCA(Session &S, std::deque<Sent> &Messages,
                     TestSimpleRemoteCA **Self = nullptr,
                     bool DropOutDuringConnect = false)
      : SimpleRemoteCA(S), Messages(Messages),
        DropOutDuringConnect(DropOutDuringConnect) {
    if (Self)
      *Self = this;
  }

  void connect(BootstrapInfo BI) override {
    // Models a transport that enabled delivery and then dropped out before it
    // got as far as accepting -- the sanctioned failed-connect path.
    if (DropOutDuringConnect)
      finishTeardown(make_error<StringError>("dropped during connect"));
    beginAccepting(BI);
  }

  void sendMessage(Opcode Op, uint64_t SeqNo, orc_rt_ControllerHandlerTag Tag,
                   WrapperFunctionBuffer Payload) override {
    Messages.push_back(Sent{Op, SeqNo, Tag, std::move(Payload)});
  }

  void beginTeardown() override {
    ++TeardownsBegun;
    // What a transport owes for a local disconnect: an orderly reason, sent
    // last.
    sendMessage(Opcode::Hangup, 0, nullptr,
                encodeHangupPayload(Error::success()));
    if (FinishTeardownOnBegin)
      finishTeardown(Error::success());
  }

  /// Clear to model a transport whose shutdown is asynchronous.
  bool FinishTeardownOnBegin = true;
  /// Set to tear down from inside connect, before beginAccepting runs.
  bool DropOutDuringConnect;

  unsigned TeardownsBegun = 0;
  // A deque, not a vector: messages() returns pointers into this, and a later
  // send must not invalidate them.
  std::deque<Sent> &Messages;
};

using SentMessages = std::deque<TestSimpleRemoteCA::Sent>;

// Messages of the given opcode, in send order.
std::vector<const TestSimpleRemoteCA::Sent *>
messages(const SentMessages &Msgs, TestSimpleRemoteCA::Opcode Op) {
  std::vector<const TestSimpleRemoteCA::Sent *> Result;
  for (auto &M : Msgs)
    if (M.Op == Op)
      Result.push_back(&M);
  return Result;
}

constexpr uint64_t opc(TestSimpleRemoteCA::Opcode Op) {
  return static_cast<uint64_t>(Op);
}

// Wrapper that echoes its arguments back as the result.
void echoWrapper(orc_rt_SessionRef S, orc_rt_WrapperFunctionBuffer ArgBytes,
                 orc_rt_WrapperFunctionReturn Return, uint64_t CallId) {
  Return(S, ArgBytes, CallId);
}

// Expect an Expected<T> to hold an error whose message equals ExpectedMsg.
template <typename T> void expectError(Expected<T> R, const char *ExpectedMsg) {
  if (R)
    ADD_FAILURE() << "expected error \"" << ExpectedMsg << "\", got a value";
  else
    EXPECT_EQ(toString(R.takeError()), ExpectedMsg);
}

} // namespace

TEST(SimpleRemoteCATest, SetupMessageRoundTrips) {
  // The setup payload's wire format is shared with the controller (it matches
  // LLVM's SPSSimpleRemoteEPCExecutorInfo), so pin the field order and types by
  // decoding a payload the encoder produced.
  using SPSSetup = SPSTuple<SPSString, uint64_t,
                            SPSSequence<SPSTuple<SPSString, SPSSequence<char>>>,
                            SPSSequence<SPSTuple<SPSString, SPSExecutorAddr>>>;

  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);

  int SomeSymbol = 0;
  SimpleSymbolTable Symbols;
  std::vector<std::pair<std::string, const void *>> SymbolDefs = {
      {"foo", &SomeSymbol}};
  cantFail(Symbols.addUnique(SymbolDefs));

  BootstrapInfo BI(S, std::move(Symbols),
                   BootstrapInfo::ValueMap{{"key", "value"}});

  // connect sends setup; take the payload from the message the base produced.
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(std::move(BI), Msgs, &CA);
  ASSERT_TRUE(CA);
  auto Setups = messages(Msgs, TestSimpleRemoteCA::Opcode::Setup);
  ASSERT_EQ(Setups.size(), 1u);
  auto &Payload = Setups[0]->Payload;

  std::string Triple;
  uint64_t PageSize = 0;
  std::unordered_map<std::string, std::string> Values;
  std::unordered_map<std::string, ExecutorAddr> DecodedSymbols;
  SPSInputBuffer IB(Payload.data(), Payload.size());
  ASSERT_TRUE(SPSSetup::AsArgList::deserialize(IB, Triple, PageSize, Values,
                                               DecodedSymbols));

  EXPECT_EQ(Triple, S.processInfo().targetTriple());
  EXPECT_EQ(PageSize, S.processInfo().pageSize());
  EXPECT_EQ(Values,
            (std::unordered_map<std::string, std::string>{{"key", "value"}}));
  EXPECT_EQ(DecodedSymbols, (std::unordered_map<std::string, ExecutorAddr>{
                                {"foo", ExecutorAddr::fromPtr(&SomeSymbol)}}));

  // The payload holds exactly the setup fields: no padding, and nothing the
  // controller would be left to interpret.
  EXPECT_EQ(static_cast<size_t>(IB.data() - Payload.data()), Payload.size());
}

TEST(SimpleRemoteCATest, HandleMessageRejectsInvalidOpcode) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  expectError(CA.handleMessage(/*OpC=*/99, /*SeqNo=*/0, ExecutorAddr(),
                               WrapperFunctionBuffer()),
              "Invalid opcode 99");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsUnexpectedSetup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Setup), 0,
                               ExecutorAddr(), WrapperFunctionBuffer()),
              "Unexpected Setup message");
}

TEST(SimpleRemoteCATest, HandleMessageAcceptsWellFormedHangup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  auto R = CA.handleMessage(
      opc(TestSimpleRemoteCA::Opcode::Hangup), 0, ExecutorAddr(),
      TestSimpleRemoteCA::encodeHangupPayload(Error::success()));
  if (!R)
    ADD_FAILURE() << "unexpected error: " << toString(R.takeError());
  else
    EXPECT_EQ(*R, TestSimpleRemoteCA::Action::End);
}

TEST(SimpleRemoteCATest, HandleMessageReportsHangupReason) {
  // A hang-up carrying a reason ends the session with that reason, rather than
  // reporting a plain End: the reactor turns it into the disconnection error.
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  expectError(CA.handleMessage(
                  opc(TestSimpleRemoteCA::Opcode::Hangup), 0, ExecutorAddr(),
                  TestSimpleRemoteCA::encodeHangupPayload(
                      make_error<StringError>("controller ran out of x"))),
              "controller ran out of x");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsMalformedHangup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);

  // A hang-up must carry no sequence number and no tag.
  expectError(CA.handleMessage(
                  opc(TestSimpleRemoteCA::Opcode::Hangup), /*SeqNo=*/5,
                  ExecutorAddr(),
                  TestSimpleRemoteCA::encodeHangupPayload(Error::success())),
              "Malformed hang-up message");
  expectError(CA.handleMessage(
                  opc(TestSimpleRemoteCA::Opcode::Hangup), 0,
                  ExecutorAddr(0x1000),
                  TestSimpleRemoteCA::encodeHangupPayload(Error::success())),
              "Malformed hang-up message");

  // It must carry a deserializable reason. An empty payload is never valid --
  // the two ends are rev-locked, so this is a bug in the peer rather than skew.
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Hangup), 0,
                               ExecutorAddr(), WrapperFunctionBuffer()),
              "Malformed hang-up message: could not deserialize reason");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsResultWithTag) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Result),
                               /*SeqNo=*/1, ExecutorAddr(0x1000),
                               WrapperFunctionBuffer()),
              "Result message should not carry a handler tag");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsResultForUnknownSequenceNumber) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  SentMessages Msgs;
  TestSimpleRemoteCA CA(S, Msgs);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Result),
                               /*SeqNo=*/7, ExecutorAddr(),
                               WrapperFunctionBuffer::copyFrom("r", 1)),
              "No pending call for sequence number 7");
}

TEST(SimpleRemoteCATest, ResultCompletesPendingCall) {
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);

  // Originate a controller call; the test double registers it as pending.
  std::optional<std::string> Res;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        ASSERT_FALSE(R.getOutOfBandError()) << R.getOutOfBandError();
        Res = std::string(R.data(), R.size());
      },
      nullptr, WrapperFunctionBuffer());

  auto Calls = messages(Msgs, TestSimpleRemoteCA::Opcode::Call);
  ASSERT_EQ(Calls.size(), 1u);
  uint64_t SeqNo = Calls[0]->SeqNo;
  ASSERT_NE(SeqNo, 0u);
  ASSERT_FALSE(Res) << "handler fired before the result arrived";

  // Deliver the matching result; handleMessage completes the call inline.
  auto R = CA->handleMessage(opc(TestSimpleRemoteCA::Opcode::Result), SeqNo,
                             ExecutorAddr(),
                             WrapperFunctionBuffer::copyFrom("a", 1));
  if (!R)
    ADD_FAILURE() << "unexpected error: " << toString(R.takeError());
  else
    EXPECT_EQ(*R, TestSimpleRemoteCA::Action::Continue);

  ASSERT_TRUE(Res);
  EXPECT_EQ(*Res, "a");
}

TEST(SimpleRemoteCATest, CallDispatchesWrapperAndReturnsResult) {
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);

  // A Call names echoWrapper via the tag; SeqNo is the call id.
  auto R = CA->handleMessage(
      opc(TestSimpleRemoteCA::Opcode::Call), /*SeqNo=*/42,
      ExecutorAddr::fromPtr(reinterpret_cast<void *>(echoWrapper)),
      WrapperFunctionBuffer::copyFrom("world", 5));
  if (!R)
    ADD_FAILURE() << "unexpected error: " << toString(R.takeError());
  else
    EXPECT_EQ(*R, TestSimpleRemoteCA::Action::Continue);

  auto Results = messages(Msgs, TestSimpleRemoteCA::Opcode::Result);
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_EQ(Results[0]->SeqNo, 42u);
  auto &Result = Results[0]->Payload;
  EXPECT_EQ(std::string(Result.data(), Result.size()), "world");
}

TEST(SimpleRemoteCATest, DisconnectDrainsPendingCalls) {
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);

  // A controller call is in flight (no result will arrive).
  std::optional<std::string> Err;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *M = R.getOutOfBandError())
          Err = M;
      },
      nullptr, WrapperFunctionBuffer());
  ASSERT_EQ(messages(Msgs, TestSimpleRemoteCA::Opcode::Call).size(), 1u);
  ASSERT_FALSE(Err) << "handler fired before disconnect";

  // Detach drives disconnect, which drains the pending call with a
  // "disconnected" error.
  S.detach([] {});

  ASSERT_TRUE(Err);
  EXPECT_EQ(*Err, "disconnected");
}

TEST(SimpleRemoteCATest, CallRacingTeardownFailsInline) {
  // Teardown has begun but the transport hasn't finished, so the call reaches
  // the CA and must be failed on the spot rather than left pending -- nothing
  // will drain it if the drain has already run.
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);
  CA->FinishTeardownOnBegin = false;

  CA->disconnect();
  ASSERT_EQ(CA->TeardownsBegun, 1u);

  std::optional<std::string> Err;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *M = R.getOutOfBandError())
          Err = M;
      },
      nullptr, WrapperFunctionBuffer());

  // The handler ran before callController returned, and nothing was sent.
  ASSERT_TRUE(Err);
  EXPECT_EQ(*Err, "disconnected");
  EXPECT_TRUE(messages(Msgs, TestSimpleRemoteCA::Opcode::Call).empty());

  CA->finishTeardown(Error::success());
}

TEST(SimpleRemoteCATest, DisconnectIsIdempotentAndDefersNotification) {
  unsigned Notifications = 0;
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) {
    ++Notifications;
    cantFail(std::move(Err));
  });

  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);
  CA->FinishTeardownOnBegin = false;

  CA->disconnect();
  CA->disconnect();
  EXPECT_EQ(CA->TeardownsBegun, 1u) << "second disconnect was not a no-op";
  EXPECT_EQ(Notifications, 0u) << "notified before the transport finished";

  // finishTeardown may destroy *CA, so it is the last thing to touch it.
  CA->finishTeardown(Error::success());
  EXPECT_EQ(Notifications, 1u);
}

TEST(SimpleRemoteCATest, TransportInitiatedTeardownNotifiesWithoutHangup) {
  // A hang-up from the controller, or a transport failure, goes straight to
  // finishTeardown: the peer already knows, so nothing is sent.
  unsigned Notifications = 0;
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) {
    ++Notifications;
    EXPECT_EQ(toString(std::move(Err)), "controller vanished");
  });

  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA);
  ASSERT_TRUE(CA);

  // Nothing sends a hang-up on this path, because beginTeardown never runs.
  EXPECT_EQ(CA->TeardownsBegun, 0u);
  CA->finishTeardown(make_error<StringError>("controller vanished"));

  EXPECT_EQ(Notifications, 1u);
}

TEST(SimpleRemoteCATest, DropOutBeforeAcceptingIsANoOp) {
  // A transport that enables delivery before calling beginAccepting can lose
  // the connection first, so beginAccepting has to tolerate running after
  // teardown has already completed: it accepts nothing and sends nothing.
  unsigned Notifications = 0;
  // Declared before the Session: its destructor drives teardown, which
  // records a hang-up.
  SentMessages Msgs;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.setOnDisconnect([&](Error Err) {
    ++Notifications;
    EXPECT_EQ(toString(std::move(Err)), "dropped during connect");
  });

  // Msgs outlives the CA, which is freed as attach returns.
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), Msgs, &CA,
                               /*DropOutDuringConnect=*/true);

  EXPECT_EQ(Notifications, 1u);
  EXPECT_TRUE(Msgs.empty()) << "setup was sent after teardown completed";
}
