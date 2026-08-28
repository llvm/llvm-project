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

#include "orc-rt/bedrock/SimpleRemoteCA.h"

#include "gtest/gtest.h"

#include "CommonTestUtils.h"

#include "orc-rt/bedrock/SimplePackedSerialization.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace orc_rt;

namespace {

// A SimpleRemoteCA with no real transport. It exposes the protected protocol
// operations for tests, records outgoing controller calls as pending results,
// and records the wrapper results the executor produces.
class TestSimpleRemoteCA : public SimpleRemoteCA {
public:
  TestSimpleRemoteCA(Session &S, TestSimpleRemoteCA **Self = nullptr)
      : SimpleRemoteCA(S) {
    if (Self)
      *Self = this;
  }

  using SimpleRemoteCA::Action;
  using SimpleRemoteCA::encodeHangupPayload;
  using SimpleRemoteCA::encodeSetupMessage;
  using SimpleRemoteCA::handleMessage;
  using SimpleRemoteCA::Opcode;

  void connect(BootstrapInfo) override {}
  void disconnect() override {
    // Mirror a real transport's teardown order: drain, then notify. A locally
    // requested disconnect is orderly, so the mode is success.
    failAllPendingCalls();
    notifyDisconnected(Error::success());
  }
  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag,
                      WrapperFunctionBuffer) override {
    LastCallSeqNo = registerPendingCall(std::move(OnComplete));
  }
  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) override {
    WrapperResults.emplace_back(CallId, std::move(ResultBytes));
  }

  uint64_t LastCallSeqNo = 0;
  std::vector<std::pair<uint64_t, WrapperFunctionBuffer>> WrapperResults;
};

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

  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);

  int SomeSymbol = 0;
  SimpleSymbolTable Symbols;
  std::vector<std::pair<std::string, const void *>> SymbolDefs = {
      {"foo", &SomeSymbol}};
  cantFail(Symbols.addUnique(SymbolDefs));

  BootstrapInfo BI(S, std::move(Symbols),
                   BootstrapInfo::ValueMap{{"key", "value"}});
  auto Payload = TestSimpleRemoteCA::encodeSetupMessage(BI);

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
  TestSimpleRemoteCA CA(S);
  expectError(CA.handleMessage(/*OpC=*/99, /*SeqNo=*/0, ExecutorAddr(),
                               WrapperFunctionBuffer()),
              "Invalid opcode 99");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsUnexpectedSetup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  TestSimpleRemoteCA CA(S);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Setup), 0,
                               ExecutorAddr(), WrapperFunctionBuffer()),
              "Unexpected Setup message");
}

TEST(SimpleRemoteCATest, HandleMessageAcceptsWellFormedHangup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  TestSimpleRemoteCA CA(S);
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
  TestSimpleRemoteCA CA(S);
  expectError(CA.handleMessage(
                  opc(TestSimpleRemoteCA::Opcode::Hangup), 0, ExecutorAddr(),
                  TestSimpleRemoteCA::encodeHangupPayload(
                      make_error<StringError>("controller ran out of x"))),
              "controller ran out of x");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsMalformedHangup) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  TestSimpleRemoteCA CA(S);

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
  TestSimpleRemoteCA CA(S);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Result),
                               /*SeqNo=*/1, ExecutorAddr(0x1000),
                               WrapperFunctionBuffer()),
              "Result message should not carry a handler tag");
}

TEST(SimpleRemoteCATest, HandleMessageRejectsResultForUnknownSequenceNumber) {
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  TestSimpleRemoteCA CA(S);
  expectError(CA.handleMessage(opc(TestSimpleRemoteCA::Opcode::Result),
                               /*SeqNo=*/7, ExecutorAddr(),
                               WrapperFunctionBuffer::copyFrom("r", 1)),
              "No pending call for sequence number 7");
}

TEST(SimpleRemoteCATest, ResultCompletesPendingCall) {
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), &CA);
  ASSERT_TRUE(CA);

  // Originate a controller call; the test double registers it as pending.
  std::optional<std::string> Res;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        ASSERT_FALSE(R.getOutOfBandError()) << R.getOutOfBandError();
        Res = std::string(R.data(), R.size());
      },
      nullptr, WrapperFunctionBuffer());

  uint64_t SeqNo = CA->LastCallSeqNo;
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
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), &CA);
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

  ASSERT_EQ(CA->WrapperResults.size(), 1u);
  EXPECT_EQ(CA->WrapperResults[0].first, 42u);
  auto &Result = CA->WrapperResults[0].second;
  EXPECT_EQ(std::string(Result.data(), Result.size()), "world");
}

TEST(SimpleRemoteCATest, DisconnectDrainsPendingCalls) {
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  TestSimpleRemoteCA *CA = nullptr;
  S.attach<TestSimpleRemoteCA>(BootstrapInfo(S), &CA);
  ASSERT_TRUE(CA);

  // A controller call is in flight (no result will arrive).
  std::optional<std::string> Err;
  S.callController(
      [&](WrapperFunctionBuffer R) {
        if (const char *M = R.getOutOfBandError())
          Err = M;
      },
      nullptr, WrapperFunctionBuffer());
  ASSERT_NE(CA->LastCallSeqNo, 0u);
  ASSERT_FALSE(Err) << "handler fired before disconnect";

  // Detach drives disconnect, which drains the pending call with a
  // "disconnected" error.
  S.detach([] {});

  ASSERT_TRUE(Err);
  EXPECT_EQ(*Err, "disconnected");
}
