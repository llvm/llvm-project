//===- SimpleRemoteCATest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the transport-independent half of the SimpleRemote protocol.
//
// These cover the payload encodings and the pending-call table on their own.
// When a call may be registered, what ends a session, and who notifies the
// Session are all the transport's to decide, and are tested with it.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"

#include "gtest/gtest.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"

#include "orc-rt/support/sps/SimplePackedSerialization.h"

#include <string>
#include <unordered_map>
#include <vector>

using namespace orc_rt;

namespace {

/// Exposes the protected utilities, and captures the handler that
/// Session::callController hands over so tests have a real one to register.
class TestCA : public SimpleRemoteCA {
public:
  using ControllerAccess::failPendingControllerCall;
  using ControllerAccess::OnControllerCallReturn;
  using SimpleRemoteCA::decodeHangup;
  using SimpleRemoteCA::encodeHangup;
  using SimpleRemoteCA::encodeSetup;
  using SimpleRemoteCA::PendingCallsMap;
  using SimpleRemoteCA::registerCall;
  using SimpleRemoteCA::takeAllCalls;
  using SimpleRemoteCA::takeCall;

  TestCA(Session &S, TestCA **Self = nullptr) : SimpleRemoteCA(S) {
    if (Self)
      *Self = this;
  }

  void connect(BootstrapInfo BI) override {}

  void disconnect() override {
    // What a transport owes on the way out: fail what is pending, then notify.
    for (auto &[SeqNo, OnComplete] : takeAllCalls())
      failPendingControllerCall(std::move(OnComplete));
    notifyDisconnected(Error::success());
  }

  void callController(OnControllerCallReturn OnComplete,
                      orc_rt_ControllerHandlerTag T,
                      WrapperFunctionBuffer ArgBytes) override {
    Captured = std::move(OnComplete);
  }

  void sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                         uint64_t CallId) override {}

  /// Single-threaded here, so no lock: a real transport takes its own.
  OnControllerCallReturn takePendingCall(uint64_t SeqNo) override {
    return takeCall(SeqNo);
  }

  /// A handler taken from the Session, for registering.
  OnControllerCallReturn Captured;
};

/// Drives Session::callController once and returns the handler it produced.
TestCA::OnControllerCallReturn borrowHandler(Session &S, TestCA &CA) {
  S.callController([](WrapperFunctionBuffer) {}, nullptr,
                   WrapperFunctionBuffer());
  return std::move(CA.Captured);
}

} // namespace

TEST(SimpleRemoteCATest, SetupMessageRoundTrips) {
  // The setup payload's wire format is shared with the controller (it matches
  // LLVM's SPSSimpleRemoteEPCExecutorInfo), so pin the field order and types by
  // decoding a payload the encoder produced.
  using SPSSetup = SPSTuple<SPSString, uint64_t,
                            SPSSequence<SPSTuple<SPSString, SPSSequence<char>>>,
                            SPSSequence<SPSTuple<SPSString, SPSExecutorAddr>>>;

  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);

  int SomeSymbol = 0;
  SimpleSymbolTable Symbols;
  std::vector<std::pair<SymbolNameSpec, const void *>> SymbolDefs = {
      {SymbolNameSpec::linker("foo"), &SomeSymbol}};
  cantFail(Symbols.addUnique(SymbolDefs));

  BootstrapInfo BI(S, std::move(Symbols),
                   BootstrapInfo::ValueMap{{"key", "value"}});
  auto Payload = TestCA::encodeSetup(BI);

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

  S.detach([] {});
}

TEST(SimpleRemoteCATest, OrderlyHangupRoundTrips) {
  // Both ends encode and decode hang-ups through these, so a success value must
  // survive the trip as a success.
  auto Err = TestCA::decodeHangup(TestCA::encodeHangup(Error::success()));
  EXPECT_FALSE(!!Err) << "an orderly hang-up is not an error";
}

TEST(SimpleRemoteCATest, HangupReasonRoundTrips) {
  auto Err = TestCA::decodeHangup(
      TestCA::encodeHangup(make_error<StringError>("controller ran out of x")));
  ASSERT_TRUE(!!Err);
  EXPECT_EQ(toString(std::move(Err)), "controller ran out of x");
}

TEST(SimpleRemoteCATest, DecodeHangupRejectsAnEmptyPayload) {
  // Never valid: the two ends are rev-locked, so this is a bug in the peer
  // rather than version skew. It comes back as an error like any other reason,
  // since both outcomes end the session.
  auto Err = TestCA::decodeHangup(WrapperFunctionBuffer());
  ASSERT_TRUE(!!Err);
  EXPECT_EQ(toString(std::move(Err)),
            "Malformed hang-up message: could not deserialize reason");
}

TEST(SimpleRemoteCATest, RegisterCallReturnsDistinctNonZeroSequenceNumbers) {
  // Zero is reserved for messages with no result to await, so a registered call
  // must never get it.
  TestCA *CA = nullptr;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.attach<TestCA>(BootstrapInfo(S), &CA);
  ASSERT_TRUE(CA);

  uint64_t First = CA->registerCall(borrowHandler(S, *CA));
  uint64_t Second = CA->registerCall(borrowHandler(S, *CA));
  EXPECT_NE(First, 0u);
  EXPECT_NE(Second, 0u);
  EXPECT_NE(First, Second);

  S.detach([] {});
}

TEST(SimpleRemoteCATest, TakeCallYieldsTheHandlerExactlyOnce) {
  TestCA *CA = nullptr;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.attach<TestCA>(BootstrapInfo(S), &CA);
  ASSERT_TRUE(CA);

  uint64_t SeqNo = CA->registerCall(borrowHandler(S, *CA));
  auto Taken = CA->takeCall(SeqNo);
  EXPECT_TRUE(!!Taken) << "the registered handler should come back";

  // A second take finds nothing: a result for a call that was never made, or
  // answered twice, which the transport reports as terminal.
  EXPECT_FALSE(!!CA->takeCall(SeqNo));
  EXPECT_FALSE(!!CA->takeCall(/*SeqNo=*/9999)) << "never registered";

  CA->failPendingControllerCall(std::move(Taken));
  S.detach([] {});
}

TEST(SimpleRemoteCATest, TakeAllCallsEmptiesTheTable) {
  TestCA *CA = nullptr;
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  S.attach<TestCA>(BootstrapInfo(S), &CA);
  ASSERT_TRUE(CA);

  uint64_t First = CA->registerCall(borrowHandler(S, *CA));
  uint64_t Second = CA->registerCall(borrowHandler(S, *CA));

  auto All = CA->takeAllCalls();
  EXPECT_EQ(All.size(), 2u);
  EXPECT_EQ(All.count(First), 1u);
  EXPECT_EQ(All.count(Second), 1u);

  // Emptied, so a second drain finds nothing and a late result finds no call.
  EXPECT_TRUE(CA->takeAllCalls().empty());
  EXPECT_FALSE(!!CA->takeCall(First));

  for (auto &[SeqNo, OnComplete] : All)
    CA->failPendingControllerCall(std::move(OnComplete));
  S.detach([] {});
}
