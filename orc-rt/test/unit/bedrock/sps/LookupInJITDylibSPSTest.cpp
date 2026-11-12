//===- LookupInJITDylibSPSTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the sps::LookupInJITDylib controller-call wrapper.
//
// Lookup semantics belong to the controller, so these tests only check that
// lookupInJITDylib carries its arguments to the controller and the results
// back intact, that it rejects a result that doesn't match the lookup set, and
// that both lookup and transport errors reach the caller. The controller-side
// handler records what it receives and returns a canned result.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/LookupInJITDylibSPS.h"
#include "orc-rt/support/sps/SPSJITDylibLookupFlags.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"
#include "orc-rt/support/sps/SPSWrapperFunction.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
#include "ErrorMatchers.h"

#include "gtest/gtest.h"

using namespace orc_rt;
using namespace orc_rt::test;

namespace {

// Declare the tag so that we can install the handler below. This must be
// kept in sync with the tag definition in LookupInJITDylibSPS.cpp.
extern "C" orc_rt_ControllerHandlerTag orc_rt_tag_sps_lookupInJITDylib;

using LookupJITDylibSPSSig = SPSExpected<SPSSymbolLookupResult>(
    SPSExecutorAddr, SPSJITDylibLookupFlags, SPSSymbolLookupSet);

// Controller-side state for a single lookup. Tests pass its address as the
// JITDylib key, which lets the (capture-less) handler reach it.
struct ControllerState {
  bool Called = false;
  std::optional<JITDylibLookupFlags> ReceivedJDLookupFlags;
  SymbolLookupSet Received;
  SymbolLookupResult ToReturn;
};

// Records the lookup flags and set, and returns ControllerState::ToReturn.
static void recordingLookup(orc_rt_SessionRef S,
                            orc_rt_WrapperFunctionBuffer ArgBytes,
                            orc_rt_WrapperFunctionReturn Return,
                            uint64_t CallId) {
  SPSWrapperFunction<LookupJITDylibSPSSig>::handle(
      S, ArgBytes, Return, CallId,
      [](move_only_function<void(Expected<SymbolLookupResult>)> OnComplete,
         void *Key, JITDylibLookupFlags JDLookupFlags,
         const SymbolLookupSet &LS) {
        auto &CS = *static_cast<ControllerState *>(Key);
        CS.Called = true;
        CS.ReceivedJDLookupFlags = JDLookupFlags;
        CS.Received = LS;
        OnComplete(std::move(CS.ToReturn));
      });
}

// Reports that the lookup itself failed, as the controller would for (e.g.) a
// missing required symbol.
static void lookupFailure(orc_rt_SessionRef S,
                          orc_rt_WrapperFunctionBuffer ArgBytes,
                          orc_rt_WrapperFunctionReturn Return,
                          uint64_t CallId) {
  SPSWrapperFunction<LookupJITDylibSPSSig>::handle(
      S, ArgBytes, Return, CallId,
      [](move_only_function<void(Expected<SymbolLookupResult>)> OnComplete,
         void *Key, JITDylibLookupFlags JDLookupFlags,
         const SymbolLookupSet &LS) {
        OnComplete(make_error<StringError>("lookup failed"));
      });
}

// Fails the call with an out-of-band error, as for a transport failure.
static void transportFailure(orc_rt_SessionRef S,
                             orc_rt_WrapperFunctionBuffer ArgBytes,
                             orc_rt_WrapperFunctionReturn Return,
                             uint64_t CallId) {
  WrapperFunctionBuffer Args(ArgBytes); // Take ownership; unused.
  Return(
      S,
      WrapperFunctionBuffer::createOutOfBandError("transport failed").release(),
      CallId);
}

// Route the lookup tag to Handler, then run lookupInJITDylib with CS as the
// JITDylib key.
static Expected<SymbolLookupResult> runLookup(orc_rt_WrapperFunction Handler,
                                              ControllerState &CS,
                                              JITDylibLookupFlags JDLookupFlags,
                                              const SymbolLookupSet &LS) {
  Session S(mockExecutorProcessInfo(), inlineDispatch, noErrors);
  DirectControllerAccess::HandlerMap HM;
  HM[&orc_rt_tag_sps_lookupInJITDylib] = Handler;
  S.attach(std::make_shared<DirectControllerAccess>(S, std::move(HM)),
           BootstrapInfo(S));

  std::future<Expected<SymbolLookupResult>> ResultF;
  sps::lookupInJITDylib(waitFor(ResultF), S, &CS, JDLookupFlags, LS);
  return ResultF.get();
}

TEST(LookupInJITDylibSPSTest, EmptyLookup) {
  ControllerState CS;
  auto Result =
      runLookup(recordingLookup, CS, JITDylibLookupFlags::MatchAllSymbols, {});
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  EXPECT_TRUE(CS.Called);
  EXPECT_TRUE(CS.Received.empty());
  EXPECT_TRUE(Result->empty());
}

TEST(LookupInJITDylibSPSTest, ArgumentsRoundTrip) {
  // Names, order, and both SymbolLookupFlags values must reach the controller
  // intact.
  SymbolLookupSet LS{{"foo", SymbolLookupFlags::RequiredSymbol},
                     {"bar", SymbolLookupFlags::WeaklyReferencedSymbol}};
  ControllerState CS;
  CS.ToReturn.resize(LS.size());

  auto Result =
      runLookup(recordingLookup, CS, JITDylibLookupFlags::MatchAllSymbols, LS);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_TRUE(CS.Called);
  ASSERT_EQ(CS.Received.size(), LS.size());
  for (size_t I = 0; I != LS.size(); ++I)
    EXPECT_EQ(CS.Received[I], LS[I]) << "at index " << I;
}

TEST(LookupInJITDylibSPSTest, JITDylibLookupFlagsRoundTrip) {
  // Both JITDylibLookupFlags values must reach the controller intact.
  for (auto JDLookupFlags : {JITDylibLookupFlags::MatchExportedSymbolsOnly,
                             JITDylibLookupFlags::MatchAllSymbols}) {
    ControllerState CS;
    auto Result = runLookup(recordingLookup, CS, JDLookupFlags, {});
    ASSERT_THAT_EXPECTED(Result, Succeeded());
    EXPECT_EQ(CS.ReceivedJDLookupFlags, JDLookupFlags);
  }
}

TEST(LookupInJITDylibSPSTest, ResultsRoundTrip) {
  // All three result states must come back as sent. In particular, "not
  // found" (nullopt) must stay distinct from "found at null".
  int X = 0;
  SymbolLookupSet LS{
      {"found", SymbolLookupFlags::RequiredSymbol},
      {"notFound", SymbolLookupFlags::RequiredSymbol},
      {"foundAtNull", SymbolLookupFlags::WeaklyReferencedSymbol}};
  ControllerState CS;
  CS.ToReturn.push_back(&X);
  CS.ToReturn.push_back(std::nullopt);
  CS.ToReturn.push_back(std::optional<const void *>(nullptr));

  auto Result =
      runLookup(recordingLookup, CS, JITDylibLookupFlags::MatchAllSymbols, LS);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  ASSERT_EQ(Result->size(), 3U);
  EXPECT_EQ((*Result)[0], std::optional<const void *>(&X));
  EXPECT_EQ((*Result)[1], std::nullopt);
  EXPECT_EQ((*Result)[2], std::optional<const void *>(nullptr));
}

TEST(LookupInJITDylibSPSTest, ResultSizeMismatchFails) {
  // A result whose length doesn't match the lookup set must be rejected, not
  // passed on to a client that will index it by lookup-set position.
  SymbolLookupSet LS{{"foo", SymbolLookupFlags::RequiredSymbol},
                     {"bar", SymbolLookupFlags::RequiredSymbol}};
  ControllerState CS;
  CS.ToReturn.resize(1);

  EXPECT_THAT_EXPECTED(
      runLookup(recordingLookup, CS, JITDylibLookupFlags::MatchAllSymbols, LS),
      FailedWithMessage(
          "lookupInJITDylib: expected 2 results from controller, got 1"));
}

TEST(LookupInJITDylibSPSTest, LookupErrorPropagates) {
  // An error from the controller's lookup is reported to the caller.
  ControllerState CS;
  EXPECT_THAT_EXPECTED(runLookup(lookupFailure, CS,
                                 JITDylibLookupFlags::MatchAllSymbols,
                                 {{"foo", SymbolLookupFlags::RequiredSymbol}}),
                       FailedWithMessage("lookup failed"));
}

TEST(LookupInJITDylibSPSTest, TransportErrorPropagates) {
  // A failure to call the controller is reported to the caller.
  ControllerState CS;
  EXPECT_THAT_EXPECTED(runLookup(transportFailure, CS,
                                 JITDylibLookupFlags::MatchAllSymbols,
                                 {{"foo", SymbolLookupFlags::RequiredSymbol}}),
                       FailedWithMessage("transport failed"));
}

} // namespace
