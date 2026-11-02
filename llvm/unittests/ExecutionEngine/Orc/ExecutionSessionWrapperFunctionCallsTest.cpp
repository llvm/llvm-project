//===- ExecutionSessionWrapperFunctionCallsTest.cpp -- Test wrapper calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/BindCallControllerHandlerSPS.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static void addAsyncWrapper(unique_function<void(int32_t)> SendResult,
                            int32_t X, int32_t Y) {
  SendResult(X + Y);
}

TEST(ExecutionSessionWrapperFunctionCalls, RegisterAsyncHandlerAndRun) {

  constexpr ExecutorAddr AddAsyncTagAddr(0x01);

  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.createBareJITDylib("JD");

  auto AddAsyncTag = ES.intern("addAsync_tag");
  cantFail(JD.define(absoluteSymbols(
      {{AddAsyncTag, {AddAsyncTagAddr, JITSymbolFlags::Exported}}})));

  cantFail(ES.registerCallControllerHandlers(
      JD, bindCallControllerHandlerSPS<int32_t(int32_t, int32_t)>(
              SymbolNameSpec::verbatim("addAsync_tag"), addAsyncWrapper)));

  std::promise<int32_t> RP;
  auto RF = RP.get_future();

  using ArgSerialization = SPSArgList<int32_t, int32_t>;
  size_t ArgBufferSize = ArgSerialization::size(1, 2);
  auto ArgBuffer = WrapperFunctionBuffer::allocate(ArgBufferSize);
  SPSOutputBuffer OB(ArgBuffer.data(), ArgBuffer.size());
  EXPECT_TRUE(ArgSerialization::serialize(OB, 1, 2));

  ES.runCallControllerHandler(
      [&](WrapperFunctionBuffer ResultBuffer) {
        int32_t Result;
        SPSInputBuffer IB(ResultBuffer.data(), ResultBuffer.size());
        EXPECT_TRUE(SPSArgList<int32_t>::deserialize(IB, Result));
        RP.set_value(Result);
      },
      AddAsyncTagAddr, std::move(ArgBuffer));

  EXPECT_EQ(RF.get(), (int32_t)3);

  cantFail(ES.endSession());
}

namespace {

class CallControllerHandlerTest : public CoreAPIsBasedStandardTest {};

} // namespace

// Returns a binding whose handler sets Called when run.
static ExecutionSession::CallControllerHandlerBinding
recordCall(SymbolNameSpec Name, bool &Called,
           SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return ExecutionSession::CallControllerHandlerBinding(
      Name,
      [&Called](ExecutionSession::CallControllerReturnFn Return,
                WrapperFunctionBuffer) {
        Called = true;
        Return(WrapperFunctionBuffer());
      },
      LF);
}

// Runs the handler registered for TagAddr in ES. Returns the out-of-band error
// reported, or the empty string if there was none.
static std::string run(ExecutionSession &ES, ExecutorAddr TagAddr) {
  std::string Err;
  ES.runCallControllerHandler(
      [&](WrapperFunctionBuffer R) {
        if (auto *Msg = R.getOutOfBandError())
          Err = Msg;
      },
      TagAddr, WrapperFunctionBuffer());
  return Err;
}

TEST_F(CallControllerHandlerTest, MissingRequiredTagFails) {
  // A missing required tag should cause registration to fail, and no handlers
  // (including those whose tags were found) should be registered.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  bool FooCalled = false, BarCalled = false;
  EXPECT_THAT_ERROR(ES.registerCallControllerHandlers(
                        JD,
                        recordCall(SymbolNameSpec::verbatim("foo"), FooCalled),
                        recordCall(SymbolNameSpec::verbatim("bar"), BarCalled)),
                    Failed());

  EXPECT_NE(run(ES, FooAddr), "");
  EXPECT_FALSE(FooCalled);
}

TEST_F(CallControllerHandlerTest, MissingWeakTagIsDropped) {
  // A missing weakly referenced tag should not cause registration to fail.
  // The handler for it is dropped, and other handlers are registered.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  bool FooCalled = false, BarCalled = false;
  EXPECT_THAT_ERROR(ES.registerCallControllerHandlers(
                        JD,
                        recordCall(SymbolNameSpec::verbatim("foo"), FooCalled),
                        recordCall(SymbolNameSpec::verbatim("bar"), BarCalled,
                                   SymbolLookupFlags::WeaklyReferencedSymbol)),
                    Succeeded());

  EXPECT_EQ(run(ES, FooAddr), "");
  EXPECT_TRUE(FooCalled);
  EXPECT_FALSE(BarCalled);
}

TEST_F(CallControllerHandlerTest, DuplicateBindingFails) {
  // Two bindings for the same tag should cause registration to fail, with no
  // handlers registered.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  bool Called1 = false, Called2 = false;
  EXPECT_THAT_ERROR(ES.registerCallControllerHandlers(
                        JD,
                        recordCall(SymbolNameSpec::verbatim("foo"), Called1),
                        recordCall(SymbolNameSpec::verbatim("foo"), Called2)),
                    Failed());

  EXPECT_NE(run(ES, FooAddr), "");
  EXPECT_FALSE(Called1);
  EXPECT_FALSE(Called2);
}

TEST_F(CallControllerHandlerTest, AlreadyRegisteredTagFails) {
  // Registering a handler for an already-registered tag should fail, leave
  // the existing handler in place, and register none of the other handlers
  // passed in the same call.
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  bool FooCalled1 = false, FooCalled2 = false, BarCalled = false;
  cantFail(ES.registerCallControllerHandlers(
      JD, recordCall(SymbolNameSpec::verbatim("foo"), FooCalled1)));

  EXPECT_THAT_ERROR(
      ES.registerCallControllerHandlers(
          JD, recordCall(SymbolNameSpec::verbatim("bar"), BarCalled),
          recordCall(SymbolNameSpec::verbatim("foo"), FooCalled2)),
      Failed());

  EXPECT_NE(run(ES, BarAddr), "");
  EXPECT_FALSE(BarCalled);

  EXPECT_EQ(run(ES, FooAddr), "");
  EXPECT_TRUE(FooCalled1);
  EXPECT_FALSE(FooCalled2);
}

TEST(CallControllerHandlerManglingTest, TagNamesAreMangled) {
  // Tag names should be mangled for the session's target: on MachO, the C
  // name "foo_tag" should resolve to the linker name "_foo_tag".
  ExecutionSession ES(std::make_unique<UnsupportedExecutorProcessControl>(
      nullptr, nullptr, "arm64-apple-darwin"));
  auto &JD = ES.createBareJITDylib("JD");

  constexpr ExecutorAddr TagAddr(0x1);
  cantFail(JD.define(absoluteSymbols(
      {{ES.intern("_foo_tag"), {TagAddr, JITSymbolFlags::Exported}}})));

  bool Called = false;
  EXPECT_THAT_ERROR(ES.registerCallControllerHandlers(
                        JD, recordCall(SymbolNameSpec::c("foo_tag"), Called)),
                    Succeeded());

  EXPECT_EQ(run(ES, TagAddr), "");
  EXPECT_TRUE(Called);

  cantFail(ES.endSession());
}
