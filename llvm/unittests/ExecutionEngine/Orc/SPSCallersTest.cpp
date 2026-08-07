//===- SPSCallersTest.cpp - Test SPS call wrappers ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/Calls.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"

#include <cstring>
#include <future>
#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;
using llvm::orc::rt::sps::Int32Int32Caller;
using llvm::orc::rt::sps::Int32VoidCaller;
using llvm::orc::rt::sps::MainCaller;
using llvm::orc::rt::sps::VoidVoidCaller;

// Test "main" function. Returns argc plus the length of the first element of
// argv (if argv is non-empty). Does not inspect argv entries beyond the first.
static int testMain(int argc, char *argv[]) {
  int Result = argc;
  if (argc > 0)
    Result += static_cast<int>(std::strlen(argv[0]));
  return Result;
}

// Executor-side "call-main" wrapper. Decodes the main-function address and the
// argument vector, then invokes the main function with a C-style (argc, argv).
static CWrapperFunctionBuffer callMainWrapper(const char *ArgData,
                                              size_t ArgSize) {
  return WrapperFunction<int64_t(SPSExecutorAddr, SPSSequence<SPSString>)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr MainFnAddr,
                std::vector<std::string> Args) -> int64_t {
               std::vector<char *> ArgV;
               ArgV.reserve(Args.size() + 1);
               for (auto &Arg : Args)
                 ArgV.push_back(Arg.data());
               ArgV.push_back(nullptr);
               auto *Main = MainFnAddr.toPtr<int(int, char **)>();
               return Main(static_cast<int>(Args.size()), ArgV.data());
             })
          .release();
}

TEST(SPSCallersTest, CallMainSyncViaDirectConstruction) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));
  ExecutorAddr MainAddr = ExecutorAddr::fromPtr(testMain);

  std::vector<std::string> Args;
  Expected<int64_t> R0 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R0, Succeeded());
  EXPECT_EQ(*R0, 0); // argc == 0, no argv[0].

  Args = {"hello"};
  Expected<int64_t> R1 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R1, Succeeded());
  EXPECT_EQ(*R1, 1 + 5); // argc == 1, strlen("hello") == 5.

  Args = {"a", "bb"};
  Expected<int64_t> R2 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R2, Succeeded());
  EXPECT_EQ(*R2, 2 + 1); // argc == 2, strlen("a") == 1.

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CallMainAsyncViaCallOperator) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));

  std::vector<std::string> Args = {"foo", "bar"};
  std::promise<MSVCPExpected<int64_t>> P;
  auto F = P.get_future();
  CallMain([&](Expected<int64_t> R) { P.set_value(std::move(R)); },
           ExecutorAddr::fromPtr(testMain), Args);

  Expected<int64_t> R = F.get();
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 2 + 3); // argc == 2, strlen("foo") == 3.

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CallMainThroughRTInterface) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));

  // Drive the caller through the runtime-agnostic rt::MainCaller interface to
  // exercise the virtual dispatch path (and to guard the interface's public
  // accessibility).
  rt::MainCaller &Base = CallMain;
  ExecutorAddr MainAddr = ExecutorAddr::fromPtr(testMain);

  // Synchronous call operator (inherited from rt::MainCaller).
  std::vector<std::string> Args = {"hello"};
  Expected<int64_t> RSync = Base(MainAddr, Args);
  ASSERT_THAT_EXPECTED(RSync, Succeeded());
  EXPECT_EQ(*RSync, 1 + 5); // argc == 1, strlen("hello") == 5.

  // Asynchronous call operator (virtual).
  std::promise<MSVCPExpected<int64_t>> P;
  auto F = P.get_future();
  Base([&](Expected<int64_t> R) { P.set_value(std::move(R)); }, MainAddr, Args);
  Expected<int64_t> RAsync = F.get();
  ASSERT_THAT_EXPECTED(RAsync, Succeeded());
  EXPECT_EQ(*RAsync, 1 + 5);

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CreateLooksUpCallMainInBootstrapJD) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  // Register the call-main wrapper in the bootstrap JITDylib under the name
  // MainCaller::Create looks for.
  auto &BootstrapJD = ES.getBootstrapJITDylib();
  cantFail(BootstrapJD.define(absoluteSymbols(
      {{ES.intern(MainCaller::CIName),
        {ExecutorAddr::fromPtr(callMainWrapper), JITSymbolFlags::Exported}}})));

  Expected<MainCaller> CallMain = MainCaller::Create(ES);
  ASSERT_THAT_EXPECTED(CallMain, Succeeded());

  std::vector<std::string> Args = {"x", "y", "z"};
  Expected<int64_t> R = (*CallMain)(ExecutorAddr::fromPtr(testMain), Args);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 3 + 1); // argc == 3, strlen("x") == 1.

  cantFail(ES.endSession());
}

// Target for VoidVoidCaller: records that it ran via a counter (there is no
// return value to observe).
static int VoidVoidCallCount = 0;
static void voidVoidTarget() { ++VoidVoidCallCount; }

// Executor-side wrapper for VoidVoidCaller. Decodes the target address and
// invokes it as a void() function.
static CWrapperFunctionBuffer callVoidVoidWrapper(const char *ArgData,
                                                  size_t ArgSize) {
  return WrapperFunction<void(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr FnAddr) { FnAddr.toPtr<void()>()(); })
      .release();
}

// Target for Int32Int32Caller: doubles its argument, so the forwarded value is
// observable in the result.
static int32_t int32Int32Target(int32_t X) { return X * 2; }

// Executor-side wrapper for Int32Int32Caller. Decodes the target address and
// the int32_t argument, invokes the target, and returns the result.
static CWrapperFunctionBuffer callInt32Int32Wrapper(const char *ArgData,
                                                    size_t ArgSize) {
  return WrapperFunction<int32_t(SPSExecutorAddr, int32_t)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr FnAddr, int32_t X) -> int32_t {
               return FnAddr.toPtr<int32_t(int32_t)>()(X);
             })
      .release();
}

// Exercises the void-return path (ErrorRetT == Error) and the empty argument
// pack, through both the synchronous and asynchronous call operators.
TEST(SPSCallersTest, VoidVoidSyncAndAsync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  VoidVoidCaller Call(ES, ExecutorAddr::fromPtr(callVoidVoidWrapper));
  ExecutorAddr TargetAddr = ExecutorAddr::fromPtr(voidVoidTarget);

  // Synchronous: the call operator returns Error, not Expected<T>.
  VoidVoidCallCount = 0;
  EXPECT_THAT_ERROR(Call(TargetAddr), Succeeded());
  EXPECT_EQ(VoidVoidCallCount, 1);

  // Asynchronous: the result is delivered as an Error.
  VoidVoidCallCount = 0;
  std::promise<MSVCPError> P;
  auto F = P.get_future();
  Call([&](Error Err) { P.set_value(std::move(Err)); }, TargetAddr);
  EXPECT_THAT_ERROR(Error(F.get()), Succeeded());
  EXPECT_EQ(VoidVoidCallCount, 1);

  cantFail(ES.endSession());
}

// Exercises a non-void caller with an argument (so argument forwarding through
// the pack is covered), and the Create / bootstrap lookup path for a caller
// other than MainCaller.
TEST(SPSCallersTest, Int32Int32SyncAndCreate) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  Int32Int32Caller Call(ES, ExecutorAddr::fromPtr(callInt32Int32Wrapper));
  Expected<int32_t> RDirect = Call(ExecutorAddr::fromPtr(int32Int32Target), 21);
  ASSERT_THAT_EXPECTED(RDirect, Succeeded());
  EXPECT_EQ(*RDirect, 42); // 21 * 2.

  auto &BootstrapJD = ES.getBootstrapJITDylib();
  cantFail(BootstrapJD.define(
      absoluteSymbols({{ES.intern(Int32Int32Caller::CIName),
                        {ExecutorAddr::fromPtr(callInt32Int32Wrapper),
                         JITSymbolFlags::Exported}}})));

  Expected<Int32Int32Caller> CreatedCall = Int32Int32Caller::Create(ES);
  ASSERT_THAT_EXPECTED(CreatedCall, Succeeded());
  Expected<int32_t> RCreated =
      (*CreatedCall)(ExecutorAddr::fromPtr(int32Int32Target), 21);
  ASSERT_THAT_EXPECTED(RCreated, Succeeded());
  EXPECT_EQ(*RCreated, 42); // 21 * 2.

  cantFail(ES.endSession());
}

// Target for Int32VoidCaller.
static int32_t int32VoidTarget() { return 42; }

// Executor-side wrapper for Int32VoidCaller. Decodes the target address,
// invokes it as an int32_t() function, and returns the result.
static CWrapperFunctionBuffer callInt32VoidWrapper(const char *ArgData,
                                                   size_t ArgSize) {
  return WrapperFunction<int32_t(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr FnAddr) -> int32_t {
               return FnAddr.toPtr<int32_t()>()();
             })
      .release();
}

// Exercises a non-void, zero-argument caller. (Void return and argument
// forwarding are covered by VoidVoidCaller and Int32Int32Caller respectively.)
TEST(SPSCallersTest, Int32VoidSync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  Int32VoidCaller Call(ES, ExecutorAddr::fromPtr(callInt32VoidWrapper));
  Expected<int32_t> R = Call(ExecutorAddr::fromPtr(int32VoidTarget));
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// operator bool reflects whether the caller has a non-null callee address, and
// the accessors return the values the caller was constructed with.
TEST(SPSCallersTest, OperatorBoolAndAccessors) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ExecutorAddr CalleeAddr = ExecutorAddr::fromPtr(callMainWrapper);
  MainCaller CallMain(ES, CalleeAddr);
  EXPECT_TRUE(static_cast<bool>(CallMain));
  EXPECT_EQ(CallMain.calleeAddr(), CalleeAddr);
  EXPECT_EQ(&CallMain.executionSession(), &ES);

  // A caller with a null callee address is falsey.
  MainCaller NullCall(ES, ExecutorAddr());
  EXPECT_FALSE(static_cast<bool>(NullCall));

  cantFail(ES.endSession());
}

// A weakly-referenced Create against a missing symbol succeeds, yielding a
// caller with a null callee (falsey) rather than an error.
TEST(SPSCallersTest, CreateWeaklyReferencedAbsent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  // Nothing is defined for MainCaller::CIName in the bootstrap JITDylib.
  Expected<MainCaller> CallMain =
      MainCaller::Create(ES, SymbolLookupFlags::WeaklyReferencedSymbol);
  ASSERT_THAT_EXPECTED(CallMain, Succeeded());
  EXPECT_FALSE(static_cast<bool>(*CallMain));
  EXPECT_EQ(CallMain->calleeAddr(), ExecutorAddr());

  cantFail(ES.endSession());
}

// A weakly-referenced Create against a present symbol resolves it, yielding a
// usable caller (truthy) bound to the registered address.
TEST(SPSCallersTest, CreateWeaklyReferencedPresent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &BootstrapJD = ES.getBootstrapJITDylib();
  ExecutorAddr CalleeAddr = ExecutorAddr::fromPtr(callMainWrapper);
  cantFail(BootstrapJD.define(
      absoluteSymbols({{ES.intern(MainCaller::CIName),
                        {CalleeAddr, JITSymbolFlags::Exported}}})));

  Expected<MainCaller> CallMain =
      MainCaller::Create(ES, SymbolLookupFlags::WeaklyReferencedSymbol);
  ASSERT_THAT_EXPECTED(CallMain, Succeeded());
  EXPECT_TRUE(static_cast<bool>(*CallMain));
  EXPECT_EQ(CallMain->calleeAddr(), CalleeAddr);

  // The resolved caller is usable.
  std::vector<std::string> Args = {"a", "bb"};
  Expected<int64_t> R = (*CallMain)(ExecutorAddr::fromPtr(testMain), Args);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 2 + 1); // argc == 2, strlen("a") == 1.

  cantFail(ES.endSession());
}

// A required (default) Create against a missing symbol fails, rather than
// yielding a null caller as the weakly-referenced form does.
TEST(SPSCallersTest, CreateRequiredAbsentFails) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  // Nothing is defined for MainCaller::CIName in the bootstrap JITDylib, and
  // the default lookup requires the symbol.
  Expected<MainCaller> CallMain = MainCaller::Create(ES);
  EXPECT_THAT_EXPECTED(CallMain, Failed());

  cantFail(ES.endSession());
}
