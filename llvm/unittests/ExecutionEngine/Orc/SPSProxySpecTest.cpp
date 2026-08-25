//===- SPSProxySpecTest.cpp - Test SPS proxy round-trips ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// End-to-end tests for the SPS proxies: that each Call*ProxySpec's dispatch
// serializes its arguments, invokes the executor-side wrapper, and
// deserializes the result. Generic Proxy behavior (independent of the
// serialization protocol) is covered by ProxyTest.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/CallProxiesSPS.h"
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

namespace sps = llvm::orc::sps;
using llvm::orc::CallInt32Int32Proxy;
using llvm::orc::CallInt32VoidProxy;
using llvm::orc::CallMainProxy;
using llvm::orc::CallVoidVoidProxy;

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

// Exercises argv marshaling: an argument vector is serialized, decoded by the
// wrapper, and the int64_t result is deserialized.
TEST(SPSProxySpecTest, CallMainSyncViaDirectConstruction) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  CallMainProxy CallMain(sps::CallMainProxySpec::dispatch,
                         ExecutorAddr::fromPtr(callMainWrapper));
  ExecutorAddr MainAddr = ExecutorAddr::fromPtr(testMain);

  std::vector<std::string> Args;
  Expected<int64_t> R0 = CallMain(ES, MainAddr, Args);
  ASSERT_THAT_EXPECTED(R0, Succeeded());
  EXPECT_EQ(*R0, 0); // argc == 0, no argv[0].

  Args = {"hello"};
  Expected<int64_t> R1 = CallMain(ES, MainAddr, Args);
  ASSERT_THAT_EXPECTED(R1, Succeeded());
  EXPECT_EQ(*R1, 1 + 5); // argc == 1, strlen("hello") == 5.

  Args = {"a", "bb"};
  Expected<int64_t> R2 = CallMain(ES, MainAddr, Args);
  ASSERT_THAT_EXPECTED(R2, Succeeded());
  EXPECT_EQ(*R2, 2 + 1); // argc == 2, strlen("a") == 1.

  cantFail(ES.endSession());
}

TEST(SPSProxySpecTest, CallMainAsyncViaCallOperator) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  CallMainProxy CallMain(sps::CallMainProxySpec::dispatch,
                         ExecutorAddr::fromPtr(callMainWrapper));

  std::vector<std::string> Args = {"foo", "bar"};
  std::promise<MSVCPExpected<int64_t>> P;
  auto F = P.get_future();
  CallMain([&](Expected<int64_t> R) { P.set_value(std::move(R)); }, ES,
           ExecutorAddr::fromPtr(testMain), Args);

  Expected<int64_t> R = F.get();
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 2 + 3); // argc == 2, strlen("foo") == 3.

  cantFail(ES.endSession());
}

// Target for CallVoidVoidProxy: records that it ran via a counter (there is no
// return value to observe).
static int VoidVoidCallCount = 0;
static void voidVoidTarget() { ++VoidVoidCallCount; }

// Executor-side wrapper for CallVoidVoidProxy. Decodes the target address and
// invokes it as a void() function.
static CWrapperFunctionBuffer callVoidVoidWrapper(const char *ArgData,
                                                  size_t ArgSize) {
  return WrapperFunction<void(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr FnAddr) { FnAddr.toPtr<void()>()(); })
      .release();
}

// Exercises the void-return path (ErrorRetT == Error) and the empty argument
// pack, through both the synchronous and asynchronous call operators.
TEST(SPSProxySpecTest, VoidVoidSyncAndAsync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  CallVoidVoidProxy Call(sps::CallVoidVoidProxySpec::dispatch,
                         ExecutorAddr::fromPtr(callVoidVoidWrapper));
  ExecutorAddr TargetAddr = ExecutorAddr::fromPtr(voidVoidTarget);

  // Synchronous: the call operator returns Error, not Expected<T>.
  VoidVoidCallCount = 0;
  EXPECT_THAT_ERROR(Call(ES, TargetAddr), Succeeded());
  EXPECT_EQ(VoidVoidCallCount, 1);

  // Asynchronous: the result is delivered as an Error.
  VoidVoidCallCount = 0;
  std::promise<MSVCPError> P;
  auto F = P.get_future();
  Call([&](Error Err) { P.set_value(std::move(Err)); }, ES, TargetAddr);
  EXPECT_THAT_ERROR(Error(F.get()), Succeeded());
  EXPECT_EQ(VoidVoidCallCount, 1);

  cantFail(ES.endSession());
}

// Target for CallInt32Int32Proxy: doubles its argument, so the forwarded value
// is observable in the result.
static int32_t int32Int32Target(int32_t X) { return X * 2; }

// Executor-side wrapper for CallInt32Int32Proxy. Decodes the target address and
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

// Exercises a non-void proxy with an argument (so argument forwarding through
// the pack is covered).
TEST(SPSProxySpecTest, Int32Int32Sync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  CallInt32Int32Proxy Call(sps::CallInt32Int32ProxySpec::dispatch,
                           ExecutorAddr::fromPtr(callInt32Int32Wrapper));
  Expected<int32_t> R = Call(ES, ExecutorAddr::fromPtr(int32Int32Target), 21);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42); // 21 * 2.

  cantFail(ES.endSession());
}

// Target for CallInt32VoidProxy.
static int32_t int32VoidTarget() { return 42; }

// Executor-side wrapper for CallInt32VoidProxy. Decodes the target address,
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

// Exercises a non-void, zero-argument proxy.
TEST(SPSProxySpecTest, Int32VoidSync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  CallInt32VoidProxy Call(sps::CallInt32VoidProxySpec::dispatch,
                          ExecutorAddr::fromPtr(callInt32VoidWrapper));
  Expected<int32_t> R = Call(ES, ExecutorAddr::fromPtr(int32VoidTarget));
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// Executor-side wrapper returning Error: fails iff its bool argument is true.
static CWrapperFunctionBuffer errorFnWrapper(const char *ArgData,
                                             size_t ArgSize) {
  return WrapperFunction<SPSError(bool)>::handle(
             ArgData, ArgSize,
             [](bool ShouldFail) -> Error {
               if (ShouldFail)
                 return make_error<StringError>("requested failure",
                                                inconvertibleErrorCode());
               return Error::success();
             })
      .release();
}

struct ErrorFnCI {
  static constexpr char Name[] = "test_sps_error_fn";
  using SPSSig = SPSError(bool);
};
using ErrorFnProxy = Proxy<Error(bool)>;
using ErrorFnProxySpec = sps::ProxySpec<ErrorFnProxy, ErrorFnCI>;

// Exercises the Error -> Error mapping across the SPS boundary, including a
// failure reported by the executor-side function itself.
TEST(SPSProxySpecTest, ErrorReturn) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ErrorFnProxy Call(ErrorFnProxySpec::dispatch,
                    ExecutorAddr::fromPtr(errorFnWrapper));

  EXPECT_THAT_ERROR(Call(ES, false), Succeeded());
  EXPECT_THAT_ERROR(Call(ES, true), Failed());

  cantFail(ES.endSession());
}

// Executor-side wrapper returning Expected<int32_t>: fails iff its argument is
// negative, else returns the argument plus one.
static CWrapperFunctionBuffer expectedFnWrapper(const char *ArgData,
                                                size_t ArgSize) {
  return WrapperFunction<SPSExpected<int32_t>(int32_t)>::handle(
             ArgData, ArgSize,
             [](int32_t X) -> Expected<int32_t> {
               if (X < 0)
                 return make_error<StringError>("negative argument",
                                                inconvertibleErrorCode());
               return X + 1;
             })
      .release();
}

struct ExpectedFnCI {
  static constexpr char Name[] = "test_sps_expected_fn";
  using SPSSig = SPSExpected<int32_t>(int32_t);
};
using ExpectedFnProxy = Proxy<Expected<int32_t>(int32_t)>;
using ExpectedFnProxySpec = sps::ProxySpec<ExpectedFnProxy, ExpectedFnCI>;

// Exercises the Expected<T> -> Expected<T> (flattening) mapping across the SPS
// boundary, for both the value and the executor-reported-error cases.
TEST(SPSProxySpecTest, ExpectedReturn) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ExpectedFnProxy Call(ExpectedFnProxySpec::dispatch,
                       ExecutorAddr::fromPtr(expectedFnWrapper));

  Expected<int32_t> R = Call(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  EXPECT_THAT_EXPECTED(Call(ES, -1), Failed());

  cantFail(ES.endSession());
}
