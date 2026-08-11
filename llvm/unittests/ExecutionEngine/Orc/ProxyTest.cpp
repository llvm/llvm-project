//===- ProxyTest.cpp - Test rt::Proxy -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for rt::Proxy that are independent of any serialization protocol.
// A trivial in-process dispatch (interpreting the callee address as a local
// function pointer, no serialization) is used throughout -- this exercises the
// Proxy plumbing directly and demonstrates that Proxy is protocol-agnostic.
// The SPS protocol itself is tested in SPSProxiesTest.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTBridge/Proxy.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"

#include <future>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

// Target invoked in-process by the test dispatch below.
int32_t addOne(int32_t X) { return X + 1; }

// A protocol-free dispatch: interpret the callee address as a local function
// pointer and call it directly. This drives rt::Proxy without any
// serialization, so the tests exercise Proxy's own logic (result plumbing,
// operator bool, lookup) rather than a particular protocol.
template <typename RetT, typename... ArgTs>
void inProcessDispatch(
    unique_function<void(typename rt::Proxy<RetT(ArgTs...)>::ErrorRetT)>
        OnComplete,
    ExecutionSession &ES, ExecutorAddr Callee, const ArgTs &...Args) {
  auto *Fn = Callee.toPtr<RetT(ArgTs...)>();
  if constexpr (std::is_void_v<RetT>) {
    Fn(Args...);
    OnComplete(Error::success());
  } else
    OnComplete(Fn(Args...));
}

using AddOneProxy = rt::Proxy<int32_t(int32_t)>;
constexpr AddOneProxy::DispatchFn AddOneDispatch =
    &inProcessDispatch<int32_t, int32_t>;

// Callee returning Error: fails iff ShouldFail. Exercises the Error -> Error
// mapping.
Error maybeFail(bool ShouldFail) {
  if (ShouldFail)
    return make_error<StringError>("requested failure",
                                   inconvertibleErrorCode());
  return Error::success();
}
using MaybeFailProxy = rt::Proxy<Error(bool)>;
constexpr MaybeFailProxy::DispatchFn MaybeFailDispatch =
    &inProcessDispatch<Error, bool>;

// Callee returning Expected<T>: fails iff Arg is negative, else returns Arg
// + 1. Exercises the Expected<T> -> Expected<T> (flattening) mapping.
Expected<int32_t> addOneOrFail(int32_t Arg) {
  if (Arg < 0)
    return make_error<StringError>("negative argument",
                                   inconvertibleErrorCode());
  return Arg + 1;
}
using AddOneOrFailProxy = rt::Proxy<Expected<int32_t>(int32_t)>;
constexpr AddOneOrFailProxy::DispatchFn AddOneOrFailDispatch =
    &inProcessDispatch<Expected<int32_t>, int32_t>;

// The callee return type maps to the client-facing (ErrorRetT) type as:
//          void -> Error
//         Error -> Error
//             T -> Expected<T>
//   Expected<T> -> Expected<T>
static_assert(std::is_same_v<rt::Proxy<void(int)>::ErrorRetT, Error>);
static_assert(std::is_same_v<rt::Proxy<Error(int)>::ErrorRetT, Error>);
static_assert(std::is_same_v<rt::Proxy<int(int)>::ErrorRetT, Expected<int>>);
static_assert(
    std::is_same_v<rt::Proxy<Expected<int>(int)>::ErrorRetT, Expected<int>>);

// A minimal ProxySpec-shaped type (static dispatch + Name) for exercising the
// proxyInit / buildProxies client path without depending on a protocol.
struct AddOneSpec {
  static constexpr const char *Name = "add_one";
  static void dispatch(unique_function<void(Expected<int32_t>)> OnComplete,
                       ExecutionSession &ES, ExecutorAddr Callee,
                       const int32_t &Arg) {
    inProcessDispatch<int32_t, int32_t>(std::move(OnComplete), ES, Callee, Arg);
  }
};

} // namespace

// The synchronous and asynchronous call operators forward the arguments to the
// dispatch function and deliver its result.
TEST(ProxyTest, SyncAndAsync) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  AddOneProxy Call(AddOneDispatch, ExecutorAddr::fromPtr(addOne));

  Expected<int32_t> RSync = Call(ES, 41);
  ASSERT_THAT_EXPECTED(RSync, Succeeded());
  EXPECT_EQ(*RSync, 42);

  std::promise<MSVCPExpected<int32_t>> P;
  auto F = P.get_future();
  Call([&](Expected<int32_t> R) { P.set_value(std::move(R)); }, ES, 41);
  Expected<int32_t> RAsync = F.get();
  ASSERT_THAT_EXPECTED(RAsync, Succeeded());
  EXPECT_EQ(*RAsync, 42);

  cantFail(ES.endSession());
}

// operator bool reflects whether the proxy has a non-null callee address, and
// calleeAddr() returns the address the proxy was constructed with.
TEST(ProxyTest, OperatorBoolAndAccessors) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ExecutorAddr CalleeAddr = ExecutorAddr::fromPtr(addOne);
  AddOneProxy Call(AddOneDispatch, CalleeAddr);
  EXPECT_TRUE(static_cast<bool>(Call));
  EXPECT_EQ(Call.calleeAddr(), CalleeAddr);

  // A default-constructed proxy has a null callee address and is falsey.
  AddOneProxy Null;
  EXPECT_FALSE(static_cast<bool>(Null));

  cantFail(ES.endSession());
}

// Create looks the callee up by name in the bootstrap JITDylib and binds a
// usable proxy to it (required-symbol, present).
TEST(ProxyTest, CreateRequiredPresent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &JD = ES.getBootstrapJITDylib();
  cantFail(JD.define(absoluteSymbols(
      {{ES.intern(AddOneSpec::Name),
        {ExecutorAddr::fromPtr(addOne), JITSymbolFlags::Exported}}})));

  Expected<AddOneProxy> Call = AddOneProxy::Create(
      AddOneDispatch, ES, AddOneSpec::Name, SymbolLookupFlags::RequiredSymbol);
  ASSERT_THAT_EXPECTED(Call, Succeeded());
  EXPECT_TRUE(static_cast<bool>(*Call));

  Expected<int32_t> R = (*Call)(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// A required (default) Create against a missing symbol fails, rather than
// yielding a null proxy as the weakly-referenced form does.
TEST(ProxyTest, CreateRequiredAbsentFails) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  Expected<AddOneProxy> Call = AddOneProxy::Create(
      AddOneDispatch, ES, AddOneSpec::Name, SymbolLookupFlags::RequiredSymbol);
  EXPECT_THAT_EXPECTED(Call, Failed());

  cantFail(ES.endSession());
}

// A weakly-referenced Create against a missing symbol succeeds, yielding a
// proxy with a null callee (falsey) rather than an error.
TEST(ProxyTest, CreateWeaklyReferencedAbsent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  Expected<AddOneProxy> Call =
      AddOneProxy::Create(AddOneDispatch, ES, AddOneSpec::Name,
                          SymbolLookupFlags::WeaklyReferencedSymbol);
  ASSERT_THAT_EXPECTED(Call, Succeeded());
  EXPECT_FALSE(static_cast<bool>(*Call));
  EXPECT_EQ(Call->calleeAddr(), ExecutorAddr());

  cantFail(ES.endSession());
}

// A weakly-referenced Create against a present symbol resolves it, yielding a
// usable proxy (truthy) bound to the registered address.
TEST(ProxyTest, CreateWeaklyReferencedPresent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &JD = ES.getBootstrapJITDylib();
  ExecutorAddr CalleeAddr = ExecutorAddr::fromPtr(addOne);
  cantFail(
      JD.define(absoluteSymbols({{ES.intern(AddOneSpec::Name),
                                  {CalleeAddr, JITSymbolFlags::Exported}}})));

  Expected<AddOneProxy> Call =
      AddOneProxy::Create(AddOneDispatch, ES, AddOneSpec::Name,
                          SymbolLookupFlags::WeaklyReferencedSymbol);
  ASSERT_THAT_EXPECTED(Call, Succeeded());
  EXPECT_TRUE(static_cast<bool>(*Call));
  EXPECT_EQ(Call->calleeAddr(), CalleeAddr);

  cantFail(ES.endSession());
}

// buildProxies resolves a set of proxies from the bootstrap JITDylib via their
// specs, exercising the proxyInit / buildProxies client entry point.
TEST(ProxyTest, BuildProxies) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &JD = ES.getBootstrapJITDylib();
  cantFail(JD.define(absoluteSymbols(
      {{ES.intern(AddOneSpec::Name),
        {ExecutorAddr::fromPtr(addOne), JITSymbolFlags::Exported}}})));

  AddOneProxy Call;
  cantFail(rt::buildProxies(ES, rt::proxyInit<AddOneSpec>(&Call)));
  ASSERT_TRUE(static_cast<bool>(Call));

  Expected<int32_t> R = Call(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// buildProxies with an explicitly-supplied dispatch function and name -- the
// proxyInit overload that takes no spec type.
TEST(ProxyTest, BuildProxiesExplicitDispatch) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &JD = ES.getBootstrapJITDylib();
  cantFail(JD.define(absoluteSymbols(
      {{ES.intern(AddOneSpec::Name),
        {ExecutorAddr::fromPtr(addOne), JITSymbolFlags::Exported}}})));

  AddOneProxy Call;
  cantFail(rt::buildProxies(
      ES, rt::proxyInit(&Call, AddOneDispatch, AddOneSpec::Name)));
  ASSERT_TRUE(static_cast<bool>(Call));

  Expected<int32_t> R = Call(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// buildProxies with a spec but an overridden lookup name -- the proxyInit
// overload that takes a spec type plus an explicit name. The symbol is defined
// only under the override name, so resolving against the spec's default Name
// would fail; success proves the override is used.
TEST(ProxyTest, BuildProxiesSpecNameOverride) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  auto &JD = ES.getBootstrapJITDylib();
  cantFail(JD.define(absoluteSymbols(
      {{ES.intern("add_one_alias"),
        {ExecutorAddr::fromPtr(addOne), JITSymbolFlags::Exported}}})));

  AddOneProxy Call;
  cantFail(
      rt::buildProxies(ES, rt::proxyInit<AddOneSpec>(&Call, "add_one_alias")));
  ASSERT_TRUE(static_cast<bool>(Call));

  Expected<int32_t> R = Call(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  cantFail(ES.endSession());
}

// buildProxies propagates the lookup flags: a weakly-referenced proxyInit for a
// missing symbol yields a null proxy rather than failing the whole build.
TEST(ProxyTest, BuildProxiesWeaklyReferencedAbsent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  AddOneProxy Call;
  cantFail(rt::buildProxies(
      ES, rt::proxyInit<AddOneSpec>(
              &Call, SymbolLookupFlags::WeaklyReferencedSymbol)));
  EXPECT_FALSE(static_cast<bool>(Call));

  cantFail(ES.endSession());
}

// A callee returning Error delivers its result as Error (not Expected<Error>),
// through both call operators, for both success and failure.
TEST(ProxyTest, ErrorReturn) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MaybeFailProxy Call(MaybeFailDispatch, ExecutorAddr::fromPtr(maybeFail));

  EXPECT_THAT_ERROR(Call(ES, false), Succeeded());
  EXPECT_THAT_ERROR(Call(ES, true), Failed());

  std::promise<MSVCPError> P;
  auto F = P.get_future();
  Call([&](Error E) { P.set_value(std::move(E)); }, ES, true);
  EXPECT_THAT_ERROR(Error(F.get()), Failed());

  cantFail(ES.endSession());
}

// A callee returning Expected<T> delivers its result flattened as Expected<T>
// (not Expected<Expected<T>>): the callee's value or error passes through
// directly.
TEST(ProxyTest, ExpectedReturn) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  AddOneOrFailProxy Call(AddOneOrFailDispatch,
                         ExecutorAddr::fromPtr(addOneOrFail));

  Expected<int32_t> R = Call(ES, 41);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 42);

  EXPECT_THAT_EXPECTED(Call(ES, -1), Failed());

  std::promise<MSVCPExpected<int32_t>> P;
  auto F = P.get_future();
  Call([&](Expected<int32_t> RA) { P.set_value(std::move(RA)); }, ES, 41);
  Expected<int32_t> RAsync = F.get();
  ASSERT_THAT_EXPECTED(RAsync, Succeeded());
  EXPECT_EQ(*RAsync, 42);

  cantFail(ES.endSession());
}
