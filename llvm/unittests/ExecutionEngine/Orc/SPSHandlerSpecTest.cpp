//===- SPSHandlerSpecTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for sps::HandlerSpec and bindCallControllerHandler.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SPSHandlerSpec.h"
#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/BindCallControllerHandler.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

struct AddDescriptor {
  static constexpr SymbolNameSpec Name = SymbolNameSpec::verbatim("add_tag");
  using SPSSig = int32_t(int32_t, int32_t);
};

using AddSpec = sps::HandlerSpec<AddDescriptor>;

void addHandler(unique_function<void(int32_t)> SendResult, int32_t X,
                int32_t Y) {
  SendResult(X + Y);
}

struct Adder {
  int32_t Offset = 0;
  void add(unique_function<void(int32_t)> SendResult, int32_t X, int32_t Y) {
    SendResult(X + Y + Offset);
  }
};

class SPSHandlerSpecTest : public CoreAPIsBasedStandardTest {
protected:
  static constexpr ExecutorAddr DefaultTagAddr{0x1000};
  static constexpr ExecutorAddr OtherTagAddr{0x2000};

  // Define a tag symbol at the given address.
  void defineTag(StringRef Name, ExecutorAddr Addr) {
    cantFail(JD.define(absoluteSymbols(
        {{ES.intern(Name), {Addr, JITSymbolFlags::Exported}}})));
  }

  // Call the handler registered for TagAddr as the executor would.
  Expected<int32_t> callAdd(ExecutorAddr TagAddr, int32_t X, int32_t Y) {
    std::promise<MSVCPExpected<int32_t>> P;
    auto F = P.get_future();
    WrapperFunction<AddDescriptor::SPSSig>::callAsync(
        [&](auto SendResult, const char *ArgData, size_t ArgSize) {
          ES.runCallControllerHandler(
              std::move(SendResult), TagAddr,
              WrapperFunctionBuffer::copyFrom(ArgData, ArgSize));
        },
        [&](Error Err, int32_t Result) {
          if (Err)
            P.set_value(std::move(Err));
          else
            P.set_value(Result);
        },
        X, Y);
    return F.get();
  }
};

} // namespace

TEST_F(SPSHandlerSpecTest, BindUnderDefaultName) {
  defineTag("add_tag", DefaultTagAddr);
  cantFail(ES.registerCallControllerHandlers(
      JD, bindCallControllerHandler<AddSpec>(addHandler)));

  EXPECT_THAT_EXPECTED(callAdd(DefaultTagAddr, 1, 2), HasValue(3));
}

TEST_F(SPSHandlerSpecTest, BindUnderOverridingName) {
  // Only the overriding tag is defined, so binding under the spec's default
  // name would fail registration.
  defineTag("other_tag", OtherTagAddr);
  cantFail(ES.registerCallControllerHandlers(
      JD, bindCallControllerHandler<AddSpec>(
              SymbolNameSpec::verbatim("other_tag"), addHandler)));

  EXPECT_THAT_EXPECTED(callAdd(OtherTagAddr, 1, 2), HasValue(3));
}

TEST_F(SPSHandlerSpecTest, BindMethodUnderDefaultName) {
  defineTag("add_tag", DefaultTagAddr);
  Adder A{10};
  cantFail(ES.registerCallControllerHandlers(
      JD, bindCallControllerHandler<AddSpec>(&A, &Adder::add)));

  EXPECT_THAT_EXPECTED(callAdd(DefaultTagAddr, 1, 2), HasValue(13));
}

TEST_F(SPSHandlerSpecTest, BindMethodUnderOverridingName) {
  defineTag("other_tag", OtherTagAddr);
  Adder A{10};
  cantFail(ES.registerCallControllerHandlers(
      JD, bindCallControllerHandler<AddSpec>(
              SymbolNameSpec::verbatim("other_tag"), &A, &Adder::add)));

  EXPECT_THAT_EXPECTED(callAdd(OtherTagAddr, 1, 2), HasValue(13));
}
