//===-- SPSAllocActionTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for AllocAction APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SPSAllocAction.h"

#include "orc-rt/ExecutorAddress.h"

#include "AllocActionTestUtils.h"
#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;

static bool AAEQ(const AllocAction &LHS, const AllocAction &RHS) {
  if (LHS.Fn != RHS.Fn)
    return false;
  if (LHS.ArgData.size() != RHS.ArgData.size())
    return false;
  return memcmp(LHS.ArgData.data(), RHS.ArgData.data(), LHS.ArgData.size()) ==
         0;
}

static bool AAPEQ(const AllocActionPair &LHS, const AllocActionPair &RHS) {
  return AAEQ(LHS.Finalize, RHS.Finalize) && AAEQ(LHS.Dealloc, RHS.Dealloc);
}

static orc_rt_WrapperFunctionBuffer noopAction(const char *ArgData,
                                               size_t ArgSize) {
  return WrapperFunctionBuffer().release();
}

TEST(SPSAllocActionTest, AllocActionSerialization) {
  AllocAction AA(noopAction, WrapperFunctionBuffer::copyFrom("hello, world!"));
  blobSerializationRoundTrip<SPSAllocAction>(AA, AAEQ);
}

TEST(SPSAllocActionTest, AllocActionPairSerialization) {
  AllocActionPair AAP;
  AAP.Finalize = {noopAction, WrapperFunctionBuffer::copyFrom("foo")};
  AAP.Dealloc = {noopAction, WrapperFunctionBuffer::copyFrom("foo")};

  blobSerializationRoundTrip<SPSAllocActionPair>(AAP, AAPEQ);
}

// Handler that returns Error::success(). Exercises the
// AllocActionSPSSerializer::serialize(Error) overload's success path.
static orc_rt_WrapperFunctionBuffer
errorSuccess_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<>::handle(
             ArgData, ArgSize, []() -> Error { return Error::success(); })
      .release();
}

// Handler that returns a StringError. Exercises the
// AllocActionSPSSerializer::serialize(Error) overload's failure path.
static orc_rt_WrapperFunctionBuffer
errorFailure_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<>::handle(
             ArgData, ArgSize,
             []() -> Error { return make_error<StringError>("test failure"); })
      .release();
}

TEST(SPSAllocActionTest, RunActionWithErrorSuccessReturn) {
  // A handler returning Error::success() should produce a non-out-of-band
  // result buffer.
  AllocAction AA(errorSuccess_sps_allocaction, WrapperFunctionBuffer());
  auto B = AA();
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(SPSAllocActionTest, RunActionWithErrorFailureReturn) {
  // A handler returning a real Error should produce an out-of-band error
  // result buffer carrying the Error's string form.
  AllocAction AA(errorFailure_sps_allocaction, WrapperFunctionBuffer());
  auto B = AA();
  ASSERT_NE(B.getOutOfBandError(), nullptr);
  EXPECT_STREQ(B.getOutOfBandError(), "test failure");
}

// Handler that takes an SPS-encoded int* argument and increments the int.
// Exercises both the SPS argument-deserialization path and the identity
// (WrapperFunctionBuffer) overload of AllocActionSPSSerializer.
static orc_rt_WrapperFunctionBuffer
increment_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<SPSExecutorAddr>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr IntPtr) {
               *IntPtr.toPtr<int *>() += 1;
               return WrapperFunctionBuffer();
             })
      .release();
}

TEST(SPSAllocActionTest, RunActionWithSPSArgsAndWFBReturn) {
  // Verifies that handlers with SPS-encoded arguments work end-to-end: SPS
  // deserializes the int*, the handler runs, and its empty WFB return is
  // passed through by AllocActionSPSSerializer's identity overload.
  int Val = 0;
  auto IncVal = *MakeAllocAction<SPSExecutorAddr>::from(
      increment_sps_allocaction, ExecutorAddr::fromPtr(&Val));
  EXPECT_TRUE(IncVal);
  auto B = IncVal();
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
  EXPECT_EQ(Val, 1);
}

TEST(SPSAllocActionTest, RunActionWithUndecodableArgs) {
  // An arg buffer that's too small for the wrapper's declared SPS signature
  // (SPSExecutorAddr expects 8 bytes here) should cause
  // AllocActionFunction::handle to return its out-of-band deserialization
  // error without invoking the handler.
  AllocAction AA(increment_sps_allocaction,
                 WrapperFunctionBuffer::allocate(/*Size=*/1));
  auto B = AA();
  ASSERT_NE(B.getOutOfBandError(), nullptr);
  EXPECT_STREQ(B.getOutOfBandError(),
               "Could not deserialize allocation action argument buffer");
}

// Test the ORC_RT_SPS_ALLOC_ACTION macro.
static Error check_values_equal(int32_t X, int32_t Y) {
  if (X == Y)
    return Error::success();
  return make_error<StringError>("X and Y differ");
}
ORC_RT_SPS_ALLOC_ACTION(macro_defined_allocaction, (int32_t, int32_t),
                        check_values_equal)

TEST(SPSAllocActionTest, RunMacroDefinedAllocActionWithErrorSuccessReturn) {
  AllocAction AA(macro_defined_allocaction,
                 *spsSerialize<SPSArgList<int32_t, int32_t>>(42, 42));
  auto B = AA();
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(SPSAllocActionTest, RunMacroDefinedAllocActionWithErrorFailureReturn) {
  AllocAction AA(macro_defined_allocaction,
                 *spsSerialize<SPSArgList<int32_t, int32_t>>(42, 7));
  auto B = AA();
  ASSERT_NE(B.getOutOfBandError(), nullptr);
  EXPECT_STREQ(B.getOutOfBandError(), "X and Y differ");
}
