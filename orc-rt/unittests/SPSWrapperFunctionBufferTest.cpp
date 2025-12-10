//===-- SPSWrapperFunctionBufferTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for WrapperFunctionBuffers.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SPSWrapperFunctionBuffer.h"

#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;

static bool WFBEQ(const WrapperFunctionBuffer &LHS,
                  const WrapperFunctionBuffer &RHS) {
  if (LHS.size() != RHS.size())
    return false;
  return memcmp(LHS.data(), RHS.data(), LHS.size()) == 0;
}

TEST(SPSWrapperFunctionBufferTest, EmptyBuffer) {
  WrapperFunctionBuffer EB;
  blobSerializationRoundTrip<SPSWrapperFunctionBuffer>(EB, WFBEQ);
}

TEST(SPSWrapperFunctionBufferTest, SmallBuffer) {
  const char *Source = "foo";
  auto EB = WrapperFunctionBuffer::copyFrom(Source);
  blobSerializationRoundTrip<SPSWrapperFunctionBuffer>(EB, WFBEQ);
}

TEST(SPSWrapperFunctionBufferTest, BigBuffer) {
  const char *Source = "The quick brown fox jumps over the lazy dog";
  auto EB = WrapperFunctionBuffer::copyFrom(Source);
  blobSerializationRoundTrip<SPSWrapperFunctionBuffer>(EB, WFBEQ);
}
