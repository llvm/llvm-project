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
