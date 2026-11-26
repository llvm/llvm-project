//===-- SPSSymbolLookupResultTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for SymbolLookupResult.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/sps/SPSSymbolLookupResult.h"

#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

#include <algorithm>

using namespace orc_rt;
using namespace orc_rt::test;

template <typename SeqT>
static bool seqEqual(const SeqT &LHS, const SeqT &RHS) {
  return LHS.size() == RHS.size() &&
         std::equal(LHS.begin(), LHS.end(), RHS.begin());
}

TEST(SPSSymbolLookupResultTest, SymbolLookupResultSerialization) {
  int X = 0;
  SymbolLookupResult R;
  R.push_back(&X);
  R.push_back(nullptr);
  R.push_back(std::nullopt);
  blobSerializationRoundTrip<SPSSymbolLookupResult, SymbolLookupResult>(
      SymbolLookupResult(), seqEqual<SymbolLookupResult>);
  blobSerializationRoundTrip<SPSSymbolLookupResult, SymbolLookupResult>(
      R, seqEqual<SymbolLookupResult>);
}
