//===-- SPSSymbolLookupSetTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for SymbolLookupFlags and SymbolLookupSet APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/sps/SPSSymbolLookupSet.h"

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

TEST(SPSSymbolLookupSetTest, SymbolLookupFlagsSerialization) {
  blobSerializationRoundTrip<bool, SymbolLookupFlags>(
      SymbolLookupFlags::RequiredSymbol);
  blobSerializationRoundTrip<bool, SymbolLookupFlags>(
      SymbolLookupFlags::WeaklyReferencedSymbol);
}

TEST(SPSSymbolLookupSetTest, SymbolLookupFlagsWireFormat) {
  // RequiredSymbol must serialize as true and WeaklyReferencedSymbol as false
  // to match the controller's RemoteSymbolLookupSetElement::Required field.
  for (auto [F, Required] :
       {std::pair{SymbolLookupFlags::RequiredSymbol, true},
        std::pair{SymbolLookupFlags::WeaklyReferencedSymbol, false}}) {
    char Buffer[sizeof(bool)];
    SPSOutputBuffer OB(Buffer, sizeof(Buffer));
    ASSERT_TRUE(
        (SPSSerializationTraits<bool, SymbolLookupFlags>::serialize(OB, F)));
    SPSInputBuffer IB(Buffer, sizeof(Buffer));
    bool B = !Required;
    ASSERT_TRUE((SPSSerializationTraits<bool, bool>::deserialize(IB, B)));
    EXPECT_EQ(B, Required);
  }
}

TEST(SPSSymbolLookupSetTest, SymbolLookupSetSerialization) {
  using SPSTag = SPSSequence<SPSTuple<SPSString, bool>>;
  blobSerializationRoundTrip<SPSTag, SymbolLookupSet>(
      SymbolLookupSet(), seqEqual<SymbolLookupSet>);
  blobSerializationRoundTrip<SPSTag, SymbolLookupSet>(
      SymbolLookupSet({{"foo", SymbolLookupFlags::RequiredSymbol},
                       {"bar", SymbolLookupFlags::WeaklyReferencedSymbol}}),
      seqEqual<SymbolLookupSet>);
}

TEST(SPSSymbolLookupSetTest, SymbolLookupResultSerialization) {
  using SPSTag = SPSSequence<SPSOptional<SPSExecutorAddr>>;
  int X = 0;
  SymbolLookupResult R;
  R.push_back(&X);
  R.push_back(nullptr);
  R.push_back(std::nullopt);
  blobSerializationRoundTrip<SPSTag, SymbolLookupResult>(
      SymbolLookupResult(), seqEqual<SymbolLookupResult>);
  blobSerializationRoundTrip<SPSTag, SymbolLookupResult>(
      R, seqEqual<SymbolLookupResult>);
}
