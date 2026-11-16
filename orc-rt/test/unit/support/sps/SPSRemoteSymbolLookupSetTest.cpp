//===-- SPSRemoteSymbolLookupSetTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for SymbolLookupFlags and RemoteSymbolLookupSet APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/sps/SPSRemoteSymbolLookupSet.h"

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

TEST(SPSRemoteSymbolLookupSetTest, SymbolLookupFlagsSerialization) {
  blobSerializationRoundTrip<bool, SymbolLookupFlags>(
      SymbolLookupFlags::RequiredSymbol);
  blobSerializationRoundTrip<bool, SymbolLookupFlags>(
      SymbolLookupFlags::WeaklyReferencedSymbol);
}

TEST(SPSRemoteSymbolLookupSetTest, SymbolLookupFlagsWireFormat) {
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

TEST(SPSRemoteSymbolLookupSetTest, RemoteSymbolLookupSetSerialization) {
  blobSerializationRoundTrip<SPSSymbolLookupSet, RemoteSymbolLookupSet>(
      RemoteSymbolLookupSet(), seqEqual<RemoteSymbolLookupSet>);
  blobSerializationRoundTrip<SPSSymbolLookupSet, RemoteSymbolLookupSet>(
      RemoteSymbolLookupSet(
          {{"foo", SymbolLookupFlags::RequiredSymbol},
           {"bar", SymbolLookupFlags::WeaklyReferencedSymbol}}),
      seqEqual<RemoteSymbolLookupSet>);
}
