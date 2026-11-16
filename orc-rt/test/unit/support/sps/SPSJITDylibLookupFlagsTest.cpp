//===-- SPSJITDylibLookupFlagsTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for JITDylibLookupFlags.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/sps/SPSJITDylibLookupFlags.h"

#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;
using namespace orc_rt::test;

TEST(SPSJITDylibLookupFlagsTest, Serialization) {
  blobSerializationRoundTrip<SPSJITDylibLookupFlags, JITDylibLookupFlags>(
      JITDylibLookupFlags::MatchExportedSymbolsOnly);
  blobSerializationRoundTrip<SPSJITDylibLookupFlags, JITDylibLookupFlags>(
      JITDylibLookupFlags::MatchAllSymbols);
}
