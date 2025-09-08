//===-- SPSMemoryFlagsTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test SPS serialization for MemoryFlags APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SPSMemoryFlags.h"

#include "SimplePackedSerializationTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(SPSMemoryFlags, TestAllocGroupSerialization) {
  for (bool Read : {false, true}) {
    for (bool Write : {false, true}) {
      for (bool Exec : {false, true}) {
        for (bool FinalizeLifetime : {false, true}) {
          AllocGroup AG((Read ? MemProt::Read : MemProt::None) |
                            (Write ? MemProt::Write : MemProt::None) |
                            (Exec ? MemProt::Exec : MemProt::None),
                        FinalizeLifetime ? MemLifetime::Finalize
                                         : MemLifetime::Standard);
          blobSerializationRoundTrip<SPSAllocGroup, AllocGroup>(AG);
        }
      }
    }
  }
}
