//===- SimplePackedSerializationTestUtils.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H
#define ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H

#include "orc-rt/SimplePackedSerialization.h"
#include "gtest/gtest.h"

template <typename SPSTagT, typename T>
static void blobSerializationRoundTrip(const T &Value) {
  using BST = orc_rt::SPSSerializationTraits<SPSTagT, T>;

  size_t Size = BST::size(Value);
  auto Buffer = std::make_unique<char[]>(Size);
  orc_rt::SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BST::serialize(OB, Value));

  orc_rt::SPSInputBuffer IB(Buffer.get(), Size);

  T DSValue;
  EXPECT_TRUE(BST::deserialize(IB, DSValue));

  EXPECT_EQ(Value, DSValue)
      << "Incorrect value after serialization/deserialization round-trip";
}

#endif // ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H
