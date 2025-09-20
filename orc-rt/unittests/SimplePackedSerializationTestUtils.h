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
#include "orc-rt/WrapperFunction.h"
#include "gtest/gtest.h"

#include <optional>

template <typename SPSTraitsT, typename... ArgTs>
static inline std::optional<orc_rt::WrapperFunctionBuffer>
spsSerialize(const ArgTs &...Args) {
  auto B = orc_rt::WrapperFunctionBuffer::allocate(SPSTraitsT::size(Args...));
  orc_rt::SPSOutputBuffer OB(B.data(), B.size());
  if (!SPSTraitsT::serialize(OB, Args...))
    return std::nullopt;
  return B;
}

template <typename SPSTraitsT, typename... ArgTs>
static bool spsDeserialize(orc_rt::WrapperFunctionBuffer &B, ArgTs &...Args) {
  orc_rt::SPSInputBuffer IB(B.data(), B.size());
  return SPSTraitsT::deserialize(IB, Args...);
}

template <typename SPSTagT, typename T, typename Comparator = std::equal_to<T>>
static inline void blobSerializationRoundTrip(const T &Value,
                                              Comparator &&C = Comparator()) {
  using BST = orc_rt::SPSSerializationTraits<SPSTagT, T>;

  size_t Size = BST::size(Value);
  auto Buffer = std::make_unique<char[]>(Size);
  orc_rt::SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BST::serialize(OB, Value));

  orc_rt::SPSInputBuffer IB(Buffer.get(), Size);

  T DSValue;
  EXPECT_TRUE(BST::deserialize(IB, DSValue));

  EXPECT_TRUE(C(Value, DSValue))
      << "Incorrect value after serialization/deserialization round-trip";
}

#endif // ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H
