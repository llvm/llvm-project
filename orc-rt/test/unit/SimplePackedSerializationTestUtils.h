//===- SimplePackedSerializationTestUtils.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H
#define ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H

#include "orc-rt/support/WrapperFunction.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"
#include "gtest/gtest.h"

#include <optional>

namespace orc_rt::test {

template <typename SPSTraitsT, typename... ArgTs>
std::optional<WrapperFunctionBuffer> spsSerialize(const ArgTs &...Args) {
  auto B = WrapperFunctionBuffer::allocate(SPSTraitsT::size(Args...));
  SPSOutputBuffer OB(B.data(), B.size());
  if (!SPSTraitsT::serialize(OB, Args...))
    return std::nullopt;
  return B;
}

template <typename SPSTraitsT, typename... ArgTs>
bool spsDeserialize(WrapperFunctionBuffer &B, ArgTs &...Args) {
  SPSInputBuffer IB(B.data(), B.size());
  return SPSTraitsT::deserialize(IB, Args...);
}

template <typename SPSTagT, typename T, typename Comparator = std::equal_to<T>>
void blobSerializationRoundTrip(const T &Value, Comparator &&C = Comparator()) {
  using BST = SPSSerializationTraits<SPSTagT, T>;

  size_t Size = BST::size(Value);
  auto Buffer = std::make_unique<char[]>(Size);
  SPSOutputBuffer OB(Buffer.get(), Size);

  EXPECT_TRUE(BST::serialize(OB, Value));

  SPSInputBuffer IB(Buffer.get(), Size);

  T DSValue;
  EXPECT_TRUE(BST::deserialize(IB, DSValue));

  EXPECT_TRUE(C(Value, DSValue))
      << "Incorrect value after serialization/deserialization round-trip";
}

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_SIMPLEPACKEDSERIALIZATIONTESTUTILS_H
