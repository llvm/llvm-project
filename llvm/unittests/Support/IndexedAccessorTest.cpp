//===- IndexedAccessorTest.cpp - Indexed Accessor Tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "gmock/gmock.h"

using namespace llvm;
using namespace llvm::detail;

namespace {
/// Simple indexed accessor range that wraps an array.
template <typename T>
struct ArrayIndexedAccessorRange
    : public indexed_accessor_range<ArrayIndexedAccessorRange<T>, T *, T> {
  ArrayIndexedAccessorRange(T *data, ptrdiff_t start, ptrdiff_t numElements)
      : indexed_accessor_range<ArrayIndexedAccessorRange<T>, T *, T>(
            data, start, numElements) {}
  using indexed_accessor_range<ArrayIndexedAccessorRange<T>, T *,
                               T>::indexed_accessor_range;

  /// See `llvm::indexed_accessor_range` for details.
  static T &dereference(T *data, ptrdiff_t index) { return data[index]; }
};
} // end anonymous namespace

template <typename T>
static void compareData(ArrayIndexedAccessorRange<T> range,
                        ArrayRef<T> referenceData) {
  ASSERT_EQ(referenceData.size(), range.size());
  ASSERT_TRUE(std::equal(range.begin(), range.end(), referenceData.begin()));
}

namespace {
TEST(AccessorRange, SliceTest) {
  int rawData[] = {0, 1, 2, 3, 4};
  ArrayRef<int> data = llvm::ArrayRef(rawData);

  ArrayIndexedAccessorRange<int> range(rawData, /*start=*/0, /*numElements=*/5);
  compareData(range, data);
  compareData(range.slice(2, 3), data.slice(2, 3));
  compareData(range.slice(0, 5), data.slice(0, 5));
}

TEST(AccessorRange, EqualTest) {
  int32_t rawData1[] = {0, 1, 2, 3, 4};
  uint64_t rawData2[] = {0, 1, 2, 3, 4};

  ArrayIndexedAccessorRange<int32_t> range1(rawData1, /*start=*/0,
                                            /*numElements=*/5);
  ArrayIndexedAccessorRange<uint64_t> range2(rawData2, /*start=*/0,
                                             /*numElements=*/5);
  EXPECT_TRUE(range1 == range2);
  EXPECT_FALSE(range1 != range2);
  EXPECT_TRUE(range2 == range1);
  EXPECT_FALSE(range2 != range1);
}
} // end anonymous namespace
