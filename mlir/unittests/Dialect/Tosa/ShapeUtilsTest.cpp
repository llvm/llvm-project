//===- ShapeUtilsTest.cpp - Tosa shape knowledge tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::tosa;

TEST(ValueKnowledgeTest, EqualityIgnoresErrors) {
  SmallVector<int64_t> dimensions(16, 7);
  ValueKnowledge valid(true, dimensions, {});
  ValueKnowledge invalid = valid;
  invalid.setHasError(true);
  EXPECT_TRUE(valid);
  EXPECT_FALSE(invalid);
  EXPECT_EQ(valid, invalid);
  EXPECT_NE(valid.sizes.getFlags(), invalid.sizes.getFlags());
}

TEST(ValueKnowledgeTest, JoinAndMeetPreserveRankAndErrors) {
  SmallVector<int64_t> dimensions(16, 7);
  ValueKnowledge ranked(true, dimensions, {});
  ValueKnowledge unranked = ValueKnowledge::getPessimisticValueState();
  for (ValueKnowledge result : {ValueKnowledge::join(ranked, unranked),
                                ValueKnowledge::join(unranked, ranked),
                                ValueKnowledge::meet(ranked, ranked)}) {
    EXPECT_TRUE(result);
    EXPECT_TRUE(result.hasRank());
    EXPECT_EQ(ArrayRef<int64_t>(result.sizes), ArrayRef<int64_t>(dimensions));
  }
  ValueKnowledge result = ValueKnowledge::meet(ranked, unranked);
  EXPECT_TRUE(result);
  EXPECT_FALSE(result.hasRank());

  ValueKnowledge conflicting(true, {8}, {});
  ValueKnowledge singleDimension(true, {7}, {});
  result = ValueKnowledge::join(singleDimension, conflicting);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.hasRank());
}
