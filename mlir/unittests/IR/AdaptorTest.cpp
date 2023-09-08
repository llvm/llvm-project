//===- AdaptorTest.cpp - Adaptor unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOpsSyntax.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace mlir;
using namespace test;

using testing::ElementsAre;

TEST(Adaptor, GenericAdaptorsOperandAccess) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();
  Builder builder(&context);

  // Has normal and Variadic arguments.
  MixedNormalVariadicOperandOp::FoldAdaptor a({});
  {
    SmallVector<int> v = {0, 1, 2, 3, 4};
    MixedNormalVariadicOperandOp::GenericAdaptor<ArrayRef<int>> b(v);
    EXPECT_THAT(b.getInput1(), ElementsAre(0, 1));
    EXPECT_EQ(b.getInput2(), 2);
    EXPECT_THAT(b.getInput3(), ElementsAre(3, 4));
  }

  // Has optional arguments.
  OIListSimple::FoldAdaptor c({}, nullptr);
  {
    // Optional arguments return the default constructed value if not present.
    // Using optional instead of plain int here to differentiate absence of
    // value from the value 0.
    SmallVector<std::optional<int>> v = {0, 4};
    OIListSimple::Properties prop;
    prop.operandSegmentSizes = {1, 0, 1};
    OIListSimple::GenericAdaptor<ArrayRef<std::optional<int>>> d(v, {}, prop,
                                                                 {});
    EXPECT_EQ(d.getArg0(), 0);
    EXPECT_EQ(d.getArg1(), std::nullopt);
    EXPECT_EQ(d.getArg2(), 4);

    // Check the property comparison operator.
    OIListSimple::Properties equivalentProp = {1, 0, 1};
    OIListSimple::Properties differentProp = {0, 0, 1};
    EXPECT_EQ(d.getProperties(), equivalentProp);
    EXPECT_NE(d.getProperties(), differentProp);
  }

  // Has VariadicOfVariadic arguments.
  FormatVariadicOfVariadicOperand::FoldAdaptor e({});
  {
    SmallVector<int> v = {0, 1, 2, 3, 4};
    FormatVariadicOfVariadicOperand::Properties prop;
    prop.operand_segments = builder.getDenseI32ArrayAttr({3, 2, 0});
    FormatVariadicOfVariadicOperand::GenericAdaptor<ArrayRef<int>> f(v, {},
                                                                     prop, {});
    SmallVector<ArrayRef<int>> operand = f.getOperand();
    ASSERT_EQ(operand.size(), (std::size_t)3);
    EXPECT_THAT(operand[0], ElementsAre(0, 1, 2));
    EXPECT_THAT(operand[1], ElementsAre(3, 4));
    EXPECT_THAT(operand[2], ElementsAre());
  }
}
