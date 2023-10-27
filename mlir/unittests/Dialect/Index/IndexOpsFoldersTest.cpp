//===- IndexOpsFoldersTest.cpp - unit tests for index op folders ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
/// Test fixture for testing operation folders.
class IndexFolderTest : public testing::Test {
public:
  IndexFolderTest() { ctx.getOrLoadDialect<index::IndexDialect>(); }

  /// Instantiate an operation, invoke its folder, and return the attribute
  /// result.
  template <typename OpT>
  void foldOp(IntegerAttr &value, Type type, ArrayRef<Attribute> operands);

protected:
  /// The MLIR context to use.
  MLIRContext ctx;
  /// A builder to use.
  OpBuilder b{&ctx};
};
} // namespace

template <typename OpT>
void IndexFolderTest::foldOp(IntegerAttr &value, Type type,
                             ArrayRef<Attribute> operands) {
  // This function returns null so that `ASSERT_*` works within it.
  OperationState state(UnknownLoc::get(&ctx), OpT::getOperationName());
  state.addTypes(type);
  OwningOpRef<OpT> op = cast<OpT>(b.create(state));
  SmallVector<OpFoldResult> results;
  LogicalResult result = op->getOperation()->fold(operands, results);
  // Propagate the failure to the test.
  if (failed(result)) {
    value = nullptr;
    return;
  }
  ASSERT_EQ(results.size(), 1u);
  value = dyn_cast_or_null<IntegerAttr>(dyn_cast<Attribute>(results.front()));
  ASSERT_TRUE(value);
}

TEST_F(IndexFolderTest, TestCastUOpFolder) {
  IntegerAttr value;
  auto fold = [&](Type type, Attribute input) {
    foldOp<index::CastUOp>(value, type, input);
  };

  // Target width less than or equal to 32 bits.
  fold(b.getIntegerType(16), b.getIndexAttr(8000000000));
  ASSERT_TRUE(value);
  EXPECT_EQ(value.getInt(), 20480u);

  // Target width greater than or equal to 64 bits.
  fold(b.getIntegerType(64), b.getIndexAttr(2000));
  ASSERT_TRUE(value);
  EXPECT_EQ(value.getInt(), 2000u);

  // Fails to fold, because truncating to 32 bits and then extending creates a
  // different value.
  fold(b.getIntegerType(64), b.getIndexAttr(8000000000));
  EXPECT_FALSE(value);

  // Target width between 32 and 64 bits.
  fold(b.getIntegerType(40), b.getIndexAttr(0x10000000010000));
  // Fold succeeds because the upper bits are truncated in the cast.
  ASSERT_TRUE(value);
  EXPECT_EQ(value.getInt(), 65536);

  // Fails to fold because the upper bits are not truncated.
  fold(b.getIntegerType(60), b.getIndexAttr(0x10000000010000));
  EXPECT_FALSE(value);
}

TEST_F(IndexFolderTest, TestCastSOpFolder) {
  IntegerAttr value;
  auto fold = [&](Type type, Attribute input) {
    foldOp<index::CastSOp>(value, type, input);
  };

  // Just test the extension cases to ensure signs are being respected.

  // Target width greater than or equal to 64 bits.
  fold(b.getIntegerType(64), b.getIndexAttr(-2000));
  ASSERT_TRUE(value);
  EXPECT_EQ(value.getInt(), -2000);

  // Target width between 32 and 64 bits.
  fold(b.getIntegerType(40), b.getIndexAttr(-0x10000000010000));
  // Fold succeeds because the upper bits are truncated in the cast.
  ASSERT_TRUE(value);
  EXPECT_EQ(value.getInt(), -65536);
}
