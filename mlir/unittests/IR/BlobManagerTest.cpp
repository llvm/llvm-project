//===- mlir/unittest/IR/BlobManagerTest.cpp - Blob management unit tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {

StringLiteral moduleStr = R"mlir(
"test.use1"() {attr = dense_resource<blob1> : tensor<1xi64> } : () -> ()

{-#
    dialect_resources: {
    builtin: {
        blob1: "0x08000000ABCDABCDABCDABCE"
    }
    }
#-}
)mlir";

TEST(DialectResourceBlobManagerTest, Lookup) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> m = parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(m);

  const auto &dialectManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(&context);
  ASSERT_NE(dialectManager.getBlobManager().lookup("blob1"), nullptr);
}

TEST(DialectResourceBlobManagerTest, GetBlobMap) {
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> m = parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(m);

  Block *block = m->getBody();
  auto &op = block->getOperations().front();
  auto resourceAttr = op.getAttrOfType<DenseResourceElementsAttr>("attr");
  ASSERT_NE(resourceAttr, nullptr);

  const auto &dialectManager =
      resourceAttr.getRawHandle().getManagerInterface(&context);

  bool blobsArePresent = false;
  dialectManager.getBlobManager().getBlobMap(
      [&](const llvm::StringMap<DialectResourceBlobManager::BlobEntry>
              &blobMap) { blobsArePresent = blobMap.contains("blob1"); });
  ASSERT_TRUE(blobsArePresent);

  // remove operations that use resources - resources must still be accessible
  block->clear();

  blobsArePresent = false;
  dialectManager.getBlobManager().getBlobMap(
      [&](const llvm::StringMap<DialectResourceBlobManager::BlobEntry>
              &blobMap) { blobsArePresent = blobMap.contains("blob1"); });
  ASSERT_TRUE(blobsArePresent);
}

TEST(DialectResourceBlobManagerTest, InsertDisambiguatesNames) {
  DialectResourceBlobManager manager;

  // Repeated insertion of the same base name appends an increasing counter.
  EXPECT_EQ(manager.insert("blob").getKey(), "blob");
  EXPECT_EQ(manager.insert("blob").getKey(), "blob_1");
  EXPECT_EQ(manager.insert("blob").getKey(), "blob_2");
  EXPECT_EQ(manager.insert("blob").getKey(), "blob_3");

  // A different base name is numbered independently.
  EXPECT_EQ(manager.insert("other").getKey(), "other");
  EXPECT_EQ(manager.insert("other").getKey(), "other_1");
}

TEST(DialectResourceBlobManagerTest, InsertSkipsNamesTakenDirectly) {
  DialectResourceBlobManager manager;

  EXPECT_EQ(manager.insert("blob").getKey(), "blob");
  // Directly claim the name the next collision would otherwise produce.
  EXPECT_EQ(manager.insert("blob_1").getKey(), "blob_1");
  // The disambiguation search must skip the taken slot rather than reuse it,
  // even though the resumed counter points at it first.
  EXPECT_EQ(manager.insert("blob").getKey(), "blob_2");
}

} // end anonymous namespace
