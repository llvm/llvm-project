//=== DistinctAttributeAllocatorTest.cpp - DistinctAttr storage alloc test ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include <thread>

using namespace mlir;

//
// Test that a DistinctAttr that is created on a separate thread does
// not have its storage deleted when the thread joins.
//
TEST(DistinctAttributeAllocatorTest, TestAttributeWellFormedAfterThreadJoin) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);
  DistinctAttr attr;

  std::thread t([&ctx, &attr]() {
    attr = DistinctAttr::create(UnitAttr::get(&ctx));
    ASSERT_TRUE(attr);
  });
  t.join();

  // If the attribute storage got deleted after the thread joins (which we don't
  // want) then trying to access it triggers an assert in Debug mode, and a
  // crash otherwise. Run this in a CrashRecoveryContext to avoid bringing down
  // the whole test suite if this test fails. Additionally, MSAN and/or TSAN
  // should raise failures here if the attribute storage was deleted.
  llvm::CrashRecoveryContext crc;
  EXPECT_TRUE(crc.RunSafely([attr]() { (void)attr.getAbstractAttribute(); }));
  EXPECT_TRUE(
      crc.RunSafely([attr]() { (void)*cast<Attribute>(attr).getImpl(); }));

  ASSERT_TRUE(attr);
}
