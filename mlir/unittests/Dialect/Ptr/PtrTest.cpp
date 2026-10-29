//===- PtrTest.cpp - Ptr dialect tests ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(Ptr, TypeOffsetIncludesAllocationPadding) {
  MLIRContext context;
  context.loadDialect<ptr::PtrDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());
  auto check = [&](Type type, uint64_t expected) {
    auto offset = ptr::TypeOffsetOp::create(builder, builder.getUnknownLoc(),
                                            builder.getIndexType(), type);
    DataLayout layout(module.get());
    EXPECT_EQ(offset.getTypeSize(layout).getFixedValue(), expected);
    EXPECT_EQ(offset.getTypeSize().getFixedValue(), expected);
  };
  check(builder.getIntegerType(24), 4);
  check(VectorType::get({3}, builder.getI32Type()), 16);
}
