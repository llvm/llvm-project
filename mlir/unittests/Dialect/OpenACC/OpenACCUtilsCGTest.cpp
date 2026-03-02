//===- OpenACCUtilsCGTest.cpp - Unit tests for OpenACC CG utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsCGTest : public ::testing::Test {
protected:
  OpenACCUtilsCGTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<acc::OpenACCDialect, DLTIDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
};

//===----------------------------------------------------------------------===//
// getDataLayout Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsCGTest, getDataLayoutNoSpecAllowDefault) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // With allowDefault=true, should return a default DataLayout
  auto dl = getDataLayout(module->getOperation(), /*allowDefault=*/true);
  EXPECT_TRUE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutNoSpecDisallowDefault) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // With allowDefault=false and no spec, should return nullopt
  auto dl = getDataLayout(module->getOperation(), /*allowDefault=*/false);
  EXPECT_FALSE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutNullOp) {
  // Null operation should return nullopt
  auto dl = getDataLayout(nullptr, /*allowDefault=*/true);
  EXPECT_FALSE(dl.has_value());
}

TEST_F(OpenACCUtilsCGTest, getDataLayoutWithSpec) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // Add a data layout spec to the module
  auto indexEntry = DataLayoutEntryAttr::get(IndexType::get(&context),
                                             b.getI32IntegerAttr(32));
  auto spec = DataLayoutSpecAttr::get(&context, {indexEntry});
  (*module)->setAttr(DLTIDialect::kDataLayoutAttrName, spec);

  // With explicit spec, should return DataLayout regardless of allowDefault
  auto dl1 = getDataLayout(module->getOperation(), /*allowDefault=*/false);
  EXPECT_TRUE(dl1.has_value());

  auto dl2 = getDataLayout(module->getOperation(), /*allowDefault=*/true);
  EXPECT_TRUE(dl2.has_value());
}
