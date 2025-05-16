//===- Offload.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/Offload.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "gmock/gmock.h"

using namespace llvm;

TEST(MLIRTarget, OffloadAPI) {
  using OffloadEntryArray = mlir::LLVM::OffloadHandler::OffloadEntryArray;
  LLVMContext llvmContext;
  Module llvmModule("offload", llvmContext);
  mlir::LLVM::OffloadHandler handler(llvmModule);
  StringRef suffix = ".mlir";
  // Check there's no entry array with `.mlir` suffix.
  OffloadEntryArray entryArray = handler.getEntryArray(suffix);
  EXPECT_EQ(entryArray, OffloadEntryArray());
  // Emit the entry array.
  handler.emitEmptyEntryArray(suffix);
  // Check there's an entry array with `.mlir` suffix.
  entryArray = handler.getEntryArray(suffix);
  ASSERT_NE(entryArray.first, nullptr);
  ASSERT_NE(entryArray.second, nullptr);
  // Check the array contains no entries.
  auto *zeroInitializer = dyn_cast_or_null<ConstantAggregateZero>(
      entryArray.first->getInitializer());
  ASSERT_NE(zeroInitializer, nullptr);
  // Insert an empty entries.
  auto emptyEntry =
      ConstantAggregateZero::get(offloading::getEntryTy(llvmModule));
  ASSERT_TRUE(succeeded(handler.insertOffloadEntry(suffix, emptyEntry)));
  // Check there's an entry in the entry array with `.mlir` suffix.
  entryArray = handler.getEntryArray(suffix);
  ASSERT_NE(entryArray.first, nullptr);
  Constant *arrayInitializer = entryArray.first->getInitializer();
  ASSERT_NE(arrayInitializer, nullptr);
  auto *arrayTy = dyn_cast_or_null<ArrayType>(arrayInitializer->getType());
  ASSERT_NE(arrayTy, nullptr);
  EXPECT_EQ(arrayTy->getNumElements(), 1u);
}
