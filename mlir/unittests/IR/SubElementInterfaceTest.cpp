//===- SubElementInterfaceTest.cpp - SubElementInterface unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "gtest/gtest.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::detail;

namespace {
TEST(SubElementInterfaceTest, Nested) {
  MLIRContext context;
  Builder builder(&context);

  BoolAttr trueAttr = builder.getBoolAttr(true);
  BoolAttr falseAttr = builder.getBoolAttr(false);
  ArrayAttr boolArrayAttr = builder.getArrayAttr({trueAttr, falseAttr});
  StringAttr strAttr = builder.getStringAttr("array");
  DictionaryAttr dictAttr =
      builder.getDictionaryAttr(builder.getNamedAttr(strAttr, boolArrayAttr));

  SmallVector<Attribute> subAttrs;
  dictAttr.walkSubAttrs([&](Attribute attr) { subAttrs.push_back(attr); });
  EXPECT_EQ(llvm::ArrayRef(subAttrs),
            ArrayRef<Attribute>({strAttr, trueAttr, falseAttr, boolArrayAttr}));
}

} // namespace
