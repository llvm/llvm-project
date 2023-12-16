//===- TypeAttrNamesTest.cpp - Type API unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file test the lookup of AbstractType / AbstractAttribute by name.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/TypeID.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
struct FooType : Type::TypeBase<FooType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "fake.foo";

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FooType)
};

struct BarAttr : Attribute::AttrBase<BarAttr, Attribute, AttributeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "fake.bar";

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BarAttr)
};

struct FakeDialect : Dialect {
  FakeDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<FakeDialect>()) {
    addTypes<FooType>();
    addAttributes<BarAttr>();
  }

  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("fake");
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FakeDialect)
};
} // namespace

TEST(AbstractType, LookupWithString) {
  MLIRContext ctx;
  ctx.loadDialect<FakeDialect>();

  // Check that we can lookup an abstract type by name.
  auto fooAbstractType = AbstractType::lookup("fake.foo", &ctx);
  EXPECT_TRUE(fooAbstractType.has_value());
  EXPECT_TRUE(fooAbstractType->get().getName() == "fake.foo");

  // Check that the abstract type is the same as the one used by the type.
  auto fooType = FooType::get(&ctx);
  EXPECT_TRUE(&fooType.getAbstractType() == &fooAbstractType->get());

  // Check that lookups of non-existing types returns nullopt.
  // Even if an attribute with the same name exists.
  EXPECT_FALSE(AbstractType::lookup("fake.bar", &ctx).has_value());
}

TEST(AbstractAttribute, LookupWithString) {
  MLIRContext ctx;
  ctx.loadDialect<FakeDialect>();

  // Check that we can lookup an abstract type by name.
  auto barAbstractAttr = AbstractAttribute::lookup("fake.bar", &ctx);
  EXPECT_TRUE(barAbstractAttr.has_value());
  EXPECT_TRUE(barAbstractAttr->get().getName() == "fake.bar");

  // Check that the abstract Attribute is the same as the one used by the
  // Attribute.
  auto barAttr = BarAttr::get(&ctx);
  EXPECT_TRUE(&barAttr.getAbstractAttribute() == &barAbstractAttr->get());

  // Check that lookups of non-existing Attributes returns nullopt.
  // Even if an attribute with the same name exists.
  EXPECT_FALSE(AbstractAttribute::lookup("fake.foo", &ctx).has_value());
}
