//===- TypeTest.cpp - Type API unit tests ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;

using testing::HasSubstr;

/// Mock implementations of a Type hierarchy
struct LeafType;

struct MiddleType : Type::TypeBase<MiddleType, Type, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "test.middle";

  static bool classof(Type ty) {
    return ty.getTypeID() == TypeID::get<LeafType>() || Base::classof(ty);
  }
};

struct LeafType : Type::TypeBase<LeafType, MiddleType, TypeStorage> {
  using Base::Base;

  static constexpr StringLiteral name = "test.leaf";
};

struct FakeDialect : Dialect {
  FakeDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<FakeDialect>()) {
    addTypes<MiddleType, LeafType>();
  }
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("fake");
  }
};

TEST(Type, Casting) {
  MLIRContext ctx;
  ctx.loadDialect<FakeDialect>();

  Type intTy = IntegerType::get(&ctx, 8);
  Type nullTy;
  MiddleType middleTy = MiddleType::get(&ctx);
  MiddleType leafTy = LeafType::get(&ctx);
  Type leaf2Ty = LeafType::get(&ctx);

  EXPECT_TRUE(isa<IntegerType>(intTy));
  EXPECT_FALSE(isa<FunctionType>(intTy));
  EXPECT_FALSE(isa_and_present<IntegerType>(nullTy));
  EXPECT_TRUE(isa<MiddleType>(middleTy));
  EXPECT_FALSE(isa<LeafType>(middleTy));
  EXPECT_TRUE(isa<MiddleType>(leafTy));
  EXPECT_TRUE(isa<LeafType>(leaf2Ty));
  EXPECT_TRUE(isa<LeafType>(leafTy));

  EXPECT_TRUE(static_cast<bool>(dyn_cast<IntegerType>(intTy)));
  EXPECT_FALSE(static_cast<bool>(dyn_cast<FunctionType>(intTy)));
  EXPECT_FALSE(static_cast<bool>(cast_if_present<FunctionType>(nullTy)));
  EXPECT_FALSE(static_cast<bool>(dyn_cast_if_present<IntegerType>(nullTy)));

  EXPECT_EQ(8u, cast<IntegerType>(intTy).getWidth());
}

TEST(Type, UserAlias) {
  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  Type intTy = IntegerType::get(&ctx, 8);
  AsmState state(&ctx);
  state.addAlias(intTy, "test.alias");
  Operation *op = Operation::create(
      UnknownLoc::get(&ctx), OperationName("test.op", &ctx), TypeRange(intTy),
      ValueRange(), NamedAttrList(), OpaqueProperties(nullptr), BlockRange(),
      /*numRegions=*/0);

  std::string str;
  llvm::raw_string_ostream os(str);
  op->print(os, state);
  EXPECT_THAT(str, HasSubstr("!test.alias = i8"));
  EXPECT_THAT(str, HasSubstr("\"test.op\"() : () -> !test.alias\n"));
  op->erase();
}
