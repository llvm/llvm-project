//===- QuantifierTest.cpp - SMT quantifier operation unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace smt;

namespace {

//===----------------------------------------------------------------------===//
// Test custom builders of ExistsOp
//===----------------------------------------------------------------------===//

TEST(QuantifierTest, ExistsBuilderWithPattern) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ExistsOp> existsOp = ExistsOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      std::nullopt,
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return boundVars;
      },
      /*weight=*/2);

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  existsOp->print(stream);

  ASSERT_STREQ(
      stream.str().str().c_str(),
      "%0 = smt.exists weight 2 {\n^bb0(%arg0: !smt.bool, "
      "%arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  smt.yield %0 : "
      "!smt.bool\n} patterns {\n^bb0(%arg0: !smt.bool, %arg1: !smt.bool):\n  "
      "smt.yield %arg0, %arg1 : !smt.bool, !smt.bool\n}\n");
}

TEST(QuantifierTest, ExistsBuilderNoPattern) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ExistsOp> existsOp = ExistsOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      ArrayRef<StringRef>{"a", "b"}, nullptr, /*weight=*/0, /*noPattern=*/true);

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  existsOp->print(stream);

  ASSERT_STREQ(stream.str().str().c_str(),
               "%0 = smt.exists [\"a\", \"b\"] no_pattern {\n^bb0(%arg0: "
               "!smt.bool, %arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  "
               "smt.yield %0 : !smt.bool\n}\n");
}

TEST(QuantifierTest, ExistsBuilderDefault) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ExistsOp> existsOp = ExistsOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      ArrayRef<StringRef>{"a", "b"});

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  existsOp->print(stream);

  ASSERT_STREQ(stream.str().str().c_str(),
               "%0 = smt.exists [\"a\", \"b\"] {\n^bb0(%arg0: !smt.bool, "
               "%arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  smt.yield "
               "%0 : !smt.bool\n}\n");
}

//===----------------------------------------------------------------------===//
// Test custom builders of ForallOp
//===----------------------------------------------------------------------===//

TEST(QuantifierTest, ForallBuilderWithPattern) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ForallOp> forallOp = ForallOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      ArrayRef<StringRef>{"a", "b"},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return boundVars;
      },
      /*weight=*/2);

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  forallOp->print(stream);

  ASSERT_STREQ(
      stream.str().str().c_str(),
      "%0 = smt.forall [\"a\", \"b\"] weight 2 {\n^bb0(%arg0: !smt.bool, "
      "%arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  smt.yield %0 : "
      "!smt.bool\n} patterns {\n^bb0(%arg0: !smt.bool, %arg1: !smt.bool):\n  "
      "smt.yield %arg0, %arg1 : !smt.bool, !smt.bool\n}\n");
}

TEST(QuantifierTest, ForallBuilderNoPattern) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ForallOp> forallOp = ForallOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      ArrayRef<StringRef>{"a", "b"}, nullptr, /*weight=*/0, /*noPattern=*/true);

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  forallOp->print(stream);

  ASSERT_STREQ(stream.str().str().c_str(),
               "%0 = smt.forall [\"a\", \"b\"] no_pattern {\n^bb0(%arg0: "
               "!smt.bool, %arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  "
               "smt.yield %0 : !smt.bool\n}\n");
}

TEST(QuantifierTest, ForallBuilderDefault) {
  MLIRContext context;
  context.loadDialect<SMTDialect>();
  Location loc(UnknownLoc::get(&context));

  OpBuilder builder(&context);
  auto boolTy = BoolType::get(&context);

  OwningOpRef<ForallOp> forallOp = ForallOp::create(
      builder, loc, TypeRange{boolTy, boolTy},
      [](OpBuilder &builder, Location loc, ValueRange boundVars) {
        return AndOp::create(builder, loc, boundVars);
      },
      std::nullopt);

  SmallVector<char, 1024> buffer;
  llvm::raw_svector_ostream stream(buffer);
  forallOp->print(stream);

  ASSERT_STREQ(stream.str().str().c_str(),
               "%0 = smt.forall {\n^bb0(%arg0: !smt.bool, "
               "%arg1: !smt.bool):\n  %0 = smt.and %arg0, %arg1\n  smt.yield "
               "%0 : !smt.bool\n}\n");
}

} // namespace
