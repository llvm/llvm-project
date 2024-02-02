//===- LoopLikeInterfaceTest.cpp - Unit tests for Loop Like Interfaces. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

struct NoZeroTripCheckLoopOp
    : public Op<NoZeroTripCheckLoopOp, LoopLikeOpInterface::Trait> {
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() {
    return "looptest.no_zero_trip_check_loop_op";
  }

  SmallVector<Region *> getLoopRegions() { return {}; }
};

struct ImplZeroTripCheckLoopOp
    : public Op<ImplZeroTripCheckLoopOp, LoopLikeOpInterface::Trait> {
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() {
    return "looptest.impl_zero_trip_check_loop_op";
  }

  SmallVector<Region *> getLoopRegions() { return {}; }

  FailureOr<LoopLikeOpInterface>
  replaceWithZeroTripCheck(RewriterBase &rewriter) {
    return cast<LoopLikeOpInterface>(this->getOperation());
  }
};

/// A dialect putting all the above together.
struct LoopTestDialect : Dialect {
  explicit LoopTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<LoopTestDialect>()) {
    addOperations<NoZeroTripCheckLoopOp, ImplZeroTripCheckLoopOp>();
  }
  static StringRef getDialectNamespace() { return "looptest"; }
};

TEST(LoopLikeOpInterface, NoReplaceWithZeroTripCheck) {
  const char *ir = R"MLIR(
  "looptest.no_zero_trip_check_loop_op"() : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<LoopTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  LoopLikeOpInterface testOp =
      cast<LoopLikeOpInterface>(module->getBody()->getOperations().front());

  IRRewriter rewriter(&ctx);
  FailureOr<LoopLikeOpInterface> result =
      testOp.replaceWithZeroTripCheck(rewriter);

  EXPECT_TRUE(failed(result));
}

TEST(LoopLikeOpInterface, ImplReplaceWithZeroTripCheck) {
  const char *ir = R"MLIR(
  "looptest.impl_zero_trip_check_loop_op"() : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<LoopTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  LoopLikeOpInterface testOp =
      cast<LoopLikeOpInterface>(module->getBody()->getOperations().front());

  IRRewriter rewriter(&ctx);
  FailureOr<LoopLikeOpInterface> result =
      testOp.replaceWithZeroTripCheck(rewriter);

  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(*result, testOp);
}
