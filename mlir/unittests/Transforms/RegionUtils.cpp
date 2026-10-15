//===- RegionUtils.cpp - Region utility unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class TestBranchOp
    : public Op<TestBranchOp, OpTrait::VariadicOperands, OpTrait::ZeroResults,
                OpTrait::OneSuccessor, OpTrait::IsTerminator,
                BranchOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBranchOp)

  using Op::Op;

  static StringRef getOperationName() { return "test.br"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  SuccessorOperands getSuccessorOperands(unsigned index) {
    assert(index == 0 && "invalid successor index");
    return SuccessorOperands(MutableOperandRange(getOperation()));
  }
};

class TestDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialect)

  static StringRef getDialectNamespace() { return "test"; }

  explicit TestDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<TestDialect>()) {
    addOperations<TestBranchOp>();
    allowUnknownOperations();
  }
};

TEST(RegionUtilsTest, ReportsDeadBlockArgumentRemoval) {
  MLIRContext context;
  context.getOrLoadDialect<TestDialect>();

  auto module = parseSourceString<ModuleOp>(R"mlir(
    "test.container"() ({
    ^bb0(%arg0: i32):
      "test.br"(%arg0)[^bb1] : (i32) -> ()
    ^bb1(%unused: i32):
      "test.br"(%unused)[^bb1] : (i32) -> ()
    }) : () -> ()
  )mlir",
                                            &context);
  ASSERT_TRUE(module);

  Operation *container = &module->getBody()->front();
  Region &region = container->getRegion(0);
  ASSERT_EQ(region.getBlocks().size(), 2u);
  ASSERT_EQ(region.back().getNumArguments(), 1u);

  IRRewriter rewriter(&context);
  EXPECT_TRUE(succeeded(runRegionDCE(rewriter, container->getRegions())));
  EXPECT_EQ(region.back().getNumArguments(), 0u);
  EXPECT_EQ(region.front().getTerminator()->getNumOperands(), 0u);
  EXPECT_EQ(region.back().getTerminator()->getNumOperands(), 0u);
}
} // namespace
