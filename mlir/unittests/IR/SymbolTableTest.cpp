//===- SymbolTableTest.cpp - SymbolTable unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

class ReplaceAllSymbolUsesTest : public ::testing::Test {
protected:
  using ReplaceFnType = llvm::function_ref<LogicalResult(
      SymbolTable, ModuleOp, Operation *, Operation *)>;

  void SetUp() override {
    ::test::registerTestDialect(registry);
    context = std::make_unique<MLIRContext>(registry);
  }

  void testReplaceAllSymbolUses(ReplaceFnType replaceFn) {
    // Set up IR and find func ops.
    OwningOpRef<ModuleOp> module =
        parseSourceString<ModuleOp>(kInput, context.get());
    SymbolTable symbolTable(module.get());
    auto opIterator = module->getBody(0)->getOperations().begin();
    auto fooOp = cast<FunctionOpInterface>(opIterator++);
    auto barOp = cast<FunctionOpInterface>(opIterator++);
    ASSERT_EQ(fooOp.getNameAttr(), "foo");
    ASSERT_EQ(barOp.getNameAttr(), "bar");

    // Call test function that does symbol replacement.
    LogicalResult res = replaceFn(symbolTable, module.get(), fooOp, barOp);
    ASSERT_TRUE(succeeded(res));
    ASSERT_TRUE(succeeded(verify(module.get())));

    // Check that it got renamed.
    bool calleeFound = false;
    fooOp->walk([&](CallOpInterface callOp) {
      StringAttr callee = callOp.getCallableForCallee()
                              .dyn_cast<SymbolRefAttr>()
                              .getLeafReference();
      EXPECT_EQ(callee, "baz");
      calleeFound = true;
    });
    EXPECT_TRUE(calleeFound);
  }

  std::unique_ptr<MLIRContext> context;

private:
  constexpr static llvm::StringLiteral kInput = R"MLIR(
      module {
        test.conversion_func_op private @foo() {
          "test.conversion_call_op"() { callee=@bar } : () -> ()
          "test.return"() : () -> ()
        }
        test.conversion_func_op private @bar()
      }
    )MLIR";

  DialectRegistry registry;
};

namespace {

TEST_F(ReplaceAllSymbolUsesTest, OperationInModuleOp) {
  // Symbol as `Operation *`, rename within module.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), module);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInModuleOp) {
  // Symbol as `StringAttr`, rename within module.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), module);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, OperationInModuleBody) {
  // Symbol as `Operation *`, rename within module body.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), &module->getRegion(0));
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInModuleBody) {
  // Symbol as `StringAttr`, rename within module body.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), &module->getRegion(0));
  });
}

TEST_F(ReplaceAllSymbolUsesTest, OperationInFuncOp) {
  // Symbol as `Operation *`, rename within function.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), fooOp);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInFuncOp) {
  // Symbol as `StringAttr`, rename within function.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), fooOp);
  });
}

} // namespace
