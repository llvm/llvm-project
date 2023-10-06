//===- SymbolTableTest.cpp - SymbolTable unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/SymbolTable.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(SymbolTableTest, ReplaceAllSymbolUses) {
  MLIRContext context;
  context.getOrLoadDialect<test::TestDialect>();

  auto testReplaceAllSymbolUses = [&](auto replaceFn) {
    const static llvm::StringLiteral input = R"MLIR(
      module {
        test.conversion_func_op private @foo() {
          "test.conversion_call_op"() { callee=@bar } : () -> ()
          "test.return"() : () -> ()
        }
        test.conversion_func_op private @bar()
      }
    )MLIR";

    // Set up IR and find func ops.
    OwningOpRef<Operation *> module = parseSourceString(input, &context);
    SymbolTable symbolTable(module.get());
    auto ops = module->getRegion(0).getBlocks().front().getOperations().begin();
    auto fooOp = cast<FunctionOpInterface>(ops++);
    auto barOp = cast<FunctionOpInterface>(ops++);
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
  };

  // Symbol as `Operation *`, rename within module.
  testReplaceAllSymbolUses(
      [&](auto symbolTable, auto module, auto fooOp, auto barOp) {
        return symbolTable.replaceAllSymbolUses(
            barOp, StringAttr::get(&context, "baz"), module);
      });

  // Symbol as `StringAttr`, rename within module.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) {
    return symbolTable.replaceAllSymbolUses(StringAttr::get(&context, "bar"),
                                            StringAttr::get(&context, "baz"),
                                            module);
  });

  // Symbol as `Operation *`, rename within module body.
  testReplaceAllSymbolUses(
      [&](auto symbolTable, auto module, auto fooOp, auto barOp) {
        return symbolTable.replaceAllSymbolUses(
            barOp, StringAttr::get(&context, "baz"), &module->getRegion(0));
      });

  // Symbol as `StringAttr`, rename within module body.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) {
    return symbolTable.replaceAllSymbolUses(StringAttr::get(&context, "bar"),
                                            StringAttr::get(&context, "baz"),
                                            &module->getRegion(0));
  });

  // Symbol as `Operation *`, rename within function.
  testReplaceAllSymbolUses(
      [&](auto symbolTable, auto module, auto fooOp, auto barOp) {
        return symbolTable.replaceAllSymbolUses(
            barOp, StringAttr::get(&context, "baz"), fooOp);
      });

  // Symbol as `StringAttr`, rename within function.
  testReplaceAllSymbolUses([&](auto symbolTable, auto module, auto fooOp,
                               auto barOp) {
    return symbolTable.replaceAllSymbolUses(StringAttr::get(&context, "bar"),
                                            StringAttr::get(&context, "baz"),
                                            fooOp);
  });
}

} // namespace
