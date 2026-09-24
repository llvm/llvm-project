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
      StringAttr callee = dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee())
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

TEST(SymbolOpInterface, NativeSymbolTraits) {
  DialectRegistry registry;
  ::test::registerTestDialect(registry);
  MLIRContext context(registry);

  constexpr static StringLiteral kInput = R"MLIR(
    "test.symbol"() <{sym_name = "symbol_name"}> : () -> ()
  )MLIR";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kInput, &context);
  auto symOp = cast<SymbolOpInterface>(module->getBody()->front());

  EXPECT_EQ(symOp.getName(), "symbol_name");
  EXPECT_TRUE(symOp.isPublic());

  symOp.setSymbolName("new_name");
  EXPECT_EQ(symOp.getName(), "new_name");
  EXPECT_EQ(symOp->getInherentAttr("sym_name").value_or(Attribute{}),
            symOp.getNameAttr());

  symOp.setPrivate();
  EXPECT_TRUE(symOp.isPrivate());
  symOp.setNested();
  EXPECT_TRUE(symOp.isNested());
  symOp.setPublic();
  EXPECT_TRUE(symOp.isPublic());
  EXPECT_FALSE(
      symOp->getInherentAttr(SymbolOpInterface::getDefaultVisibilityAttrName())
          .value_or(Attribute{}));
}

TEST_F(ReplaceAllSymbolUsesTest, OperationInModuleOp) {
  // Symbol as `Operation *`, rename within module.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), module);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInModuleOp) {
  // Symbol as `StringAttr`, rename within module.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), module);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, OperationInModuleBody) {
  // Symbol as `Operation *`, rename within module body.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), &module->getRegion(0));
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInModuleBody) {
  // Symbol as `StringAttr`, rename within module body.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), &module->getRegion(0));
  });
}

TEST_F(ReplaceAllSymbolUsesTest, OperationInFuncOp) {
  // Symbol as `Operation *`, rename within function.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        barOp, StringAttr::get(context.get(), "baz"), fooOp);
  });
}

TEST_F(ReplaceAllSymbolUsesTest, StringAttrInFuncOp) {
  // Symbol as `StringAttr`, rename within function.
  testReplaceAllSymbolUses([&](const auto &symbolTable, auto module, auto fooOp,
                               auto barOp) -> LogicalResult {
    return symbolTable.replaceAllSymbolUses(
        StringAttr::get(context.get(), "bar"),
        StringAttr::get(context.get(), "baz"), fooOp);
  });
}

TEST(SymbolOpInterface, Visibility) {
  DialectRegistry registry;
  ::test::registerTestDialect(registry);
  MLIRContext context(registry);

  constexpr static StringLiteral kInput = R"MLIR(
    "test.overridden_symbol_visibility"() {sym_name = "symbol_name"} : () -> ()
  )MLIR";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kInput, &context);
  auto symOp = cast<SymbolOpInterface>(module->getBody()->front());

  ASSERT_TRUE(symOp.isPrivate());
  ASSERT_FALSE(symOp.isPublic());
  ASSERT_FALSE(symOp.isNested());
  ASSERT_TRUE(symOp.canDiscardOnUseEmpty());
  ASSERT_EQ(SymbolTable::getSymbolVisibility(symOp),
            SymbolTable::Visibility::Private);

  std::string diagStr;
  context.getDiagEngine().registerHandler(
      [&](Diagnostic &diag) { diagStr += diag.str(); });

  std::string expectedDiag;
  symOp.setPublic();
  expectedDiag += "'test.overridden_symbol_visibility' op cannot change "
                  "visibility of symbol to public";
  symOp.setNested();
  expectedDiag += "'test.overridden_symbol_visibility' op cannot change "
                  "visibility of symbol to nested";
  symOp.setPrivate();
  expectedDiag += "'test.overridden_symbol_visibility' op cannot change "
                  "visibility of symbol to private";
  SymbolTable::setSymbolVisibility(symOp, SymbolTable::Visibility::Nested);
  expectedDiag += "'test.overridden_symbol_visibility' op cannot change "
                  "visibility of symbol to nested";

  ASSERT_EQ(diagStr, expectedDiag);
  ASSERT_FALSE(
      symOp->hasAttr(SymbolOpInterface::getDefaultVisibilityAttrName()));
}

TEST(SymbolUserMap, AllUsesVisible) {
  DialectRegistry registry;
  ::test::registerTestDialect(registry);
  MLIRContext context(registry);

  constexpr static StringLiteral kInput = R"MLIR(
    module @root {
      module @exposed {
        "test.symbol"() <{sym_name = "public"}> : () -> ()
        "test.symbol"() <{sym_name = "private", sym_visibility = "private"}> : () -> ()
        "test.symbol"() <{sym_name = "nested", sym_visibility = "nested"}> : () -> ()
        "test.symbol"() <{sym_name = "local_user"}> {use = [@nested, @public]} : () -> ()
        module @child attributes {sym_visibility = "nested"} {
          "test.symbol"() <{sym_name = "leaf", sym_visibility = "nested"}> : () -> ()
        }
        module @hidden attributes {sym_visibility = "private"} {
          "test.symbol"() <{sym_name = "leaf", sym_visibility = "nested"}> : () -> ()
          "test.symbol"() <{sym_name = "public"}> : () -> ()
        }
      }
      "test.symbol"() <{sym_name = "outside", sym_visibility = "private"}>
          {use = [@exposed::@nested, @exposed::@public]} : () -> ()
    }
  )MLIR";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kInput, &context);
  ASSERT_TRUE(module);
  SymbolTableCollection tables;
  Operation *exposed = SymbolTable::lookupSymbolIn(*module, "exposed");
  Operation *publicSymbol = SymbolTable::lookupSymbolIn(exposed, "public");
  Operation *privateSymbol = SymbolTable::lookupSymbolIn(exposed, "private");
  Operation *nested = SymbolTable::lookupSymbolIn(exposed, "nested");
  Operation *localUser = SymbolTable::lookupSymbolIn(exposed, "local_user");
  Operation *outside = SymbolTable::lookupSymbolIn(*module, "outside");
  Operation *child = SymbolTable::lookupSymbolIn(exposed, "child");
  Operation *childLeaf = SymbolTable::lookupSymbolIn(child, "leaf");
  Operation *hidden = SymbolTable::lookupSymbolIn(exposed, "hidden");
  Operation *hiddenLeaf = SymbolTable::lookupSymbolIn(hidden, "leaf");
  Operation *hiddenPublic = SymbolTable::lookupSymbolIn(hidden, "public");

  // A detached, named root contains all IR users, including public symbol
  // users.
  SymbolUserMap wholeMap(tables, *module);
  EXPECT_TRUE(wholeMap.areAllUsesVisible(publicSymbol));
  EXPECT_EQ(wholeMap.getUsers(publicSymbol).size(), 2u);
  EXPECT_TRUE(wholeMap.areAllUsesVisible(privateSymbol));
  EXPECT_TRUE(wholeMap.areAllUsesVisible(nested));
  EXPECT_TRUE(wholeMap.areAllUsesVisible(childLeaf));
  EXPECT_TRUE(wholeMap.areAllUsesVisible(hiddenLeaf));
  EXPECT_EQ(wholeMap.getUsers(nested).size(), 2u);

  // A map of an exposed table omits outside users, even with local users.
  SymbolUserMap exposedMap(tables, exposed);
  EXPECT_FALSE(exposedMap.areAllUsesVisible(publicSymbol));
  ASSERT_EQ(exposedMap.getUsers(publicSymbol).size(), 1u);
  EXPECT_EQ(exposedMap.getUsers(publicSymbol).front(), localUser);
  EXPECT_TRUE(exposedMap.areAllUsesVisible(privateSymbol));
  EXPECT_FALSE(exposedMap.areAllUsesVisible(nested));
  ASSERT_EQ(exposedMap.getUsers(nested).size(), 1u);
  EXPECT_EQ(exposedMap.getUsers(nested).front(), localUser);
  EXPECT_FALSE(exposedMap.areAllUsesVisible(childLeaf));
  EXPECT_TRUE(exposedMap.useEmpty(childLeaf));
  EXPECT_TRUE(exposedMap.areAllUsesVisible(hiddenLeaf));
  EXPECT_TRUE(exposedMap.areAllUsesVisible(hiddenPublic));
  EXPECT_FALSE(exposedMap.areAllUsesVisible(outside));
  EXPECT_FALSE(exposedMap.areAllUsesVisible(exposed));

  // A private table hides its nested symbols even when it is the map root.
  SymbolUserMap hiddenMap(tables, hidden);
  EXPECT_FALSE(hiddenMap.areAllUsesVisible(hidden));
  EXPECT_TRUE(hiddenMap.areAllUsesVisible(hiddenLeaf));
  EXPECT_TRUE(hiddenMap.areAllUsesVisible(hiddenPublic));
}

} // namespace
