//===- OpImplementationTest.cpp - Operation implementation tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
static unsigned countSubstr(StringRef str, StringRef sub) {
  unsigned count = 0;
  size_t pos = 0;
  while ((pos = str.find(sub, pos)) != StringRef::npos) {
    ++count;
    pos += sub.size();
  }
  return count;
}

TEST(AlwaysUseAliasesTest, SharedStateDoesNotLeakAliases) {
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  const char *src = R"mlir(
    builtin.module {
      func.func @f(%arg0: tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>) -> tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32> {
        return %arg0 : tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
      }
      func.func @g(%arg0: tuple<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>) -> tuple<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32> {
        return %arg0 : tuple<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>
      }
    }
  )mlir";
  ParserConfig config(&context);
  OwningOpRef<Operation *> module = parseSourceString(src, config);
  ASSERT_TRUE(module);

  SmallVector<func::FuncOp> funcs;
  module->walk([&](func::FuncOp func) { funcs.push_back(func); });
  ASSERT_EQ(funcs.size(), 2u);

  // Print both with one shared state: each print must define exactly its
  // own alias, with nothing leaking across.
  OpPrintingFlags flags;
  flags.alwaysUseAliases();
  AsmState state(module.get(), flags);
  std::string first, second;
  {
    llvm::raw_string_ostream os(first);
    funcs[0]->print(os, state);
  }
  {
    llvm::raw_string_ostream os(second);
    funcs[1]->print(os, state);
  }
  EXPECT_EQ(countSubstr(first, "!tuple ="), 1u);
  EXPECT_NE(first.find("tuple<i32,"), std::string::npos);
  EXPECT_EQ(first.find("f32"), std::string::npos);
  EXPECT_EQ(countSubstr(second, "!tuple ="), 1u);
  EXPECT_NE(second.find("tuple<f32,"), std::string::npos);
  EXPECT_EQ(second.find("i32"), std::string::npos);
}
TEST(AsmDialectResourceHandleTest, DefaultHandle) {
  AsmDialectResourceHandle handle;
  EXPECT_EQ(handle.getResource(), nullptr);
  EXPECT_EQ(handle.getTypeID(), TypeID::get<void>());
  EXPECT_EQ(handle.getDialect(), nullptr);
}
} // namespace
