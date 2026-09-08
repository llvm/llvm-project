//===- LLVMToolSessionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LLVMDriver.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

std::unique_ptr<LLVMToolSession> Session;
unsigned CompilerCalls;
unsigned LinkerCalls;

int linkerMain(int Argc, char **Argv, const ToolContext &Context) {
  ++LinkerCalls;
  EXPECT_EQ(Argc, 3);
  EXPECT_STREQ(Argv[0], "wasm-ld");
  EXPECT_TRUE(Context.getCallableTool("clang"));
  return 0;
}

int compilerMain(int Argc, char **Argv, const ToolContext &Context) {
  ++CompilerCalls;
  EXPECT_EQ(Argc, 2);
  EXPECT_STREQ(Argv[0], "clang");
  const char *LinkArgs[] = {"wasm-ld", Argv[1], "out.wasm"};
  return Context.callTool(LinkArgs);
}

TEST(LLVMToolSessionTest, SupportsSequentialNestedToolCalls) {
  const char *First[] = {"clang", "first.cpp"};
  const char *Second[] = {"clang", "second.cpp"};

  EXPECT_EQ(Session->callTool(First), 0);
  EXPECT_EQ(Session->callTool(Second), 0);
  EXPECT_EQ(CompilerCalls, 2u);
  EXPECT_EQ(LinkerCalls, 2u);
}

TEST(LLVMToolSessionTest, ReportsUnknownTools) {
  const char *Args[] = {"not-an-llvm-tool"};
  EXPECT_EQ(Session->callTool(Args), -1);
}

} // namespace

int main(int Argc, char **Argv) {
  const CallableTool Tools[] = {
      {"clang", compilerMain},
      {"wasm-ld", linkerMain},
  };
  Session = std::make_unique<LLVMToolSession>(Argc, Argv, Tools);
  testing::InitGoogleTest(&Argc, Argv);
  int Result = RUN_ALL_TESTS();
  Session.reset();
  return Result;
}
