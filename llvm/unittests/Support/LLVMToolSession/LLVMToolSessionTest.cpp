//===- LLVMToolSessionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

std::unique_ptr<LLVMToolSession> Session;
unsigned CompilerCalls;
unsigned LinkerCalls;
unsigned WrapperCalls;

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
  ErrorOr<int> Result = Context.callTool(LinkArgs);
  EXPECT_TRUE(Result);
  return Result ? *Result : 1;
}

int clangWrapperMain(int Argc, char **Argv, const ToolContext &Context) {
  ++WrapperCalls;
  EXPECT_EQ(Argc, 1);
  EXPECT_STREQ(Argv[0], "/tmp/clang-wrapper.exe");
  EXPECT_STREQ(Context.PrependArg, "clang-wrapper");
  return 0;
}

int resetOptionsMain(int, char **, const ToolContext &) {
  cl::ResetAllOptionOccurrences();
  return 0;
}

TEST(LLVMToolSessionTest, SupportsSequentialNestedToolCalls) {
  unsigned CompilerCallsBefore = CompilerCalls;
  unsigned LinkerCallsBefore = LinkerCalls;
  const char *First[] = {"clang", "first.cpp"};
  const char *Second[] = {"clang", "second.cpp"};

  ErrorOr<int> FirstResult = Session->callTool(First);
  ASSERT_TRUE(FirstResult);
  EXPECT_EQ(*FirstResult, 0);
  ErrorOr<int> SecondResult = Session->callTool(Second);
  ASSERT_TRUE(SecondResult);
  EXPECT_EQ(*SecondResult, 0);
  EXPECT_EQ(CompilerCalls, CompilerCallsBefore + 2);
  EXPECT_EQ(LinkerCalls, LinkerCallsBefore + 2);
}

TEST(LLVMToolSessionTest, ReportsUnknownTools) {
  const char *Args[] = {"not-a-tool"};
  ErrorOr<int> Result = Session->callTool(Args);
  EXPECT_FALSE(Result);
  EXPECT_EQ(Result.getError(),
            make_error_code(std::errc::no_such_file_or_directory));
}

TEST(LLVMToolSessionTest, SupportsStatefulCallableTools) {
  const char *Args[] = {"stateful-tool"};
  ErrorOr<int> Result = Session->callTool(Args);
  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, 42);
}

TEST(LLVMToolSessionTest, PrefersExactToolNameBeforeFuzzyMatch) {
  const char *Args[] = {"/tmp/clang-wrapper.exe"};
  ErrorOr<int> Result = Session->callTool(Args);
  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, 0);
  EXPECT_EQ(WrapperCalls, 1u);
}

TEST(LLVMToolSessionTest, SurvivesCommandLineOptionReset) {
  const char *ResetArgs[] = {"reset-options"};
  ErrorOr<int> ResetResult = Session->callTool(ResetArgs);
  ASSERT_TRUE(ResetResult);
  EXPECT_EQ(*ResetResult, 0);

  unsigned CompilerCallsBefore = CompilerCalls;
  const char *CompilerArgs[] = {"clang", "after-reset.cpp"};
  ErrorOr<int> CompilerResult = Session->callTool(CompilerArgs);
  ASSERT_TRUE(CompilerResult);
  EXPECT_EQ(*CompilerResult, 0);
  EXPECT_EQ(CompilerCalls, CompilerCallsBefore + 1);
}

} // namespace

int main(int Argc, char **Argv) {
  int StatefulResult = 42;
  const CallableTool Tools[] = {
      {"clang", compilerMain},
      {"clang-wrapper", clangWrapperMain},
      {"reset-options", resetOptionsMain},
      {"stateful-tool",
       [&StatefulResult](int, char **, const ToolContext &) {
         return StatefulResult;
       }},
      {"wasm-ld", linkerMain},
  };
  Session = std::make_unique<LLVMToolSession>(Argc, Argv, Tools);
  testing::InitGoogleTest(&Argc, Argv);
  int Result = RUN_ALL_TESTS();
  Session.reset();
  return Result;
}
