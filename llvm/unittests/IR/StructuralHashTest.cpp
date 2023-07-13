//===- llvm/unittest/IR/StructuralHashTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructuralHash.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

std::unique_ptr<Module> parseIR(LLVMContext &Context, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Context);
  if (!M)
    Err.print("StructuralHashTest", errs());
  return M;
}

TEST(StructuralHashTest, Empty) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, Basic) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M0 = parseIR(Ctx, "");
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f() { ret void }");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define void @f() { ret void }");
  std::unique_ptr<Module> M3 = parseIR(Ctx, "@g = global i32 2");
  std::unique_ptr<Module> M4 = parseIR(Ctx, "@g = global i32 2");
  EXPECT_NE(StructuralHash(*M0), StructuralHash(*M1));
  EXPECT_NE(StructuralHash(*M0), StructuralHash(*M3));
  EXPECT_NE(StructuralHash(*M1), StructuralHash(*M3));
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_EQ(StructuralHash(*M3), StructuralHash(*M4));
}

TEST(StructuralHashTest, BasicFunction) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, "define void @f() {\n"
                                           "  ret void\n"
                                           "}\n"
                                           "define void @g() {\n"
                                           "  ret void\n"
                                           "}\n"
                                           "define i32 @h(i32 %i) {\n"
                                           "  ret i32 %i\n"
                                           "}\n");
  EXPECT_EQ(StructuralHash(*M->getFunction("f")),
            StructuralHash(*M->getFunction("g")));
  EXPECT_NE(StructuralHash(*M->getFunction("f")),
            StructuralHash(*M->getFunction("h")));
}

TEST(StructuralHashTest, Declaration) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M0 = parseIR(Ctx, "");
  std::unique_ptr<Module> M1 = parseIR(Ctx, "declare void @f()");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "@g = external global i32");
  EXPECT_EQ(StructuralHash(*M0), StructuralHash(*M1));
  EXPECT_EQ(StructuralHash(*M0), StructuralHash(*M2));
}

TEST(StructuralHashTest, GlobalType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "@g = global i32 1");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "@g = global float 1.0");
  EXPECT_NE(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, Function) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f() { ret void }");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define void @f(i32) { ret void }");
  EXPECT_NE(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, FunctionRetType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f() { ret void }");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i32 @f() { ret i32 0 }");
  // FIXME: should be different
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, InstructionOpCode) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %a = load i32, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  std::unique_ptr<Module> M2 =
      parseIR(Ctx, "define void @f(ptr %p) {\n"
                   "  %a = getelementptr i8, ptr %p, i32 1\n"
                   "  ret void\n"
                   "}\n");
  EXPECT_NE(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, InstructionType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %a = load i32, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %a = load i64, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  // FIXME: should be different
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, IgnoredMetadata) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "@a = global i32 1\n");
  // clang-format off
  std::unique_ptr<Module> M2 = parseIR(
      Ctx, R"(
        @a = global i32 1
        @llvm.embedded.object = private constant [4 x i8] c"BC\C0\00", section ".llvm.lto", align 1, !exclude !0
        @llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"

        !llvm.embedded.objects = !{!1}

        !0 = !{}
        !1 = !{ptr @llvm.embedded.object, !".llvm.lto"}
        )");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
}
} // end anonymous namespace
