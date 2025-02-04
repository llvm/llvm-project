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
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

using testing::Contains;
using testing::Key;
using testing::Pair;
using testing::SizeIs;

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
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
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

TEST(StructuralHashTest, InstructionSubType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %a = load i32, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %a = load i64, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, InstructionType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %1 = load i32, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define void @f(ptr %p) {\n"
                                            "  %1 = load float, ptr %p\n"
                                            "  ret void\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
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
  // clang-format on
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
}

TEST(StructuralHashTest, ComparisonInstructionPredicate) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i1 @f(i64 %a, i64 %b) {\n"
                                            "  %1 = icmp eq i64 %a, %b\n"
                                            "  ret i1 %1\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i1 @f(i64 %a, i64 %b) {\n"
                                            "  %1 = icmp ne i64 %a, %b\n"
                                            " ret i1 %1\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, IntrinsicInstruction) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 =
      parseIR(Ctx, "define float @f(float %a) {\n"
                   "  %b = call float @llvm.sin.f32(float %a)\n"
                   "  ret float %b\n"
                   "}\n"
                   "declare float @llvm.sin.f32(float)\n");
  std::unique_ptr<Module> M2 =
      parseIR(Ctx, "define float @f(float %a) {\n"
                   "  %b = call float @llvm.cos.f32(float %a)\n"
                   "  ret float %b\n"
                   "}\n"
                   "declare float @llvm.cos.f32(float)\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, CallInstruction) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i64 @f(i64 %a) {\n"
                                            "  %b = call i64 @f1(i64 %a)\n"
                                            "  ret i64 %b\n"
                                            "}\n"
                                            "declare i64 @f1(i64)");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i64 @f(i64 %a) {\n"
                                            "  %b = call i64 @f2(i64 %a)\n"
                                            "  ret i64 %b\n"
                                            "}\n"
                                            "declare i64 @f2(i64)");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, ConstantInteger) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i64 @f1() {\n"
                                            "  ret i64 1\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i64 @f2() {\n"
                                            "  ret i64 2\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, BigConstantInteger) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i128 @f1() {\n"
                                            "  ret i128 18446744073709551616\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i128 @f2() {\n"
                                            "  ret i128 18446744073709551617\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, ArgumentNumber) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i64 @f1(i64 %a, i64 %b) {\n"
                                            "  ret i64 %a\n"
                                            "}\n");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i64 @f2(i64 %a, i64 %b) {\n"
                                            "  ret i64 %b\n"
                                            "}\n");
  EXPECT_EQ(StructuralHash(*M1), StructuralHash(*M2));
  EXPECT_NE(StructuralHash(*M1, true), StructuralHash(*M2, true));
}

TEST(StructuralHashTest, Differences) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M1 = parseIR(Ctx, "define i64 @f(i64 %a) {\n"
                                            "  %c = add i64 %a, 1\n"
                                            "  %b = call i64 @f1(i64 %c)\n"
                                            "  ret i64 %b\n"
                                            "}\n"
                                            "declare i64 @f1(i64)");
  auto *F1 = M1->getFunction("f");
  std::unique_ptr<Module> M2 = parseIR(Ctx, "define i64 @g(i64 %a) {\n"
                                            "  %c = add i64 %a, 1\n"
                                            "  %b = call i64 @f2(i64 %c)\n"
                                            "  ret i64 %b\n"
                                            "}\n"
                                            "declare i64 @f2(i64)");
  auto *F2 = M2->getFunction("g");

  // They are originally different when not ignoring any operand.
  EXPECT_NE(StructuralHash(*F1, true), StructuralHash(*F2, true));
  EXPECT_NE(StructuralHashWithDifferences(*F1, nullptr).FunctionHash,
            StructuralHashWithDifferences(*F2, nullptr).FunctionHash);

  // When we ignore the call target f1 vs f2, they have the same hash.
  auto IgnoreOp = [&](const Instruction *I, unsigned OpndIdx) {
    return I->getOpcode() == Instruction::Call && OpndIdx == 1;
  };
  auto FuncHashInfo1 = StructuralHashWithDifferences(*F1, IgnoreOp);
  auto FuncHashInfo2 = StructuralHashWithDifferences(*F2, IgnoreOp);
  EXPECT_EQ(FuncHashInfo1.FunctionHash, FuncHashInfo2.FunctionHash);

  // There are total 3 instructions.
  EXPECT_THAT(*FuncHashInfo1.IndexInstruction, SizeIs(3));
  EXPECT_THAT(*FuncHashInfo2.IndexInstruction, SizeIs(3));

  // The only 1 operand (the call target) has been ignored.
  EXPECT_THAT(*FuncHashInfo1.IndexOperandHashMap, SizeIs(1u));
  EXPECT_THAT(*FuncHashInfo2.IndexOperandHashMap, SizeIs(1u));

  // The index pair of instruction and operand (1, 1) is a key in the map.
  ASSERT_THAT(*FuncHashInfo1.IndexOperandHashMap, Contains(Key(Pair(1, 1))));
  ASSERT_THAT(*FuncHashInfo2.IndexOperandHashMap, Contains(Key(Pair(1, 1))));

  // The indexed instruciton must be the call instruction as shown in the
  // IgnoreOp above.
  EXPECT_EQ(FuncHashInfo1.IndexInstruction->lookup(1)->getOpcode(),
            Instruction::Call);
  EXPECT_EQ(FuncHashInfo2.IndexInstruction->lookup(1)->getOpcode(),
            Instruction::Call);

  // The ignored operand hashes (for f1 vs. f2) are different.
  EXPECT_NE(FuncHashInfo1.IndexOperandHashMap->lookup({1, 1}),
            FuncHashInfo2.IndexOperandHashMap->lookup({1, 1}));
}

} // end anonymous namespace
