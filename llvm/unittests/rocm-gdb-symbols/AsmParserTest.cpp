//===- llvm/unittest/rocm-dgb-symbols/AsmParserTest.cpp - AsmParser tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Local.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DIExprAsmParserTest : public testing::Test {
protected:
  LLVMContext Context;
  Type *Int64Ty = Type::getInt64Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int16Ty = Type::getInt16Ty(Context);
  Type *Int8Ty = Type::getInt8Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  std::unique_ptr<Module> M;
  const DIExpr *Expr;

  void parseNamedDIExpr(const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, Context);
    if (!M)
      GTEST_SKIP();
    bool BrokenDebugInfo = false;
    bool HardError = verifyModule(*M, &errs(), &BrokenDebugInfo);
    if (HardError || BrokenDebugInfo)
      GTEST_SKIP();
    const NamedMDNode *N = M->getNamedMetadata("named");
    if (!N || N->getNumOperands() != 1u || !isa<const DIExpr>(N->getOperand(0)))
      GTEST_SKIP();
    Expr = cast<const DIExpr>(N->getOperand(0));
  }
};

TEST_F(DIExprAsmParserTest, Empty) {
  parseNamedDIExpr(R"(!named = !{!DIExpr()})");
  DIExprBuilder Builder = Expr->builder();
  ASSERT_EQ(std::distance(Builder.begin(), Builder.end()), 0u);
}

TEST_F(DIExprAsmParserTest, Referrer) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpReferrer(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Referrer(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, Arg) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpArg(3, float))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Arg(3, FloatTy)}));
}

TEST_F(DIExprAsmParserTest, TypeObject) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpTypeObject(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::TypeObject(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, Constant) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpConstant(float 2.0))})");
  DIExprBuilder Builder = Expr->builder();
  ASSERT_EQ(SmallVector<DIOp::Variant>(Builder.range()),
            SmallVector<DIOp::Variant>(
                {DIOp::Constant(ConstantFP::get(Context, APFloat(2.0f)))}));
}

TEST_F(DIExprAsmParserTest, Reinterpret) {
  parseNamedDIExpr(
      R"(!named = !{!DIExpr(DIOpReinterpret(i32 addrspace(5)*))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>(
                {DIOp::Reinterpret(PointerType::get(Context, 5))}));
}

TEST_F(DIExprAsmParserTest, BitOffset) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpBitOffset(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::BitOffset(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, ByteOffset) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpByteOffset(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::ByteOffset(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, Composite) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpComposite(2, i8))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Composite(2, Int8Ty)}));
}

TEST_F(DIExprAsmParserTest, Extend) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpExtend(2))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Extend(2)}));
}

TEST_F(DIExprAsmParserTest, Select) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpSelect())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Select()}));
}

TEST_F(DIExprAsmParserTest, AddrOf) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpAddrOf(7))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::AddrOf(7)}));
}

TEST_F(DIExprAsmParserTest, Deref) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpDeref(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Deref(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, Read) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpRead())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Read()}));
}

TEST_F(DIExprAsmParserTest, Add) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpAdd())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Add()}));
}

TEST_F(DIExprAsmParserTest, Sub) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpSub())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Sub()}));
}

TEST_F(DIExprAsmParserTest, Mul) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpMul())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Mul()}));
}

TEST_F(DIExprAsmParserTest, Div) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpDiv())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Div()}));
}

TEST_F(DIExprAsmParserTest, Shr) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpShr())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Shr()}));
}

TEST_F(DIExprAsmParserTest, Shl) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpShl())})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::Shl()}));
}

TEST_F(DIExprAsmParserTest, PushLane) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(DIOpPushLane(i32))})");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>({DIOp::PushLane(Int32Ty)}));
}

TEST_F(DIExprAsmParserTest, MultipleOps) {
  parseNamedDIExpr(R"(!named = !{!DIExpr(
    DIOpArg(0, i8),
    DIOpArg(1, i8),
    DIOpAdd(),
    DIOpArg(2, i8),
    DIOpComposite(2, i16),
    DIOpReinterpret(i8 addrspace(1)*)
  )}
)");
  ASSERT_EQ(SmallVector<DIOp::Variant>(Expr->builder().range()),
            SmallVector<DIOp::Variant>(
                {DIOp::Arg(0, Int8Ty), DIOp::Arg(1, Int8Ty), DIOp::Add(),
                 DIOp::Arg(2, Int8Ty), DIOp::Composite(2, Int16Ty),
                 DIOp::Reinterpret(PointerType::get(Int8Ty, 1))}));
}

} // end namespace
