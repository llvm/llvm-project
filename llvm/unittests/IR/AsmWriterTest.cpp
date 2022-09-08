//===- llvm/unittest/IR/AsmWriter.cpp - AsmWriter tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(AsmWriterTest, DebugPrintDetachedInstruction) {

  // PR24852: Ensure that an instruction can be printed even when it
  // has metadata attached but no parent.
  LLVMContext Ctx;
  auto Ty = Type::getInt32Ty(Ctx);
  auto Poison = PoisonValue::get(Ty);
  std::unique_ptr<BinaryOperator> Add(BinaryOperator::CreateAdd(Poison, Poison));
  Add->setMetadata(
      "", MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(Ty, 1))}));
  std::string S;
  raw_string_ostream OS(S);
  Add->print(OS);
  std::size_t r = OS.str().find("<badref> = add i32 poison, poison, !<empty");
  EXPECT_TRUE(r != std::string::npos);
}

TEST(AsmWriterTest, DebugPrintDetachedArgument) {
  LLVMContext Ctx;
  auto Ty = Type::getInt32Ty(Ctx);
  auto Arg = new Argument(Ty);

  std::string S;
  raw_string_ostream OS(S);
  Arg->print(OS);
  EXPECT_EQ(S, "i32 <badref>");
  delete Arg;
}

TEST(AsmWriterTest, DumpDIExpression) {
  LLVMContext Ctx;
  uint64_t Ops[] = {
    dwarf::DW_OP_constu, 4,
    dwarf::DW_OP_minus,
    dwarf::DW_OP_deref,
  };
  DIExpression *Expr = DIExpression::get(Ctx, Ops);
  std::string S;
  raw_string_ostream OS(S);
  Expr->print(OS);
  EXPECT_EQ("!DIExpression(DW_OP_constu, 4, DW_OP_minus, DW_OP_deref)",
            OS.str());
}

}
