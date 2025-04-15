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
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using ::testing::HasSubstr;

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
  EXPECT_THAT(S, HasSubstr("<badref> = add i32 poison, poison, !<empty"));
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
  EXPECT_EQ("!DIExpression(DW_OP_constu, 4, DW_OP_minus, DW_OP_deref)", S);
}

TEST(AsmWriterTest, PrintAddrspaceWithNullOperand) {
  LLVMContext Ctx;
  Module M("test module", Ctx);
  SmallVector<Type *, 3> FArgTypes;
  FArgTypes.push_back(Type::getInt64Ty(Ctx));
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), FArgTypes, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "", &M);
  Argument *Arg0 = F->getArg(0);
  Value *Args[] = {Arg0};
  std::unique_ptr<CallInst> Call(CallInst::Create(F, Args));
  // This will make Call's operand null.
  Call->dropAllReferences();

  std::string S;
  raw_string_ostream OS(S);
  Call->print(OS);
  EXPECT_THAT(S, HasSubstr("<cannot get addrspace!>"));
}

TEST(AsmWriterTest, PrintNullOperandBundle) {
  LLVMContext C;
  Type *Int32Ty = Type::getInt32Ty(C);
  FunctionType *FnTy = FunctionType::get(Int32Ty, Int32Ty, /*isVarArg=*/false);
  Value *Callee = Constant::getNullValue(PointerType::getUnqual(C));
  Value *Args[] = {ConstantInt::get(Int32Ty, 42)};
  std::unique_ptr<BasicBlock> NormalDest(BasicBlock::Create(C));
  std::unique_ptr<BasicBlock> UnwindDest(BasicBlock::Create(C));
  OperandBundleDef Bundle("bundle", UndefValue::get(Int32Ty));
  std::unique_ptr<InvokeInst> Invoke(
      InvokeInst::Create(FnTy, Callee, NormalDest.get(), UnwindDest.get(), Args,
                         Bundle, "result"));
  // Makes the operand bundle null.
  Invoke->dropAllReferences();
  Invoke->setNormalDest(NormalDest.get());
  Invoke->setUnwindDest(UnwindDest.get());

  std::string S;
  raw_string_ostream OS(S);
  Invoke->print(OS);
  EXPECT_THAT(S, HasSubstr("<null operand bundle!>"));
}
}
