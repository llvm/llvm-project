//===- llvm/unittest/IR/AsmWriter.cpp - AsmWriter tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/FunctionInstructionPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/SourceMgr.h"
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

TEST(AsmWriterTest, PersistentBasicBlockPrint) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    @0 = global i32 0

    declare void @use(ptr)

    define void @f() !dbg !6 {
      call void @use(ptr @0), !annotation !12
        #dbg_value(i32 0, !9, !DIExpression(DW_OP_constu, 4), !11)
      ret void, !annotation !12
    }

    define void @g() {
      ret void
    }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "x", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
    !12 = !{!"annotation"}
  )",
                                                  Err, Ctx);
  ASSERT_NE(M, nullptr);

  std::string First;
  raw_string_ostream FirstOS(First);
  M->getFunction("f")->getEntryBlock().print(FirstOS);

  std::string Second;
  raw_string_ostream SecondOS(Second);
  M->getFunction("f")->getEntryBlock().print(SecondOS);

  EXPECT_EQ(First, Second);
  EXPECT_THAT(First, HasSubstr("call void @use(ptr @0), !annotation !"));
  EXPECT_THAT(First, HasSubstr("#dbg_value(i32 0, !"));
  EXPECT_THAT(First, HasSubstr("!DIExpression(DW_OP_constu, 4)"));
  EXPECT_THAT(First, HasSubstr("ret void, !annotation !"));

  MDNode *Earlier = MDNode::getDistinct(Ctx, MDString::get(Ctx, "earlier"));
  M->getFunction("f")->getEntryBlock().getTerminator()->setMetadata("order",
                                                                    Earlier);
  MDNode *Later = MDNode::getDistinct(Ctx, MDString::get(Ctx, "later"));
  M->getFunction("g")->getEntryBlock().getTerminator()->setMetadata("order",
                                                                    Later);

  std::string LaterOutput;
  raw_string_ostream LaterOS(LaterOutput);
  M->getFunction("g")->getEntryBlock().print(LaterOS);

  std::string EarlierOutput;
  raw_string_ostream EarlierOS(EarlierOutput);
  M->getFunction("f")->getEntryBlock().print(EarlierOS);

  StringRef MetadataPrefix = "!order !";
  size_t EarlierPos = EarlierOutput.find(MetadataPrefix);
  size_t LaterPos = LaterOutput.find(MetadataPrefix);
  ASSERT_NE(EarlierPos, StringRef::npos);
  ASSERT_NE(LaterPos, StringRef::npos);
  StringRef EarlierIDText =
      StringRef(EarlierOutput).drop_front(EarlierPos + MetadataPrefix.size());
  StringRef LaterIDText =
      StringRef(LaterOutput).drop_front(LaterPos + MetadataPrefix.size());
  unsigned EarlierID;
  unsigned LaterID;
  ASSERT_FALSE(EarlierIDText.consumeInteger(10, EarlierID));
  ASSERT_FALSE(LaterIDText.consumeInteger(10, LaterID));
  EXPECT_LT(EarlierID, LaterID);
}

TEST(AsmWriterTest, PersistentPrintTemporaryMetadata) {
  LLVMContext Ctx;
  Module M("test", Ctx);
  Function *F = Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false),
      Function::ExternalLinkage, "f", M);
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  ReturnInst *Ret = ReturnInst::Create(Ctx, BB);

  MDNode *Permanent = MDNode::getDistinct(Ctx, MDString::get(Ctx, "permanent"));
  TempMDNode Temporary = MDNode::getTemporary(Ctx, Permanent);
  Ret->setMetadata("permanent", Permanent);
  Ret->setMetadata("temporary", Temporary.get());

  std::string FunctionText;
  raw_string_ostream FunctionOS(FunctionText);
  F->print(FunctionOS);
  EXPECT_THAT(FunctionText, HasSubstr("!permanent !0"));
  EXPECT_THAT(FunctionText, HasSubstr("!temporary !1"));

  std::string BasicBlockText;
  raw_string_ostream BasicBlockOS(BasicBlockText);
  BB->print(BasicBlockOS);
  EXPECT_THAT(BasicBlockText, HasSubstr("!permanent !0"));
  EXPECT_THAT(BasicBlockText, HasSubstr("!temporary !1"));

  std::string ModuleText;
  raw_string_ostream ModuleOS(ModuleText);
  M.print(ModuleOS, nullptr);
  EXPECT_THAT(ModuleText, HasSubstr("!0 = distinct !{!\"permanent\"}"));
  EXPECT_THAT(ModuleText, HasSubstr("!1 = <temporary!> !{!0}"));

  MDNode *Later = MDNode::getDistinct(Ctx, MDString::get(Ctx, "later"));
  Ret->setMetadata("later", Later);
  std::string LaterText;
  raw_string_ostream LaterOS(LaterText);
  F->print(LaterOS);
  EXPECT_THAT(LaterText, HasSubstr("!later !2"));
  EXPECT_THAT(LaterText, HasSubstr("!temporary !1"));
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

TEST(AsmWriterTest, FunctionInstructionPrinterMatchesInstructionPrint) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
    declare i32 @callee(i32)

    define i32 @test(i32 %arg, ptr %ptr) {
    entry:
      %0 = add nsw i32 %arg, 1, !annotation !0
      %named = call i32 @callee(i32 %0) [ "tag"(i32 %arg) ], !prof !1
      store atomic i32 %named, ptr %ptr release, align 4, !annotation !0
      ret i32 %named
    }

    !0 = !{!"attached metadata"}
    !1 = !{!"branch_weights", i32 10}
  )",
                                                  Err, Ctx);
  ASSERT_TRUE(M);
  Function &F = *M->getFunction("test");
  // Exercise an instruction that exceeds the printer's internal buffer.
  F.front().front().setName(std::string(2048, 'x'));

  for (bool IsForDebug : {false, true}) {
    std::string Expected;
    raw_string_ostream ExpectedOS(Expected);
    ModuleSlotTracker ExpectedMST(M.get());
    ExpectedOS << "before:";
    for (const Instruction &I : instructions(F)) {
      I.print(ExpectedOS, ExpectedMST, IsForDebug);
      ExpectedOS << ":after\n";
    }

    std::string Actual;
    raw_string_ostream ActualOS(Actual);
    ModuleSlotTracker ActualMST(M.get());
    FunctionInstructionPrinter Printer(ActualOS, ActualMST, F, IsForDebug);
    ActualOS << "before:";
    for (const Instruction &I : instructions(F)) {
      Printer.printInstruction(I);
      ActualOS << ":after\n";
    }

    EXPECT_EQ(Expected, Actual);
  }
}
}
