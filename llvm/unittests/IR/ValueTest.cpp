//===- llvm/unittest/IR/ValueTest.cpp - Value unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Value.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(ValueTest, UsedInBasicBlock) {
  LLVMContext C;

  const char *ModuleString = "define void @f(i32 %x, i32 %y) {\n"
                             "bb0:\n"
                             "  %y1 = add i32 %y, 1\n"
                             "  %y2 = add i32 %y, 1\n"
                             "  %y3 = add i32 %y, 1\n"
                             "  %y4 = add i32 %y, 1\n"
                             "  %y5 = add i32 %y, 1\n"
                             "  %y6 = add i32 %y, 1\n"
                             "  %y7 = add i32 %y, 1\n"
                             "  %y8 = add i32 %x, 1\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);

  Function *F = M->getFunction("f");

  EXPECT_FALSE(F->isUsedInBasicBlock(&F->front()));
  EXPECT_TRUE(std::next(F->arg_begin())->isUsedInBasicBlock(&F->front()));
  EXPECT_TRUE(F->arg_begin()->isUsedInBasicBlock(&F->front()));
}

TEST(GlobalTest, CreateAddressSpace) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("TestModule", Ctx));
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  GlobalVariable *Dummy0
    = new GlobalVariable(*M,
                         Int32Ty,
                         true,
                         GlobalValue::ExternalLinkage,
                         Constant::getAllOnesValue(Int32Ty),
                         "dummy",
                         nullptr,
                         GlobalVariable::NotThreadLocal,
                         1);

  EXPECT_TRUE(Value::MaximumAlignment == 4294967296ULL);
  Dummy0->setAlignment(Align(4294967296ULL));
  EXPECT_EQ(Dummy0->getAlignment(), 4294967296ULL);

  // Make sure the address space isn't dropped when returning this.
  Constant *Dummy1 = M->getOrInsertGlobal("dummy", Int32Ty);
  EXPECT_EQ(Dummy0, Dummy1);
  EXPECT_EQ(1u, Dummy1->getType()->getPointerAddressSpace());


  // This one requires a bitcast, but the address space must also stay the same.
  GlobalVariable *DummyCast0
    = new GlobalVariable(*M,
                         Int32Ty,
                         true,
                         GlobalValue::ExternalLinkage,
                         Constant::getAllOnesValue(Int32Ty),
                         "dummy_cast",
                         nullptr,
                         GlobalVariable::NotThreadLocal,
                         1);

  // Make sure the address space isn't dropped when returning this.
  Constant *DummyCast1 = M->getOrInsertGlobal("dummy_cast", Int8Ty);
  EXPECT_EQ(DummyCast0, DummyCast1);
  EXPECT_EQ(1u, DummyCast1->getType()->getPointerAddressSpace());
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG

TEST(GlobalTest, AlignDeath) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("TestModule", Ctx));
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  GlobalVariable *Var =
      new GlobalVariable(*M, Int32Ty, true, GlobalValue::ExternalLinkage,
                         Constant::getAllOnesValue(Int32Ty), "var", nullptr,
                         GlobalVariable::NotThreadLocal, 1);

  EXPECT_DEATH(Var->setAlignment(Align(8589934592ULL)),
               "Alignment is greater than MaximumAlignment");
}
#endif
#endif

TEST(ValueTest, printSlots) {
  // Check that Value::print() and Value::printAsOperand() work with and
  // without a slot tracker.
  LLVMContext C;

  const char *ModuleString = "@g0 = external global %500\n"
                             "@g1 = external global %900\n"
                             "\n"
                             "%900 = type { i32, i32 }\n"
                             "%500 = type { i32 }\n"
                             "\n"
                             "define void @f(i32 %x, i32 %y) {\n"
                             "entry:\n"
                             "  %0 = add i32 %y, 1\n"
                             "  %1 = add i32 %y, 1\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);

  Function *F = M->getFunction("f");
  ASSERT_TRUE(F);
  ASSERT_FALSE(F->empty());
  BasicBlock &BB = F->getEntryBlock();
  ASSERT_EQ(3u, BB.size());

  Instruction *I0 = &*BB.begin();
  ASSERT_TRUE(I0);
  Instruction *I1 = &*++BB.begin();
  ASSERT_TRUE(I1);

  GlobalVariable *G0 = M->getGlobalVariable("g0");
  ASSERT_TRUE(G0);
  GlobalVariable *G1 = M->getGlobalVariable("g1");
  ASSERT_TRUE(G1);

  ModuleSlotTracker MST(M.get());

#define CHECK_PRINT(INST, STR)                                                 \
  do {                                                                         \
    {                                                                          \
      std::string S;                                                           \
      raw_string_ostream OS(S);                                                \
      INST->print(OS);                                                         \
      EXPECT_EQ(STR, OS.str());                                                \
    }                                                                          \
    {                                                                          \
      std::string S;                                                           \
      raw_string_ostream OS(S);                                                \
      INST->print(OS, MST);                                                    \
      EXPECT_EQ(STR, OS.str());                                                \
    }                                                                          \
  } while (false)
  CHECK_PRINT(I0, "  %0 = add i32 %y, 1");
  CHECK_PRINT(I1, "  %1 = add i32 %y, 1");
#undef CHECK_PRINT

#define CHECK_PRINT_AS_OPERAND(INST, TYPE, STR)                                \
  do {                                                                         \
    {                                                                          \
      std::string S;                                                           \
      raw_string_ostream OS(S);                                                \
      INST->printAsOperand(OS, TYPE);                                          \
      EXPECT_EQ(StringRef(STR), StringRef(OS.str()));                          \
    }                                                                          \
    {                                                                          \
      std::string S;                                                           \
      raw_string_ostream OS(S);                                                \
      INST->printAsOperand(OS, TYPE, MST);                                     \
      EXPECT_EQ(StringRef(STR), StringRef(OS.str()));                          \
    }                                                                          \
  } while (false)
  CHECK_PRINT_AS_OPERAND(I0, false, "%0");
  CHECK_PRINT_AS_OPERAND(I1, false, "%1");
  CHECK_PRINT_AS_OPERAND(I0, true, "i32 %0");
  CHECK_PRINT_AS_OPERAND(I1, true, "i32 %1");
  CHECK_PRINT_AS_OPERAND(G0, true, "ptr @g0");
  CHECK_PRINT_AS_OPERAND(G1, true, "ptr @g1");
#undef CHECK_PRINT_AS_OPERAND
}

TEST(ValueTest, getLocalSlots) {
  // Verify that the getLocalSlot method returns the correct slot numbers.
  LLVMContext C;
  const char *ModuleString = "define void @f(i32 %x, i32 %y) {\n"
                             "entry:\n"
                             "  %0 = add i32 %y, 1\n"
                             "  %1 = add i32 %y, 1\n"
                             "  br label %2\n"
                             "\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);

  Function *F = M->getFunction("f");
  ASSERT_TRUE(F);
  ASSERT_FALSE(F->empty());
  BasicBlock &EntryBB = F->getEntryBlock();
  ASSERT_EQ(3u, EntryBB.size());
  BasicBlock *BB2 = &*++F->begin();
  ASSERT_TRUE(BB2);

  Instruction *I0 = &*EntryBB.begin();
  ASSERT_TRUE(I0);
  Instruction *I1 = &*++EntryBB.begin();
  ASSERT_TRUE(I1);

  ModuleSlotTracker MST(M.get());
  MST.incorporateFunction(*F);
  EXPECT_EQ(MST.getLocalSlot(I0), 0);
  EXPECT_EQ(MST.getLocalSlot(I1), 1);
  EXPECT_EQ(MST.getLocalSlot(&EntryBB), -1);
  EXPECT_EQ(MST.getLocalSlot(BB2), 2);
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST(ValueTest, getLocalSlotDeath) {
  LLVMContext C;
  const char *ModuleString = "define void @f(i32 %x, i32 %y) {\n"
                             "entry:\n"
                             "  %0 = add i32 %y, 1\n"
                             "  %1 = add i32 %y, 1\n"
                             "  br label %2\n"
                             "\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, C);

  Function *F = M->getFunction("f");
  ASSERT_TRUE(F);
  ASSERT_FALSE(F->empty());
  BasicBlock *BB2 = &*++F->begin();
  ASSERT_TRUE(BB2);

  ModuleSlotTracker MST(M.get());
  EXPECT_DEATH(MST.getLocalSlot(BB2), "No function incorporated");
}
#endif

TEST(ValueTest, replaceUsesOutsideBlock) {
  // Check that Value::replaceUsesOutsideBlock(New, BB) replaces uses outside
  // BB, including dbg.* uses of MetadataAsValue(ValueAsMetadata(this)).
  const auto *IR = R"(
    define i32 @f() !dbg !6 {
    entry:
      %a = add i32 0, 1, !dbg !15
      %b = add i32 0, 2, !dbg !15
      %c = add i32 %a, 2, !dbg !15
      call void @llvm.dbg.value(metadata i32 %a, metadata !9, metadata !DIExpression()), !dbg !15
      br label %exit, !dbg !15

    exit:
      call void @llvm.dbg.value(metadata i32 %a, metadata !11, metadata !DIExpression()), !dbg !16
      ret i32 %a, !dbg !16
    }

    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "test.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9, !11}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_signed)
    !11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 2, type: !12)
    !12 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_signed)
    !15 = !DILocation(line: 1, column: 1, scope: !6)
    !16 = !DILocation(line: 5, column: 1, scope: !6)
  )";
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  if (!M)
    Err.print("ValueTest", errs());

  auto GetNext = [](auto *I) { return &*++I->getIterator(); };

  Function *F = M->getFunction("f");
  // Entry.
  BasicBlock *Entry = &F->front();
  Instruction *A = &Entry->front();
  Instruction *B = GetNext(A);
  Instruction *C = GetNext(B);
  auto *EntryDbg = cast<DbgValueInst>(GetNext(C));
  // Exit.
  BasicBlock *Exit = GetNext(Entry);
  auto *ExitDbg = cast<DbgValueInst>(&Exit->front());
  Instruction *Ret = GetNext(ExitDbg);

  A->replaceUsesOutsideBlock(B, Entry);
  // These users are in Entry so shouldn't be changed.
  ASSERT_TRUE(C->getOperand(0) == cast<Value>(A));
  ASSERT_TRUE(EntryDbg->getValue(0) == cast<Value>(A));
  // These users are outside Entry so should be changed.
  ASSERT_TRUE(ExitDbg->getValue(0) == cast<Value>(B));
  ASSERT_TRUE(Ret->getOperand(0) == cast<Value>(B));
}
} // end anonymous namespace
