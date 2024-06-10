//===- llvm/unittest/IR/BasicBlockTest.cpp - BasicBlock unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockDbgInfoTest", errs());
  return Mod;
}

namespace {

// We can occasionally moveAfter an instruction so that it moves to the
// position that it already resides at. This is fine -- but gets complicated
// with dbg.value intrinsics. By moving an instruction, we can end up changing
// nothing but the location of debug-info intrinsics. That has to be modelled
// by DbgVariableRecords, the dbg.value replacement.
TEST(BasicBlockDbgInfoTest, InsertAfterSelf) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      %c = add i16 %b, 1, !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  // Fetch the entry block.
  BasicBlock &BB = M->getFunction("f")->getEntryBlock();

  Instruction *Inst1 = &*BB.begin();
  Instruction *Inst2 = &*std::next(BB.begin());
  Instruction *RetInst = &*std::next(Inst2->getIterator());
  EXPECT_TRUE(Inst1->hasDbgRecords());
  EXPECT_TRUE(Inst2->hasDbgRecords());
  EXPECT_FALSE(RetInst->hasDbgRecords());

  // If we move Inst2 to be after Inst1, then it comes _immediately_ after. Were
  // we in dbg.value form we would then have:
  //    dbg.value
  //    %b = add
  //    %c = add
  //    dbg.value
  // Check that this is replicated by DbgVariableRecords.
  Inst2->moveAfter(Inst1);

  // Inst1 should only have one DbgVariableRecord on it.
  EXPECT_TRUE(Inst1->hasDbgRecords());
  auto Range1 = Inst1->getDbgRecordRange();
  EXPECT_EQ(std::distance(Range1.begin(), Range1.end()), 1u);
  // Inst2 should have none.
  EXPECT_FALSE(Inst2->hasDbgRecords());
  // While the return inst should now have one on it.
  EXPECT_TRUE(RetInst->hasDbgRecords());
  auto Range2 = RetInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(Range2.begin(), Range2.end()), 1u);
}

TEST(BasicBlockDbgInfoTest, SplitBasicBlockBefore) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"---(
    define dso_local void @func() #0 !dbg !10 {
      %1 = alloca i32, align 4
      tail call void @llvm.dbg.declare(metadata ptr %1, metadata !14, metadata !DIExpression()), !dbg !16
      store i32 2, ptr %1, align 4, !dbg !16
      ret void, !dbg !17
    }

    declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

    attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
    !llvm.ident = !{!9}

    !0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "dummy", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "dummy", directory: "dummy")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !4 = !{i32 1, !"wchar_size", i32 4}
    !5 = !{i32 8, !"PIC Level", i32 2}
    !6 = !{i32 7, !"PIE Level", i32 2}
    !7 = !{i32 7, !"uwtable", i32 2}
    !8 = !{i32 7, !"frame-pointer", i32 2}
    !9 = !{!"dummy"}
    !10 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
    !11 = !DISubroutineType(types: !12)
    !12 = !{null}
    !13 = !{}
    !14 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 2, type: !15)
    !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
    !16 = !DILocation(line: 2, column: 6, scope: !10)
    !17 = !DILocation(line: 3, column: 2, scope: !10)
  )---");
  ASSERT_TRUE(M);

  Function *F = M->getFunction("func");

  BasicBlock &BB = F->getEntryBlock();
  auto I = std::prev(BB.end(), 2);
  BB.splitBasicBlockBefore(I, "before");

  BasicBlock &BBBefore = F->getEntryBlock();
  auto I2 = std::prev(BBBefore.end(), 2);
  ASSERT_TRUE(I2->hasDbgRecords());
}

TEST(BasicBlockDbgInfoTest, MarkerOperations) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  // Fetch the entry block,
  BasicBlock &BB = M->getFunction("f")->getEntryBlock();
  EXPECT_EQ(BB.size(), 2u);

  // Fetch out our two markers,
  Instruction *Instr1 = &*BB.begin();
  Instruction *Instr2 = Instr1->getNextNode();
  DbgMarker *Marker1 = Instr1->DebugMarker;
  DbgMarker *Marker2 = Instr2->DebugMarker;
  // There's no TrailingDbgRecords marker allocated yet.
  DbgMarker *EndMarker = nullptr;

  // Check that the "getMarker" utilities operate as expected.
  EXPECT_EQ(BB.getMarker(Instr1->getIterator()), Marker1);
  EXPECT_EQ(BB.getMarker(Instr2->getIterator()), Marker2);
  EXPECT_EQ(BB.getNextMarker(Instr1), Marker2);
  EXPECT_EQ(BB.getNextMarker(Instr2), EndMarker); // Is nullptr.

  // There should be two DbgVariableRecords,
  EXPECT_EQ(Marker1->StoredDbgRecords.size(), 1u);
  EXPECT_EQ(Marker2->StoredDbgRecords.size(), 1u);

  // Unlink them and try to re-insert them through the basic block.
  DbgRecord *DVR1 = &*Marker1->StoredDbgRecords.begin();
  DbgRecord *DVR2 = &*Marker2->StoredDbgRecords.begin();
  DVR1->removeFromParent();
  DVR2->removeFromParent();
  EXPECT_TRUE(Marker1->StoredDbgRecords.empty());
  EXPECT_TRUE(Marker2->StoredDbgRecords.empty());

  // This should appear in Marker1.
  BB.insertDbgRecordBefore(DVR1, BB.begin());
  EXPECT_EQ(Marker1->StoredDbgRecords.size(), 1u);
  EXPECT_EQ(DVR1, &*Marker1->StoredDbgRecords.begin());

  // This should attach to Marker2.
  BB.insertDbgRecordAfter(DVR2, &*BB.begin());
  EXPECT_EQ(Marker2->StoredDbgRecords.size(), 1u);
  EXPECT_EQ(DVR2, &*Marker2->StoredDbgRecords.begin());

  // Now, how about removing instructions? That should cause any
  // DbgVariableRecords to "fall down".
  Instr1->removeFromParent();
  Marker1 = nullptr;
  // DbgVariableRecords should now be in Marker2.
  EXPECT_EQ(BB.size(), 1u);
  EXPECT_EQ(Marker2->StoredDbgRecords.size(), 2u);
  // They should also be in the correct order.
  SmallVector<DbgRecord *, 2> DVRs;
  for (DbgRecord &DVR : Marker2->getDbgRecordRange())
    DVRs.push_back(&DVR);
  EXPECT_EQ(DVRs[0], DVR1);
  EXPECT_EQ(DVRs[1], DVR2);

  // If we remove the end instruction, the DbgVariableRecords should fall down
  // into the trailing marker.
  EXPECT_EQ(BB.getTrailingDbgRecords(), nullptr);
  Instr2->removeFromParent();
  EXPECT_TRUE(BB.empty());
  EndMarker = BB.getTrailingDbgRecords();
  ASSERT_NE(EndMarker, nullptr);
  EXPECT_EQ(EndMarker->StoredDbgRecords.size(), 2u);
  // Again, these should arrive in the correct order.

  DVRs.clear();
  for (DbgRecord &DVR : EndMarker->getDbgRecordRange())
    DVRs.push_back(&DVR);
  EXPECT_EQ(DVRs[0], DVR1);
  EXPECT_EQ(DVRs[1], DVR2);

  // Inserting a normal instruction at the beginning: shouldn't dislodge the
  // DbgVariableRecords. It's intended to not go at the start.
  Instr1->insertBefore(BB, BB.begin());
  EXPECT_EQ(EndMarker->StoredDbgRecords.size(), 2u);
  Instr1->removeFromParent();

  // Inserting at end(): should dislodge the DbgVariableRecords, if they were
  // dbg.values then they would sit "above" the new instruction.
  Instr1->insertBefore(BB, BB.end());
  EXPECT_EQ(Instr1->DebugMarker->StoredDbgRecords.size(), 2u);
  // We should de-allocate the trailing marker when something is inserted
  // at end().
  EXPECT_EQ(BB.getTrailingDbgRecords(), nullptr);

  // Remove Instr1: now the DbgVariableRecords will fall down again,
  Instr1->removeFromParent();
  EndMarker = BB.getTrailingDbgRecords();
  EXPECT_EQ(EndMarker->StoredDbgRecords.size(), 2u);

  // Inserting a terminator, however it's intended, should dislodge the
  // trailing DbgVariableRecords, as it's the clear intention of the caller that
  // this be the final instr in the block, and DbgVariableRecords aren't allowed
  // to live off the end forever.
  Instr2->insertBefore(BB, BB.begin());
  EXPECT_EQ(Instr2->DebugMarker->StoredDbgRecords.size(), 2u);
  EXPECT_EQ(BB.getTrailingDbgRecords(), nullptr);

  // Teardown,
  Instr1->insertBefore(BB, BB.begin());
}

TEST(BasicBlockDbgInfoTest, HeadBitOperations) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %c = add i16 %a, 1, !dbg !11
      %d = add i16 %a, 1, !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  // Test that the movement of debug-data when using moveBefore etc and
  // insertBefore etc are governed by the "head" bit of iterators.
  BasicBlock &BB = M->getFunction("f")->getEntryBlock();

  // Test that the head bit behaves as expected: it should be set when the
  // code wants the _start_ of the block, but not otherwise.
  EXPECT_TRUE(BB.getFirstInsertionPt().getHeadBit());
  BasicBlock::iterator BeginIt = BB.begin();
  EXPECT_TRUE(BeginIt.getHeadBit());
  // If you launder the instruction pointer through dereferencing and then
  // get the iterator again with getIterator, the head bit is lost. This is
  // deliberate: if you're calling getIterator, then you're requesting an
  // iterator for the position of _this_ instruction, not "the start of this
  // block".
  BasicBlock::iterator BeginIt2 = BeginIt->getIterator();
  EXPECT_FALSE(BeginIt2.getHeadBit());

  // Fetch some instruction pointers.
  Instruction *BInst = &*BeginIt;
  Instruction *CInst = BInst->getNextNode();
  Instruction *DInst = CInst->getNextNode();
  // CInst should have debug-info.
  ASSERT_TRUE(CInst->DebugMarker);
  EXPECT_FALSE(CInst->DebugMarker->StoredDbgRecords.empty());

  // If we move "c" to the start of the block, just normally, then the
  // DbgVariableRecords should fall down to "d".
  CInst->moveBefore(BB, BeginIt2);
  EXPECT_TRUE(!CInst->DebugMarker ||
              CInst->DebugMarker->StoredDbgRecords.empty());
  ASSERT_TRUE(DInst->DebugMarker);
  EXPECT_FALSE(DInst->DebugMarker->StoredDbgRecords.empty());

  // Wheras if we move D to the start of the block with moveBeforePreserving,
  // the DbgVariableRecords should move with it.
  DInst->moveBeforePreserving(BB, BB.begin());
  EXPECT_FALSE(DInst->DebugMarker->StoredDbgRecords.empty());
  EXPECT_EQ(&*BB.begin(), DInst);

  // Similarly, moveAfterPreserving "D" to "C" should move DbgVariableRecords
  // with "D".
  DInst->moveAfterPreserving(CInst);
  EXPECT_FALSE(DInst->DebugMarker->StoredDbgRecords.empty());

  // (move back to the start...)
  DInst->moveBeforePreserving(BB, BB.begin());

  // Current order of insts: "D -> C -> B -> Ret". DbgVariableRecords on "D".
  // If we move "C" to the beginning of the block, it should go before the
  // DbgVariableRecords. They'll stay on "D".
  CInst->moveBefore(BB, BB.begin());
  EXPECT_TRUE(!CInst->DebugMarker ||
              CInst->DebugMarker->StoredDbgRecords.empty());
  EXPECT_FALSE(DInst->DebugMarker->StoredDbgRecords.empty());
  EXPECT_EQ(&*BB.begin(), CInst);
  EXPECT_EQ(CInst->getNextNode(), DInst);

  // Move back.
  CInst->moveBefore(BInst);
  EXPECT_EQ(&*BB.begin(), DInst);

  // Current order of insts: "D -> C -> B -> Ret". DbgVariableRecords on "D".
  // Now move CInst to the position of DInst, but using getIterator instead of
  // BasicBlock::begin. This signals that we want the "C" instruction to be
  // immediately before "D", with any DbgVariableRecords on "D" now moving to
  // "C". It's the equivalent of moving an instruction to the position between a
  // run of dbg.values and the next instruction.
  CInst->moveBefore(BB, DInst->getIterator());
  // CInst gains the DbgVariableRecords.
  EXPECT_TRUE(!DInst->DebugMarker ||
              DInst->DebugMarker->StoredDbgRecords.empty());
  EXPECT_FALSE(CInst->DebugMarker->StoredDbgRecords.empty());
  EXPECT_EQ(&*BB.begin(), CInst);
}

TEST(BasicBlockDbgInfoTest, InstrDbgAccess) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %c = add i16 %a, 1, !dbg !11
      %d = add i16 %a, 1, !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  // Check that DbgVariableRecords can be accessed from Instructions without
  // digging into the depths of DbgMarkers.
  BasicBlock &BB = M->getFunction("f")->getEntryBlock();

  Instruction *BInst = &*BB.begin();
  Instruction *CInst = BInst->getNextNode();
  Instruction *DInst = CInst->getNextNode();

  ASSERT_FALSE(BInst->DebugMarker);
  ASSERT_TRUE(CInst->DebugMarker);
  ASSERT_EQ(CInst->DebugMarker->StoredDbgRecords.size(), 1u);
  DbgRecord *DVR1 = &*CInst->DebugMarker->StoredDbgRecords.begin();
  ASSERT_TRUE(DVR1);
  EXPECT_FALSE(BInst->hasDbgRecords());

  // Clone DbgVariableRecords from one inst to another. Other arguments to clone
  // are tested in DbgMarker test.
  auto Range1 = BInst->cloneDebugInfoFrom(CInst);
  EXPECT_EQ(BInst->DebugMarker->StoredDbgRecords.size(), 1u);
  DbgRecord *DVR2 = &*BInst->DebugMarker->StoredDbgRecords.begin();
  EXPECT_EQ(std::distance(Range1.begin(), Range1.end()), 1u);
  EXPECT_EQ(&*Range1.begin(), DVR2);
  EXPECT_NE(DVR1, DVR2);

  // We should be able to get a range over exactly the same information.
  auto Range2 = BInst->getDbgRecordRange();
  EXPECT_EQ(Range1.begin(), Range2.begin());
  EXPECT_EQ(Range1.end(), Range2.end());

  // We should be able to query if there are DbgVariableRecords,
  EXPECT_TRUE(BInst->hasDbgRecords());
  EXPECT_TRUE(CInst->hasDbgRecords());
  EXPECT_FALSE(DInst->hasDbgRecords());

  // Dropping should be easy,
  BInst->dropDbgRecords();
  EXPECT_FALSE(BInst->hasDbgRecords());
  EXPECT_EQ(BInst->DebugMarker->StoredDbgRecords.size(), 0u);

  // And we should be able to drop individual DbgVariableRecords.
  CInst->dropOneDbgRecord(DVR1);
  EXPECT_FALSE(CInst->hasDbgRecords());
  EXPECT_EQ(CInst->DebugMarker->StoredDbgRecords.size(), 0u);
}

/* Let's recall the big illustration from BasicBlock::spliceDebugInfo:

                                               Dest
                                                 |
   this-block:    A----A----A                ====A----A----A----A---A---A
    Src-block                ++++B---B---B---B:::C
                                 |               |
                                First           Last

  in all it's glory. Depending on the bit-configurations for the iterator head
  / tail bits on the three named iterators, there are eight ways for a splice to
  occur. To save the amount of thinking needed to pack this into one unit test,
  just test the same IR eight times with difference splices. The IR shall be
  thus:

    define i16 @f(i16 %a) !dbg !6 {
    entry:
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      br label %exit, !dbg !11

    exit:
      call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
      %c = add i16 %b, 1, !dbg !11
      ret i16 0, !dbg !11
    }

  The iterators will be:
    Dest: exit block, "c" instruction.
    First: entry block, "b" instruction.
    Last: entry block, branch instruction.

  The numbered configurations will be:

       |    Dest-Head   |   First-Head   |   Last-tail
   ----+----------------+----------------+------------
    0  |      false     |     false      |     false
    1  |      true      |     false      |     false
    2  |      false     |     true       |     false
    3  |      true      |     true       |     false
    4  |      false     |     false      |     true
    5  |      true      |     false      |     true
    6  |      false     |     true       |     true
    7  |      true      |     true       |     true

  Each numbered test scenario will also have a short explanation indicating what
  this bit configuration represents.
*/

static const std::string SpliceTestIR = R"(
    define i16 @f(i16 %a) !dbg !6 {
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      br label %exit, !dbg !11

    exit:
      call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
      %c = add i16 %b, 1, !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)";

class DbgSpliceTest : public ::testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<Module> M;
  BasicBlock *BBEntry, *BBExit;
  BasicBlock::iterator Dest, First, Last;
  Instruction *BInst, *Branch, *CInst;
  DbgVariableRecord *DVRA, *DVRB, *DVRConst;

  void SetUp() override {
    M = parseIR(C, SpliceTestIR.c_str());

    BBEntry = &M->getFunction("f")->getEntryBlock();
    BBExit = BBEntry->getNextNode();

    Dest = BBExit->begin();
    First = BBEntry->begin();
    Last = BBEntry->getTerminator()->getIterator();
    BInst = &*First;
    Branch = &*Last;
    CInst = &*Dest;

    DVRA =
        cast<DbgVariableRecord>(&*BInst->DebugMarker->StoredDbgRecords.begin());
    DVRB = cast<DbgVariableRecord>(
        &*Branch->DebugMarker->StoredDbgRecords.begin());
    DVRConst =
        cast<DbgVariableRecord>(&*CInst->DebugMarker->StoredDbgRecords.begin());
  }

  bool InstContainsDbgVariableRecord(Instruction *I, DbgVariableRecord *DVR) {
    for (DbgRecord &D : I->getDbgRecordRange()) {
      if (&D == DVR) {
        // Confirm too that the links between the records are correct.
        EXPECT_EQ(DVR->Marker, I->DebugMarker);
        EXPECT_EQ(I->DebugMarker->MarkedInstr, I);
        return true;
      }
    }
    return false;
  }

  bool CheckDVROrder(Instruction *I,
                     SmallVector<DbgVariableRecord *> CheckVals) {
    SmallVector<DbgRecord *> Vals;
    for (DbgRecord &D : I->getDbgRecordRange())
      Vals.push_back(&D);

    EXPECT_EQ(Vals.size(), CheckVals.size());
    if (Vals.size() != CheckVals.size())
      return false;

    for (unsigned int I = 0; I < Vals.size(); ++I) {
      EXPECT_EQ(Vals[I], CheckVals[I]);
      // Provide another expectation failure to let us localise what goes wrong,
      // by returning a flag to the caller.
      if (Vals[I] != CheckVals[I])
        return false;
    }
    return true;
  }
};

TEST_F(DbgSpliceTest, DbgSpliceTest0) {
  Dest.setHeadBit(false);
  First.setHeadBit(false);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, not including leading dbg.value, to Last, including the
    trailing dbg.value. Place at Dest, between the constant dbg.value and %c.
    %b, and the following dbg.value, should move, to:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Dest, in exit block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRB));

  // DVRA, should have "fallen" onto the branch, remained in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRA));

  // DVRConst should be on the moved %b instruction.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRConst));
}

TEST_F(DbgSpliceTest, DbgSpliceTest1) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, not including leading dbg.value, to Last, including the
    trailing dbg.value. Place at the head of Dest, i.e. at the very start of
    BBExit, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
First     %b = add i16 %a, 1, !dbg !11
DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata
!DIExpression()), !dbg !11 DVRConst  call void @llvm.dbg.value(metadata i16 0,
metadata !9, metadata !DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1,
!dbg !11 ret i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRB));

  // DVRA, should have "fallen" onto the branch, remained in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRA));

  // DVRConst should be behind / after the moved instructions, remain on CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));

  // Order of DVRB and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(CInst, {DVRB, DVRConst}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest2) {
  Dest.setHeadBit(false);
  First.setHeadBit(true);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from head of First, which includes the leading dbg.value, to Last,
    including the trailing dbg.value. Place in front of Dest, but after any
    debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 DVRA      call void @llvm.dbg.value(metadata i16 %a,
metadata !9, metadata !DIExpression()), !dbg !11 First     %b = add i16 %a, 1,
!dbg !11 DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9,
metadata !DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret
i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);

  // DVRB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRB));

  // DVRA, should have transferred with the spliced instructions, remains on
  // the "b" inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRA));

  // DVRConst should be ahead of the moved instructions, ahead of BInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRConst));

  // Order of DVRA and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(BInst, {DVRConst, DVRA}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest3) {
  Dest.setHeadBit(true);
  First.setHeadBit(true);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from head of First, which includes the leading dbg.value, to Last,
    including the trailing dbg.value. Place at head of Dest, before any
    debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
Last      br label %exit, !dbg !11

BBExit  exit:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9,
metadata !DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret
i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);

  // DVRB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRB));

  // DVRA, should have transferred with the spliced instructions, remains on
  // the "b" inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRA));

  // DVRConst should be behind the moved instructions, ahead of CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));

  // Order of DVRB and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(CInst, {DVRB, DVRConst}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest4) {
  Dest.setHeadBit(false);
  First.setHeadBit(false);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, not including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at Dest,
    after any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 DVRB      call void @llvm.dbg.value(metadata i16 %b,
metadata !9, metadata !DIExpression()), !dbg !11 Last      br label %exit, !dbg
!11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 Dest      %c =
add i16 %b, 1, !dbg !11 ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have remained in entry block, falls onto Branch inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRA));

  // DVRConst should be ahead of the moved instructions, BInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRConst));

  // Order of DVRA and DVRA should be thus:
  EXPECT_TRUE(CheckDVROrder(Branch, {DVRA, DVRB}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest5) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, not including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at head
    of Dest, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 DVRB      call void @llvm.dbg.value(metadata i16 %b,
metadata !9, metadata !DIExpression()), !dbg !11 Last      br label %exit, !dbg
!11

BBExit  exit:
First     %b = add i16 %a, 1, !dbg !11
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have remained in entry block, falls onto Branch inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRA));

  // DVRConst should be behind of the moved instructions, on CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));

  // Order of DVRA and DVRB should be thus:
  EXPECT_TRUE(CheckDVROrder(Branch, {DVRA, DVRB}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest6) {
  Dest.setHeadBit(false);
  First.setHeadBit(true);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at Dest,
    after any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 DVRA      call void @llvm.dbg.value(metadata i16 %a,
metadata !9, metadata !DIExpression()), !dbg !11 First     %b = add i16 %a, 1,
!dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have transferred to BBExit, on B inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRA));

  // DVRConst should be ahead of the moved instructions, on BInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRConst));

  // Order of DVRA and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(BInst, {DVRConst, DVRA}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest7) {
  Dest.setHeadBit(true);
  First.setHeadBit(true);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from First, including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at head
    of Dest, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRConst  call
void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()),
!dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have transferred to BBExit, on B inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(BInst, DVRA));

  // DVRConst should be after of the moved instructions, on CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));
}

// But wait, there's more! What if you splice a range that is empty, but
// implicitly contains debug-info? In the dbg.value design for debug-info,
// this would be an explicit range, but in DbgVariableRecord debug-info, it
// isn't. Check that if we try to do that, with differing head-bit values, that
// DbgVariableRecords are transferred.
// Test with empty transfers to Dest, with head bit set and not set.

TEST_F(DbgSpliceTest, DbgSpliceEmpty0) {
  Dest.setHeadBit(false);
  First.setHeadBit(false);
  Last.setHeadBit(false);
  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from BBEntry.getFirstInsertionPt to First -- this implicitly is a
    splice of DVRA, but the iterators are pointing at the same instruction. The
    only difference is the setting of the head bit. Becomes;

        define i16 @f(i16 %a) !dbg !6 {
First     %b = add i16 %a, 1, !dbg !11
DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 DVRA      call void @llvm.dbg.value(metadata i16 %a,
metadata !9, metadata !DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1,
!dbg !11 ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, BBEntry->getFirstInsertionPt(), First);
  EXPECT_EQ(BInst->getParent(), BBEntry);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have transferred to BBExit, on C inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRA));

  // DVRConst should be ahead of the moved DbgVariableRecord, on CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));

  // Order of DVRA and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(CInst, {DVRConst, DVRA}));
}

TEST_F(DbgSpliceTest, DbgSpliceEmpty1) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setHeadBit(false);
  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 First     %b = add i16 %a, 1, !dbg !11 DVRB      call
void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()),
!dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata
!DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1, !dbg !11 ret i16 0,
!dbg !11
        }

    Splice from BBEntry.getFirstInsertionPt to First -- this implicitly is a
    splice of DVRA, but the iterators are pointing at the same instruction. The
    only difference is the setting of the head bit. Insert at head of Dest,
    i.e. before DVRConst. Becomes;

        define i16 @f(i16 %a) !dbg !6 {
First     %b = add i16 %a, 1, !dbg !11
DVRB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata
!DIExpression()), !dbg !11 Last      br label %exit, !dbg !11

BBExit  exit:
DVRA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata
!DIExpression()), !dbg !11 DVRConst  call void @llvm.dbg.value(metadata i16 0,
metadata !9, metadata !DIExpression()), !dbg !11 Dest      %c = add i16 %b, 1,
!dbg !11 ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, BBEntry->getFirstInsertionPt(), First);
  EXPECT_EQ(BInst->getParent(), BBEntry);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DVRB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDbgVariableRecord(Branch, DVRB));

  // DVRA, should have transferred to BBExit, on C inst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRA));

  // DVRConst should be ahead of the moved DbgVariableRecord, on CInst.
  EXPECT_TRUE(InstContainsDbgVariableRecord(CInst, DVRConst));

  // Order of DVRA and DVRConst should be thus:
  EXPECT_TRUE(CheckDVROrder(CInst, {DVRA, DVRConst}));
}

// If we splice new instructions into a block with trailing DbgVariableRecords,
// then the trailing DbgVariableRecords should get flushed back out.
TEST(BasicBlockDbgInfoTest, DbgSpliceTrailing) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      br label %exit

    exit:
      %b = add i16 %a, 1, !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  BasicBlock &Entry = M->getFunction("f")->getEntryBlock();
  BasicBlock &Exit = *Entry.getNextNode();

  // Begin by forcing entry block to have dangling DbgVariableRecord.
  Entry.getTerminator()->eraseFromParent();
  ASSERT_NE(Entry.getTrailingDbgRecords(), nullptr);
  EXPECT_TRUE(Entry.empty());

  // Now transfer the entire contents of the exit block into the entry.
  Entry.splice(Entry.end(), &Exit, Exit.begin(), Exit.end());

  // The trailing DbgVariableRecord should have been placed at the front of
  // what's been spliced in.
  Instruction *BInst = &*Entry.begin();
  ASSERT_TRUE(BInst->DebugMarker);
  EXPECT_EQ(BInst->DebugMarker->StoredDbgRecords.size(), 1u);
}

// When we remove instructions from the program, adjacent DbgVariableRecords
// coalesce together into one DbgMarker. In "old" dbg.value mode you could
// re-insert the removed instruction back into the middle of a sequence of
// dbg.values. Test that this can be replicated correctly by DbgVariableRecords
TEST(BasicBlockDbgInfoTest, RemoveInstAndReinsert) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      %qux = sub i16 %a, 0
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %foo = add i16 %a, %a
      call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 1
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  BasicBlock &Entry = M->getFunction("f")->getEntryBlock();

  // Fetch the relevant instructions from the converted function.
  Instruction *SubInst = &*Entry.begin();
  ASSERT_TRUE(isa<BinaryOperator>(SubInst));
  Instruction *AddInst = SubInst->getNextNode();
  ASSERT_TRUE(isa<BinaryOperator>(AddInst));
  Instruction *RetInst = AddInst->getNextNode();
  ASSERT_TRUE(isa<ReturnInst>(RetInst));

  // add and sub should both have one DbgVariableRecord on add and ret.
  EXPECT_FALSE(SubInst->hasDbgRecords());
  EXPECT_TRUE(AddInst->hasDbgRecords());
  EXPECT_TRUE(RetInst->hasDbgRecords());
  auto R1 = AddInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R1.begin(), R1.end()), 1u);
  auto R2 = RetInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R2.begin(), R2.end()), 1u);

  // The Supported (TM) code sequence for removing then reinserting insts
  // after another instruction:
  std::optional<DbgVariableRecord::self_iterator> Pos =
      AddInst->getDbgReinsertionPosition();
  AddInst->removeFromParent();

  // We should have a re-insertion position.
  ASSERT_TRUE(Pos);
  // Both DbgVariableRecords should now be attached to the ret inst.
  auto R3 = RetInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R3.begin(), R3.end()), 2u);

  // Re-insert and re-insert.
  AddInst->insertAfter(SubInst);
  Entry.reinsertInstInDbgRecords(AddInst, Pos);
  // We should be back into a position of having one DbgVariableRecord on add
  // and ret.
  EXPECT_FALSE(SubInst->hasDbgRecords());
  EXPECT_TRUE(AddInst->hasDbgRecords());
  EXPECT_TRUE(RetInst->hasDbgRecords());
  auto R4 = AddInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R4.begin(), R4.end()), 1u);
  auto R5 = RetInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R5.begin(), R5.end()), 1u);
}

// Test instruction removal and re-insertion, this time with one
// DbgVariableRecord that should hop up one instruction.
TEST(BasicBlockDbgInfoTest, RemoveInstAndReinsertForOneDbgVariableRecord) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      %qux = sub i16 %a, 0
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      %foo = add i16 %a, %a
      ret i16 1
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  BasicBlock &Entry = M->getFunction("f")->getEntryBlock();

  // Fetch the relevant instructions from the converted function.
  Instruction *SubInst = &*Entry.begin();
  ASSERT_TRUE(isa<BinaryOperator>(SubInst));
  Instruction *AddInst = SubInst->getNextNode();
  ASSERT_TRUE(isa<BinaryOperator>(AddInst));
  Instruction *RetInst = AddInst->getNextNode();
  ASSERT_TRUE(isa<ReturnInst>(RetInst));

  // There should be one DbgVariableRecord.
  EXPECT_FALSE(SubInst->hasDbgRecords());
  EXPECT_TRUE(AddInst->hasDbgRecords());
  EXPECT_FALSE(RetInst->hasDbgRecords());
  auto R1 = AddInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R1.begin(), R1.end()), 1u);

  // The Supported (TM) code sequence for removing then reinserting insts:
  std::optional<DbgVariableRecord::self_iterator> Pos =
      AddInst->getDbgReinsertionPosition();
  AddInst->removeFromParent();

  // No re-insertion position as there were no DbgVariableRecords on the ret.
  ASSERT_FALSE(Pos);
  // The single DbgVariableRecord should now be attached to the ret inst.
  EXPECT_TRUE(RetInst->hasDbgRecords());
  auto R2 = RetInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R2.begin(), R2.end()), 1u);

  // Re-insert and re-insert.
  AddInst->insertAfter(SubInst);
  Entry.reinsertInstInDbgRecords(AddInst, Pos);
  // We should be back into a position of having one DbgVariableRecord on the
  // AddInst.
  EXPECT_FALSE(SubInst->hasDbgRecords());
  EXPECT_TRUE(AddInst->hasDbgRecords());
  EXPECT_FALSE(RetInst->hasDbgRecords());
  auto R3 = AddInst->getDbgRecordRange();
  EXPECT_EQ(std::distance(R3.begin(), R3.end()), 1u);
}

// Similar to the above, what if we splice into an empty block with debug-info,
// with debug-info at the start of the moving range, that we intend to be
// transferred. The dbg.value of %a should remain at the start, but come ahead
// of the i16 0 dbg.value.
TEST(BasicBlockDbgInfoTest, DbgSpliceToEmpty1) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      br label %exit

    exit:
      call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 1, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  Function &F = *M->getFunction("f");
  BasicBlock &Entry = F.getEntryBlock();
  BasicBlock &Exit = *Entry.getNextNode();

  // Begin by forcing entry block to have dangling DbgVariableRecord.
  Entry.getTerminator()->eraseFromParent();
  ASSERT_NE(Entry.getTrailingDbgRecords(), nullptr);
  EXPECT_TRUE(Entry.empty());

  // Now transfer the entire contents of the exit block into the entry. This
  // includes both dbg.values.
  Entry.splice(Entry.end(), &Exit, Exit.begin(), Exit.end());

  // We should now have two dbg.values on the first instruction, and they
  // should be in the correct order of %a, then 0.
  Instruction *BInst = &*Entry.begin();
  ASSERT_TRUE(BInst->hasDbgRecords());
  EXPECT_EQ(BInst->DebugMarker->StoredDbgRecords.size(), 2u);
  SmallVector<DbgVariableRecord *, 2> DbgVariableRecords;
  for (DbgRecord &DVR : BInst->getDbgRecordRange())
    DbgVariableRecords.push_back(cast<DbgVariableRecord>(&DVR));

  EXPECT_EQ(DbgVariableRecords[0]->getVariableLocationOp(0), F.getArg(0));
  Value *SecondDVRValue = DbgVariableRecords[1]->getVariableLocationOp(0);
  ASSERT_TRUE(isa<ConstantInt>(SecondDVRValue));
  EXPECT_EQ(cast<ConstantInt>(SecondDVRValue)->getZExtValue(), 0ull);

  // No trailing DbgVariableRecords in the entry block now.
  EXPECT_EQ(Entry.getTrailingDbgRecords(), nullptr);
}

// Similar test again, but this time: splice the contents of exit into entry,
// with the intention of leaving the first dbg.value (i16 0) behind.
TEST(BasicBlockDbgInfoTest, DbgSpliceToEmpty2) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
      br label %exit

    exit:
      call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 1, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  Function &F = *M->getFunction("f");
  BasicBlock &Entry = F.getEntryBlock();
  BasicBlock &Exit = *Entry.getNextNode();

  // Begin by forcing entry block to have dangling DbgVariableRecord.
  Entry.getTerminator()->eraseFromParent();
  ASSERT_NE(Entry.getTrailingDbgRecords(), nullptr);
  EXPECT_TRUE(Entry.empty());

  // Now transfer into the entry block -- fetching the first instruction with
  // begin and then calling getIterator clears the "head" bit, meaning that the
  // range to move will not include any leading DbgVariableRecords.
  Entry.splice(Entry.end(), &Exit, Exit.begin()->getIterator(), Exit.end());

  // We should now have one dbg.values on the first instruction, %a.
  Instruction *BInst = &*Entry.begin();
  ASSERT_TRUE(BInst->hasDbgRecords());
  EXPECT_EQ(BInst->DebugMarker->StoredDbgRecords.size(), 1u);
  SmallVector<DbgVariableRecord *, 2> DbgVariableRecords;
  for (DbgRecord &DVR : BInst->getDbgRecordRange())
    DbgVariableRecords.push_back(cast<DbgVariableRecord>(&DVR));

  EXPECT_EQ(DbgVariableRecords[0]->getVariableLocationOp(0), F.getArg(0));
  // No trailing DbgVariableRecords in the entry block now.
  EXPECT_EQ(Entry.getTrailingDbgRecords(), nullptr);

  // We should have nothing left in the exit block...
  EXPECT_TRUE(Exit.empty());
  // ... except for some dangling DbgVariableRecords.
  EXPECT_NE(Exit.getTrailingDbgRecords(), nullptr);
  EXPECT_FALSE(Exit.getTrailingDbgRecords()->empty());
  Exit.getTrailingDbgRecords()->eraseFromParent();
  Exit.deleteTrailingDbgRecords();
}

// What if we moveBefore end() -- there might be no debug-info there, in which
// case we shouldn't crash.
TEST(BasicBlockDbgInfoTest, DbgMoveToEnd) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
    entry:
      br label %exit

    exit:
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  Function &F = *M->getFunction("f");
  BasicBlock &Entry = F.getEntryBlock();
  BasicBlock &Exit = *Entry.getNextNode();

  // Move the return to the end of the entry block.
  Instruction *Br = Entry.getTerminator();
  Instruction *Ret = Exit.getTerminator();
  EXPECT_EQ(Entry.getTrailingDbgRecords(), nullptr);
  Ret->moveBefore(Entry, Entry.end());
  Br->eraseFromParent();

  // There should continue to not be any debug-info anywhere.
  EXPECT_EQ(Entry.getTrailingDbgRecords(), nullptr);
  EXPECT_EQ(Exit.getTrailingDbgRecords(), nullptr);
  EXPECT_FALSE(Ret->hasDbgRecords());
}

} // End anonymous namespace.
