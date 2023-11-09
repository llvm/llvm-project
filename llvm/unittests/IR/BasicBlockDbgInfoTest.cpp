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

extern cl::opt<bool> UseNewDbgInfoFormat;

// None of these tests are meaningful or do anything if we do not have the
// experimental "head" bit compiled into ilist_iterator (aka
// ilist_iterator_w_bits), thus there's no point compiling these tests in.
#ifdef EXPERIMENTAL_DEBUGINFO_ITERATORS

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("BasicBlockDbgInfoTest", errs());
  return Mod;
}

namespace {

TEST(BasicBlockDbgInfoTest, MarkerOperations) {
  LLVMContext C;
  UseNewDbgInfoFormat = true;

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
  // Convert the module to "new" form debug-info.
  M->convertToNewDbgValues();
  EXPECT_EQ(BB.size(), 2u);

  // Fetch out our two markers,
  Instruction *Instr1 = &*BB.begin();
  Instruction *Instr2 = Instr1->getNextNode();
  DPMarker *Marker1 = Instr1->DbgMarker;
  DPMarker *Marker2 = Instr2->DbgMarker;
  // There's no TrailingDPValues marker allocated yet.
  DPMarker *EndMarker = nullptr;

  // Check that the "getMarker" utilities operate as expected.
  EXPECT_EQ(BB.getMarker(Instr1->getIterator()), Marker1);
  EXPECT_EQ(BB.getMarker(Instr2->getIterator()), Marker2);
  EXPECT_EQ(BB.getNextMarker(Instr1), Marker2);
  EXPECT_EQ(BB.getNextMarker(Instr2), EndMarker); // Is nullptr.

  // There should be two DPValues,
  EXPECT_EQ(Marker1->StoredDPValues.size(), 1u);
  EXPECT_EQ(Marker2->StoredDPValues.size(), 1u);

  // Unlink them and try to re-insert them through the basic block.
  DPValue *DPV1 = &*Marker1->StoredDPValues.begin();
  DPValue *DPV2 = &*Marker2->StoredDPValues.begin();
  DPV1->removeFromParent();
  DPV2->removeFromParent();
  EXPECT_TRUE(Marker1->StoredDPValues.empty());
  EXPECT_TRUE(Marker2->StoredDPValues.empty());

  // This should appear in Marker1.
  BB.insertDPValueBefore(DPV1, BB.begin());
  EXPECT_EQ(Marker1->StoredDPValues.size(), 1u);
  EXPECT_EQ(DPV1, &*Marker1->StoredDPValues.begin());

  // This should attach to Marker2.
  BB.insertDPValueAfter(DPV2, &*BB.begin());
  EXPECT_EQ(Marker2->StoredDPValues.size(), 1u);
  EXPECT_EQ(DPV2, &*Marker2->StoredDPValues.begin());

  // Now, how about removing instructions? That should cause any DPValues to
  // "fall down".
  Instr1->removeFromParent();
  Marker1 = nullptr;
  // DPValues should now be in Marker2.
  EXPECT_EQ(BB.size(), 1u);
  EXPECT_EQ(Marker2->StoredDPValues.size(), 2u);
  // They should also be in the correct order.
  SmallVector<DPValue *, 2> DPVs;
  for (DPValue &DPV : Marker2->getDbgValueRange())
    DPVs.push_back(&DPV);
  EXPECT_EQ(DPVs[0], DPV1);
  EXPECT_EQ(DPVs[1], DPV2);

  // If we remove the end instruction, the DPValues should fall down into
  // the trailing marker.
  EXPECT_EQ(BB.getTrailingDPValues(), nullptr);
  Instr2->removeFromParent();
  EXPECT_TRUE(BB.empty());
  EndMarker = BB.getTrailingDPValues();;
  ASSERT_NE(EndMarker, nullptr);
  EXPECT_EQ(EndMarker->StoredDPValues.size(), 2u);
  // Again, these should arrive in the correct order.

  DPVs.clear();
  for (DPValue &DPV : EndMarker->getDbgValueRange())
    DPVs.push_back(&DPV);
  EXPECT_EQ(DPVs[0], DPV1);
  EXPECT_EQ(DPVs[1], DPV2);

  // Inserting a normal instruction at the beginning: shouldn't dislodge the
  // DPValues. It's intended to not go at the start.
  Instr1->insertBefore(BB, BB.begin());
  EXPECT_EQ(EndMarker->StoredDPValues.size(), 2u);
  Instr1->removeFromParent();

  // Inserting at end(): should dislodge the DPValues, if they were dbg.values
  // then they would sit "above" the new instruction.
  Instr1->insertBefore(BB, BB.end());
  EXPECT_EQ(Instr1->DbgMarker->StoredDPValues.size(), 2u);
  // However we won't de-allocate the trailing marker until a terminator is
  // inserted.
  EXPECT_EQ(EndMarker->StoredDPValues.size(), 0u);
  EXPECT_EQ(BB.getTrailingDPValues(), EndMarker);

  // Remove Instr1: now the DPValues will fall down again,
  Instr1->removeFromParent();
  EndMarker = BB.getTrailingDPValues();;
  EXPECT_EQ(EndMarker->StoredDPValues.size(), 2u);

  // Inserting a terminator, however it's intended, should dislodge the
  // trailing DPValues, as it's the clear intention of the caller that this be
  // the final instr in the block, and DPValues aren't allowed to live off the
  // end forever.
  Instr2->insertBefore(BB, BB.begin());
  EXPECT_EQ(Instr2->DbgMarker->StoredDPValues.size(), 2u);
  EXPECT_EQ(BB.getTrailingDPValues(), nullptr);

  // Teardown,
  Instr1->insertBefore(BB, BB.begin());

  UseNewDbgInfoFormat = false;
}

TEST(BasicBlockDbgInfoTest, HeadBitOperations) {
  LLVMContext C;
  UseNewDbgInfoFormat = true;

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
  // Convert the module to "new" form debug-info.
  M->convertToNewDbgValues();

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
  ASSERT_TRUE(CInst->DbgMarker);
  EXPECT_FALSE(CInst->DbgMarker->StoredDPValues.empty());

  // If we move "c" to the start of the block, just normally, then the DPValues
  // should fall down to "d".
  CInst->moveBefore(BB, BeginIt2);
  EXPECT_TRUE(!CInst->DbgMarker || CInst->DbgMarker->StoredDPValues.empty());
  ASSERT_TRUE(DInst->DbgMarker);
  EXPECT_FALSE(DInst->DbgMarker->StoredDPValues.empty());

  // Wheras if we move D to the start of the block with moveBeforePreserving,
  // the DPValues should move with it.
  DInst->moveBeforePreserving(BB, BB.begin());
  EXPECT_FALSE(DInst->DbgMarker->StoredDPValues.empty());
  EXPECT_EQ(&*BB.begin(), DInst);

  // Similarly, moveAfterPreserving "D" to "C" should move DPValues with "D".
  DInst->moveAfterPreserving(CInst);
  EXPECT_FALSE(DInst->DbgMarker->StoredDPValues.empty());

  // (move back to the start...)
  DInst->moveBeforePreserving(BB, BB.begin());

  // Current order of insts: "D -> C -> B -> Ret". DPValues on "D".
  // If we move "C" to the beginning of the block, it should go before the
  // DPValues. They'll stay on "D".
  CInst->moveBefore(BB, BB.begin());
  EXPECT_TRUE(!CInst->DbgMarker || CInst->DbgMarker->StoredDPValues.empty());
  EXPECT_FALSE(DInst->DbgMarker->StoredDPValues.empty());
  EXPECT_EQ(&*BB.begin(), CInst);
  EXPECT_EQ(CInst->getNextNode(), DInst);

  // Move back.
  CInst->moveBefore(BInst);
  EXPECT_EQ(&*BB.begin(), DInst);

  // Current order of insts: "D -> C -> B -> Ret". DPValues on "D".
  // Now move CInst to the position of DInst, but using getIterator instead of
  // BasicBlock::begin. This signals that we want the "C" instruction to be
  // immediately before "D", with any DPValues on "D" now moving to "C".
  // It's the equivalent of moving an instruction to the position between a
  // run of dbg.values and the next instruction.
  CInst->moveBefore(BB, DInst->getIterator());
  // CInst gains the DPValues.
  EXPECT_TRUE(!DInst->DbgMarker || DInst->DbgMarker->StoredDPValues.empty());
  EXPECT_FALSE(CInst->DbgMarker->StoredDPValues.empty());
  EXPECT_EQ(&*BB.begin(), CInst);

  UseNewDbgInfoFormat = false;
}

TEST(BasicBlockDbgInfoTest, InstrDbgAccess) {
  LLVMContext C;
  UseNewDbgInfoFormat = true;

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

  // Check that DPValues can be accessed from Instructions without digging
  // into the depths of DPMarkers.
  BasicBlock &BB = M->getFunction("f")->getEntryBlock();
  // Convert the module to "new" form debug-info.
  M->convertToNewDbgValues();

  Instruction *BInst = &*BB.begin();
  Instruction *CInst = BInst->getNextNode();
  Instruction *DInst = CInst->getNextNode();

  ASSERT_TRUE(BInst->DbgMarker);
  ASSERT_TRUE(CInst->DbgMarker);
  ASSERT_EQ(CInst->DbgMarker->StoredDPValues.size(), 1u);
  DPValue *DPV1 = &*CInst->DbgMarker->StoredDPValues.begin();
  ASSERT_TRUE(DPV1);
  EXPECT_EQ(BInst->DbgMarker->StoredDPValues.size(), 0u);

  // Clone DPValues from one inst to another. Other arguments to clone are
  // tested in DPMarker test.
  auto Range1 = BInst->cloneDebugInfoFrom(CInst);
  EXPECT_EQ(BInst->DbgMarker->StoredDPValues.size(), 1u);
  DPValue *DPV2 = &*BInst->DbgMarker->StoredDPValues.begin();
  EXPECT_EQ(std::distance(Range1.begin(), Range1.end()), 1u);
  EXPECT_EQ(&*Range1.begin(), DPV2);
  EXPECT_NE(DPV1, DPV2);

  // We should be able to get a range over exactly the same information.
  auto Range2 = BInst->getDbgValueRange();
  EXPECT_EQ(Range1.begin(), Range2.begin());
  EXPECT_EQ(Range1.end(), Range2.end());

  // We should be able to query if there are DPValues,
  EXPECT_TRUE(BInst->hasDbgValues());
  EXPECT_TRUE(CInst->hasDbgValues());
  EXPECT_FALSE(DInst->hasDbgValues());

  // Dropping should be easy,
  BInst->dropDbgValues();
  EXPECT_FALSE(BInst->hasDbgValues());
  EXPECT_EQ(BInst->DbgMarker->StoredDPValues.size(), 0u);

  // And we should be able to drop individual DPValues.
  CInst->dropOneDbgValue(DPV1);
  EXPECT_FALSE(CInst->hasDbgValues());
  EXPECT_EQ(CInst->DbgMarker->StoredDPValues.size(), 0u);

  UseNewDbgInfoFormat = false;
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
  DPValue *DPVA, *DPVB, *DPVConst;

  void SetUp() override {
    UseNewDbgInfoFormat = true;
    M = parseIR(C, SpliceTestIR.c_str());
    M->convertToNewDbgValues();

    BBEntry = &M->getFunction("f")->getEntryBlock();
    BBExit = BBEntry->getNextNode();

    Dest = BBExit->begin();
    First = BBEntry->begin();
    Last = BBEntry->getTerminator()->getIterator();
    BInst = &*First;
    Branch = &*Last;
    CInst = &*Dest;

    DPVA = &*BInst->DbgMarker->StoredDPValues.begin();
    DPVB = &*Branch->DbgMarker->StoredDPValues.begin();
    DPVConst = &*CInst->DbgMarker->StoredDPValues.begin();
  }

  void TearDown() override { UseNewDbgInfoFormat = false; }

  bool InstContainsDPValue(Instruction *I, DPValue *DPV) {
    for (DPValue &D : I->getDbgValueRange()) {
      if (&D == DPV) {
        // Confirm too that the links between the records are correct.
        EXPECT_EQ(DPV->Marker, I->DbgMarker);
        EXPECT_EQ(I->DbgMarker->MarkedInstr, I);
        return true;
      }
    }
    return false;
  }

  bool CheckDPVOrder(Instruction *I, SmallVector<DPValue *> CheckVals) {
    SmallVector<DPValue *> Vals;
    for (DPValue &D : I->getDbgValueRange())
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
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, not including leading dbg.value, to Last, including the
    trailing dbg.value. Place at Dest, between the constant dbg.value and %c.
    %b, and the following dbg.value, should move, to:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Dest, in exit block.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVB));

  // DPVA, should have "fallen" onto the branch, remained in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVA));

  // DPVConst should be on the moved %b instruction.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVConst));
}

TEST_F(DbgSpliceTest, DbgSpliceTest1) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, not including leading dbg.value, to Last, including the
    trailing dbg.value. Place at the head of Dest, i.e. at the very start of
    BBExit, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVB));

  // DPVA, should have "fallen" onto the branch, remained in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVA));

  // DPVConst should be behind / after the moved instructions, remain on CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));

  // Order of DPVB and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(CInst, {DPVB, DPVConst}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest2) {
  Dest.setHeadBit(false);
  First.setHeadBit(true);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from head of First, which includes the leading dbg.value, to Last,
    including the trailing dbg.value. Place in front of Dest, but after any
    debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }


  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);

  // DPVB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVB));

  // DPVA, should have transferred with the spliced instructions, remains on
  // the "b" inst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVA));

  // DPVConst should be ahead of the moved instructions, ahead of BInst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVConst));

  // Order of DPVA and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(BInst, {DPVConst, DPVA}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest3) {
  Dest.setHeadBit(true);
  First.setHeadBit(true);
  Last.setTailBit(false);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from head of First, which includes the leading dbg.value, to Last,
    including the trailing dbg.value. Place at head of Dest, before any
    debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
Last      br label %exit, !dbg !11

BBExit  exit:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);

  // DPVB: should be on CInst, in exit block.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVB));

  // DPVA, should have transferred with the spliced instructions, remains on
  // the "b" inst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVA));

  // DPVConst should be behind the moved instructions, ahead of CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));

  // Order of DPVB and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(CInst, {DPVB, DPVConst}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest4) {
  Dest.setHeadBit(false);
  First.setHeadBit(false);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, not including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at Dest,
    after any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have remained in entry block, falls onto Branch inst.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVA));

  // DPVConst should be ahead of the moved instructions, BInst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVConst));

  // Order of DPVA and DPVA should be thus:
  EXPECT_TRUE(CheckDPVOrder(Branch, {DPVA, DPVB}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest5) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, not including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at head
    of Dest, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
First     %b = add i16 %a, 1, !dbg !11
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have remained in entry block, falls onto Branch inst.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVA));

  // DPVConst should be behind of the moved instructions, on CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));

  // Order of DPVA and DPVB should be thus:
  EXPECT_TRUE(CheckDPVOrder(Branch, {DPVA, DPVB}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest6) {
  Dest.setHeadBit(false);
  First.setHeadBit(true);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at Dest,
    after any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have transferred to BBExit, on B inst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVA));

  // DPVConst should be ahead of the moved instructions, on BInst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVConst));

  // Order of DPVA and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(BInst, {DPVConst, DPVA}));
}

TEST_F(DbgSpliceTest, DbgSpliceTest7) {
  Dest.setHeadBit(true);
  First.setHeadBit(true);
  Last.setTailBit(true);

  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from First, including the leading dbg.value, to Last, but NOT
    including the trailing dbg.value because the tail bit is set. Place at head
    of Dest, before any debug-info there. Becomes:

        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, First, Last);
  EXPECT_EQ(BInst->getParent(), BBExit);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have transferred to BBExit, on B inst.
  EXPECT_TRUE(InstContainsDPValue(BInst, DPVA));

  // DPVConst should be after of the moved instructions, on CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));
}

// But wait, there's more! What if you splice a range that is empty, but
// implicitly contains debug-info? In the dbg.value design for debug-info,
// this would be an explicit range, but in DPValue debug-info, it isn't.
// Check that if we try to do that, with differing head-bit values, that
// DPValues are transferred.
// Test with empty transfers to Dest, with head bit set and not set.

TEST_F(DbgSpliceTest, DbgSpliceEmpty0) {
  Dest.setHeadBit(false);
  First.setHeadBit(false);
  Last.setHeadBit(false);
  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from BBEntry.getFirstInsertionPt to First -- this implicitly is a
    splice of DPVA, but the iterators are pointing at the same instruction. The
    only difference is the setting of the head bit. Becomes;

        define i16 @f(i16 %a) !dbg !6 {
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, BBEntry->getFirstInsertionPt(), First);
  EXPECT_EQ(BInst->getParent(), BBEntry);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have transferred to BBExit, on C inst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVA));

  // DPVConst should be ahead of the moved DPValue, on CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));

  // Order of DPVA and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(CInst, {DPVConst, DPVA}));
}

TEST_F(DbgSpliceTest, DbgSpliceEmpty1) {
  Dest.setHeadBit(true);
  First.setHeadBit(false);
  Last.setHeadBit(false);
  /*
        define i16 @f(i16 %a) !dbg !6 {
BBEntry entry:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

    Splice from BBEntry.getFirstInsertionPt to First -- this implicitly is a
    splice of DPVA, but the iterators are pointing at the same instruction. The
    only difference is the setting of the head bit. Insert at head of Dest,
    i.e. before DPVConst. Becomes;

        define i16 @f(i16 %a) !dbg !6 {
First     %b = add i16 %a, 1, !dbg !11
DPVB      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
Last      br label %exit, !dbg !11

BBExit  exit:
DPVA      call void @llvm.dbg.value(metadata i16 %a, metadata !9, metadata !DIExpression()), !dbg !11
DPVConst  call void @llvm.dbg.value(metadata i16 0, metadata !9, metadata !DIExpression()), !dbg !11
Dest      %c = add i16 %b, 1, !dbg !11
          ret i16 0, !dbg !11
        }

  */
  BBExit->splice(Dest, BBEntry, BBEntry->getFirstInsertionPt(), First);
  EXPECT_EQ(BInst->getParent(), BBEntry);
  EXPECT_EQ(CInst->getParent(), BBExit);
  EXPECT_EQ(Branch->getParent(), BBEntry);

  // DPVB: should be on Branch as before, remain in entry block.
  EXPECT_TRUE(InstContainsDPValue(Branch, DPVB));

  // DPVA, should have transferred to BBExit, on C inst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVA));

  // DPVConst should be ahead of the moved DPValue, on CInst.
  EXPECT_TRUE(InstContainsDPValue(CInst, DPVConst));

  // Order of DPVA and DPVConst should be thus:
  EXPECT_TRUE(CheckDPVOrder(CInst, {DPVA, DPVConst}));
}

// If we splice new instructions into a block with trailing DPValues, then
// the trailing DPValues should get flushed back out.
TEST(BasicBlockDbgInfoTest, DbgSpliceTrailing) {
  LLVMContext C;
  UseNewDbgInfoFormat = true;

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
  M->convertToNewDbgValues();

  // Begin by forcing entry block to have dangling DPValue.
  Entry.getTerminator()->eraseFromParent();
  ASSERT_NE(Entry.getTrailingDPValues(), nullptr);
  EXPECT_TRUE(Entry.empty());

  // Now transfer the entire contents of the exit block into the entry.
  Entry.splice(Entry.end(), &Exit, Exit.begin(), Exit.end());

  // The trailing DPValue should have been placed at the front of what's been
  // spliced in.
  Instruction *BInst = &*Entry.begin();
  ASSERT_TRUE(BInst->DbgMarker);
  EXPECT_EQ(BInst->DbgMarker->StoredDPValues.size(), 1u);

  UseNewDbgInfoFormat = false;
}

} // End anonymous namespace.
#endif // EXPERIMENTAL_DEBUGINFO_ITERATORS
