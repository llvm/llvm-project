//===- DebugSSAUpdater.cpp - Unit tests for debug variable tracking -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DebugSSAUpdater.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("DebugSSAUpdaterTests", errs());
  return Mod;
}

namespace {

// Verify that two conflicting live-in values result in no live-in range for a
// block.
TEST(DebugSSAUpdater, EmptyPHIRange) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C,
              R"(define i32 @foo(i32 %a, i1 %b) !dbg !7 {
entry:
    #dbg_value(i32 %a, !6, !DIExpression(), !10)
  br i1 %b, label %if.then, label %if.else, !dbg !11

if.then:
  %c = add i32 %a, 10, !dbg !12
    #dbg_value(i32 %c, !6, !DIExpression(), !13)
  br label %exit, !dbg !14

if.else:
  %d = mul i32 %a, 3, !dbg !15
    #dbg_value(i32 %d, !6, !DIExpression(), !16)
  br label %exit, !dbg !17

exit:
  %res = phi i32 [ %c, %if.then ], [ %d, %if.else ], !dbg !18
  ret i32 %res, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 20.0.0")
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 20.0.0"}
!6 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 11, type: !8)
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 10, type: !9, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 10, scope: !7)
!11 = !DILocation(line: 11, scope: !7)
!12 = !DILocation(line: 12, scope: !7)
!13 = !DILocation(line: 13, scope: !7)
!14 = !DILocation(line: 14, scope: !7)
!15 = !DILocation(line: 15, scope: !7)
!16 = !DILocation(line: 16, scope: !7)
!17 = !DILocation(line: 17, scope: !7)
!18 = !DILocation(line: 18, scope: !7)
!19 = !DILocation(line: 19, scope: !7)
)");

  Function *Foo = &*M->begin();
  DebugVariableAggregate VarA(cast<DbgVariableRecord>(
      Foo->begin()->begin()->getDbgRecordRange().begin()));
  DbgValueRangeTable DbgValueRanges;
  DbgValueRanges.addVariable(Foo, VarA);
  BasicBlock *ExitBlock = &Foo->back();
  // We should have 5 ranges: 1 in the entry block, and 2 in each `if` block,
  // while there should be no range for the exit block.
  EXPECT_EQ(DbgValueRanges.getVariableRanges(VarA).size(), 5u);
  EXPECT_TRUE(none_of(DbgValueRanges.getVariableRanges(VarA),
                      [&](DbgRangeEntry VarRange) {
                        return VarRange.Start->getParent() == ExitBlock;
                      }));
}

// Verify that we correctly set live-in variable values through loops.
TEST(DebugSSAUpdater, LoopPHI) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C,
              R"(define i32 @foo(i32 %a, i32 %max) !dbg !7 {
entry:
    #dbg_value(i32 %a, !6, !DIExpression(), !10)
  %cond.entry = icmp slt i32 %a, %max, !dbg !11
  br i1 %cond.entry, label %loop, label %exit, !dbg !12

loop:
  %loop.a = phi i32 [ %a, %entry ], [ %inc, %loop ]
  %inc = add i32 %loop.a, 1, !dbg !13
  %cond.loop = icmp slt i32 %inc, %max, !dbg !14
  br i1 %cond.loop, label %loop, label %exit, !dbg !15

exit:
  %res = phi i32 [ %a, %entry ], [ %loop.a, %loop ]
  ret i32 %res, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 20.0.0")
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 20.0.0"}
!6 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 11, type: !8)
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 10, type: !9, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 10, scope: !7)
!11 = !DILocation(line: 11, scope: !7)
!12 = !DILocation(line: 12, scope: !7)
!13 = !DILocation(line: 13, scope: !7)
!14 = !DILocation(line: 14, scope: !7)
!15 = !DILocation(line: 15, scope: !7)
!16 = !DILocation(line: 16, scope: !7)
)");

  Function *Foo = &*M->begin();
  DebugVariableAggregate VarA(cast<DbgVariableRecord>(
      Foo->begin()->begin()->getDbgRecordRange().begin()));
  DbgValueRangeTable DbgValueRanges;
  DbgValueRanges.addVariable(Foo, VarA);
  // We should have 3 ranges: 1 in the entry block, and 1 live-in entry for each
  // of the loops.
  EXPECT_EQ(DbgValueRanges.getVariableRanges(VarA).size(), 3u);
  EXPECT_TRUE(
      all_of(DbgValueRanges.getVariableRanges(VarA),
             [&](DbgRangeEntry VarRange) { return !VarRange.Value.IsUndef; }));
}

// Verify that when a variable has only undef debug values, it has no live
// ranges.
TEST(DebugSSAUpdater, AllUndefVar) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C,
              R"(define i32 @foo(i32 %a, i1 %b) !dbg !7 {
entry:
    #dbg_value(i32 poison, !6, !DIExpression(), !10)
  br i1 %b, label %if.then, label %if.else, !dbg !11

if.then:
  %c = add i32 %a, 10, !dbg !12
    #dbg_value(i32 poison, !6, !DIExpression(), !13)
  br label %exit, !dbg !14

if.else:
  %d = mul i32 %a, 3, !dbg !15
    #dbg_value(i32 poison, !6, !DIExpression(), !16)
  br label %exit, !dbg !17

exit:
  %res = phi i32 [ %c, %if.then ], [ %d, %if.else ], !dbg !18
  ret i32 %res, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 20.0.0")
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 20.0.0"}
!6 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 11, type: !8)
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 10, type: !9, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 10, scope: !7)
!11 = !DILocation(line: 11, scope: !7)
!12 = !DILocation(line: 12, scope: !7)
!13 = !DILocation(line: 13, scope: !7)
!14 = !DILocation(line: 14, scope: !7)
!15 = !DILocation(line: 15, scope: !7)
!16 = !DILocation(line: 16, scope: !7)
!17 = !DILocation(line: 17, scope: !7)
!18 = !DILocation(line: 18, scope: !7)
!19 = !DILocation(line: 19, scope: !7)
)");

  Function *Foo = &*M->begin();
  DebugVariableAggregate VarA(cast<DbgVariableRecord>(
      Foo->begin()->begin()->getDbgRecordRange().begin()));
  DbgValueRangeTable DbgValueRanges;
  DbgValueRanges.addVariable(Foo, VarA);
  // There should be no variable ranges emitted for a variable that has only
  // undef dbg_values.
  EXPECT_EQ(DbgValueRanges.getVariableRanges(VarA).size(), 0u);
}
} // namespace
