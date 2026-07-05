; RUN: llc --stop-after=virtregrewriter -experimental-debug-variable-locations=false -o - %s | FileCheck %s

; Check that any debug value with 64+ unique machine location operands is set
; undef by LiveDebugVariables.
; This test uses constant i32 values in the DIArgList instead of SSA values, as
; it allows the test case to be much simpler. If at some point we are able to
; simplify such constant expressions, this test will fail; if that happens,
; there is another version of this test that uses SSA values at:
; https://reviews.llvm.org/D101373?id=340953

; CHECK: ![[VAR_A:[0-9]+]] = !DILocalVariable(name: "a"

; CHECK-NOT: DBG_VALUE
; CHECK: DBG_VALUE_LIST ![[VAR_A]], !DIExpression(DW_OP_LLVM_arg, 0), $noreg
; CHECK-NOT: DBG_VALUE

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @_Z3fooi(i32 returned %b) local_unnamed_addr !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata !DIArgList(i32 1, i32 126, i32 124, i32 122, i32 120, i32 118, i32 116, i32 114, i32 112, i32 110, i32 108, i32 106, i32 104, i32 102, i32 100, i32 98, i32 96, i32 94, i32 92, i32 90, i32 88, i32 86, i32 84, i32 82, i32 80, i32 78, i32 76, i32 74, i32 72, i32 70, i32 68, i32 66, i32 64, i32 62, i32 60, i32 58, i32 56, i32 54, i32 52, i32 50, i32 48, i32 46, i32 44, i32 42, i32 40, i32 38, i32 36, i32 34, i32 32, i32 30, i32 28, i32 26, i32 24, i32 22, i32 20, i32 18, i32 16, i32 14, i32 12, i32 10, i32 8, i32 6, i32 4, i32 2), metadata !14, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 63, DW_OP_plus, DW_OP_LLVM_arg, 62, DW_OP_plus, DW_OP_LLVM_arg, 61, DW_OP_plus, DW_OP_LLVM_arg, 60, DW_OP_plus, DW_OP_LLVM_arg, 59, DW_OP_plus, DW_OP_LLVM_arg, 58, DW_OP_plus, DW_OP_LLVM_arg, 57, DW_OP_plus, DW_OP_LLVM_arg, 56, DW_OP_plus, DW_OP_LLVM_arg, 55, DW_OP_plus, DW_OP_LLVM_arg, 54, DW_OP_plus, DW_OP_LLVM_arg, 53, DW_OP_plus, DW_OP_LLVM_arg, 52, DW_OP_plus, DW_OP_LLVM_arg, 51, DW_OP_plus, DW_OP_LLVM_arg, 50, DW_OP_plus, DW_OP_LLVM_arg, 49, DW_OP_plus, DW_OP_LLVM_arg, 48, DW_OP_plus, DW_OP_LLVM_arg, 47, DW_OP_plus, DW_OP_LLVM_arg, 46, DW_OP_plus, DW_OP_LLVM_arg, 45, DW_OP_plus, DW_OP_LLVM_arg, 44, DW_OP_plus, DW_OP_LLVM_arg, 43, DW_OP_plus, DW_OP_LLVM_arg, 42, DW_OP_plus, DW_OP_LLVM_arg, 41, DW_OP_plus, DW_OP_LLVM_arg, 40, DW_OP_plus, DW_OP_LLVM_arg, 39, DW_OP_plus, DW_OP_LLVM_arg, 38, DW_OP_plus, DW_OP_LLVM_arg, 37, DW_OP_plus, DW_OP_LLVM_arg, 36, DW_OP_plus, DW_OP_LLVM_arg, 35, DW_OP_plus, DW_OP_LLVM_arg, 34, DW_OP_plus, DW_OP_LLVM_arg, 33, DW_OP_plus, DW_OP_LLVM_arg, 32, DW_OP_plus, DW_OP_LLVM_arg, 31, DW_OP_plus, DW_OP_LLVM_arg, 30, DW_OP_plus, DW_OP_LLVM_arg, 29, DW_OP_plus, DW_OP_LLVM_arg, 28, DW_OP_plus, DW_OP_LLVM_arg, 27, DW_OP_plus, DW_OP_LLVM_arg, 26, DW_OP_plus, DW_OP_LLVM_arg, 25, DW_OP_plus, DW_OP_LLVM_arg, 24, DW_OP_plus, DW_OP_LLVM_arg, 23, DW_OP_plus, DW_OP_LLVM_arg, 22, DW_OP_plus, DW_OP_LLVM_arg, 21, DW_OP_plus, DW_OP_LLVM_arg, 20, DW_OP_plus, DW_OP_LLVM_arg, 19, DW_OP_plus, DW_OP_LLVM_arg, 18, DW_OP_plus, DW_OP_LLVM_arg, 17, DW_OP_plus, DW_OP_LLVM_arg, 16, DW_OP_plus, DW_OP_LLVM_arg, 15, DW_OP_plus, DW_OP_LLVM_arg, 14, DW_OP_plus, DW_OP_LLVM_arg, 13, DW_OP_plus, DW_OP_LLVM_arg, 12, DW_OP_plus, DW_OP_LLVM_arg, 11, DW_OP_plus, DW_OP_LLVM_arg, 10, DW_OP_plus, DW_OP_LLVM_arg, 9, DW_OP_plus, DW_OP_LLVM_arg, 8, DW_OP_plus, DW_OP_LLVM_arg, 7, DW_OP_plus, DW_OP_LLVM_arg, 6, DW_OP_plus, DW_OP_LLVM_arg, 5, DW_OP_plus, DW_OP_LLVM_arg, 4, DW_OP_plus, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !15
  ret i32 %b, !dbg !16
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "live-debug-vars-loc-limit.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang version 13.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "b", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 2, type: !11)
!15 = !DILocation(line: 0, scope: !8)
!16 = !DILocation(line: 3, column: 3, scope: !8)
