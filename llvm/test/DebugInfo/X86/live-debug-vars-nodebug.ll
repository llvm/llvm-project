; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:   | FileCheck %s --check-prefix=NORMAL-INPUT

; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-before=finalize-isel \
; RUN:   -experimental-debug-variable-locations \
; RUN:   | FileCheck %s --check-prefix=EXPER-INPUT

; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-after=livedebugvars \
; RUN:   | FileCheck %s --check-prefix=OUTPUT

; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-after=livedebugvars \
; RUN:   -experimental-debug-variable-locations \
; RUN:   | FileCheck %s --check-prefix=OUTPUT

; This test checks that LiveDebugVariables strips all debug instructions
; from nodebug functions. Such instructions occur when a function with debug
; info is inlined into a nodebug function.

; The test first verifies that DBG_VALUE/DBG_LABEL instructions are present in
; the input to LiveDebugVariables. It then verifies that after the pass is ran
; no debug instructions are present.

; When -experimental-debug-variable-locations is enabled, certain variable
; locations are represented by DBG_INSTR_REF instead of DBG_VALUE. The test
; verifies that a DBG_INSTR_REF is emitted by the option, and that it is also
; stripped.

; Generated from:
;
; extern int foobar();
; 
; int bar(int a) {
;   int b = 10;
;   b += foobar();
; label:
;   if (a) goto label;
;   return b;
; }
; 
; __attribute__((nodebug))
; int foo(int a) {
;   return bar(a);
; }

; NORMAL-INPUT-DAG: DBG_VALUE
; NORMAL-INPUT-DAG: DBG_LABEL

; EXPER-INPUT-DAG: DBG_INSTR_REF
; EXPER-INPUT-DAG: DBG_LABEL

; OUTPUT-NOT: DBG_VALUE
; OUTPUT-NOT: DBG_INSTR_REF
; OUTPUT-NOT: DBG_LABEL

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @_Z3fooi(i32 %a) local_unnamed_addr #0 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 10, metadata !13, metadata !DIExpression()), !dbg !15
  %call.i = tail call i32 @_Z6foobarv()
  call void @llvm.dbg.value(metadata i32 undef, metadata !13, metadata !DIExpression()), !dbg !15
  %tobool.not.i = icmp eq i32 %a, 0
  br i1 %tobool.not.i, label %_Z3bari.exit, label %label.i

label.i:                                          ; preds = %entry, %label.i
  call void @llvm.dbg.label(metadata !14), !dbg !18
  br label %label.i

_Z3bari.exit:                                     ; preds = %entry
  %add.i = add nsw i32 %call.i, 10
  call void @llvm.dbg.value(metadata i32 %add.i, metadata !13, metadata !DIExpression()), !dbg !15
  ret i32 %add.i
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2
declare dso_local i32 @_Z6foobarv() local_unnamed_addr #1
declare void @llvm.dbg.label(metadata) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "f.cpp", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 3, type: !10)
!13 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 4, type: !10)
!14 = !DILabel(scope: !7, name: "label", file: !1, line: 6)
!15 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 6, column: 1, scope: !7)
