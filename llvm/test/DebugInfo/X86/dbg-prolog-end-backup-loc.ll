; RUN: llc %s -o - | FileCheck %s

;; This test has had source-locations removed from the prologue, to simulate
;; heavily-optimised scenarios where a lot of debug-info gets dropped. Check
;; that we can pick a "worst-case" prologue_end position, of the first
;; instruction that does any meaningful computation (the add). It's better to
;; put the prologue_end flag here rather than deeper into the loop, past the
;; early-exit check.
;;
;; Generated from this code at -O2 -g in clang, with source locations then
;; deleted.
;;
;; int glob = 0;
;; int foo(int arg, int sum) {
;;   arg += sum;
;;   while (arg) {
;;     glob--;
;;     arg %= glob;
;;   }
;;   return 0;
;; }

; CHECK-LABEL: foo:
;; Scope-line location:
; CHECK:       .loc    0 2 0
;; Entry block:
; CHECK:        movl    %edi, %edx
; CHECK-NEXT:   .loc    0 2 0 prologue_end
; CHECK-NEXT:   addl    %esi, %edx
; CHECK-NEXT:   je      .LBB0_4
; CHECK-LABEL: # %bb.1:

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glob = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local noundef i32 @foo(i32 noundef %arg, i32 noundef %sum) local_unnamed_addr !dbg !9 {
entry:
  %add = add nsw i32 %sum, %arg
  %tobool.not4 = icmp eq i32 %add, 0
  br i1 %tobool.not4, label %while.end, label %while.body.preheader

while.body.preheader:
  %glob.promoted = load i32, ptr @glob, align 4
  br label %while.body, !dbg !14

while.body:
  %arg.addr.06 = phi i32 [ %rem, %while.body ], [ %add, %while.body.preheader ]
  %dec35 = phi i32 [ %dec, %while.body ], [ %glob.promoted, %while.body.preheader ]
  %dec = add nsw i32 %dec35, -1, !dbg !15
  %rem = srem i32 %arg.addr.06, %dec, !dbg !17
  %tobool.not = icmp eq i32 %rem, 0, !dbg !14
  br i1 %tobool.not, label %while.cond.while.end_crit_edge, label %while.body, !dbg !14

while.cond.while.end_crit_edge:
  store i32 %dec, ptr @glob, align 4, !dbg !15
  br label %while.end, !dbg !14

while.end:
  ret i32 0, !dbg !18
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang"}
!9 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{!5, !5, !5}
!12 = !{}
!13 = !DILocation(line: 3, column: 7, scope: !9)
!14 = !DILocation(line: 4, column: 3, scope: !9)
!15 = !DILocation(line: 5, column: 9, scope: !16)
!16 = distinct !DILexicalBlock(scope: !9, file: !3, line: 4, column: 15)
!17 = !DILocation(line: 6, column: 9, scope: !16)
!18 = !DILocation(line: 8, column: 3, scope: !9)
