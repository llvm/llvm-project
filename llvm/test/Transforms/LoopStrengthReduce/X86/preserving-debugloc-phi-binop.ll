; RUN: opt -S -passes=loop-reduce -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; Check that LoopStrengthReduce's OptimizeShadowIV() propagates the debug
; locations of the old phi (`%accum`) and binop (`%accum.next`) instruction
; to the new phi and binop instruction, respectively.

target datalayout = "n8:16:32:64"

define i32 @foobar6() !dbg !5 {
; CHECK-LABEL: define i32 @foobar6(
; CHECK:  loop:
; CHECK:    [[IV_S_:%.*]] = phi double [ -3.220000e+03, %[[ENTRY:.*]] ], [ [[IV_S_NEXT_:%.*]], %loop ], !dbg [[DBG9:![0-9]+]]
; CHECK:    [[IV_S_NEXT_]] = fadd double [[IV_S_]], 0x41624E65A0000000, !dbg [[DBG11:![0-9]+]]
; CHECK:  exit:
;
entry:
  br label %loop, !dbg !8

loop:                                             ; preds = %loop, %entry
  %accum = phi i32 [ -3220, %entry ], [ %accum.next, %loop ], !dbg !9
  %iv = phi i32 [ 12, %entry ], [ %iv.next, %loop ], !dbg !10
  %tmp1 = sitofp i32 %accum to double, !dbg !11
  tail call void @foo(double %tmp1), !dbg !12
  %accum.next = add nsw i32 %accum, 9597741, !dbg !13
  %iv.next = add nuw nsw i32 %iv, 1, !dbg !14
  %exitcond = icmp ugt i32 %iv, 235, !dbg !15
  br i1 %exitcond, label %exit, label %loop, !dbg !16

exit:                                             ; preds = %loop
  ret i32 %accum.next, !dbg !17
}

declare void @foo(double)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG9]] = !DILocation(line: 2,
; CHECK: [[DBG11]] = !DILocation(line: 6,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.ll", directory: "/")
!2 = !{i32 10}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foobar6", linkageName: "foobar6", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)
!16 = !DILocation(line: 9, column: 1, scope: !5)
!17 = !DILocation(line: 10, column: 1, scope: !5)
