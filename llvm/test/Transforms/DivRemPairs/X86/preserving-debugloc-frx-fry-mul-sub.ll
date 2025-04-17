; RUN: opt -S -passes=div-rem-pairs -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; Check that DivRemPairs's optimizeDivRem() correctly propagates debug locations
; of `%div` and `%rem` to new freeze, mul and sub instructions.

define i128 @dont_hoist_urem(i128 %a, i128 %b) !dbg !5 {
; CHECK-LABEL: define i128 @dont_hoist_urem(
; CHECK-SAME: i128 [[A:%.*]], i128 [[B:%.*]])
; CHECK:       entry:
; CHECK-NEXT:    [[A_FROZEN:%.*]] = freeze i128 [[A]], !dbg [[DBG8:![0-9]+]]
; CHECK-NEXT:    [[B_FROZEN:%.*]] = freeze i128 [[B]], !dbg [[DBG8]]
; CHECK:       if:
; CHECK-NEXT:    [[TMP0:%.*]] = mul i128 [[DIV:%.*]], [[B_FROZEN]], !dbg [[DBG11:![0-9]+]]
; CHECK-NEXT:    [[REM_DECOMPOSED:%.*]] = sub i128 [[A_FROZEN]], [[TMP0:%.*]], !dbg [[DBG11]]
; CHECK:       end:
entry:
  %div = udiv i128 %a, %b, !dbg !8
  %cmp = icmp eq i128 %div, 42, !dbg !9
  br i1 %cmp, label %if, label %end, !dbg !10

if:                                               ; preds = %entry
  %rem = urem i128 %a, %b, !dbg !11
  br label %end, !dbg !12

end:                                              ; preds = %if, %entry
  %ret = phi i128 [ %rem, %if ], [ 3, %entry ], !dbg !13
  ret i128 %ret, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "frx_fry_.preserve.ll", directory: "/")
!2 = !{i32 7}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "dont_hoist_urem", linkageName: "dont_hoist_urem", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
; CHECK: [[DBG8]] = !DILocation(line: 1,
; CHECK: [[DBG11]] = !DILocation(line: 4,
