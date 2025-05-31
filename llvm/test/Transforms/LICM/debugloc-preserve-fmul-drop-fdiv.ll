; RUN: opt -passes=licm -verify-memoryssa -S < %s | FileCheck %s

; JumpThreading's hoistRegion() replaces the `fdiv` (%v6), of which the second
; operand (%v) is a loop invariant, with a loop invariant `fdiv` and a `fmul`.
; This test checks that the debug location propagates to the new `fmul` from
; the old `fdiv` it replaces in block `loop` and the debug location drop of new
; `fdiv`, which is hoisted to block `entry` after being created.

define zeroext i1 @invariant_denom(double %v) !dbg !5 {
; CHECK:       entry:
; CHECK-NEXT:    [[TMP0:%.*]] = fdiv fast double 1.000000e+00, [[V:%.*]]{{$}}
; CHECK:       loop:
; CHECK:         [[TMP1:%.*]] = fmul fast double {{.*}}, !dbg [[DBG12:![0-9]+]]
; CHECK:       [[DBG12]] = !DILocation(line: 5,
;
entry:
  br label %loop, !dbg !8

loop:                                             ; preds = %loop, %entry
  %v3 = phi i32 [ 0, %entry ], [ %v11, %loop ], !dbg !9
  %v4 = phi i32 [ 0, %entry ], [ %v12, %loop ], !dbg !10
  %v5 = uitofp i32 %v4 to double, !dbg !11
  %v6 = fdiv fast double %v5, %v, !dbg !12
  %v7 = fptoui double %v6 to i64, !dbg !13
  %v8 = and i64 %v7, 1, !dbg !14
  %v9 = xor i64 %v8, 1, !dbg !15
  %v10 = trunc i64 %v9 to i32, !dbg !16
  %v11 = add i32 %v10, %v3, !dbg !17
  %v12 = add nuw i32 %v4, 1, !dbg !18
  %v13 = icmp eq i32 %v12, -1, !dbg !19
  br i1 %v13, label %end, label %loop, !dbg !20

end:                                              ; preds = %loop
  %v15 = phi i32 [ %v11, %loop ], !dbg !21
  %v16 = icmp ne i32 %v15, 0, !dbg !22
  ret i1 %v16, !dbg !23
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 16}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "invariant_denom", linkageName: "invariant_denom", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!18 = !DILocation(line: 11, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
!20 = !DILocation(line: 13, column: 1, scope: !5)
!21 = !DILocation(line: 14, column: 1, scope: !5)
!22 = !DILocation(line: 15, column: 1, scope: !5)
!23 = !DILocation(line: 16, column: 1, scope: !5)
