; RUN: opt -S -passes=lower-constant-intrinsics < %s | FileCheck %s

; Check that LowerConstantIntrinsics's replaceConditionalBranchesOnConstant() correctly
; propagates the debug location from the old br instruction to the new one.

; Function Attrs: nounwind
define i32 @test_branch(i32 %in) !dbg !5 {
; CHECK-LABEL: define i32 @test_branch(
; CHECK:           br label %[[FALSE:.*]], !dbg [[DBG8:![0-9]+]]
; CHECK:       [[FALSE]]:
;
  %v = call i1 @llvm.is.constant.i32(i32 %in), !dbg !8
  br i1 %v, label %True, label %False, !dbg !9

True:                                             ; preds = %0
  %call1 = tail call i32 @subfun_1(), !dbg !10
  ret i32 %call1, !dbg !11

False:                                            ; preds = %0
  %call2 = tail call i32 @subfun_2(), !dbg !12
  ret i32 %call2, !dbg !13
}

declare i32 @subfun_1()
declare i32 @subfun_2()

declare i1 @llvm.is.constant.i32(i32)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG8]] = !DILocation(line: 2,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.ll", directory: "/")
!2 = !{i32 6}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_branch", linkageName: "test_branch", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
