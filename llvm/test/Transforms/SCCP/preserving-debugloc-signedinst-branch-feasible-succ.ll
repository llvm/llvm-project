; Test that the debug information is propagated correctly to the new instructions
; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

define double @sdiv_ashr_sitofp_dbg_pres(i7 %y) !dbg !5 {
; CHECK-LABEL: define double @sdiv_ashr_sitofp_dbg_pres(
; CHECK:    [[SDIV:%.*]] = udiv i8 42, [[ZEXT1:%.*]], !dbg [[DBG9:![0-9]+]]
; CHECK:    [[ASHR:%.*]] = lshr i8 42, [[SDIV]], !dbg [[DBG10:![0-9]+]]
; CHECK:    [[SITOFP:%.*]] = uitofp nneg i16 [[ZEXT2:%.*]] to double, !dbg [[DBG12:![0-9]+]]
;
  %zext1 = zext i7 %y to i8, !dbg !8
  %sdiv = sdiv i8 42, %zext1, !dbg !9
  %ashr = ashr i8 42, %sdiv, !dbg !10
  %zext2 = zext i8 %ashr to i16, !dbg !11
  %sitofp = sitofp i16 %zext2 to double, !dbg !12
  ret double %sitofp, !dbg !13
}

define i32 @test_duplicate_successors_phi(i1 %c, i32 %x) !dbg !14 {
; CHECK-LABEL: define i32 @test_duplicate_successors_phi(
; CHECK:       switch:
; CHECK-NEXT:    br label %[[SWITCH_DEFAULT:.*]], !dbg [[DBG16:![0-9]+]]
;
entry:
  br i1 %c, label %switch, label %end, !dbg !15

switch:                                           ; preds = %entry
  switch i32 -1, label %switch.default [
  i32 0, label %end
  i32 1, label %end
  ], !dbg !16

switch.default:                                   ; preds = %switch
  ret i32 -1, !dbg !17

end:                                              ; preds = %switch, %switch, %entry
  %phi = phi i32 [ %x, %entry ], [ 1, %switch ], [ 1, %switch ], !dbg !18
  ret i32 %phi, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

;.
; CHECK: [[DBG9]] = !DILocation(line: 2
; CHECK: [[DBG10]] = !DILocation(line: 3
; CHECK: [[DBG12]] = !DILocation(line: 5
; CHECK: [[DBG16]] = !DILocation(line: 8
;.


!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "sccp.ll", directory: "/")
!2 = !{i32 11}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "sdiv_ashr_sitofp_dbg_pres", linkageName: "sdiv_ashr_sitofp_dbg_pres", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = distinct !DISubprogram(name: "test_duplicate_successors_phi", linkageName: "test_duplicate_successors_phi", scope: null, file: !1, line: 7, type: !6, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 7, column: 1, scope: !14)
!16 = !DILocation(line: 8, column: 1, scope: !14)
!17 = !DILocation(line: 9, column: 1, scope: !14)
!18 = !DILocation(line: 10, column: 1, scope: !14)
!19 = !DILocation(line: 11, column: 1, scope: !14)
