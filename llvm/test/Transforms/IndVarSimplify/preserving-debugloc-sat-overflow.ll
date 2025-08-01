; Test that the debug information is propagated correctly to the new instructions
; RUN: opt < %s -passes=indvars -S | FileCheck %s

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.uadd.sat.i32(i32, i32) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #0

define void @f_uadd(ptr %a) !dbg !5 {
; CHECK-LABEL: define void @f_uadd(
; CHECK:    [[TMP0:%.*]] = add nuw nsw i32 [[I_04:%.*]], 1, !dbg [[DBG14:![0-9]+]]
;
entry:
  br label %for.body, !dbg !8

for.cond.cleanup:                                 ; preds = %cont
  ret void, !dbg !9

for.body:                                         ; preds = %cont, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %2, %cont ], !dbg !10
  %idxprom = sext i32 %i.04 to i64, !dbg !11
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom, !dbg !12
  store i8 0, ptr %arrayidx, align 1, !dbg !13
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %i.04, i32 1), !dbg !14
  %1 = extractvalue { i32, i1 } %0, 1, !dbg !15
  br i1 %1, label %trap, label %cont, !dbg !16, !nosanitize !7

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !dbg !17, !nosanitize !7
  unreachable, !dbg !18, !nosanitize !7

cont:                                             ; preds = %for.body
  %2 = extractvalue { i32, i1 } %0, 0, !dbg !19
  %cmp = icmp slt i32 %2, 16, !dbg !20
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !21
}

define void @uadd_sat(ptr %p) !dbg !22 {
; CHECK-LABEL: define void @uadd_sat(
; CHECK:    [[SAT1:%.*]] = add nuw nsw i32 [[I:%.*]], 1, !dbg [[DBG23:![0-9]+]]
;
entry:
  br label %loop, !dbg !23

loop:                                             ; preds = %loop, %entry
  %i = phi i32 [ 0, %entry ], [ %i.inc, %loop ], !dbg !24
  %sat = call i32 @llvm.uadd.sat.i32(i32 %i, i32 1), !dbg !25
  store volatile i32 %sat, ptr %p, align 4, !dbg !26
  %i.inc = add nuw nsw i32 %i, 1, !dbg !27
  %cmp = icmp ne i32 %i.inc, 100, !dbg !28
  br i1 %cmp, label %loop, label %end, !dbg !29

end:                                              ; preds = %loop
  ret void, !dbg !30
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

;.
; CHECK: [[DBG14]] = !DILocation(line: 12,
; CHECK: [[DBG23]] = !DILocation(line: 17,
;.

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "eliminatesat.ll", directory: "/")
!2 = !{i32 22}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f_uadd", linkageName: "f_uadd", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!22 = distinct !DISubprogram(name: "uadd_sat", linkageName: "uadd_sat", scope: null, file: !1, line: 15, type: !6, scopeLine: 15, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILocation(line: 15, column: 1, scope: !22)
!24 = !DILocation(line: 16, column: 1, scope: !22)
!25 = !DILocation(line: 17, column: 1, scope: !22)
!26 = !DILocation(line: 18, column: 1, scope: !22)
!27 = !DILocation(line: 19, column: 1, scope: !22)
!28 = !DILocation(line: 20, column: 1, scope: !22)
!29 = !DILocation(line: 21, column: 1, scope: !22)
!30 = !DILocation(line: 22, column: 1, scope: !22)
