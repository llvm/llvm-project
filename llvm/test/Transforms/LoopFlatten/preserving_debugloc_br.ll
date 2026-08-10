; RUN: opt -S -passes="loop(loop-flatten)" < %s | FileCheck %s

; Check that LoopFlatten's DoFlattenLoopPair() propagates the debug location of the
; original terminator to the new branch instruction.

define i32 @test1(i32 %val, ptr nocapture %A) !dbg !5 {
; CHECK-LABEL: define i32 @test1(
; CHECK-LABEL: for.body3:
; CHECK:         br label %for.inc6, !dbg [[DBG22:![0-9]+]]
; CHECK-LABEL: for.inc6:
;
entry:
  br label %for.body, !dbg !8

for.body:                                         ; preds = %for.inc6, %entry
  %i.018 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ], !dbg !9
  %mul = mul nuw nsw i32 %i.018, 20, !dbg !10
  br label %for.body3, !dbg !11

for.body3:                                        ; preds = %for.body3, %for.body
  %j.017 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ], !dbg !12
  %add = add nuw nsw i32 %j.017, %mul, !dbg !13
  %arrayidx = getelementptr inbounds i16, ptr %A, i32 %add, !dbg !14
  %0 = load i16, ptr %arrayidx, align 2, !dbg !15
  %conv16 = zext i16 %0 to i32, !dbg !16
  %add4 = add i32 %conv16, %val, !dbg !17
  %conv5 = trunc i32 %add4 to i16, !dbg !18
  store i16 %conv5, ptr %arrayidx, align 2, !dbg !19
  %inc = add nuw nsw i32 %j.017, 1, !dbg !20
  %exitcond = icmp ne i32 %inc, 20, !dbg !21
  br i1 %exitcond, label %for.body3, label %for.inc6, !dbg !22

for.inc6:                                         ; preds = %for.body3
  %inc7 = add nuw nsw i32 %i.018, 1, !dbg !23
  %exitcond19 = icmp ne i32 %inc7, 10, !dbg !24
  br i1 %exitcond19, label %for.body, label %for.end8, !dbg !25

for.end8:                                         ; preds = %for.inc6
  ret i32 10, !dbg !26
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG22]] = !DILocation(line: 15,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 19}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!24 = !DILocation(line: 17, column: 1, scope: !5)
!25 = !DILocation(line: 18, column: 1, scope: !5)
!26 = !DILocation(line: 19, column: 1, scope: !5)
