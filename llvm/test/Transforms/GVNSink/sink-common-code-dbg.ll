; RUN: opt < %s -passes=gvn-sink -S | FileCheck %s

; Test that GVNSink correctly merges the debug locations of sinked instruction

define zeroext i1 @test18(i32 %flag, i32 %blksA, i32 %blksB, i32 %nblks) !dbg !5 {
; CHECK:       if.end:
; CHECK-NEXT:    [[CMP2_SINK:%.*]] = phi i1 [ %cmp2, %if.then2 ], [ %cmp, %if.then ], [ %cmp3, %if.then3 ]
; CHECK-NEXT:    [[FROMBOOL4:%.*]] = zext i1 [[CMP2_SINK]] to i8, !dbg [[DBG17:![0-9]+]]
;
entry:
  switch i32 %flag, label %if.then3 [
  i32 0, label %if.then
  i32 1, label %if.then2
  ], !dbg !8

if.then:                                          ; preds = %entry
  %cmp = icmp uge i32 %blksA, %nblks, !dbg !9
  %frombool1 = zext i1 %cmp to i8, !dbg !10
  br label %if.end, !dbg !11

if.then2:                                         ; preds = %entry
  %add = add i32 %nblks, %blksB, !dbg !12
  %cmp2 = icmp ule i32 %add, %blksA, !dbg !13
  %frombool3 = zext i1 %cmp2 to i8, !dbg !14
  br label %if.end, !dbg !15

if.then3:                                         ; preds = %entry
  %add2 = add i32 %nblks, %blksA, !dbg !16
  %cmp3 = icmp ule i32 %add2, %blksA, !dbg !17
  %frombool4 = zext i1 %cmp3 to i8, !dbg !18
  br label %if.end, !dbg !19

if.end:                                           ; preds = %if.then3, %if.then2, %if.then
  %obeys.0 = phi i8 [ %frombool1, %if.then ], [ %frombool3, %if.then2 ], [ %frombool4, %if.then3 ], !dbg !20
  %tobool4 = icmp ne i8 %obeys.0, 0, !dbg !21
  ret i1 %tobool4, !dbg !22
}

define zeroext i1 @test_pr30244(i1 zeroext %flag, i1 zeroext %flag2, i32 %blksA, i32 %blksB, i32 %nblks) !dbg !23 {
; CHECK:       if.end.gvnsink.split:
; CHECK-NEXT:    [[CMP2_SINK:%.*]] = phi i1 [ %cmp2, %if.then2 ], [ %cmp, %if.then ]
; CHECK-NEXT:    [[FROMBOOL1:%.*]] = zext i1 [[CMP2_SINK]] to i8, !dbg [[DBG29:![0-9]+]]
;
entry:
  %p = alloca i8, align 1, !dbg !24
  br i1 %flag, label %if.then, label %if.else, !dbg !25

if.then:                                          ; preds = %entry
  %cmp = icmp uge i32 %blksA, %nblks, !dbg !26
  %frombool1 = zext i1 %cmp to i8, !dbg !27
  store i8 %frombool1, ptr %p, align 1, !dbg !28
  br label %if.end, !dbg !29

if.else:                                          ; preds = %entry
  br i1 %flag2, label %if.then2, label %if.end, !dbg !30

if.then2:                                         ; preds = %if.else
  %add = add i32 %nblks, %blksB, !dbg !31
  %cmp2 = icmp ule i32 %add, %blksA, !dbg !32
  %frombool3 = zext i1 %cmp2 to i8, !dbg !33
  store i8 %frombool3, ptr %p, align 1, !dbg !34
  br label %if.end, !dbg !35

if.end:                                           ; preds = %if.then2, %if.else, %if.then
  ret i1 true, !dbg !36
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "sink-common-code.ll", directory: "/")
!2 = !{i32 28}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test18", linkageName: "test18", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!23 = distinct !DISubprogram(name: "test_pr30244", linkageName: "test_pr30244", scope: null, file: !1, line: 16, type: !6, scopeLine: 16, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 16, column: 1, scope: !23)
!25 = !DILocation(line: 17, column: 1, scope: !23)
!26 = !DILocation(line: 18, column: 1, scope: !23)
!27 = !DILocation(line: 19, column: 1, scope: !23)
!28 = !DILocation(line: 20, column: 1, scope: !23)
!29 = !DILocation(line: 21, column: 1, scope: !23)
!30 = !DILocation(line: 22, column: 1, scope: !23)
!31 = !DILocation(line: 23, column: 1, scope: !23)
!32 = !DILocation(line: 24, column: 1, scope: !23)
!33 = !DILocation(line: 25, column: 1, scope: !23)
!34 = !DILocation(line: 26, column: 1, scope: !23)
!35 = !DILocation(line: 27, column: 1, scope: !23)
!36 = !DILocation(line: 28, column: 1, scope: !23)
;.
; CHECK: [[DBG17]] = !DILocation(line: 0
; CHECK: [[DBG29]] = !DILocation(line: 0
;.
