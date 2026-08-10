; Test that the debug information is propagated correctly to the new instructions
; RUN: opt < %s -passes=indvars -S | FileCheck %s

define void @test_srem_urem(ptr %a) !dbg !5 {
; CHECK-LABEL: define void @test_srem_urem(
; CHECK:    [[REM_UREM:%.*]] = urem i32 [[I_01:%.*]], 2, !dbg [[DBG10:![0-9]+]]
;
entry:
  br label %for.body, !dbg !8

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !9
  %rem = srem i32 %i.01, 2, !dbg !10
  %idxprom = sext i32 %rem to i64, !dbg !11
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom, !dbg !12
  store i32 %i.01, ptr %arrayidx, align 4, !dbg !13
  %inc = add nsw i32 %i.01, 1, !dbg !14
  %cmp = icmp slt i32 %inc, 64, !dbg !15
  br i1 %cmp, label %for.body, label %for.end, !dbg !16

for.end:                                          ; preds = %for.body
  ret void, !dbg !17
}

define void @test_sdiv_udiv(ptr %a) !dbg !18 {
; CHECK-LABEL: define void @test_sdiv_udiv(
; CHECK:    [[DIV_UDIV:%.*]] = udiv i32 [[I_01:%.*]], 2, !dbg [[DBG21:![0-9]+]]
;
entry:
  br label %for.body, !dbg !19

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !20
  %div = sdiv i32 %i.01, 2, !dbg !21
  %idxprom = sext i32 %div to i64, !dbg !22
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom, !dbg !23
  store i32 %i.01, ptr %arrayidx, align 4, !dbg !24
  %inc = add nsw i32 %i.01, 1, !dbg !25
  %cmp = icmp slt i32 %inc, 64, !dbg !26
  br i1 %cmp, label %for.body, label %for.end, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !28
}

; Function Attrs: nounwind
define i32 @test_rem_num_zero(i64 %arg1) #0 !dbg !29 {
; CHECK-LABEL: define i32 @test_rem_num_zero(
; CHECK:    [[TMP0:%.*]] = icmp eq i64 [[T12:%.*]],  %arg1
; CHECK:    [[IV_REM:%.*]] = select i1 [[TMP0]], i64 0, i64 [[T12]], !dbg [[DBG36:![0-9]+]]
;
bb:
  %t = icmp sgt i64 %arg1, 0, !dbg !30
  br i1 %t, label %bb4, label %bb54, !dbg !31

bb4:                                              ; preds = %bb
  br label %bb5, !dbg !32

bb5:                                              ; preds = %bb49, %bb4
  %t6 = phi i64 [ %t51, %bb49 ], [ 0, %bb4 ], !dbg !33
  %t7 = phi i32 [ %t50, %bb49 ], [ 0, %bb4 ], !dbg !34
  %t12 = add nsw i64 %t6, 1, !dbg !35
  %t13 = srem i64 %t12, %arg1, !dbg !36
  %t14 = icmp sgt i64 %arg1, 0, !dbg !37
  br i1 %t14, label %bb15, label %bb49, !dbg !38

bb15:                                             ; preds = %bb5
  br label %bb16, !dbg !39

bb16:                                             ; preds = %bb44, %bb15
  %t17 = phi i64 [ %t46, %bb44 ], [ 0, %bb15 ], !dbg !40
  %t18 = phi i32 [ %t45, %bb44 ], [ %t7, %bb15 ], !dbg !41
  %t19 = icmp sgt i64 %arg1, 0, !dbg !42
  br i1 %t19, label %bb20, label %bb44, !dbg !43

bb20:                                             ; preds = %bb16
  br label %bb21, !dbg !44

bb21:                                             ; preds = %bb21, %bb20
  %t25 = mul i64 %t13, %arg1, !dbg !45
  %t42 = icmp slt i64 %t25, %arg1, !dbg !46
  br i1 %t42, label %bb21, label %bb43, !dbg !47

bb43:                                             ; preds = %bb21
  br label %bb44, !dbg !48

bb44:                                             ; preds = %bb43, %bb16
  %t45 = phi i32 [ %t18, %bb16 ], [ 0, %bb43 ], !dbg !49
  %t46 = add nsw i64 %t17, 1, !dbg !50
  %t47 = icmp slt i64 %t46, %arg1, !dbg !51
  br i1 %t47, label %bb16, label %bb48, !dbg !52

bb48:                                             ; preds = %bb44
  br label %bb49, !dbg !53

bb49:                                             ; preds = %bb48, %bb5
  %t50 = phi i32 [ %t7, %bb5 ], [ %t45, %bb48 ], !dbg !54
  %t51 = add nsw i64 %t6, 1, !dbg !55
  %t52 = icmp slt i64 %t51, %arg1, !dbg !56
  br i1 %t52, label %bb5, label %bb53, !dbg !57

bb53:                                             ; preds = %bb49
  br label %bb54, !dbg !58

bb54:                                             ; preds = %bb53, %bb
  %t55 = phi i32 [ 0, %bb ], [ %t50, %bb53 ], !dbg !59
  ret i32 %t55, !dbg !60
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

;.
; CHECK: [[DBG10]] = !DILocation(line: 3,
; CHECK: [[DBG21]] = !DILocation(line: 13,
; CHECK: [[DBG36]] = !DILocation(line: 27,
;.

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "indvars.ll", directory: "/")
!2 = !{i32 51}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_srem_urem", linkageName: "test_srem_urem", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!18 = distinct !DISubprogram(name: "test_sdiv_udiv", linkageName: "test_sdiv_udiv", scope: null, file: !1, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = !DILocation(line: 11, column: 1, scope: !18)
!20 = !DILocation(line: 12, column: 1, scope: !18)
!21 = !DILocation(line: 13, column: 1, scope: !18)
!22 = !DILocation(line: 14, column: 1, scope: !18)
!23 = !DILocation(line: 15, column: 1, scope: !18)
!24 = !DILocation(line: 16, column: 1, scope: !18)
!25 = !DILocation(line: 17, column: 1, scope: !18)
!26 = !DILocation(line: 18, column: 1, scope: !18)
!27 = !DILocation(line: 19, column: 1, scope: !18)
!28 = !DILocation(line: 20, column: 1, scope: !18)
!29 = distinct !DISubprogram(name: "test_rem_num_zero", linkageName: "test_rem_num_zero", scope: null, file: !1, line: 21, type: !6, scopeLine: 21, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!30 = !DILocation(line: 21, column: 1, scope: !29)
!31 = !DILocation(line: 22, column: 1, scope: !29)
!32 = !DILocation(line: 23, column: 1, scope: !29)
!33 = !DILocation(line: 24, column: 1, scope: !29)
!34 = !DILocation(line: 25, column: 1, scope: !29)
!35 = !DILocation(line: 26, column: 1, scope: !29)
!36 = !DILocation(line: 27, column: 1, scope: !29)
!37 = !DILocation(line: 28, column: 1, scope: !29)
!38 = !DILocation(line: 29, column: 1, scope: !29)
!39 = !DILocation(line: 30, column: 1, scope: !29)
!40 = !DILocation(line: 31, column: 1, scope: !29)
!41 = !DILocation(line: 32, column: 1, scope: !29)
!42 = !DILocation(line: 33, column: 1, scope: !29)
!43 = !DILocation(line: 34, column: 1, scope: !29)
!44 = !DILocation(line: 35, column: 1, scope: !29)
!45 = !DILocation(line: 36, column: 1, scope: !29)
!46 = !DILocation(line: 37, column: 1, scope: !29)
!47 = !DILocation(line: 38, column: 1, scope: !29)
!48 = !DILocation(line: 39, column: 1, scope: !29)
!49 = !DILocation(line: 40, column: 1, scope: !29)
!50 = !DILocation(line: 41, column: 1, scope: !29)
!51 = !DILocation(line: 42, column: 1, scope: !29)
!52 = !DILocation(line: 43, column: 1, scope: !29)
!53 = !DILocation(line: 44, column: 1, scope: !29)
!54 = !DILocation(line: 45, column: 1, scope: !29)
!55 = !DILocation(line: 46, column: 1, scope: !29)
!56 = !DILocation(line: 47, column: 1, scope: !29)
!57 = !DILocation(line: 48, column: 1, scope: !29)
!58 = !DILocation(line: 49, column: 1, scope: !29)
!59 = !DILocation(line: 50, column: 1, scope: !29)
!60 = !DILocation(line: 51, column: 1, scope: !29)
