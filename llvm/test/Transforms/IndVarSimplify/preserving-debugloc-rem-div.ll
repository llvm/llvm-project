; Test that the debug information is propagated correctly to the new instructions
; RUN: opt < %s -passes=indvars -S | FileCheck %s

define void @test_srem_urem(ptr %a) !dbg !5 {
; CHECK-LABEL: define void @test_srem_urem(
; CHECK:    [[REM_UREM:%.*]] = urem i32 [[I_01:%.*]], 2, !dbg [[DBG20:![0-9]+]]
; CHECK:      #dbg_value(i32 [[REM_UREM]], [[META11:![0-9]+]], !DIExpression(), [[DBG20]])
;
entry:
  br label %for.body, !dbg !18

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %i.01, metadata !9, metadata !DIExpression()), !dbg !19
  %rem = srem i32 %i.01, 2, !dbg !20
  tail call void @llvm.dbg.value(metadata i32 %rem, metadata !11, metadata !DIExpression()), !dbg !20
  %idxprom = sext i32 %rem to i64, !dbg !21
  tail call void @llvm.dbg.value(metadata i64 %idxprom, metadata !12, metadata !DIExpression()), !dbg !21
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom, !dbg !22
  tail call void @llvm.dbg.value(metadata ptr %arrayidx, metadata !14, metadata !DIExpression()), !dbg !22
  store i32 %i.01, ptr %arrayidx, align 4, !dbg !23
  %inc = add nsw i32 %i.01, 1, !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !15, metadata !DIExpression()), !dbg !24
  %cmp = icmp slt i32 %inc, 64, !dbg !25
  tail call void @llvm.dbg.value(metadata i1 %cmp, metadata !16, metadata !DIExpression()), !dbg !25
  br i1 %cmp, label %for.body, label %for.end, !dbg !26

for.end:                                          ; preds = %for.body
  ret void, !dbg !27
}

define void @test_sdiv_udiv(ptr %a) !dbg !28 {
; CHECK-LABEL: define void @test_sdiv_udiv(
; CHECK:    [[DIV_UDIV:%.*]] = udiv i32 [[I_01:%.*]], 2, !dbg [[DBG38:![0-9]+]]
; CHECK:      #dbg_value(i32 [[DIV_UDIV]], [[META31:![0-9]+]], !DIExpression(), [[DBG38]])
;
entry:
  br label %for.body, !dbg !36

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %i.01, metadata !30, metadata !DIExpression()), !dbg !37
  %div = sdiv i32 %i.01, 2, !dbg !38
  tail call void @llvm.dbg.value(metadata i32 %div, metadata !31, metadata !DIExpression()), !dbg !38
  %idxprom = sext i32 %div to i64, !dbg !39
  tail call void @llvm.dbg.value(metadata i64 %idxprom, metadata !32, metadata !DIExpression()), !dbg !39
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom, !dbg !40
  tail call void @llvm.dbg.value(metadata ptr %arrayidx, metadata !33, metadata !DIExpression()), !dbg !40
  store i32 %i.01, ptr %arrayidx, align 4, !dbg !41
  %inc = add nsw i32 %i.01, 1, !dbg !42
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !34, metadata !DIExpression()), !dbg !42
  %cmp = icmp slt i32 %inc, 64, !dbg !43
  tail call void @llvm.dbg.value(metadata i1 %cmp, metadata !35, metadata !DIExpression()), !dbg !43
  br i1 %cmp, label %for.body, label %for.end, !dbg !44

for.end:                                          ; preds = %for.body
  ret void, !dbg !45
}

; Function Attrs: nounwind
define i32 @test_rem_num_zero(i64 %arg1) #0 !dbg !46 {
; CHECK-LABEL: define i32 @test_rem_num_zero(
; CHECK:    [[TMP0:%.*]] = icmp eq i64 [[T12:%.*]], %arg1
; CHECK:    [[IV_REM:%.*]] = select i1 [[TMP0]], i64 0, i64 [[T12]], !dbg [[DBG72:![0-9]+]]
; CHECK:      #dbg_value(i64 [[IV_REM]], [[META52:![0-9]+]], !DIExpression(), [[DBG72]])
;
bb:
  %t = icmp sgt i64 %arg1, 0, !dbg !66
  tail call void @llvm.dbg.value(metadata i1 %t, metadata !48, metadata !DIExpression()), !dbg !66
  br i1 %t, label %bb4, label %bb54, !dbg !67

bb4:                                              ; preds = %bb
  br label %bb5, !dbg !68

bb5:                                              ; preds = %bb49, %bb4
  %t6 = phi i64 [ %t51, %bb49 ], [ 0, %bb4 ], !dbg !69
  %t7 = phi i32 [ %t50, %bb49 ], [ 0, %bb4 ], !dbg !70
  tail call void @llvm.dbg.value(metadata i64 %t6, metadata !49, metadata !DIExpression()), !dbg !69
  tail call void @llvm.dbg.value(metadata i32 %t7, metadata !50, metadata !DIExpression()), !dbg !70
  %t12 = add nsw i64 %t6, 1, !dbg !71
  tail call void @llvm.dbg.value(metadata i64 %t12, metadata !51, metadata !DIExpression()), !dbg !71
  %t13 = srem i64 %t12, %arg1, !dbg !72
  tail call void @llvm.dbg.value(metadata i64 %t13, metadata !52, metadata !DIExpression()), !dbg !72
  %t14 = icmp sgt i64 %arg1, 0, !dbg !73
  tail call void @llvm.dbg.value(metadata i1 %t14, metadata !53, metadata !DIExpression()), !dbg !73
  br i1 %t14, label %bb15, label %bb49, !dbg !74

bb15:                                             ; preds = %bb5
  br label %bb16, !dbg !75

bb16:                                             ; preds = %bb44, %bb15
  %t17 = phi i64 [ %t46, %bb44 ], [ 0, %bb15 ], !dbg !76
  %t18 = phi i32 [ %t45, %bb44 ], [ %t7, %bb15 ], !dbg !77
  tail call void @llvm.dbg.value(metadata i64 %t17, metadata !54, metadata !DIExpression()), !dbg !76
  tail call void @llvm.dbg.value(metadata i32 %t18, metadata !55, metadata !DIExpression()), !dbg !77
  %t19 = icmp sgt i64 %arg1, 0, !dbg !78
  tail call void @llvm.dbg.value(metadata i1 %t19, metadata !56, metadata !DIExpression()), !dbg !78
  br i1 %t19, label %bb20, label %bb44, !dbg !79

bb20:                                             ; preds = %bb16
  br label %bb21, !dbg !80

bb21:                                             ; preds = %bb21, %bb20
  %t25 = mul i64 %t13, %arg1, !dbg !81
  tail call void @llvm.dbg.value(metadata i64 %t25, metadata !57, metadata !DIExpression()), !dbg !81
  %t42 = icmp slt i64 %t25, %arg1, !dbg !82
  tail call void @llvm.dbg.value(metadata i1 %t42, metadata !58, metadata !DIExpression()), !dbg !82
  br i1 %t42, label %bb21, label %bb43, !dbg !83

bb43:                                             ; preds = %bb21
  br label %bb44, !dbg !84

bb44:                                             ; preds = %bb43, %bb16
  %t45 = phi i32 [ %t18, %bb16 ], [ 0, %bb43 ], !dbg !85
  tail call void @llvm.dbg.value(metadata i32 %t45, metadata !59, metadata !DIExpression()), !dbg !85
  %t46 = add nsw i64 %t17, 1, !dbg !86
  tail call void @llvm.dbg.value(metadata i64 %t46, metadata !60, metadata !DIExpression()), !dbg !86
  %t47 = icmp slt i64 %t46, %arg1, !dbg !87
  tail call void @llvm.dbg.value(metadata i1 %t47, metadata !61, metadata !DIExpression()), !dbg !87
  br i1 %t47, label %bb16, label %bb48, !dbg !88

bb48:                                             ; preds = %bb44
  br label %bb49, !dbg !89

bb49:                                             ; preds = %bb48, %bb5
  %t50 = phi i32 [ %t7, %bb5 ], [ %t45, %bb48 ], !dbg !90
  tail call void @llvm.dbg.value(metadata i32 %t50, metadata !62, metadata !DIExpression()), !dbg !90
  %t51 = add nsw i64 %t6, 1, !dbg !91
  tail call void @llvm.dbg.value(metadata i64 %t51, metadata !63, metadata !DIExpression()), !dbg !91
  %t52 = icmp slt i64 %t51, %arg1, !dbg !92
  tail call void @llvm.dbg.value(metadata i1 %t52, metadata !64, metadata !DIExpression()), !dbg !92
  br i1 %t52, label %bb5, label %bb53, !dbg !93

bb53:                                             ; preds = %bb49
  br label %bb54, !dbg !94

bb54:                                             ; preds = %bb53, %bb
  %t55 = phi i32 [ 0, %bb ], [ %t50, %bb53 ], !dbg !95
  tail call void @llvm.dbg.value(metadata i32 %t55, metadata !65, metadata !DIExpression()), !dbg !95
  ret i32 %t55, !dbg !96
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

;.
; CHECK: [[DBG20]] = !DILocation(line: 3,
; CHECK: [[DBG38]] = !DILocation(line: 13,
; CHECK: [[DBG72]] = !DILocation(line: 27,
;.

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "indvars.ll", directory: "/")
!2 = !{i32 51}
!3 = !{i32 30}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_srem_urem", linkageName: "test_srem_urem", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !12, !14, !15, !16}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 3, type: !10)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !13)
!13 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !13)
!15 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 7, type: !10)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 8, type: !17)
!17 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 1, column: 1, scope: !5)
!19 = !DILocation(line: 2, column: 1, scope: !5)
!20 = !DILocation(line: 3, column: 1, scope: !5)
!21 = !DILocation(line: 4, column: 1, scope: !5)
!22 = !DILocation(line: 5, column: 1, scope: !5)
!23 = !DILocation(line: 6, column: 1, scope: !5)
!24 = !DILocation(line: 7, column: 1, scope: !5)
!25 = !DILocation(line: 8, column: 1, scope: !5)
!26 = !DILocation(line: 9, column: 1, scope: !5)
!27 = !DILocation(line: 10, column: 1, scope: !5)
!28 = distinct !DISubprogram(name: "test_sdiv_udiv", linkageName: "test_sdiv_udiv", scope: null, file: !1, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!29 = !{!30, !31, !32, !33, !34, !35}
!30 = !DILocalVariable(name: "7", scope: !28, file: !1, line: 12, type: !10)
!31 = !DILocalVariable(name: "8", scope: !28, file: !1, line: 13, type: !10)
!32 = !DILocalVariable(name: "9", scope: !28, file: !1, line: 14, type: !13)
!33 = !DILocalVariable(name: "10", scope: !28, file: !1, line: 15, type: !13)
!34 = !DILocalVariable(name: "11", scope: !28, file: !1, line: 17, type: !10)
!35 = !DILocalVariable(name: "12", scope: !28, file: !1, line: 18, type: !17)
!36 = !DILocation(line: 11, column: 1, scope: !28)
!37 = !DILocation(line: 12, column: 1, scope: !28)
!38 = !DILocation(line: 13, column: 1, scope: !28)
!39 = !DILocation(line: 14, column: 1, scope: !28)
!40 = !DILocation(line: 15, column: 1, scope: !28)
!41 = !DILocation(line: 16, column: 1, scope: !28)
!42 = !DILocation(line: 17, column: 1, scope: !28)
!43 = !DILocation(line: 18, column: 1, scope: !28)
!44 = !DILocation(line: 19, column: 1, scope: !28)
!45 = !DILocation(line: 20, column: 1, scope: !28)
!46 = distinct !DISubprogram(name: "test_rem_num_zero", linkageName: "test_rem_num_zero", scope: null, file: !1, line: 21, type: !6, scopeLine: 21, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !47)
!47 = !{!48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65}
!48 = !DILocalVariable(name: "13", scope: !46, file: !1, line: 21, type: !17)
!49 = !DILocalVariable(name: "14", scope: !46, file: !1, line: 24, type: !13)
!50 = !DILocalVariable(name: "15", scope: !46, file: !1, line: 25, type: !10)
!51 = !DILocalVariable(name: "16", scope: !46, file: !1, line: 26, type: !13)
!52 = !DILocalVariable(name: "17", scope: !46, file: !1, line: 27, type: !13)
!53 = !DILocalVariable(name: "18", scope: !46, file: !1, line: 28, type: !17)
!54 = !DILocalVariable(name: "19", scope: !46, file: !1, line: 31, type: !13)
!55 = !DILocalVariable(name: "20", scope: !46, file: !1, line: 32, type: !10)
!56 = !DILocalVariable(name: "21", scope: !46, file: !1, line: 33, type: !17)
!57 = !DILocalVariable(name: "22", scope: !46, file: !1, line: 36, type: !13)
!58 = !DILocalVariable(name: "23", scope: !46, file: !1, line: 37, type: !17)
!59 = !DILocalVariable(name: "24", scope: !46, file: !1, line: 40, type: !10)
!60 = !DILocalVariable(name: "25", scope: !46, file: !1, line: 41, type: !13)
!61 = !DILocalVariable(name: "26", scope: !46, file: !1, line: 42, type: !17)
!62 = !DILocalVariable(name: "27", scope: !46, file: !1, line: 45, type: !10)
!63 = !DILocalVariable(name: "28", scope: !46, file: !1, line: 46, type: !13)
!64 = !DILocalVariable(name: "29", scope: !46, file: !1, line: 47, type: !17)
!65 = !DILocalVariable(name: "30", scope: !46, file: !1, line: 50, type: !10)
!66 = !DILocation(line: 21, column: 1, scope: !46)
!67 = !DILocation(line: 22, column: 1, scope: !46)
!68 = !DILocation(line: 23, column: 1, scope: !46)
!69 = !DILocation(line: 24, column: 1, scope: !46)
!70 = !DILocation(line: 25, column: 1, scope: !46)
!71 = !DILocation(line: 26, column: 1, scope: !46)
!72 = !DILocation(line: 27, column: 1, scope: !46)
!73 = !DILocation(line: 28, column: 1, scope: !46)
!74 = !DILocation(line: 29, column: 1, scope: !46)
!75 = !DILocation(line: 30, column: 1, scope: !46)
!76 = !DILocation(line: 31, column: 1, scope: !46)
!77 = !DILocation(line: 32, column: 1, scope: !46)
!78 = !DILocation(line: 33, column: 1, scope: !46)
!79 = !DILocation(line: 34, column: 1, scope: !46)
!80 = !DILocation(line: 35, column: 1, scope: !46)
!81 = !DILocation(line: 36, column: 1, scope: !46)
!82 = !DILocation(line: 37, column: 1, scope: !46)
!83 = !DILocation(line: 38, column: 1, scope: !46)
!84 = !DILocation(line: 39, column: 1, scope: !46)
!85 = !DILocation(line: 40, column: 1, scope: !46)
!86 = !DILocation(line: 41, column: 1, scope: !46)
!87 = !DILocation(line: 42, column: 1, scope: !46)
!88 = !DILocation(line: 43, column: 1, scope: !46)
!89 = !DILocation(line: 44, column: 1, scope: !46)
!90 = !DILocation(line: 45, column: 1, scope: !46)
!91 = !DILocation(line: 46, column: 1, scope: !46)
!92 = !DILocation(line: 47, column: 1, scope: !46)
!93 = !DILocation(line: 48, column: 1, scope: !46)
!94 = !DILocation(line: 49, column: 1, scope: !46)
!95 = !DILocation(line: 50, column: 1, scope: !46)
!96 = !DILocation(line: 51, column: 1, scope: !46)
