; RUN: opt < %s -passes=sample-profile --overwrite-existing-weights -sample-profile-file=%S/Inputs/pseudo-probe-missing-probe.prof -S | FileCheck %s

; CHECK:  br i1 %tobool, label %if.then, label %if.else, !dbg ![[#]], !prof ![[#PROF:]]
; CHECK:  [[#PROF]] = !{!"branch_weights", i32 14904, i32 1820}
; Verify the else branch is not set to a zero count
; CHECK-NOT:  [[#PROF]] = !{!"branch_weights", i32 16724, i32 0}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(i32 noundef %i) #0 !dbg !47 !prof !52 {
entry:
    #dbg_value(i32 %i, !51, !DIExpression(), !53)
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !54
  %0 = load volatile i32, ptr @x, align 4, !dbg !54, !tbaa !55
  %add = add nsw i32 %0, 5, !dbg !54
  store volatile i32 %add, ptr @x, align 4, !dbg !54, !tbaa !55
  ret void, !dbg !59
}

; Function Attrs: nounwind uwtable
define dso_local void @baz(i32 noundef %i) #1 !dbg !60 !prof !63 {
entry:
    #dbg_value(i32 %i, !62, !DIExpression(), !64)
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !65
  %rem = srem i32 %i, 100, !dbg !67
  %tobool = icmp ne i32 %rem, 0, !dbg !67
  br i1 %tobool, label %if.then, label %if.end, !dbg !68

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 2, i32 0, i64 -1), !dbg !69
  %0 = load volatile i32, ptr @x, align 4, !dbg !69, !tbaa !55
  %inc = add nsw i32 %0, 1, !dbg !69
  store volatile i32 %inc, ptr @x, align 4, !dbg !69, !tbaa !55
  br label %if.end, !dbg !70

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 3, i32 0, i64 -1), !dbg !71
  %1 = load volatile i32, ptr @x, align 4, !dbg !71, !tbaa !55
  %add = add nsw i32 %1, 2, !dbg !71
  store volatile i32 %add, ptr @x, align 4, !dbg !71, !tbaa !55
  %rem1 = srem i32 %i, 2, !dbg !72
  %tobool2 = icmp ne i32 %rem1, 0, !dbg !72
  br i1 %tobool2, label %if.then3, label %if.else, !dbg !74

if.then3:                                         ; preds = %if.end
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 4, i32 0, i64 -1), !dbg !75
  %2 = load volatile i32, ptr @x, align 4, !dbg !75, !tbaa !55
  %inc4 = add nsw i32 %2, 1, !dbg !75
  store volatile i32 %inc4, ptr @x, align 4, !dbg !75, !tbaa !55
  br label %if.end11, !dbg !76

if.else:                                          ; preds = %if.end
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 5, i32 0, i64 -1), !dbg !77
  %rem5 = srem i32 %i, 3, !dbg !79
  %tobool6 = icmp ne i32 %rem5, 0, !dbg !79
  br i1 %tobool6, label %if.then7, label %if.else9, !dbg !80

if.then7:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 6, i32 0, i64 -1), !dbg !81
  %3 = load volatile i32, ptr @x, align 4, !dbg !81, !tbaa !55
  %add8 = add nsw i32 %3, 2, !dbg !81
  store volatile i32 %add8, ptr @x, align 4, !dbg !81, !tbaa !55
  br label %if.end11, !dbg !82

if.else9:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 7, i32 0, i64 -1), !dbg !83
  %4 = load volatile i32, ptr @x, align 4, !dbg !83, !tbaa !55
  %dec = add nsw i32 %4, -1, !dbg !83
  store volatile i32 %dec, ptr @x, align 4, !dbg !83, !tbaa !55
  br label %if.end11

if.end11:                                         ; preds = %if.then7, %if.else9, %if.then3
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 9, i32 0, i64 -1), !dbg !84
  ret void, !dbg !84
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #1 !dbg !85 !prof !90 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !91
    #dbg_value(i32 0, !89, !DIExpression(), !92)
  br label %while.cond, !dbg !93

while.cond:                                       ; preds = %if.end, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ], !dbg !92
    #dbg_value(i32 %i.0, !89, !DIExpression(), !92)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !94
  %inc = add nsw i32 %i.0, 1, !dbg !94
    #dbg_value(i32 %inc, !89, !DIExpression(), !92)
  %cmp = icmp slt i32 %i.0, 160000000, !dbg !95
  br i1 %cmp, label %while.body, label %while.end, !dbg !93, !prof !96

while.body:                                       ; preds = %while.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !97
  %rem = srem i32 %inc, 10, !dbg !100
  %tobool = icmp ne i32 %rem, 0, !dbg !100
  br i1 %tobool, label %if.then, label %if.else, !dbg !101, !prof !102

if.then:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !103
  call void @bar(i32 noundef %inc), !dbg !104, !prof !106
  br label %if.end, !dbg !107

if.else:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !108
    #dbg_value(i32 %inc, !62, !DIExpression(), !109)
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !112
  %rem.i = srem i32 %inc, 100, !dbg !113
  %tobool.i = icmp ne i32 %rem.i, 0, !dbg !113
  br i1 %tobool.i, label %if.then.i, label %if.end.i, !dbg !114, !prof !115

if.then.i:                                        ; preds = %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 2, i32 0, i64 -1), !dbg !116
  %0 = load volatile i32, ptr @x, align 4, !dbg !116, !tbaa !55
  %inc.i = add nsw i32 %0, 1, !dbg !116
  store volatile i32 %inc.i, ptr @x, align 4, !dbg !116, !tbaa !55
  br label %if.end.i, !dbg !117

if.end.i:                                         ; preds = %if.then.i, %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 3, i32 0, i64 -1), !dbg !118
  %1 = load volatile i32, ptr @x, align 4, !dbg !118, !tbaa !55
  %add.i = add nsw i32 %1, 2, !dbg !118
  store volatile i32 %add.i, ptr @x, align 4, !dbg !118, !tbaa !55
  %rem1.i = srem i32 %inc, 2, !dbg !119
  %tobool2.i = icmp ne i32 %rem1.i, 0, !dbg !119
  br i1 %tobool2.i, label %if.then3.i, label %if.else.i, !dbg !120, !prof !121

if.then3.i:                                       ; preds = %if.end.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 4, i32 0, i64 -1), !dbg !122
  %2 = load volatile i32, ptr @x, align 4, !dbg !122, !tbaa !55
  %inc4.i = add nsw i32 %2, 1, !dbg !122
  store volatile i32 %inc4.i, ptr @x, align 4, !dbg !122, !tbaa !55
  br label %baz.exit, !dbg !123

if.else.i:                                        ; preds = %if.end.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 5, i32 0, i64 -1), !dbg !124
  %rem5.i = srem i32 %inc, 3, !dbg !125
  %tobool6.i = icmp ne i32 %rem5.i, 0, !dbg !125
  br i1 %tobool6.i, label %if.then7.i, label %if.else9.i, !dbg !126, !prof !127

if.then7.i:                                       ; preds = %if.else.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 6, i32 0, i64 -1), !dbg !128
  %3 = load volatile i32, ptr @x, align 4, !dbg !128, !tbaa !55
  %add8.i = add nsw i32 %3, 2, !dbg !128
  store volatile i32 %add8.i, ptr @x, align 4, !dbg !128, !tbaa !55
  br label %baz.exit, !dbg !129

if.else9.i:                                       ; preds = %if.else.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 7, i32 0, i64 -1), !dbg !130
  %4 = load volatile i32, ptr @x, align 4, !dbg !130, !tbaa !55
  %dec.i = add nsw i32 %4, -1, !dbg !130
  store volatile i32 %dec.i, ptr @x, align 4, !dbg !130, !tbaa !55
  br label %baz.exit

baz.exit:                                         ; preds = %if.then3.i, %if.then7.i, %if.else9.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 9, i32 0, i64 -1), !dbg !131
  br label %if.end

if.end:                                           ; preds = %baz.exit, %if.then
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !93
  br label %while.cond, !dbg !93, !llvm.loop !132

while.end:                                        ; preds = %while.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !135
  ret i32 0, !dbg !135
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13, !14}
!llvm.ident = !{!43}
!llvm.pseudo_probe_desc = !{!44, !45, !46}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "b67c15e928f76c51702a59639dbebb4c")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!14 = !{i32 1, !"ProfileSummary", !15}
!15 = !{!16, !17, !18, !19, !20, !21, !22, !23, !24, !25}
!16 = !{!"ProfileFormat", !"SampleProfile"}
!17 = !{!"TotalCount", i64 105360}
!18 = !{!"MaxCount", i64 16724}
!19 = !{!"MaxInternalCount", i64 0}
!20 = !{!"MaxFunctionCount", i64 15026}
!21 = !{!"NumCounts", i64 14}
!22 = !{!"NumFunctions", i64 2}
!23 = !{!"IsPartialProfile", i64 0}
!24 = !{!"PartialProfileRatio", double 0.000000e+00}
!25 = !{!"DetailedSummary", !26}
!26 = !{!27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42}
!27 = !{i32 10000, i64 16724, i32 3}
!28 = !{i32 100000, i64 16724, i32 3}
!29 = !{i32 200000, i64 16724, i32 3}
!30 = !{i32 300000, i64 16724, i32 3}
!31 = !{i32 400000, i64 16724, i32 3}
!32 = !{i32 500000, i64 15026, i32 5}
!33 = !{i32 600000, i64 15026, i32 5}
!34 = !{i32 700000, i64 15026, i32 5}
!35 = !{i32 800000, i64 14342, i32 6}
!36 = !{i32 900000, i64 1882, i32 8}
!37 = !{i32 950000, i64 1872, i32 10}
!38 = !{i32 990000, i64 1550, i32 12}
!39 = !{i32 999000, i64 1550, i32 12}
!40 = !{i32 999900, i64 1550, i32 12}
!41 = !{i32 999990, i64 1550, i32 12}
!42 = !{i32 999999, i64 1550, i32 12}
!43 = !{!"clang version 20.0.0"}
!44 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!45 = !{i64 7546896869197086323, i64 191430930410, !"baz"}
!46 = !{i64 -2624081020897602054, i64 563091374530180, !"main"}
!47 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 3, type: !48, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !50)
!48 = !DISubroutineType(types: !49)
!49 = !{null, !6}
!50 = !{!51}
!51 = !DILocalVariable(name: "i", arg: 1, scope: !47, file: !3, line: 3, type: !6)
!52 = !{!"function_entry_count", i64 15026}
!53 = !DILocation(line: 0, scope: !47)
!54 = !DILocation(line: 4, column: 5, scope: !47)
!55 = !{!56, !56, i64 0}
!56 = !{!"int", !57, i64 0}
!57 = !{!"omnipotent char", !58, i64 0}
!58 = !{!"Simple C/C++ TBAA"}
!59 = !DILocation(line: 8, column: 1, scope: !47)
!60 = distinct !DISubprogram(name: "baz", scope: !3, file: !3, line: 10, type: !48, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !61)
!61 = !{!62}
!62 = !DILocalVariable(name: "i", arg: 1, scope: !60, file: !3, line: 10, type: !6)
!63 = !{!"function_entry_count", i64 -1}
!64 = !DILocation(line: 0, scope: !60)
!65 = !DILocation(line: 11, column: 6, scope: !66)
!66 = distinct !DILexicalBlock(scope: !60, file: !3, line: 11, column: 6)
!67 = !DILocation(line: 11, column: 7, scope: !66)
!68 = !DILocation(line: 11, column: 6, scope: !60)
!69 = !DILocation(line: 12, column: 6, scope: !66)
!70 = !DILocation(line: 12, column: 5, scope: !66)
!71 = !DILocation(line: 14, column: 5, scope: !60)
!72 = !DILocation(line: 15, column: 9, scope: !73)
!73 = distinct !DILexicalBlock(scope: !60, file: !3, line: 15, column: 7)
!74 = !DILocation(line: 15, column: 7, scope: !60)
!75 = !DILocation(line: 16, column: 7, scope: !73)
!76 = !DILocation(line: 16, column: 6, scope: !73)
!77 = !DILocation(line: 17, column: 12, scope: !78)
!78 = distinct !DILexicalBlock(scope: !73, file: !3, line: 17, column: 12)
!79 = !DILocation(line: 17, column: 14, scope: !78)
!80 = !DILocation(line: 17, column: 12, scope: !73)
!81 = !DILocation(line: 18, column: 7, scope: !78)
!82 = !DILocation(line: 18, column: 6, scope: !78)
!83 = !DILocation(line: 20, column: 7, scope: !78)
!84 = !DILocation(line: 21, column: 1, scope: !60)
!85 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 23, type: !86, scopeLine: 23, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !88)
!86 = !DISubroutineType(types: !87)
!87 = !{!6}
!88 = !{!89}
!89 = !DILocalVariable(name: "i", scope: !85, file: !3, line: 24, type: !6)
!90 = !{!"function_entry_count", i64 1}
!91 = !DILocation(line: 24, column: 7, scope: !85)
!92 = !DILocation(line: 0, scope: !85)
!93 = !DILocation(line: 25, column: 3, scope: !85)
!94 = !DILocation(line: 25, column: 11, scope: !85)
!95 = !DILocation(line: 25, column: 14, scope: !85)
!96 = !{!"branch_weights", i32 16724, i32 1}
!97 = !DILocation(line: 26, column: 8, scope: !98)
!98 = distinct !DILexicalBlock(scope: !99, file: !3, line: 26, column: 8)
!99 = distinct !DILexicalBlock(scope: !85, file: !3, line: 25, column: 30)
!100 = !DILocation(line: 26, column: 10, scope: !98)
!101 = !DILocation(line: 26, column: 8, scope: !99)
!102 = !{!"branch_weights", i32 14852, i32 1872}
!103 = !DILocation(line: 27, column: 10, scope: !98)
!104 = !DILocation(line: 27, column: 6, scope: !105)
!105 = !DILexicalBlockFile(scope: !98, file: !3, discriminator: 455082031)
!106 = !{!"branch_weights", i32 14852}
!107 = !DILocation(line: 27, column: 6, scope: !98)
!108 = !DILocation(line: 29, column: 10, scope: !98)
!109 = !DILocation(line: 0, scope: !60, inlinedAt: !110)
!110 = distinct !DILocation(line: 29, column: 6, scope: !111)
!111 = !DILexicalBlockFile(scope: !98, file: !3, discriminator: 455082047)
!112 = !DILocation(line: 11, column: 6, scope: !66, inlinedAt: !110)
!113 = !DILocation(line: 11, column: 7, scope: !66, inlinedAt: !110)
!114 = !DILocation(line: 11, column: 6, scope: !60, inlinedAt: !110)
!115 = !{!"branch_weights", i32 1736, i32 136}
!116 = !DILocation(line: 12, column: 6, scope: !66, inlinedAt: !110)
!117 = !DILocation(line: 12, column: 5, scope: !66, inlinedAt: !110)
!118 = !DILocation(line: 14, column: 5, scope: !60, inlinedAt: !110)
!119 = !DILocation(line: 15, column: 9, scope: !73, inlinedAt: !110)
!120 = !DILocation(line: 15, column: 7, scope: !60, inlinedAt: !110)
!121 = !{!"branch_weights", i32 0, i32 1872}
!122 = !DILocation(line: 16, column: 7, scope: !73, inlinedAt: !110)
!123 = !DILocation(line: 16, column: 6, scope: !73, inlinedAt: !110)
!124 = !DILocation(line: 17, column: 12, scope: !78, inlinedAt: !110)
!125 = !DILocation(line: 17, column: 14, scope: !78, inlinedAt: !110)
!126 = !DILocation(line: 17, column: 12, scope: !73, inlinedAt: !110)
!127 = !{!"branch_weights", i32 936, i32 936}
!128 = !DILocation(line: 18, column: 7, scope: !78, inlinedAt: !110)
!129 = !DILocation(line: 18, column: 6, scope: !78, inlinedAt: !110)
!130 = !DILocation(line: 20, column: 7, scope: !78, inlinedAt: !110)
!131 = !DILocation(line: 21, column: 1, scope: !60, inlinedAt: !110)
!132 = distinct !{!132, !93, !133, !134}
!133 = !DILocation(line: 30, column: 3, scope: !85)
!134 = !{!"llvm.loop.mustprogress"}
!135 = !DILocation(line: 31, column: 3, scope: !85)
