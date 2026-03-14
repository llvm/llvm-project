; RUN: opt < %s -passes=sample-profile  -sample-profile-file=%S/Inputs/pseudo-probe-missing-probe.prof -S | FileCheck %s

; CHECK:  br i1 %tobool.not.i, label %if.end.i, label %if.then.i, !dbg ![[#]], !prof ![[#PROF:]]

; CHECK:  [[#PROF]] = !{!"branch_weights", i32 918, i32 918}
; Verify the else branch is not set to a zero count
; CHECK-NOT:  [[#PROF]] = !{!"branch_weights", i32 1698, i32 0}

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @bar(i32 %i) local_unnamed_addr #0 !dbg !18 {
entry:
    #dbg_value(i32 poison, !22, !DIExpression(), !23)
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !24
  %0 = load volatile i32, ptr @x, align 4, !dbg !24, !tbaa !25
  %add = add nsw i32 %0, 5, !dbg !24
  store volatile i32 %add, ptr @x, align 4, !dbg !24, !tbaa !25
  ret void, !dbg !29
}

; Function Attrs: nofree norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @baz(i32 noundef %i) local_unnamed_addr #1 !dbg !30 {
entry:
    #dbg_value(i32 %i, !32, !DIExpression(), !33)
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !34
  %rem = srem i32 %i, 100, !dbg !36
  %tobool.not = icmp eq i32 %rem, 0, !dbg !36
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !37

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 2, i32 0, i64 -1), !dbg !38
  %0 = load volatile i32, ptr @x, align 4, !dbg !38, !tbaa !25
  %inc = add nsw i32 %0, 1, !dbg !38
  store volatile i32 %inc, ptr @x, align 4, !dbg !38, !tbaa !25
  br label %if.end, !dbg !39

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 3, i32 0, i64 -1), !dbg !40
  %1 = load volatile i32, ptr @x, align 4, !dbg !40, !tbaa !25
  %add = add nsw i32 %1, 2, !dbg !40
  store volatile i32 %add, ptr @x, align 4, !dbg !40, !tbaa !25
  %2 = and i32 %i, 1, !dbg !41
  %tobool2.not = icmp eq i32 %2, 0, !dbg !41
  br i1 %tobool2.not, label %if.else, label %if.end11, !dbg !43

if.else:                                          ; preds = %if.end
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 5, i32 0, i64 -1), !dbg !44
  %rem5 = srem i32 %i, 3, !dbg !46
  %tobool6.not = icmp eq i32 %rem5, 0, !dbg !46
  %spec.select = select i1 %tobool6.not, i32 -1, i32 2, !dbg !47
  br label %if.end11, !dbg !47

if.end11:                                         ; preds = %if.else, %if.end
  %.sink14 = phi i32 [ 1, %if.end ], [ %spec.select, %if.else ]
  %3 = load volatile i32, ptr @x, align 4, !dbg !48, !tbaa !25
  %add8 = add nsw i32 %3, %.sink14, !dbg !48
  store volatile i32 %add8, ptr @x, align 4, !dbg !48, !tbaa !25
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 9, i32 0, i64 -1), !dbg !49
  ret void, !dbg !49
}

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 !dbg !50 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !55
    #dbg_value(i32 0, !54, !DIExpression(), !56)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !57
  br label %while.body, !dbg !58

while.body:                                       ; preds = %entry, %if.end
  %inc7 = phi i32 [ 1, %entry ], [ %inc, %if.end ]
  %i.06 = phi i32 [ 0, %entry ], [ %inc7, %if.end ]
    #dbg_value(i32 %i.06, !54, !DIExpression(), !56)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !59
  %rem = urem i32 %inc7, 10, !dbg !62
  %tobool.not = icmp eq i32 %rem, 0, !dbg !62
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !63

if.then:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !64
  tail call void @bar(i32 poison), !dbg !65
  br label %if.end, !dbg !67

if.else:                                          ; preds = %while.body
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !68
    #dbg_value(i32 %inc7, !32, !DIExpression(), !69)
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !72
  %rem.i4 = urem i32 %inc7, 100, !dbg !73
  %tobool.not.i = icmp eq i32 %rem.i4, 0, !dbg !73
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i, !dbg !74

if.then.i:                                        ; preds = %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 2, i32 0, i64 -1), !dbg !75
  %0 = load volatile i32, ptr @x, align 4, !dbg !75, !tbaa !25
  %inc.i = add nsw i32 %0, 1, !dbg !75
  store volatile i32 %inc.i, ptr @x, align 4, !dbg !75, !tbaa !25
  br label %if.end.i, !dbg !76

if.end.i:                                         ; preds = %if.then.i, %if.else
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 3, i32 0, i64 -1), !dbg !77
  %1 = load volatile i32, ptr @x, align 4, !dbg !77, !tbaa !25
  %add.i = add nsw i32 %1, 2, !dbg !77
  store volatile i32 %add.i, ptr @x, align 4, !dbg !77, !tbaa !25
  %2 = and i32 %i.06, 1, !dbg !78
  %tobool2.not.i.not = icmp eq i32 %2, 0, !dbg !78
  br i1 %tobool2.not.i.not, label %baz.exit, label %if.else.i, !dbg !79

if.else.i:                                        ; preds = %if.end.i
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 5, i32 0, i64 -1), !dbg !80
  %rem5.i5 = urem i32 %inc7, 3, !dbg !81
  %tobool6.not.i = icmp eq i32 %rem5.i5, 0, !dbg !81
  %spec.select.i = select i1 %tobool6.not.i, i32 -1, i32 2, !dbg !82
  br label %baz.exit, !dbg !82

baz.exit:                                         ; preds = %if.end.i, %if.else.i
  %.sink14.i = phi i32 [ 1, %if.end.i ], [ %spec.select.i, %if.else.i ]
  %3 = load volatile i32, ptr @x, align 4, !dbg !83, !tbaa !25
  %add8.i = add nsw i32 %3, %.sink14.i, !dbg !83
  store volatile i32 %add8.i, ptr @x, align 4, !dbg !83, !tbaa !25
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 9, i32 0, i64 -1), !dbg !84
  br label %if.end

if.end:                                           ; preds = %baz.exit, %if.then
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !58
    #dbg_value(i32 %inc7, !54, !DIExpression(), !56)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !57
  %inc = add nuw nsw i32 %inc7, 1, !dbg !57
    #dbg_value(i32 %inc, !54, !DIExpression(), !56)
  %exitcond.not = icmp eq i32 %inc, 160000001, !dbg !85
  br i1 %exitcond.not, label %while.end, label %while.body, !dbg !58, !llvm.loop !86

while.end:                                        ; preds = %if.end
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !89
  ret i32 0, !dbg !89
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree norecurse nounwind memory(readwrite, argmem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nofree norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile"}
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16, !17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home", checksumkind: CSK_MD5, checksum: "b67c15e928f76c51702a59639dbebb4c")
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
!14 = !{!"clang version 20.0.0"}
!15 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!16 = !{i64 7546896869197086323, i64 191430930410, !"baz"}
!17 = !{i64 -2624081020897602054, i64 563091374530180, !"main"}
!18 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 3, type: !19, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !6}
!21 = !{!22}
!22 = !DILocalVariable(name: "i", arg: 1, scope: !18, file: !3, line: 3, type: !6)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocation(line: 4, column: 5, scope: !18)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 8, column: 1, scope: !18)
!30 = distinct !DISubprogram(name: "baz", scope: !3, file: !3, line: 10, type: !19, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31)
!31 = !{!32}
!32 = !DILocalVariable(name: "i", arg: 1, scope: !30, file: !3, line: 10, type: !6)
!33 = !DILocation(line: 0, scope: !30)
!34 = !DILocation(line: 11, column: 6, scope: !35)
!35 = distinct !DILexicalBlock(scope: !30, file: !3, line: 11, column: 6)
!36 = !DILocation(line: 11, column: 7, scope: !35)
!37 = !DILocation(line: 11, column: 6, scope: !30)
!38 = !DILocation(line: 12, column: 6, scope: !35)
!39 = !DILocation(line: 12, column: 5, scope: !35)
!40 = !DILocation(line: 14, column: 5, scope: !30)
!41 = !DILocation(line: 15, column: 9, scope: !42)
!42 = distinct !DILexicalBlock(scope: !30, file: !3, line: 15, column: 7)
!43 = !DILocation(line: 15, column: 7, scope: !30)
!44 = !DILocation(line: 17, column: 12, scope: !45)
!45 = distinct !DILexicalBlock(scope: !42, file: !3, line: 17, column: 12)
!46 = !DILocation(line: 17, column: 14, scope: !45)
!47 = !DILocation(line: 17, column: 12, scope: !42)
!48 = !DILocation(line: 0, scope: !42)
!49 = !DILocation(line: 21, column: 1, scope: !30)
!50 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 23, type: !51, scopeLine: 23, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !53)
!51 = !DISubroutineType(types: !52)
!52 = !{!6}
!53 = !{!54}
!54 = !DILocalVariable(name: "i", scope: !50, file: !3, line: 24, type: !6)
!55 = !DILocation(line: 24, column: 7, scope: !50)
!56 = !DILocation(line: 0, scope: !50)
!57 = !DILocation(line: 25, column: 11, scope: !50)
!58 = !DILocation(line: 25, column: 3, scope: !50)
!59 = !DILocation(line: 26, column: 8, scope: !60)
!60 = distinct !DILexicalBlock(scope: !61, file: !3, line: 26, column: 8)
!61 = distinct !DILexicalBlock(scope: !50, file: !3, line: 25, column: 30)
!62 = !DILocation(line: 26, column: 10, scope: !60)
!63 = !DILocation(line: 26, column: 8, scope: !61)
!64 = !DILocation(line: 27, column: 10, scope: !60)
!65 = !DILocation(line: 27, column: 6, scope: !66)
!66 = !DILexicalBlockFile(scope: !60, file: !3, discriminator: 455082031)
!67 = !DILocation(line: 27, column: 6, scope: !60)
!68 = !DILocation(line: 29, column: 10, scope: !60)
!69 = !DILocation(line: 0, scope: !30, inlinedAt: !70)
!70 = distinct !DILocation(line: 29, column: 6, scope: !71)
!71 = !DILexicalBlockFile(scope: !60, file: !3, discriminator: 455082047)
!72 = !DILocation(line: 11, column: 6, scope: !35, inlinedAt: !70)
!73 = !DILocation(line: 11, column: 7, scope: !35, inlinedAt: !70)
!74 = !DILocation(line: 11, column: 6, scope: !30, inlinedAt: !70)
!75 = !DILocation(line: 12, column: 6, scope: !35, inlinedAt: !70)
!76 = !DILocation(line: 12, column: 5, scope: !35, inlinedAt: !70)
!77 = !DILocation(line: 14, column: 5, scope: !30, inlinedAt: !70)
!78 = !DILocation(line: 15, column: 9, scope: !42, inlinedAt: !70)
!79 = !DILocation(line: 15, column: 7, scope: !30, inlinedAt: !70)
!80 = !DILocation(line: 17, column: 12, scope: !45, inlinedAt: !70)
!81 = !DILocation(line: 17, column: 14, scope: !45, inlinedAt: !70)
!82 = !DILocation(line: 17, column: 12, scope: !42, inlinedAt: !70)
!83 = !DILocation(line: 0, scope: !42, inlinedAt: !70)
!84 = !DILocation(line: 21, column: 1, scope: !30, inlinedAt: !70)
!85 = !DILocation(line: 25, column: 14, scope: !50)
!86 = distinct !{!86, !58, !87, !88}
!87 = !DILocation(line: 30, column: 3, scope: !50)
!88 = !{!"llvm.loop.mustprogress"}
!89 = !DILocation(line: 31, column: 3, scope: !50)
