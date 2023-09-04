; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-lto.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-impl 2>&1 | FileCheck %s


; CHECK: Run stale profile matching for main

; CHECK: Location is matched from 1 to 1
; CHECK: Location is matched from 2 to 2
; CHECK: Location is matched from 4 to 4
; CHECK: Location is matched from 6 to 6
; CHECK: Location is matched from 7 to 7
; CHECK: Location is matched from 8 to 8
; CHECK: Location is matched from 10 to 10

; CHECK: Callsite with callee:foo is matched from 12 to 6
; CHECK: Location is rematched backwards from 7 to 1
; CHECK: Location is rematched backwards from 8 to 2
; CHECK: Location is rematched backwards from 10 to 4
; CHECK: Callsite with callee:bar is matched from 13 to 7
; CHECK: Callsite with callee:foo is matched from 14 to 8
; CHECK: Callsite with callee:bar is matched from 15 to 9


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = internal global i32 1, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !49 !prof !55 {
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !56
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !57
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 6148914874488455168), !dbg !58
  br label %1, !dbg !61

1:                                                ; preds = %17, %0
  %2 = phi i32 [ 0, %0 ], [ %28, %17 ]
  call void @llvm.dbg.value(metadata i32 %2, metadata !53, metadata !DIExpression()), !dbg !57
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !63
  %3 = load volatile i32, ptr @x, align 4, !dbg !65, !tbaa !66
  call void @llvm.dbg.value(metadata i32 %2, metadata !70, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 %3, metadata !75, metadata !DIExpression()), !dbg !76
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !79
  %4 = urem i32 %2, 10, !dbg !81
  %5 = icmp eq i32 %4, 0, !dbg !81
  %6 = zext i1 %5 to i32, !dbg !82
  %7 = add nsw i32 %3, %6, !dbg !82
  %8 = call i32 @bar(i32 noundef %7) #4, !dbg !83, !prof !84
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !85
  %9 = load volatile i32, ptr @x, align 4, !dbg !86, !tbaa !66
  %10 = add nsw i32 %9, %8, !dbg !86
  store volatile i32 %10, ptr @x, align 4, !dbg !86, !tbaa !66
  %11 = load volatile i32, ptr @x, align 4, !dbg !87, !tbaa !66
  %12 = call i32 @bar(i32 noundef %11) #4, !dbg !88, !prof !90
  %13 = load volatile i32, ptr @x, align 4, !dbg !91, !tbaa !66
  %14 = add nsw i32 %13, %12, !dbg !91
  store volatile i32 %14, ptr @x, align 4, !dbg !91, !tbaa !66
  %15 = load volatile i32, ptr @x, align 4, !dbg !92, !tbaa !66
  %16 = icmp slt i32 %15, 0, !dbg !94
  br i1 %16, label %30, label %17, !dbg !95, !prof !96

17:                                               ; preds = %1
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !97
  %18 = load volatile i32, ptr @x, align 4, !dbg !98, !tbaa !66
  call void @llvm.dbg.value(metadata i32 %2, metadata !70, metadata !DIExpression()), !dbg !99
  call void @llvm.dbg.value(metadata i32 %18, metadata !75, metadata !DIExpression()), !dbg !99
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !102
  %19 = zext i1 %5 to i32, !dbg !103
  %20 = add nsw i32 %18, %19, !dbg !103
  %21 = call i32 @bar(i32 noundef %20) #4, !dbg !104, !prof !84
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 4, i32 0, i64 -1), !dbg !105
  %22 = load volatile i32, ptr @x, align 4, !dbg !106, !tbaa !66
  %23 = add nsw i32 %22, %21, !dbg !106
  store volatile i32 %23, ptr @x, align 4, !dbg !106, !tbaa !66
  %24 = load volatile i32, ptr @x, align 4, !dbg !107, !tbaa !66
  %25 = call i32 @bar(i32 noundef %24) #4, !dbg !108, !prof !90
  %26 = load volatile i32, ptr @x, align 4, !dbg !110, !tbaa !66
  %27 = add nsw i32 %26, %25, !dbg !110
  store volatile i32 %27, ptr @x, align 4, !dbg !110, !tbaa !66
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !111
  %28 = add nuw nsw i32 %2, 1, !dbg !112
  call void @llvm.dbg.value(metadata i32 %28, metadata !53, metadata !DIExpression()), !dbg !57
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -6148914324732641280), !dbg !58
  %29 = icmp eq i32 %28, 1000000, !dbg !114
  br i1 %29, label %30, label %1, !dbg !61, !prof !116, !llvm.loop !117

30:                                               ; preds = %17, %1
  %31 = phi i32 [ 1, %1 ], [ 0, %17 ]
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !121
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 10, i32 0, i64 -1), !dbg !122
  ret i32 %31, !dbg !122
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define available_externally dso_local i32 @bar(i32 noundef %0) local_unnamed_addr #3 !dbg !123 !prof !128 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !127, metadata !DIExpression()), !dbg !129
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !130
  %2 = add nsw i32 %0, 1, !dbg !131
  ret i32 %2, !dbg !132
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!2, !7}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!46, !46}
!llvm.pseudo_probe_desc = !{!47, !48}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "file1.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "03dfbda098b1285e8c9837f184747483")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !8, producer: "clang version 18.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!8 = !DIFile(filename: "file2.c", directory: "/home", checksumkind: CSK_MD5, checksum: "bd15f56a5a8604ec94485c3dbf9d0cdd")
!9 = !{i32 7, !"Dwarf Version", i32 5}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"PIE Level", i32 2}
!14 = !{i32 7, !"uwtable", i32 2}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!17 = !{i32 1, !"ProfileSummary", !18}
!18 = !{!19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!19 = !{!"ProfileFormat", !"SampleProfile"}
!20 = !{!"TotalCount", i64 9258}
!21 = !{!"MaxCount", i64 2401}
!22 = !{!"MaxInternalCount", i64 0}
!23 = !{!"MaxFunctionCount", i64 2401}
!24 = !{!"NumCounts", i64 18}
!25 = !{!"NumFunctions", i64 2}
!26 = !{!"IsPartialProfile", i64 0}
!27 = !{!"PartialProfileRatio", double 0.000000e+00}
!28 = !{!"DetailedSummary", !29}
!29 = !{!30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45}
!30 = !{i32 10000, i64 2401, i32 1}
!31 = !{i32 100000, i64 2401, i32 1}
!32 = !{i32 200000, i64 2401, i32 1}
!33 = !{i32 300000, i64 636, i32 2}
!34 = !{i32 400000, i64 618, i32 4}
!35 = !{i32 500000, i64 614, i32 8}
!36 = !{i32 600000, i64 614, i32 8}
!37 = !{i32 700000, i64 614, i32 8}
!38 = !{i32 800000, i64 589, i32 10}
!39 = !{i32 900000, i64 554, i32 11}
!40 = !{i32 950000, i64 526, i32 12}
!41 = !{i32 990000, i64 59, i32 16}
!42 = !{i32 999000, i64 59, i32 16}
!43 = !{i32 999900, i64 59, i32 16}
!44 = !{i32 999990, i64 59, i32 16}
!45 = !{i32 999999, i64 59, i32 16}
!46 = !{!"clang version 18.0.0"}
!47 = !{i64 6699318081062747564, i64 563022570642068, !"foo"}
!48 = !{i64 -2624081020897602054, i64 1126124211449298, !"main"}
!49 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !50, scopeLine: 12, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !52)
!50 = !DISubroutineType(types: !51)
!51 = !{!6}
!52 = !{!53}
!53 = !DILocalVariable(name: "i", scope: !54, file: !3, line: 13, type: !6)
!54 = distinct !DILexicalBlock(scope: !49, file: !3, line: 13, column: 3)
!55 = !{!"function_entry_count", i64 608}
!56 = !DILocation(line: 13, column: 11, scope: !54)
!57 = !DILocation(line: 0, scope: !54)
!58 = !DILocation(line: 13, column: 18, scope: !59)
!59 = !DILexicalBlockFile(scope: !60, file: !3, discriminator: 0)
!60 = distinct !DILexicalBlock(scope: !54, file: !3, line: 13, column: 3)
!61 = !DILocation(line: 13, column: 3, scope: !62)
!62 = !DILexicalBlockFile(scope: !54, file: !3, discriminator: 2)
!63 = !DILocation(line: 14, column: 15, scope: !64)
!64 = distinct !DILexicalBlock(scope: !60, file: !3, line: 13, column: 40)
!65 = !DILocation(line: 14, column: 18, scope: !64)
!66 = !{!67, !67, i64 0}
!67 = !{!"int", !68, i64 0}
!68 = !{!"omnipotent char", !69, i64 0}
!69 = !{!"Simple C/C++ TBAA"}
!70 = !DILocalVariable(name: "i", arg: 1, scope: !71, file: !3, line: 6, type: !6)
!71 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 6, type: !72, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !74)
!72 = !DISubroutineType(types: !73)
!73 = !{!6, !6, !6}
!74 = !{!70, !75}
!75 = !DILocalVariable(name: "p", arg: 2, scope: !71, file: !3, line: 6, type: !6)
!76 = !DILocation(line: 0, scope: !71, inlinedAt: !77)
!77 = distinct !DILocation(line: 14, column: 11, scope: !78)
!78 = !DILexicalBlockFile(scope: !64, file: !3, discriminator: 186646631)
!79 = !DILocation(line: 7, column: 6, scope: !80, inlinedAt: !77)
!80 = distinct !DILexicalBlock(scope: !71, file: !3, line: 7, column: 6)
!81 = !DILocation(line: 7, column: 8, scope: !80, inlinedAt: !77)
!82 = !DILocation(line: 7, column: 6, scope: !71, inlinedAt: !77)
!83 = !DILocation(line: 0, scope: !80, inlinedAt: !77)
!84 = !{!"branch_weights", i64 614}
!85 = !DILocation(line: 9, column: 1, scope: !71, inlinedAt: !77)
!86 = !DILocation(line: 14, column: 8, scope: !64)
!87 = !DILocation(line: 15, column: 15, scope: !64)
!88 = !DILocation(line: 15, column: 11, scope: !89)
!89 = !DILexicalBlockFile(scope: !64, file: !3, discriminator: 186646639)
!90 = !{!"branch_weights", i32 614}
!91 = !DILocation(line: 15, column: 8, scope: !64)
!92 = !DILocation(line: 16, column: 9, scope: !93)
!93 = distinct !DILexicalBlock(scope: !64, file: !3, line: 16, column: 9)
!94 = !DILocation(line: 16, column: 11, scope: !93)
!95 = !DILocation(line: 16, column: 9, scope: !64)
!96 = !{!"branch_weights", i32 0, i32 614}
!97 = !DILocation(line: 18, column: 15, scope: !64)
!98 = !DILocation(line: 18, column: 18, scope: !64)
!99 = !DILocation(line: 0, scope: !71, inlinedAt: !100)
!100 = distinct !DILocation(line: 18, column: 11, scope: !101)
!101 = !DILexicalBlockFile(scope: !64, file: !3, discriminator: 186646647)
!102 = !DILocation(line: 7, column: 6, scope: !80, inlinedAt: !100)
!103 = !DILocation(line: 7, column: 6, scope: !71, inlinedAt: !100)
!104 = !DILocation(line: 0, scope: !80, inlinedAt: !100)
!105 = !DILocation(line: 9, column: 1, scope: !71, inlinedAt: !100)
!106 = !DILocation(line: 18, column: 8, scope: !64)
!107 = !DILocation(line: 19, column: 15, scope: !64)
!108 = !DILocation(line: 19, column: 11, scope: !109)
!109 = !DILexicalBlockFile(scope: !64, file: !3, discriminator: 186646655)
!110 = !DILocation(line: 19, column: 8, scope: !64)
!111 = !DILocation(line: 13, column: 36, scope: !59)
!112 = !DILocation(line: 13, column: 36, scope: !113)
!113 = !DILexicalBlockFile(scope: !60, file: !3, discriminator: 6)
!114 = !DILocation(line: 13, column: 20, scope: !115)
!115 = !DILexicalBlockFile(scope: !60, file: !3, discriminator: 2)
!116 = !{!"branch_weights", i32 608, i32 614}
!117 = distinct !{!117, !118, !119, !120}
!118 = !DILocation(line: 13, column: 3, scope: !54)
!119 = !DILocation(line: 20, column: 3, scope: !54)
!120 = !{!"llvm.loop.mustprogress"}
!121 = !DILocation(line: 0, scope: !49)
!122 = !DILocation(line: 22, column: 1, scope: !49)
!123 = distinct !DISubprogram(name: "bar", scope: !8, file: !8, line: 1, type: !124, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !126)
!124 = !DISubroutineType(types: !125)
!125 = !{!6, !6}
!126 = !{!127}
!127 = !DILocalVariable(name: "x", arg: 1, scope: !123, file: !8, line: 1, type: !6)
!128 = !{!"function_entry_count", i64 2401}
!129 = !DILocation(line: 0, scope: !123)
!130 = !DILocation(line: 5, column: 10, scope: !123)
!131 = !DILocation(line: 5, column: 12, scope: !123)
!132 = !DILocation(line: 5, column: 3, scope: !123)
