; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile-mismatch.prof -report-profile-staleness -S 2>%t
; RUN: FileCheck %s --input-file %t

; CHECK: (1/3) of functions' profile are invalid and (10/50) of samples are discarded due to function hash mismatch.
; CHECK: (2/3) of callsites' profile are invalid and (20/30) of samples are discarded due to callsite location mismatch.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 noundef %x) #0 !dbg !16 {
entry:
  %y = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 %x, metadata !20, metadata !DIExpression()), !dbg !22
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %y), !dbg !23
  call void @llvm.dbg.declare(metadata ptr %y, metadata !21, metadata !DIExpression()), !dbg !24
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1), !dbg !25
  %add = add nsw i32 %x, 1, !dbg !26
  store volatile i32 %add, ptr %y, align 4, !dbg !24, !tbaa !27
  %y.0. = load volatile i32, ptr %y, align 4, !dbg !31, !tbaa !27
  %add1 = add nsw i32 %y.0., 1, !dbg !32
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %y), !dbg !33
  ret i32 %add1, !dbg !34
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %x) #3 !dbg !35 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !37, metadata !DIExpression()), !dbg !38
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !39
  %add = add nsw i32 %x, 2, !dbg !40
  ret i32 %add, !dbg !41
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @matched(i32 noundef %x) #3 !dbg !42 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !44, metadata !DIExpression()), !dbg !45
  call void @llvm.pseudoprobe(i64 -5844448289301669773, i64 1, i32 0, i64 -1), !dbg !46
  %add = add nsw i32 %x, 3, !dbg !47
  ret i32 %add, !dbg !48
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !49 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !59
  call void @llvm.dbg.value(metadata i32 0, metadata !53, metadata !DIExpression()), !dbg !60
  br label %for.cond, !dbg !61

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc8, %for.cond.cleanup3 ], !dbg !60
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !53, metadata !DIExpression()), !dbg !60
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !62
  %cmp = icmp ult i32 %i.0, 1000, !dbg !64
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !65

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !67
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 11, i32 0, i64 -1), !dbg !68
  ret i32 0, !dbg !68

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !69
  call void @llvm.dbg.value(metadata i32 0, metadata !55, metadata !DIExpression()), !dbg !70
  br label %for.cond1, !dbg !71

for.cond1:                                        ; preds = %for.body4, %for.body
  %a.0 = phi i32 [ 0, %for.body ], [ %inc, %for.body4 ], !dbg !70
  call void @llvm.dbg.value(metadata i32 %a.0, metadata !55, metadata !DIExpression()), !dbg !70
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 5, i32 0, i64 -1), !dbg !72
  %cmp2 = icmp ult i32 %a.0, 10000, !dbg !75
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3, !dbg !76

for.cond.cleanup3:                                ; preds = %for.cond1
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !67
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !78
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 10, i32 0, i64 -1), !dbg !79
  %inc8 = add nuw nsw i32 %i.0, 1, !dbg !79
  call void @llvm.dbg.value(metadata i32 %inc8, metadata !53, metadata !DIExpression()), !dbg !60
  br label %for.cond, !dbg !81, !llvm.loop !82

for.body4:                                        ; preds = %for.cond1
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !86
  %0 = load volatile i32, ptr @x, align 4, !dbg !86, !tbaa !27
  %call = call i32 @matched(i32 noundef %0), !dbg !88
  store volatile i32 %call, ptr @x, align 4, !dbg !90, !tbaa !27
  %1 = load volatile i32, ptr @x, align 4, !dbg !91, !tbaa !27
  %call5 = call i32 @foo(i32 noundef %1), !dbg !92
  store volatile i32 %call5, ptr @x, align 4, !dbg !94, !tbaa !27
  %2 = load volatile i32, ptr @x, align 4, !dbg !95, !tbaa !27
  %call6 = call i32 @bar(i32 noundef %2), !dbg !96
  store volatile i32 %call6, ptr @x, align 4, !dbg !98, !tbaa !27
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !99
  %inc = add nuw nsw i32 %a.0, 1, !dbg !99
  call void @llvm.dbg.value(metadata i32 %inc, metadata !55, metadata !DIExpression()), !dbg !70
  br label %for.cond1, !dbg !101, !llvm.loop !102
}

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #4 = { inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}
!llvm.pseudo_probe_desc = !{!12, !13, !14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{!""}
!12 = !{i64 6699318081062747564, i64 4294967295, !"foo"}
!13 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!14 = !{i64 -5844448289301669773, i64 4294967295, !"matched"}
!15 = !{i64 -2624081020897602054, i64 844635331715433, !"main"}
!16 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 2, type: !17, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!6, !6}
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "x", arg: 1, scope: !16, file: !3, line: 2, type: !6)
!21 = !DILocalVariable(name: "y", scope: !16, file: !3, line: 3, type: !5)
!22 = !DILocation(line: 0, scope: !16)
!23 = !DILocation(line: 3, column: 3, scope: !16)
!24 = !DILocation(line: 3, column: 16, scope: !16)
!25 = !DILocation(line: 3, column: 20, scope: !16)
!26 = !DILocation(line: 3, column: 22, scope: !16)
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !DILocation(line: 4, column: 10, scope: !16)
!32 = !DILocation(line: 4, column: 12, scope: !16)
!33 = !DILocation(line: 5, column: 1, scope: !16)
!34 = !DILocation(line: 4, column: 3, scope: !16)
!35 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 7, type: !17, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !36)
!36 = !{!37}
!37 = !DILocalVariable(name: "x", arg: 1, scope: !35, file: !3, line: 7, type: !6)
!38 = !DILocation(line: 0, scope: !35)
!39 = !DILocation(line: 8, column: 10, scope: !35)
!40 = !DILocation(line: 8, column: 12, scope: !35)
!41 = !DILocation(line: 8, column: 3, scope: !35)
!42 = distinct !DISubprogram(name: "matched", scope: !3, file: !3, line: 11, type: !17, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !43)
!43 = !{!44}
!44 = !DILocalVariable(name: "x", arg: 1, scope: !42, file: !3, line: 11, type: !6)
!45 = !DILocation(line: 0, scope: !42)
!46 = !DILocation(line: 12, column: 10, scope: !42)
!47 = !DILocation(line: 12, column: 12, scope: !42)
!48 = !DILocation(line: 12, column: 3, scope: !42)
!49 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 15, type: !50, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !52)
!50 = !DISubroutineType(types: !51)
!51 = !{!6}
!52 = !{!53, !55}
!53 = !DILocalVariable(name: "i", scope: !54, file: !3, line: 16, type: !6)
!54 = distinct !DILexicalBlock(scope: !49, file: !3, line: 16, column: 3)
!55 = !DILocalVariable(name: "a", scope: !56, file: !3, line: 17, type: !6)
!56 = distinct !DILexicalBlock(scope: !57, file: !3, line: 17, column: 5)
!57 = distinct !DILexicalBlock(scope: !58, file: !3, line: 16, column: 34)
!58 = distinct !DILexicalBlock(scope: !54, file: !3, line: 16, column: 3)
!59 = !DILocation(line: 16, column: 12, scope: !54)
!60 = !DILocation(line: 0, scope: !54)
!61 = !DILocation(line: 16, column: 8, scope: !54)
!62 = !DILocation(line: 16, column: 19, scope: !63)
!63 = !DILexicalBlockFile(scope: !58, file: !3, discriminator: 2)
!64 = !DILocation(line: 16, column: 21, scope: !63)
!65 = !DILocation(line: 16, column: 3, scope: !66)
!66 = !DILexicalBlockFile(scope: !54, file: !3, discriminator: 2)
!67 = !DILocation(line: 0, scope: !49)
!68 = !DILocation(line: 23, column: 1, scope: !49)
!69 = !DILocation(line: 17, column: 14, scope: !56)
!70 = !DILocation(line: 0, scope: !56)
!71 = !DILocation(line: 17, column: 10, scope: !56)
!72 = !DILocation(line: 17, column: 21, scope: !73)
!73 = !DILexicalBlockFile(scope: !74, file: !3, discriminator: 2)
!74 = distinct !DILexicalBlock(scope: !56, file: !3, line: 17, column: 5)
!75 = !DILocation(line: 17, column: 23, scope: !73)
!76 = !DILocation(line: 17, column: 5, scope: !77)
!77 = !DILexicalBlockFile(scope: !56, file: !3, discriminator: 2)
!78 = !DILocation(line: 22, column: 3, scope: !57)
!79 = !DILocation(line: 16, column: 30, scope: !80)
!80 = !DILexicalBlockFile(scope: !58, file: !3, discriminator: 4)
!81 = !DILocation(line: 16, column: 3, scope: !80)
!82 = distinct !{!82, !83, !84, !85}
!83 = !DILocation(line: 16, column: 3, scope: !54)
!84 = !DILocation(line: 22, column: 3, scope: !54)
!85 = !{!"llvm.loop.mustprogress"}
!86 = !DILocation(line: 18, column: 19, scope: !87)
!87 = distinct !DILexicalBlock(scope: !74, file: !3, line: 17, column: 37)
!88 = !DILocation(line: 18, column: 11, scope: !89)
!89 = !DILexicalBlockFile(scope: !87, file: !3, discriminator: 186646631)
!90 = !DILocation(line: 18, column: 9, scope: !87)
!91 = !DILocation(line: 19, column: 15, scope: !87)
!92 = !DILocation(line: 19, column: 11, scope: !93)
!93 = !DILexicalBlockFile(scope: !87, file: !3, discriminator: 186646639)
!94 = !DILocation(line: 19, column: 9, scope: !87)
!95 = !DILocation(line: 20, column: 15, scope: !87)
!96 = !DILocation(line: 20, column: 11, scope: !97)
!97 = !DILexicalBlockFile(scope: !87, file: !3, discriminator: 186646647)
!98 = !DILocation(line: 20, column: 9, scope: !87)
!99 = !DILocation(line: 17, column: 33, scope: !100)
!100 = !DILexicalBlockFile(scope: !74, file: !3, discriminator: 4)
!101 = !DILocation(line: 17, column: 5, scope: !100)
!102 = distinct !{!102, !103, !104, !85}
!103 = !DILocation(line: 17, column: 5, scope: !56)
!104 = !DILocation(line: 21, column: 5, scope: !56)
