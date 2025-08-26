; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming.prof --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -persist-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 --func-profile-similarity-threshold=70 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl --min-call-count-for-cg-matching=10 --min-func-count-for-cg-matching=10 2>&1 | FileCheck %s  --check-prefix=TINY-FUNC

; Verify find new IR functions.
; CHECK: Function new_block_only is not in profile or profile symbol list.
; CHECK: Function new_foo is not in profile or profile symbol list.

; CHECK: Run stale profile matching for main
; CHECK: The similarity between new_foo(IR) and foo(profile) is 0.75
; CHECK: Function:new_foo matches profile:foo
; CHECK: Run stale profile matching for cold_func
; CHECK: The checksums for new_block_only(IR) and block_only(Profile) match.
; CHECK: Function:new_block_only matches profile:block_only
; CHECK: Run stale profile matching for test_noninline
; CHECK: Run stale profile matching for baz
; CHECK: Run stale profile matching for bar

; CHECK: (2/3) of functions' profile are matched and (55/81) of samples are reused by call graph matching.

; Verify the matched function is updated correctly by checking the inlining.
; CHECK: 'new_foo' inlined into 'main' to match profiling context with (cost=110, threshold=3000) at callsite main:2:7.5;
; CHECK: 'new_block_only' inlined into 'main' to match profiling context with (cost=75, threshold=3000) at callsite baz:1:3.2 @ main:3:7.6
; CHECK: 'new_block_only' inlined into 'main' to match profiling context with (cost=75, threshold=3000) at callsite baz:1:3.2 @ new_foo:2:3.3 @ main:2:7.5;
; CHECK: 'new_foo' inlined into 'test_noninline' to match profiling context with (cost=110, threshold=3000) at callsite test_noninline:1:3.2;

; CHECK: !"NumCallGraphRecoveredProfiledFunc", i64 2, !"NumCallGraphRecoveredFuncSamples", i64 55

; TINY-FUNC-NOT: Function:new_foo matches profile:foo
; TINY-FUNC-NOT: Function:new_block_only matches profile:block_only


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %x) #0 !dbg !22 {
entry:
    #dbg_value(i32 %x, !26, !DIExpression(), !27)
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !28
  %add = add nsw i32 %x, 1, !dbg !29
  ret i32 %add, !dbg !30
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define dso_local void @new_block_only() #2 !dbg !31 {
entry:
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 1, i32 0, i64 -1), !dbg !34
  %0 = load volatile i32, ptr @x, align 4, !dbg !34, !tbaa !36
  %cmp = icmp eq i32 %0, 9999, !dbg !40
  br i1 %cmp, label %if.then, label %if.else, !dbg !41

if.then:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 2, i32 0, i64 -1), !dbg !42
  %1 = load volatile i32, ptr @x, align 4, !dbg !42, !tbaa !36
  %add = add nsw i32 %1, 1000, !dbg !42
  store volatile i32 %add, ptr @x, align 4, !dbg !42, !tbaa !36
  br label %if.end10, !dbg !43

if.else:                                          ; preds = %entry
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 3, i32 0, i64 -1), !dbg !44
  %2 = load volatile i32, ptr @x, align 4, !dbg !44, !tbaa !36
  %cmp1 = icmp eq i32 %2, 999, !dbg !46
  br i1 %cmp1, label %if.then2, label %if.else4, !dbg !47

if.then2:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 4, i32 0, i64 -1), !dbg !48
  %3 = load volatile i32, ptr @x, align 4, !dbg !48, !tbaa !36
  %add3 = add nsw i32 %3, 100, !dbg !48
  store volatile i32 %add3, ptr @x, align 4, !dbg !48, !tbaa !36
  br label %if.end10, !dbg !49

if.else4:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 5, i32 0, i64 -1), !dbg !50
  %4 = load volatile i32, ptr @x, align 4, !dbg !50, !tbaa !36
  %cmp5 = icmp eq i32 %4, 99, !dbg !52
  br i1 %cmp5, label %if.then6, label %if.else8, !dbg !53

if.then6:                                         ; preds = %if.else4
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 6, i32 0, i64 -1), !dbg !54
  %5 = load volatile i32, ptr @x, align 4, !dbg !54, !tbaa !36
  %add7 = add nsw i32 %5, 10, !dbg !54
  store volatile i32 %add7, ptr @x, align 4, !dbg !54, !tbaa !36
  br label %if.end10, !dbg !55

if.else8:                                         ; preds = %if.else4
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 7, i32 0, i64 -1), !dbg !56
  %6 = load volatile i32, ptr @x, align 4, !dbg !56, !tbaa !36
  %inc = add nsw i32 %6, 1, !dbg !56
  store volatile i32 %inc, ptr @x, align 4, !dbg !56, !tbaa !36
  br label %if.end10

if.end10:                                         ; preds = %if.then2, %if.else8, %if.then6, %if.then
  call void @llvm.pseudoprobe(i64 2964250471062803127, i64 10, i32 0, i64 -1), !dbg !57
  ret void, !dbg !57
}

; Function Attrs: nounwind uwtable
define dso_local void @baz() #2 !dbg !58 {
entry:
  call void @llvm.pseudoprobe(i64 7546896869197086323, i64 1, i32 0, i64 -1), !dbg !59
  call void @new_block_only(), !dbg !60
  ret void, !dbg !62
}

; Function Attrs: nounwind uwtable
define dso_local void @new_foo() #2 !dbg !63 {
entry:
  call void @llvm.pseudoprobe(i64 5381804724291869009, i64 1, i32 0, i64 -1), !dbg !64
  %0 = load volatile i32, ptr @x, align 4, !dbg !64, !tbaa !36
  %call = call i32 @bar(i32 noundef %0), !dbg !65
  %1 = load volatile i32, ptr @x, align 4, !dbg !67, !tbaa !36
  %add = add nsw i32 %1, %call, !dbg !67
  store volatile i32 %add, ptr @x, align 4, !dbg !67, !tbaa !36
  call void @baz(), !dbg !68
  %2 = load volatile i32, ptr @x, align 4, !dbg !70, !tbaa !36
  %call1 = call i32 @bar(i32 noundef %2), !dbg !71
  %3 = load volatile i32, ptr @x, align 4, !dbg !73, !tbaa !36
  %add2 = add nsw i32 %3, %call1, !dbg !73
  store volatile i32 %add2, ptr @x, align 4, !dbg !73, !tbaa !36
  ret void, !dbg !74
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @test_noninline() #0 !dbg !75 {
entry:
  call void @llvm.pseudoprobe(i64 -5610330892148506720, i64 1, i32 0, i64 -1), !dbg !76
  call void @new_foo(), !dbg !77
  ret void, !dbg !79
}

; Function Attrs: nounwind uwtable
define dso_local void @cold_func() #2 !dbg !80 {
entry:
  call void @llvm.pseudoprobe(i64 2711072140522378707, i64 1, i32 0, i64 -1), !dbg !81
  call void @new_block_only(), !dbg !82
  ret void, !dbg !84
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #2 !dbg !85 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !91
    #dbg_value(i32 0, !89, !DIExpression(), !92)
  br label %for.cond, !dbg !93

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !94
    #dbg_value(i32 %i.0, !89, !DIExpression(), !92)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !95
  %cmp = icmp slt i32 %i.0, 1000000, !dbg !97
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !98

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !99
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 9, i32 0, i64 -1), !dbg !100
  call void @cold_func(), !dbg !101
  ret i32 0, !dbg !103

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !104
  call void @new_foo(), !dbg !106
  call void @baz(), !dbg !108
  call void @test_noninline(), !dbg !110
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 8, i32 0, i64 -1), !dbg !112
  %inc = add nsw i32 %i.0, 1, !dbg !112
    #dbg_value(i32 %inc, !89, !DIExpression(), !92)
  br label %for.cond, !dbg !113, !llvm.loop !114
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr nocapture) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #4

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16, !17, !18, !19, !20, !21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git (https://github.com/llvm/llvm-project.git 2e1509152224d8ffbeac84c489920dcbaeefc2b2)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test_rename.c", directory: "/home/wlei/local/toytest/rename", checksumkind: CSK_MD5, checksum: "b07f600b3cdefd40bd44932bc13c33f5")
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
!14 = !{!"clang version 19.0.0git (https://github.com/llvm/llvm-project.git 2e1509152224d8ffbeac84c489920dcbaeefc2b2)"}
!15 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!16 = !{i64 2964250471062803127, i64 206551239323, !"new_block_only"}
!17 = !{i64 7546896869197086323, i64 281479271677951, !"baz"}
!18 = !{i64 5381804724291869009, i64 844429225099263, !"new_foo"}
!19 = !{i64 -5610330892148506720, i64 281479271677951, !"test_noninline"}
!20 = !{i64 2711072140522378707, i64 281479271677951, !"cold_func"}
!21 = !{i64 -2624081020897602054, i64 1126003093360596, !"main"}
!22 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 3, type: !23, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{!6, !6}
!25 = !{!26}
!26 = !DILocalVariable(name: "x", arg: 1, scope: !22, file: !3, line: 3, type: !6)
!27 = !DILocation(line: 0, scope: !22)
!28 = !DILocation(line: 4, column: 10, scope: !22)
!29 = !DILocation(line: 4, column: 12, scope: !22)
!30 = !DILocation(line: 4, column: 3, scope: !22)
!31 = distinct !DISubprogram(name: "new_block_only", scope: !3, file: !3, line: 7, type: !32, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !DILocation(line: 8, column: 6, scope: !35)
!35 = distinct !DILexicalBlock(scope: !31, file: !3, line: 8, column: 6)
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocation(line: 8, column: 8, scope: !35)
!41 = !DILocation(line: 8, column: 6, scope: !31)
!42 = !DILocation(line: 9, column: 7, scope: !35)
!43 = !DILocation(line: 9, column: 5, scope: !35)
!44 = !DILocation(line: 10, column: 12, scope: !45)
!45 = distinct !DILexicalBlock(scope: !35, file: !3, line: 10, column: 12)
!46 = !DILocation(line: 10, column: 14, scope: !45)
!47 = !DILocation(line: 10, column: 12, scope: !35)
!48 = !DILocation(line: 11, column: 7, scope: !45)
!49 = !DILocation(line: 11, column: 5, scope: !45)
!50 = !DILocation(line: 12, column: 12, scope: !51)
!51 = distinct !DILexicalBlock(scope: !45, file: !3, line: 12, column: 12)
!52 = !DILocation(line: 12, column: 14, scope: !51)
!53 = !DILocation(line: 12, column: 12, scope: !45)
!54 = !DILocation(line: 13, column: 7, scope: !51)
!55 = !DILocation(line: 13, column: 5, scope: !51)
!56 = !DILocation(line: 15, column: 6, scope: !51)
!57 = !DILocation(line: 16, column: 1, scope: !31)
!58 = distinct !DISubprogram(name: "baz", scope: !3, file: !3, line: 18, type: !32, scopeLine: 18, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!59 = !DILocation(line: 19, column: 3, scope: !58)
!60 = !DILocation(line: 19, column: 3, scope: !61)
!61 = !DILexicalBlockFile(scope: !58, file: !3, discriminator: 186646551)
!62 = !DILocation(line: 20, column: 1, scope: !58)
!63 = distinct !DISubprogram(name: "new_foo", scope: !3, file: !3, line: 22, type: !32, scopeLine: 22, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!64 = !DILocation(line: 23, column: 12, scope: !63)
!65 = !DILocation(line: 23, column: 8, scope: !66)
!66 = !DILexicalBlockFile(scope: !63, file: !3, discriminator: 186646551)
!67 = !DILocation(line: 23, column: 5, scope: !63)
!68 = !DILocation(line: 24, column: 3, scope: !69)
!69 = !DILexicalBlockFile(scope: !63, file: !3, discriminator: 186646559)
!70 = !DILocation(line: 25, column: 12, scope: !63)
!71 = !DILocation(line: 25, column: 8, scope: !72)
!72 = !DILexicalBlockFile(scope: !63, file: !3, discriminator: 186646567)
!73 = !DILocation(line: 25, column: 5, scope: !63)
!74 = !DILocation(line: 26, column: 1, scope: !63)
!75 = distinct !DISubprogram(name: "test_noninline", scope: !3, file: !3, line: 28, type: !32, scopeLine: 28, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!76 = !DILocation(line: 29, column: 3, scope: !75)
!77 = !DILocation(line: 29, column: 3, scope: !78)
!78 = !DILexicalBlockFile(scope: !75, file: !3, discriminator: 186646551)
!79 = !DILocation(line: 30, column: 1, scope: !75)
!80 = distinct !DISubprogram(name: "cold_func", scope: !3, file: !3, line: 32, type: !32, scopeLine: 32, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!81 = !DILocation(line: 32, column: 20, scope: !80)
!82 = !DILocation(line: 32, column: 20, scope: !83)
!83 = !DILexicalBlockFile(scope: !80, file: !3, discriminator: 186646551)
!84 = !DILocation(line: 32, column: 37, scope: !80)
!85 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 34, type: !86, scopeLine: 34, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !88)
!86 = !DISubroutineType(types: !87)
!87 = !{!6}
!88 = !{!89}
!89 = !DILocalVariable(name: "i", scope: !90, file: !3, line: 35, type: !6)
!90 = distinct !DILexicalBlock(scope: !85, file: !3, line: 35, column: 3)
!91 = !DILocation(line: 35, column: 12, scope: !90)
!92 = !DILocation(line: 0, scope: !90)
!93 = !DILocation(line: 35, column: 8, scope: !90)
!94 = !DILocation(line: 35, scope: !90)
!95 = !DILocation(line: 35, column: 19, scope: !96)
!96 = distinct !DILexicalBlock(scope: !90, file: !3, line: 35, column: 3)
!97 = !DILocation(line: 35, column: 21, scope: !96)
!98 = !DILocation(line: 35, column: 3, scope: !90)
!99 = !DILocation(line: 0, scope: !85)
!100 = !DILocation(line: 40, column: 3, scope: !85)
!101 = !DILocation(line: 40, column: 3, scope: !102)
!102 = !DILexicalBlockFile(scope: !85, file: !3, discriminator: 186646615)
!103 = !DILocation(line: 41, column: 1, scope: !85)
!104 = !DILocation(line: 36, column: 7, scope: !105)
!105 = distinct !DILexicalBlock(scope: !96, file: !3, line: 35, column: 41)
!106 = !DILocation(line: 36, column: 7, scope: !107)
!107 = !DILexicalBlockFile(scope: !105, file: !3, discriminator: 186646575)
!108 = !DILocation(line: 37, column: 7, scope: !109)
!109 = !DILexicalBlockFile(scope: !105, file: !3, discriminator: 186646583)
!110 = !DILocation(line: 38, column: 7, scope: !111)
!111 = !DILexicalBlockFile(scope: !105, file: !3, discriminator: 186646591)
!112 = !DILocation(line: 35, column: 37, scope: !96)
!113 = !DILocation(line: 35, column: 3, scope: !96)
!114 = distinct !{!114, !98, !115, !116}
!115 = !DILocation(line: 39, column: 3, scope: !90)
!116 = !{!"llvm.loop.mustprogress"}
