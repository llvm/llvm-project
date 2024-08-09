; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-toplev-func.prof --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 2>&1 | FileCheck %s -check-prefix=CHECK-TEXT
; RUN: llvm-profdata merge --sample %S/Inputs/pseudo-probe-stale-profile-toplev-func.prof -extbinary -o %t.extbinary
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.extbinary --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 2>&1 | FileCheck %s -check-prefix=CHECK-EXTBIN

; CHECK-TEXT: Run stale profile matching for main
; CHECK-TEXT-NOT: Read top-level function foo for call-graph matching
; CHECK-TEXT: The checksums for foo_rename(IR) and foo(Profile) match.
; CHECK-TEXT: Function:foo_rename matches profile:foo
; CHECK-TEXT: Run stale profile matching for foo_rename
; CHECK-TEXT-NOT: Top-level function foo is recovered and re-read by the sample reader.
; CHECK-TEXT: (1/3) of functions' profile are matched and (2724522/3177413) of samples are reused by call graph matching.

; CHECK-TEXT: Processing Function main
; CHECK-TEXT:     5:  call void @foo_rename(), !dbg ![[#]] - weight: 51
; CHECK-TEXT: Processing Function foo_rename
; CHECK-TEXT:     11:  %call = call i32 @bar(i32 noundef %5), !dbg ![[#]] - weight: 452687


; CHECK-EXTBIN: Run stale profile matching for main
; CHECK-EXTBIN: Read top-level function foo for call-graph matching
; CHECK-EXTBIN: The checksums for foo_rename(IR) and foo(Profile) match.
; CHECK-EXTBIN: Function:foo_rename matches profile:foo
; CHECK-EXTBIN: Run stale profile matching for foo_rename
; CHECK-EXTBIN: Top-level function foo is recovered and re-read by the sample reader.
; CHECK-EXTBIN: (1/3) of functions' profile are matched and (2724522/3177413) of samples are reused by call graph matching.

; CHECK-EXTBIN: Processing Function main
; CHECK-EXTBIN:     5:  call void @foo_rename(), !dbg ![[#]] - weight: 51
; CHECK-EXTBIN: Processing Function foo_rename
; CHECK-EXTBIN:     11:  %call = call i32 @bar(i32 noundef %5), !dbg ![[#]] - weight: 452687


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @bar(i32 noundef %x) #0 !dbg !18 {
entry:
    #dbg_value(i32 %x, !22, !DIExpression(), !23)
  call void @llvm.pseudoprobe(i64 -2012135647395072713, i64 1, i32 0, i64 -1), !dbg !24
  %add = add nsw i32 %x, 1, !dbg !25
  ret i32 %add, !dbg !26
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo_rename() #0 !dbg !27 {
entry:
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 1, i32 0, i64 -1), !dbg !33
    #dbg_value(i32 0, !31, !DIExpression(), !34)
  br label %for.cond, !dbg !35

for.cond:                                         ; preds = %if.end7, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc9, %if.end7 ], !dbg !36
    #dbg_value(i32 %i.0, !31, !DIExpression(), !34)
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 2, i32 0, i64 -1), !dbg !37
  %cmp = icmp slt i32 %i.0, 10000, !dbg !39
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !40

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 3, i32 0, i64 -1), !dbg !41
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 13, i32 0, i64 -1), !dbg !42
  ret void, !dbg !42

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 4, i32 0, i64 -1), !dbg !43
  %0 = load volatile i32, ptr @x, align 4, !dbg !43, !tbaa !46
  %rem = srem i32 %0, 3, !dbg !50
  %cmp1 = icmp eq i32 %rem, 1, !dbg !51
  br i1 %cmp1, label %if.then, label %if.else, !dbg !52

if.then:                                          ; preds = %for.body
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 5, i32 0, i64 -1), !dbg !53
  %1 = load volatile i32, ptr @x, align 4, !dbg !53, !tbaa !46
  %add = add nsw i32 %1, 100, !dbg !53
  store volatile i32 %add, ptr @x, align 4, !dbg !53, !tbaa !46
  br label %if.end7, !dbg !54

if.else:                                          ; preds = %for.body
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 6, i32 0, i64 -1), !dbg !55
  %2 = load volatile i32, ptr @x, align 4, !dbg !55, !tbaa !46
  %rem2 = srem i32 %2, 2, !dbg !57
  %cmp3 = icmp eq i32 %rem2, 1, !dbg !58
  br i1 %cmp3, label %if.then4, label %if.else6, !dbg !59

if.then4:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 7, i32 0, i64 -1), !dbg !60
  %3 = load volatile i32, ptr @x, align 4, !dbg !60, !tbaa !46
  %add5 = add nsw i32 %3, 10, !dbg !60
  store volatile i32 %add5, ptr @x, align 4, !dbg !60, !tbaa !46
  br label %if.end7, !dbg !61

if.else6:                                         ; preds = %if.else
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 8, i32 0, i64 -1), !dbg !62
  %4 = load volatile i32, ptr @x, align 4, !dbg !62, !tbaa !46
  %inc = add nsw i32 %4, 1, !dbg !62
  store volatile i32 %inc, ptr @x, align 4, !dbg !62, !tbaa !46
  br label %if.end7

if.end7:                                          ; preds = %if.then4, %if.else6, %if.then
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 10, i32 0, i64 -1), !dbg !63
  %5 = load volatile i32, ptr @x, align 4, !dbg !63, !tbaa !46
  %call = call i32 @bar(i32 noundef %5), !dbg !64
  %6 = load volatile i32, ptr @x, align 4, !dbg !66, !tbaa !46
  %add8 = add nsw i32 %6, %call, !dbg !66
  store volatile i32 %add8, ptr @x, align 4, !dbg !66, !tbaa !46
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 12, i32 0, i64 -1), !dbg !67
  %inc9 = add nsw i32 %i.0, 1, !dbg !67
    #dbg_value(i32 %inc9, !31, !DIExpression(), !34)
  br label %for.cond, !dbg !68, !llvm.loop !69
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #2 !dbg !72 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !78
    #dbg_value(i32 0, !76, !DIExpression(), !79)
  br label %for.cond, !dbg !80

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !81
    #dbg_value(i32 %i.0, !76, !DIExpression(), !79)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !82
  %cmp = icmp slt i32 %i.0, 100000, !dbg !84
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !85

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !86
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !87
  ret i32 0, !dbg !87

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !88
  call void @foo_rename(), !dbg !90
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !92
  %inc = add nsw i32 %i.0, 1, !dbg !92
    #dbg_value(i32 %inc, !76, !DIExpression(), !79)
  br label %for.cond, !dbg !93, !llvm.loop !94
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16, !17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test_rename.c", directory: "/home", checksumkind: CSK_MD5, checksum: "5c9304100fda7763e5a474c768d3b005")
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
!16 = !{i64 -2115950948644264162, i64 281718392333557, !"foo_rename"}
!17 = !{i64 -2624081020897602054, i64 281582264815352, !"main"}
!18 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 3, type: !19, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!6, !6}
!21 = !{!22}
!22 = !DILocalVariable(name: "x", arg: 1, scope: !18, file: !3, line: 3, type: !6)
!23 = !DILocation(line: 0, scope: !18)
!24 = !DILocation(line: 4, column: 10, scope: !18)
!25 = !DILocation(line: 4, column: 12, scope: !18)
!26 = !DILocation(line: 4, column: 3, scope: !18)
!27 = distinct !DISubprogram(name: "foo_rename", scope: !3, file: !3, line: 7, type: !28, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !{!31}
!31 = !DILocalVariable(name: "i", scope: !32, file: !3, line: 8, type: !6)
!32 = distinct !DILexicalBlock(scope: !27, file: !3, line: 8, column: 3)
!33 = !DILocation(line: 8, column: 12, scope: !32)
!34 = !DILocation(line: 0, scope: !32)
!35 = !DILocation(line: 8, column: 8, scope: !32)
!36 = !DILocation(line: 8, scope: !32)
!37 = !DILocation(line: 8, column: 19, scope: !38)
!38 = distinct !DILexicalBlock(scope: !32, file: !3, line: 8, column: 3)
!39 = !DILocation(line: 8, column: 21, scope: !38)
!40 = !DILocation(line: 8, column: 3, scope: !32)
!41 = !DILocation(line: 0, scope: !27)
!42 = !DILocation(line: 17, column: 1, scope: !27)
!43 = !DILocation(line: 9, column: 10, scope: !44)
!44 = distinct !DILexicalBlock(scope: !45, file: !3, line: 9, column: 10)
!45 = distinct !DILexicalBlock(scope: !38, file: !3, line: 8, column: 39)
!46 = !{!47, !47, i64 0}
!47 = !{!"int", !48, i64 0}
!48 = !{!"omnipotent char", !49, i64 0}
!49 = !{!"Simple C/C++ TBAA"}
!50 = !DILocation(line: 9, column: 12, scope: !44)
!51 = !DILocation(line: 9, column: 16, scope: !44)
!52 = !DILocation(line: 9, column: 10, scope: !45)
!53 = !DILocation(line: 10, column: 10, scope: !44)
!54 = !DILocation(line: 10, column: 8, scope: !44)
!55 = !DILocation(line: 11, column: 16, scope: !56)
!56 = distinct !DILexicalBlock(scope: !44, file: !3, line: 11, column: 16)
!57 = !DILocation(line: 11, column: 18, scope: !56)
!58 = !DILocation(line: 11, column: 22, scope: !56)
!59 = !DILocation(line: 11, column: 16, scope: !44)
!60 = !DILocation(line: 12, column: 10, scope: !56)
!61 = !DILocation(line: 12, column: 8, scope: !56)
!62 = !DILocation(line: 14, column: 9, scope: !56)
!63 = !DILocation(line: 15, column: 15, scope: !45)
!64 = !DILocation(line: 15, column: 11, scope: !65)
!65 = !DILexicalBlockFile(scope: !45, file: !3, discriminator: 455082079)
!66 = !DILocation(line: 15, column: 8, scope: !45)
!67 = !DILocation(line: 8, column: 35, scope: !38)
!68 = !DILocation(line: 8, column: 3, scope: !38)
!69 = distinct !{!69, !40, !70, !71}
!70 = !DILocation(line: 16, column: 3, scope: !32)
!71 = !{!"llvm.loop.mustprogress"}
!72 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 19, type: !73, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !75)
!73 = !DISubroutineType(types: !74)
!74 = !{!6}
!75 = !{!76}
!76 = !DILocalVariable(name: "i", scope: !77, file: !3, line: 20, type: !6)
!77 = distinct !DILexicalBlock(scope: !72, file: !3, line: 20, column: 3)
!78 = !DILocation(line: 20, column: 12, scope: !77)
!79 = !DILocation(line: 0, scope: !77)
!80 = !DILocation(line: 20, column: 8, scope: !77)
!81 = !DILocation(line: 20, scope: !77)
!82 = !DILocation(line: 20, column: 19, scope: !83)
!83 = distinct !DILexicalBlock(scope: !77, file: !3, line: 20, column: 3)
!84 = !DILocation(line: 20, column: 21, scope: !83)
!85 = !DILocation(line: 20, column: 3, scope: !77)
!86 = !DILocation(line: 0, scope: !72)
!87 = !DILocation(line: 23, column: 1, scope: !72)
!88 = !DILocation(line: 21, column: 7, scope: !89)
!89 = distinct !DILexicalBlock(scope: !83, file: !3, line: 20, column: 40)
!90 = !DILocation(line: 21, column: 7, scope: !91)
!91 = !DILexicalBlockFile(scope: !89, file: !3, discriminator: 455082031)
!92 = !DILocation(line: 20, column: 36, scope: !83)
!93 = !DILocation(line: 20, column: 3, scope: !83)
!94 = distinct !{!94, !85, !95, !71}
!95 = !DILocation(line: 22, column: 3, scope: !77)
