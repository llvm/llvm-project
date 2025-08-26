; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-toplev-func.prof --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 --load-func-profile-for-cg-matching 2>&1 | FileCheck %s -check-prefix=CHECK-TEXT
; RUN: llvm-profdata merge --sample %S/Inputs/pseudo-probe-stale-profile-toplev-func.prof -extbinary -o %t.extbinary
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.extbinary --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 --load-func-profile-for-cg-matching 2>&1 | FileCheck %s -check-prefix=CHECK-EXTBIN

; CHECK-TEXT: Run stale profile matching for main
; CHECK-TEXT-NOT: Read top-level function foo for call-graph matching
; CHECK-TEXT: The checksums for foo_rename(IR) and foo(Profile) match.
; CHECK-TEXT: Function:foo_rename matches profile:foo
; CHECK-TEXT: Run stale profile matching for foo_rename
; CHECK-TEXT: (1/3) of functions' profile are matched and (2724522/3177413) of samples are reused by call graph matching.

; CHECK-TEXT: Processing Function main
; CHECK-TEXT:     5:  call void @foo_rename(), !dbg ![[#]] - weight: 51
; CHECK-TEXT: Processing Function foo_rename
; CHECK-TEXT:     2:  %call = call i32 @bar(i32 noundef %0), !dbg ![[#]] - weight: 452674


; CHECK-EXTBIN: Run stale profile matching for main
; CHECK-EXTBIN: Read top-level function foo for call-graph matching
; CHECK-EXTBIN: The checksums for foo_rename(IR) and foo(Profile) match.
; CHECK-EXTBIN: Function:foo_rename matches profile:foo
; CHECK-EXTBIN: Run stale profile matching for foo_rename
; CHECK-EXTBIN: (1/3) of functions' profile are matched and (2724522/3177413) of samples are reused by call graph matching.

; CHECK-EXTBIN: Processing Function main
; CHECK-EXTBIN:     5:  call void @foo_rename(), !dbg ![[#]] - weight: 51
; CHECK-EXTBIN: Processing Function foo_rename
; CHECK-EXTBIN:     2:  %call = call i32 @bar(i32 noundef %0), !dbg ![[#]] - weight: 452674


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
  call void @llvm.pseudoprobe(i64 -2115950948644264162, i64 1, i32 0, i64 -1), !dbg !30
  %0 = load volatile i32, ptr @x, align 4, !dbg !30, !tbaa !31
  %call = call i32 @bar(i32 noundef %0), !dbg !35
  %1 = load volatile i32, ptr @x, align 4, !dbg !37, !tbaa !31
  %add = add nsw i32 %1, %call, !dbg !37
  store volatile i32 %add, ptr @x, align 4, !dbg !37, !tbaa !31
  ret void, !dbg !38
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #1 !dbg !39 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !45
    #dbg_value(i32 0, !43, !DIExpression(), !46)
  br label %for.cond, !dbg !47

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !48
    #dbg_value(i32 %i.0, !43, !DIExpression(), !46)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !49
  %cmp = icmp slt i32 %i.0, 100000, !dbg !51
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !52

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !53
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !54
  ret i32 0, !dbg !54

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !55
  call void @foo_rename(), !dbg !57
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !59
  %inc = add nsw i32 %i.0, 1, !dbg !59
    #dbg_value(i32 %inc, !43, !DIExpression(), !46)
  br label %for.cond, !dbg !60, !llvm.loop !61
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr nocapture) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr nocapture) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16, !17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test_rename.c", directory: "/home", checksumkind: CSK_MD5, checksum: "11a33a83e4d190ebda0792d0610f0c67")
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
!16 = !{i64 -2115950948644264162, i64 281479271677951, !"foo_rename"}
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
!27 = distinct !DISubprogram(name: "foo_rename", scope: !3, file: !3, line: 7, type: !28, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !DILocation(line: 8, column: 15, scope: !27)
!31 = !{!32, !32, i64 0}
!32 = !{!"int", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 8, column: 11, scope: !36)
!36 = !DILexicalBlockFile(scope: !27, file: !3, discriminator: 455082007)
!37 = !DILocation(line: 8, column: 8, scope: !27)
!38 = !DILocation(line: 9, column: 1, scope: !27)
!39 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !40, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !42)
!40 = !DISubroutineType(types: !41)
!41 = !{!6}
!42 = !{!43}
!43 = !DILocalVariable(name: "i", scope: !44, file: !3, line: 12, type: !6)
!44 = distinct !DILexicalBlock(scope: !39, file: !3, line: 12, column: 3)
!45 = !DILocation(line: 12, column: 12, scope: !44)
!46 = !DILocation(line: 0, scope: !44)
!47 = !DILocation(line: 12, column: 8, scope: !44)
!48 = !DILocation(line: 12, scope: !44)
!49 = !DILocation(line: 12, column: 19, scope: !50)
!50 = distinct !DILexicalBlock(scope: !44, file: !3, line: 12, column: 3)
!51 = !DILocation(line: 12, column: 21, scope: !50)
!52 = !DILocation(line: 12, column: 3, scope: !44)
!53 = !DILocation(line: 0, scope: !39)
!54 = !DILocation(line: 15, column: 1, scope: !39)
!55 = !DILocation(line: 13, column: 7, scope: !56)
!56 = distinct !DILexicalBlock(scope: !50, file: !3, line: 12, column: 40)
!57 = !DILocation(line: 13, column: 7, scope: !58)
!58 = !DILexicalBlockFile(scope: !56, file: !3, discriminator: 455082031)
!59 = !DILocation(line: 12, column: 36, scope: !50)
!60 = !DILocation(line: 12, column: 3, scope: !50)
!61 = distinct !{!61, !52, !62, !63}
!62 = !DILocation(line: 14, column: 3, scope: !44)
!63 = !{!"llvm.loop.mustprogress"}
