; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-renaming-recursive.prof --salvage-stale-profile --salvage-unused-profile -report-profile-staleness -persist-profile-staleness -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl -pass-remarks=inline --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 2>&1 | FileCheck %s

; CHECK: Run stale profile matching for main
; CHECK: Function:foo_new matches profile:foo
; CHECK: Run stale profile matching for foo_new
; CHECK: Function:bar_new matches profile:bar
; CHECK: Run stale profile matching for bar_new

; CHECK: Function processing order:
; CHECK: main
; CHECK: foo_new
; CHECK: bar_new

; CHECK: 'foo_new' inlined into 'main' to match profiling context with (cost=0, threshold=3000) at callsite main:2:7;
; CHECK: 'bar_new' inlined into 'main' to match profiling context with (cost=-15, threshold=3000) at callsite foo_new:1:3 @ main:2:7;



target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: nounwind uwtable
define dso_local void @bar_new() #0 !dbg !18 {
entry:
  call void @llvm.pseudoprobe(i64 8236371237083957767, i64 1, i32 0, i64 -1), !dbg !21
  %0 = load volatile i32, ptr @x, align 4, !dbg !21, !tbaa !22
  %inc = add nsw i32 %0, 1, !dbg !21
  store volatile i32 %inc, ptr @x, align 4, !dbg !21, !tbaa !22
  ret void, !dbg !26
}

; Function Attrs: nounwind uwtable
define dso_local void @foo_new() #0 !dbg !27 {
entry:
  call void @llvm.pseudoprobe(i64 -837213161392124280, i64 1, i32 0, i64 -1), !dbg !28
  call void @bar_new(), !dbg !29
  ret void, !dbg !31
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !32 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !38
    #dbg_value(i32 0, !36, !DIExpression(), !39)
  br label %for.cond, !dbg !40

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !41
    #dbg_value(i32 %i.0, !36, !DIExpression(), !39)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !42
  %cmp = icmp slt i32 %i.0, 1000000, !dbg !44
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !45

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !46
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !47
  ret i32 0, !dbg !47

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !48
  call void @foo_new(), !dbg !50
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !52
  %inc = add nsw i32 %i.0, 1, !dbg !52
    #dbg_value(i32 %inc, !36, !DIExpression(), !39)
  br label %for.cond, !dbg !53, !llvm.loop !54
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16, !17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "48867dcc5b42e2991317c585b7545860")
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
!14 = !{!"clang version 19.0.0"}
!15 = !{i64 8236371237083957767, i64 4294967295, !"bar_new"}
!16 = !{i64 -837213161392124280, i64 281479271677951, !"foo_new"}
!17 = !{i64 -2624081020897602054, i64 281582264815352, !"main"}
!18 = distinct !DISubprogram(name: "bar_new", scope: !3, file: !3, line: 3, type: !19, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocation(line: 4, column: 4, scope: !18)
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 5, column: 1, scope: !18)
!27 = distinct !DISubprogram(name: "foo_new", scope: !3, file: !3, line: 7, type: !19, scopeLine: 7, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!28 = !DILocation(line: 8, column: 3, scope: !27)
!29 = !DILocation(line: 8, column: 3, scope: !30)
!30 = !DILexicalBlockFile(scope: !27, file: !3, discriminator: 455082007)
!31 = !DILocation(line: 9, column: 1, scope: !27)
!32 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !33, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !35)
!33 = !DISubroutineType(types: !34)
!34 = !{!6}
!35 = !{!36}
!36 = !DILocalVariable(name: "i", scope: !37, file: !3, line: 12, type: !6)
!37 = distinct !DILexicalBlock(scope: !32, file: !3, line: 12, column: 3)
!38 = !DILocation(line: 12, column: 12, scope: !37)
!39 = !DILocation(line: 0, scope: !37)
!40 = !DILocation(line: 12, column: 8, scope: !37)
!41 = !DILocation(line: 12, scope: !37)
!42 = !DILocation(line: 12, column: 19, scope: !43)
!43 = distinct !DILexicalBlock(scope: !37, file: !3, line: 12, column: 3)
!44 = !DILocation(line: 12, column: 21, scope: !43)
!45 = !DILocation(line: 12, column: 3, scope: !37)
!46 = !DILocation(line: 0, scope: !32)
!47 = !DILocation(line: 15, column: 1, scope: !32)
!48 = !DILocation(line: 13, column: 7, scope: !49)
!49 = distinct !DILexicalBlock(scope: !43, file: !3, line: 12, column: 41)
!50 = !DILocation(line: 13, column: 7, scope: !51)
!51 = !DILexicalBlockFile(scope: !49, file: !3, discriminator: 455082031)
!52 = !DILocation(line: 12, column: 37, scope: !43)
!53 = !DILocation(line: 12, column: 3, scope: !43)
!54 = distinct !{!54, !45, !55, !56}
!55 = !DILocation(line: 14, column: 3, scope: !37)
!56 = !{!"llvm.loop.mustprogress"}
