; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-name-similarity.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

; CHECK: Function _Z3fool is not in profile or profile symbol list.
; CHECK: Run stale profile matching for main
; CHECK: The functions _Z3fool(IR) and _Z3fooi(Profile) share the same base name: foo
; CHECK: Function:_Z3fool matches profile:_Z3fooi
; CHECK: Run stale profile matching for _Z3fool


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: mustprogress noinline nounwind uwtable
define dso_local void @_Z3fool(i64 noundef %y) #0 !dbg !17 {
entry:
    #dbg_value(i64 %y, !22, !DIExpression(), !23)
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 1, i32 0, i64 -1), !dbg !24
  %0 = load volatile i32, ptr @x, align 4, !dbg !25, !tbaa !26
  %conv = sext i32 %0 to i64, !dbg !25
  %add = add nsw i64 %conv, %y, !dbg !25
  %conv1 = trunc i64 %add to i32, !dbg !25
  store volatile i32 %conv1, ptr @x, align 4, !dbg !25, !tbaa !26
  ret void, !dbg !30
}

; Function Attrs: mustprogress norecurse nounwind uwtable
define dso_local noundef i32 @main() #1 !dbg !31 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !37
    #dbg_value(i32 0, !35, !DIExpression(), !38)
  br label %for.cond, !dbg !39

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !40
    #dbg_value(i32 %i.0, !35, !DIExpression(), !38)
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !41
  %cmp = icmp slt i32 %i.0, 1000000, !dbg !43
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !44

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 3, i32 0, i64 -1), !dbg !45
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 7, i32 0, i64 -1), !dbg !46
  ret i32 0, !dbg !46

for.body:                                         ; preds = %for.cond
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !47
  %conv = sext i32 %i.0 to i64, !dbg !47
  call void @_Z3fool(i64 noundef %conv), !dbg !49
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 6, i32 0, i64 -1), !dbg !51
  %inc = add nsw i32 %i.0, 1, !dbg !51
    #dbg_value(i32 %inc, !35, !DIExpression(), !38)
  br label %for.cond, !dbg !52, !llvm.loop !53
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #3

attributes #0 = { mustprogress noinline nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { mustprogress norecurse nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}
!llvm.pseudo_probe_desc = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 21.0.0git (https://github.com/llvm/llvm-project.git c9f1d2cbf18990311ea1287cc154e3784a10a3b0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test_rename.c", directory: "/home", checksumkind: CSK_MD5, checksum: "2991f6c78cef4c393285c97c0f5dabc4")
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
!14 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git c9f1d2cbf18990311ea1287cc154e3784a10a3b0)"}
!15 = !{i64 5326982120444056491, i64 4294967295, !"_Z3fool"}
!16 = !{i64 -2624081020897602054, i64 281582264815352, !"main"}
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !3, file: !3, line: 3, type: !18, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!21 = !{!22}
!22 = !DILocalVariable(name: "y", arg: 1, scope: !17, file: !3, line: 3, type: !20)
!23 = !DILocation(line: 0, scope: !17)
!24 = !DILocation(line: 4, column: 9, scope: !17)
!25 = !DILocation(line: 4, column: 6, scope: !17)
!26 = !{!27, !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !DILocation(line: 5, column: 1, scope: !17)
!31 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !32, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !34)
!32 = !DISubroutineType(types: !33)
!33 = !{!6}
!34 = !{!35}
!35 = !DILocalVariable(name: "i", scope: !36, file: !3, line: 8, type: !6)
!36 = distinct !DILexicalBlock(scope: !31, file: !3, line: 8, column: 3)
!37 = !DILocation(line: 8, column: 12, scope: !36)
!38 = !DILocation(line: 0, scope: !36)
!39 = !DILocation(line: 8, column: 8, scope: !36)
!40 = !DILocation(line: 8, scope: !36)
!41 = !DILocation(line: 8, column: 19, scope: !42)
!42 = distinct !DILexicalBlock(scope: !36, file: !3, line: 8, column: 3)
!43 = !DILocation(line: 8, column: 21, scope: !42)
!44 = !DILocation(line: 8, column: 3, scope: !36)
!45 = !DILocation(line: 0, scope: !31)
!46 = !DILocation(line: 11, column: 1, scope: !31)
!47 = !DILocation(line: 9, column: 11, scope: !48)
!48 = distinct !DILexicalBlock(scope: !42, file: !3, line: 8, column: 41)
!49 = !DILocation(line: 9, column: 7, scope: !50)
!50 = !DILexicalBlockFile(scope: !48, file: !3, discriminator: 455082031)
!51 = !DILocation(line: 8, column: 37, scope: !42)
!52 = !DILocation(line: 8, column: 3, scope: !42)
!53 = distinct !{!53, !44, !54, !55}
!54 = !DILocation(line: 10, column: 3, scope: !36)
!55 = !{!"llvm.loop.mustprogress"}
