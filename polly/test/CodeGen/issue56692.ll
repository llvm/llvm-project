; RUN: opt %loadNPMPolly -polly-parallel -polly-parallel-force -polly-omp-backend=LLVM -polly-codegen-verify -passes=polly-codegen -S < %s | FileCheck %s
; https://github.com/llvm/llvm-project/issues/56692
;
; CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}), !dbg ![[OPTLOC:[0-9]+]]
; CHECK: call void @__kmpc_dispatch_init_8({{.*}}), !dbg ![[OPTLOC]]
;
; CHECK: ![[OPTLOC]] = !DILocation(line: 0, scope: !{{[0-9]+}})

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo(i32 noundef %n, ptr noalias noundef nonnull align 8 %A) #0 !dbg !9 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !18, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata ptr %A, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !23
  %cmp3 = icmp sgt i32 %n, 0, !dbg !24
  br i1 %cmp3, label %for.body.lr.ph, label %for.end, !dbg !26

for.body.lr.ph:                                   ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !24
  br label %for.body, !dbg !26

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !20, metadata !DIExpression()), !dbg !23
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %indvars.iv, !dbg !27
  %0 = load double, ptr %arrayidx, align 8, !dbg !27, !tbaa !29
  %mul = fmul double %0, 4.200000e+01, !dbg !33
  store double %mul, ptr %arrayidx, align 8, !dbg !34, !tbaa !29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !35
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !20, metadata !DIExpression()), !dbg !23
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count, !dbg !24
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !26, !llvm.loop !36

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !26

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void, !dbg !41
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nounwind uwtable
define internal void @__kmpc_fork_call(ptr noundef %q, i32 noundef %nargs, ptr noundef %microtask, ...) #0 !dbg !42 {
entry:
  call void @llvm.dbg.value(metadata ptr %q, metadata !52, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata i32 %nargs, metadata !53, metadata !DIExpression()), !dbg !55
  call void @llvm.dbg.value(metadata ptr %microtask, metadata !54, metadata !DIExpression()), !dbg !55
  ret void, !dbg !56
}

; Function Attrs: nounwind uwtable
define internal i32 @__kmpc_dispatch_next_8(ptr noundef %loc, i32 noundef %gtid, ptr noundef %p_last, ptr noundef %p_lb, ptr noundef %p_ub, ptr noundef %p_st) #0 !dbg !57 {
entry:
  call void @llvm.dbg.value(metadata ptr %loc, metadata !70, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata i32 %gtid, metadata !71, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata ptr %p_last, metadata !72, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata ptr %p_lb, metadata !73, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata ptr %p_ub, metadata !74, metadata !DIExpression()), !dbg !76
  call void @llvm.dbg.value(metadata ptr %p_st, metadata !75, metadata !DIExpression()), !dbg !76
  ret i32 0, !dbg !77
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 15.0.0 (/home/meinersbur/src/llvm-project/clang 4e94f6653150511de434fa7e29b684ae7f0e52b6)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "parallel.c", directory: "/home/meinersbur/build/llvm-project/release_clang", checksumkind: CSK_MD5, checksum: "f66d96502f5555302321720f0cab6b0d")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 15.0.0 (/home/meinersbur/src/llvm-project/clang 4e94f6653150511de434fa7e29b684ae7f0e52b6)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 18, type: !10, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!17 = !{!18, !19, !20}
!18 = !DILocalVariable(name: "n", arg: 1, scope: !9, file: !1, line: 18, type: !12)
!19 = !DILocalVariable(name: "A", arg: 2, scope: !9, file: !1, line: 18, type: !13)
!20 = !DILocalVariable(name: "i", scope: !21, file: !1, line: 19, type: !12)
!21 = distinct !DILexicalBlock(scope: !9, file: !1, line: 19, column: 5)
!22 = !DILocation(line: 0, scope: !9)
!23 = !DILocation(line: 0, scope: !21)
!24 = !DILocation(line: 19, column: 23, scope: !25)
!25 = distinct !DILexicalBlock(scope: !21, file: !1, line: 19, column: 5)
!26 = !DILocation(line: 19, column: 5, scope: !21)
!27 = !DILocation(line: 20, column: 21, scope: !28)
!28 = distinct !DILexicalBlock(scope: !25, file: !1, line: 19, column: 33)
!29 = !{!30, !30, i64 0}
!30 = !{!"double", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 20, column: 19, scope: !28)
!34 = !DILocation(line: 20, column: 14, scope: !28)
!35 = !DILocation(line: 19, column: 28, scope: !25)
!36 = distinct !{!36, !26, !37, !38}
!37 = !DILocation(line: 21, column: 5, scope: !21)
!38 = !{!"llvm.loop.mustprogress"}
!39 = !DILocation(line: 23, column: 5, scope: !9)
!40 = !DILocation(line: 24, column: 5, scope: !9)
!41 = !DILocation(line: 25, column: 1, scope: !9)
!42 = distinct !DISubprogram(name: "__kmpc_fork_call", scope: !1, file: !1, line: 9, type: !43, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !51)
!43 = !DISubroutineType(types: !44)
!44 = !{null, !45, !47, !45, null}
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64)
!46 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!47 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !48, line: 26, baseType: !49)
!48 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h", directory: "", checksumkind: CSK_MD5, checksum: "55bcbdc3159515ebd91d351a70d505f4")
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !50, line: 41, baseType: !12)
!50 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "", checksumkind: CSK_MD5, checksum: "d108b5f93a74c50510d7d9bc0ab36df9")
!51 = !{!52, !53, !54}
!52 = !DILocalVariable(name: "q", arg: 1, scope: !42, file: !1, line: 9, type: !45)
!53 = !DILocalVariable(name: "nargs", arg: 2, scope: !42, file: !1, line: 9, type: !47)
!54 = !DILocalVariable(name: "microtask", arg: 3, scope: !42, file: !1, line: 9, type: !45)
!55 = !DILocation(line: 0, scope: !42)
!56 = !DILocation(line: 10, column: 1, scope: !42)
!57 = distinct !DISubprogram(name: "__kmpc_dispatch_next_8", scope: !1, file: !1, line: 12, type: !58, scopeLine: 14, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !69)
!58 = !DISubroutineType(types: !59)
!59 = !{!12, !60, !62, !63, !64, !64, !64}
!60 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !61, size: 64)
!61 = !DICompositeType(tag: DW_TAG_structure_type, name: "ident_t", file: !1, line: 5, flags: DIFlagFwdDecl)
!62 = !DIDerivedType(tag: DW_TAG_typedef, name: "kmp_int32", file: !1, line: 6, baseType: !47)
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !62, size: 64)
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !65, size: 64)
!65 = !DIDerivedType(tag: DW_TAG_typedef, name: "kmp_int64", file: !1, line: 7, baseType: !66)
!66 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !48, line: 27, baseType: !67)
!67 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int64_t", file: !50, line: 44, baseType: !68)
!68 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!69 = !{!70, !71, !72, !73, !74, !75}
!70 = !DILocalVariable(name: "loc", arg: 1, scope: !57, file: !1, line: 12, type: !60)
!71 = !DILocalVariable(name: "gtid", arg: 2, scope: !57, file: !1, line: 12, type: !62)
!72 = !DILocalVariable(name: "p_last", arg: 3, scope: !57, file: !1, line: 13, type: !63)
!73 = !DILocalVariable(name: "p_lb", arg: 4, scope: !57, file: !1, line: 13, type: !64)
!74 = !DILocalVariable(name: "p_ub", arg: 5, scope: !57, file: !1, line: 14, type: !64)
!75 = !DILocalVariable(name: "p_st", arg: 6, scope: !57, file: !1, line: 14, type: !64)
!76 = !DILocation(line: 0, scope: !57)
!77 = !DILocation(line: 15, column: 37, scope: !57)
