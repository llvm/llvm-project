; RUN: opt -passes=openmp-opt-cgscc -pass-remarks-analysis=openmp-opt -openmp-print-icv-values -disable-output < %s 2>&1 | FileCheck %s

; ModuleID = 'icv_remarks.c'
source_filename = "icv_remarks.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @.str }, align 8
@1 = private unnamed_addr constant [26 x i8] c";icv_remarks.c;foo;18;1;;\00", align 1

; CHECK-DAG: remark: icv_remarks.c:12:0: OpenMP ICV nthreads Value: IMPLEMENTATION_DEFINED
; CHECK-DAG: remark: icv_remarks.c:12:0: OpenMP ICV active_levels Value: 0
; CHECK-DAG: remark: icv_remarks.c:12:0: OpenMP ICV cancel Value: 0
define dso_local void @foo(i32 %a) local_unnamed_addr #0 !dbg !17 {
entry:
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 dereferenceable(24) %.kmpc_loc.addr, ptr nonnull align 8 dereferenceable(24) @0, i64 16, i1 false)
  call void @llvm.dbg.value(metadata i32 %a, metadata !19, metadata !DIExpression()), !dbg !21
  tail call void @omp_set_num_threads(i32 %a) #1, !dbg !22
  %call = tail call i32 @omp_get_max_threads() #1, !dbg !23
  call void @llvm.dbg.value(metadata i32 %call, metadata !20, metadata !DIExpression()), !dbg !21
  tail call void @use(i32 %call) #1, !dbg !24
  %0 = getelementptr inbounds %struct.ident_t, ptr %.kmpc_loc.addr, i64 0, i32 4, !dbg !25
  store ptr @1, ptr %0, align 8, !dbg !25, !tbaa !26
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull %.kmpc_loc.addr, i32 0, ptr @.omp_outlined.) #1, !dbg !25
  ret void, !dbg !32
}

declare !dbg !4 dso_local void @omp_set_num_threads(i32) local_unnamed_addr #1

declare !dbg !9 dso_local i32 @omp_get_max_threads() local_unnamed_addr #1

declare !dbg !12 dso_local void @use(i32) local_unnamed_addr #2

; CHECK-DAG: remark: icv_remarks.c:18:0: OpenMP ICV nthreads Value: IMPLEMENTATION_DEFINED
; CHECK-DAG: remark: icv_remarks.c:18:0: OpenMP ICV active_levels Value: 0
; CHECK-DAG: remark: icv_remarks.c:18:0: OpenMP ICV cancel Value: 0
define internal void @.omp_outlined.(ptr noalias nocapture readnone %.global_tid., ptr noalias nocapture readnone %.bound_tid.) #3 !dbg !33 {
entry:
  call void @llvm.dbg.value(metadata ptr %.global_tid., metadata !41, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata ptr %.bound_tid., metadata !42, metadata !DIExpression()), !dbg !43
  call void @llvm.dbg.value(metadata ptr undef, metadata !44, metadata !DIExpression()) #1, !dbg !50
  call void @llvm.dbg.value(metadata ptr undef, metadata !47, metadata !DIExpression()) #1, !dbg !50
  tail call void @omp_set_num_threads(i32 10) #1, !dbg !52
  %call.i = tail call i32 @omp_get_max_threads() #1, !dbg !53
  call void @llvm.dbg.value(metadata i32 %call.i, metadata !48, metadata !DIExpression()) #1, !dbg !54
  tail call void @use(i32 %call.i) #1, !dbg !55
  ret void, !dbg !56
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #4

declare !callback !57 dso_local void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #1

declare void @llvm.dbg.value(metadata, metadata, metadata) #5

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind willreturn }
attributes #5 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15, !59}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 73cea83a6f5ab521edf3cccfc603534776d691ec)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "icv_remarks.c", directory: "/tmp")
!2 = !{}
!3 = !{!4, !9, !12}
!4 = !DISubprogram(name: "omp_set_num_threads", scope: !5, file: !5, line: 57, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DIFile(filename: "/usr/local/lib/clang/11.0.0/include/omp.h", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DISubprogram(name: "omp_get_max_threads", scope: !5, file: !5, line: 67, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!8}
!12 = !DISubprogram(name: "use", scope: !1, file: !1, line: 10, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 73cea83a6f5ab521edf3cccfc603534776d691ec)"}
!17 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 12, type: !6, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!18 = !{!19, !20}
!19 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !1, line: 12, type: !8)
!20 = !DILocalVariable(name: "num", scope: !17, file: !1, line: 15, type: !8)
!21 = !DILocation(line: 0, scope: !17)
!22 = !DILocation(line: 13, column: 3, scope: !17)
!23 = !DILocation(line: 15, column: 13, scope: !17)
!24 = !DILocation(line: 17, column: 3, scope: !17)
!25 = !DILocation(line: 18, column: 1, scope: !17)
!26 = !{!27, !31, i64 16}
!27 = !{!"ident_t", !28, i64 0, !28, i64 4, !28, i64 8, !28, i64 12, !31, i64 16}
!28 = !{!"int", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !{!"any pointer", !29, i64 0}
!32 = !DILocation(line: 24, column: 1, scope: !17)
!33 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 18, type: !34, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !40)
!34 = !DISubroutineType(types: !35)
!35 = !{null, !36, !36}
!36 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !37)
!37 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !38)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !39, size: 64)
!39 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!40 = !{!41, !42}
!41 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !33, type: !36, flags: DIFlagArtificial)
!42 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !33, type: !36, flags: DIFlagArtificial)
!43 = !DILocation(line: 0, scope: !33)
!44 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !45, type: !36, flags: DIFlagArtificial)
!45 = distinct !DISubprogram(name: ".omp_outlined._debug__", scope: !1, file: !1, line: 19, type: !34, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46)
!46 = !{!44, !47, !48}
!47 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !45, type: !36, flags: DIFlagArtificial)
!48 = !DILocalVariable(name: "num1", scope: !49, file: !1, line: 21, type: !8)
!49 = distinct !DILexicalBlock(scope: !45, file: !1, line: 19, column: 3)
!50 = !DILocation(line: 0, scope: !45, inlinedAt: !51)
!51 = distinct !DILocation(line: 18, column: 1, scope: !33)
!52 = !DILocation(line: 20, column: 5, scope: !49, inlinedAt: !51)
!53 = !DILocation(line: 21, column: 16, scope: !49, inlinedAt: !51)
!54 = !DILocation(line: 0, scope: !49, inlinedAt: !51)
!55 = !DILocation(line: 22, column: 5, scope: !49, inlinedAt: !51)
!56 = !DILocation(line: 18, column: 1, scope: !33)
!57 = !{!58}
!58 = !{i64 2, i64 -1, i64 -1, i1 true}
!59 = !{i32 7, !"openmp", i32 50}
