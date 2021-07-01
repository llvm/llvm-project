; RUN: opt -S -pass-remarks=openmp-opt -attributor -openmp-opt-cgscc -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -pass-remarks=openmp-opt -passes='attributor,cgscc(openmp-opt-cgscc)' -disable-output < %s 2>&1 | FileCheck %s
; ModuleID = 'parallel_deletion_remarks.ll'
source_filename = "parallel_deletion_remarks.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8

; void delete_parallel(void) {
; #pragma omp parallel
; 	{ unknown_willreturn(); }
; #pragma omp parallel
; 	{ readonly_willreturn(); }
; #pragma omp parallel
; 	{ readnone_willreturn(); }
; #pragma omp parallel
; 	{}
; }
;
; This will delete all but the first parallel region

; CHECK: remark: parallel_deletion_remarks.c:10:1: Parallel region in delete_parallel deleted
; CHECK: remark: parallel_deletion_remarks.c:12:1: Parallel region in delete_parallel deleted
; CHECK: remark: parallel_deletion_remarks.c:14:1: Parallel region in delete_parallel deleted
define dso_local void @delete_parallel() local_unnamed_addr !dbg !15 {
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*)), !dbg !18
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..2 to void (i32*, i32*, ...)*)), !dbg !19
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..4 to void (i32*, i32*, ...)*)), !dbg !20
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @.omp_outlined..6 to void (i32*, i32*, ...)*)), !dbg !21
  ret void, !dbg !22
}

declare !callback !23 void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) local_unnamed_addr

; Function Attrs: willreturn
declare !dbg !4 void @unknown_willreturn(...) #0

; Function Attrs: readonly willreturn
declare !dbg !7 void @readonly_willreturn(...) #1

; Function Attrs: readnone willreturn
declare !dbg !8 void @readnone_willreturn(...) #2

define internal void @.omp_outlined.(i32* noalias nocapture readnone %0, i32* noalias nocapture readnone %1) !dbg !25 {
  call void (...) @unknown_willreturn(), !dbg !36
  ret void, !dbg !36
}

define internal void @.omp_outlined..2(i32* noalias nocapture readnone %0, i32* noalias nocapture readnone %1) !dbg !37 {
  call void (...) @readonly_willreturn(), !dbg !41
  ret void, !dbg !41
}

define internal void @.omp_outlined..4(i32* noalias nocapture readnone %0, i32* noalias nocapture readnone %1) !dbg !42 {
  call void (...) @readnone_willreturn(), !dbg !46
  ret void, !dbg !46
}

define internal void @.omp_outlined..6(i32* noalias nocapture %0, i32* noalias nocapture %1) !dbg !47 {
  ret void, !dbg !51
}

attributes #0 = { willreturn }
attributes #1 = { readonly willreturn }
attributes #2 = { readnone willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11, !12, !13, !52}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "parallel_deletion_remarks.c", directory: "/tmp")
!2 = !{}
!3 = !{!4, !7, !8}
!4 = !DISubprogram(name: "unknown_willreturn", scope: !1, file: !1, line: 3, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, null}
!7 = !DISubprogram(name: "readonly_willreturn", scope: !1, file: !1, line: 4, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!8 = !DISubprogram(name: "readnone_willreturn", scope: !1, file: !1, line: 5, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"PIC Level", i32 2}
!13 = !{i32 7, !"PIE Level", i32 2}
!14 = !{!"clang version 10.0.0 "}
!15 = distinct !DISubprogram(name: "delete_parallel", scope: !1, file: !1, line: 7, type: !16, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DILocation(line: 8, column: 1, scope: !15)
!19 = !DILocation(line: 10, column: 1, scope: !15)
!20 = !DILocation(line: 12, column: 1, scope: !15)
!21 = !DILocation(line: 14, column: 1, scope: !15)
!22 = !DILocation(line: 16, column: 1, scope: !15)
!23 = !{!24}
!24 = !{i64 2, i64 -1, i64 -1, i1 true}
!25 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 9, type: !26, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !33)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !28, !28}
!28 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !29)
!29 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!31 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !32)
!32 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!33 = !{!34, !35}
!34 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !25, type: !28, flags: DIFlagArtificial)
!35 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !25, type: !28, flags: DIFlagArtificial)
!36 = !DILocation(line: 9, column: 2, scope: !25)
!37 = distinct !DISubprogram(name: ".omp_outlined..2", scope: !1, file: !1, line: 11, type: !26, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !38)
!38 = !{!39, !40}
!39 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !37, type: !28, flags: DIFlagArtificial)
!40 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !37, type: !28, flags: DIFlagArtificial)
!41 = !DILocation(line: 11, column: 2, scope: !37)
!42 = distinct !DISubprogram(name: ".omp_outlined..4", scope: !1, file: !1, line: 13, type: !26, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !43)
!43 = !{!44, !45}
!44 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !42, type: !28, flags: DIFlagArtificial)
!45 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !42, type: !28, flags: DIFlagArtificial)
!46 = !DILocation(line: 13, column: 2, scope: !42)
!47 = distinct !DISubprogram(name: ".omp_outlined..6", scope: !1, file: !1, line: 15, type: !26, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !48)
!48 = !{!49, !50}
!49 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !47, type: !28, flags: DIFlagArtificial)
!50 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !47, type: !28, flags: DIFlagArtificial)
!51 = !DILocation(line: 15, column: 2, scope: !47)
!52 = !{i32 7, !"openmp", i32 50}
