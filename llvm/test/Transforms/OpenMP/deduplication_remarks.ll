; RUN: opt -openmpopt -pass-remarks=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -passes=openmpopt -pass-remarks=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; ModuleID = 'deduplication_remarks.c'
source_filename = "deduplication_remarks.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 34, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str0, i32 0, i32 0) }, align 8
@.str0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1

; CHECK: remark: deduplication_remarks.c:9:10: OpenMP runtime call __kmpc_global_thread_num moved to deduplication_remarks.c:5:10
; CHECK: remark: deduplication_remarks.c:7:10: OpenMP runtime call __kmpc_global_thread_num deduplicated
; CHECK: remark: deduplication_remarks.c:5:10: OpenMP runtime call __kmpc_global_thread_num deduplicated
define dso_local void @deduplicate() local_unnamed_addr !dbg !14 {
  %1 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0), !dbg !21
  call void @useI32(i32 %1), !dbg !23
  %2 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0), !dbg !24
  call void @useI32(i32 %2), !dbg !25
  %3 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @0), !dbg !26
  call void @useI32(i32 %3), !dbg !27
  ret void, !dbg !28
}

declare i32 @__kmpc_global_thread_num(%struct.ident_t*)

declare !dbg !4 void @useI32(i32) local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "deduplication_remarks.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "useI32", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{!"clang version 10.0.0 "}
!14 = distinct !DISubprogram(name: "deduplicate", scope: !1, file: !1, line: 4, type: !15, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18, !19, !20}
!18 = !DILocalVariable(name: "x", scope: !14, file: !1, line: 5, type: !7)
!19 = !DILocalVariable(name: "y", scope: !14, file: !1, line: 7, type: !7)
!20 = !DILocalVariable(name: "z", scope: !14, file: !1, line: 9, type: !7)
!21 = !DILocation(line: 5, column: 10, scope: !14)
!22 = !DILocation(line: 0, scope: !14)
!23 = !DILocation(line: 6, column: 2, scope: !14)
!24 = !DILocation(line: 7, column: 10, scope: !14)
!25 = !DILocation(line: 8, column: 2, scope: !14)
!26 = !DILocation(line: 9, column: 10, scope: !14)
!27 = !DILocation(line: 10, column: 2, scope: !14)
!28 = !DILocation(line: 13, column: 1, scope: !14)
