; RUN: opt -S -aa-pipeline= -passes='attributor,cgscc(openmp-opt-cgscc)' -openmp-opt-enable-merging < %s | FileCheck %s

; void merge_seq(int a) {
; #pragma omp parallel          // line 13
;   use(a);
;   ++a;                        // line 16, sequentialized
; #pragma omp parallel          // line 17
;   use(a);
;   use(a);                     // line 20, after both regions
; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

; CHECK-LABEL: define internal void @merge_seq..omp_par
; CHECK:       omp.par.merged:
; CHECK-NEXT:    call void (ptr, ptr, ...) @.omp_outlined.(
; CHECK-NEXT:    call i32 @__kmpc_global_thread_num(ptr {{.*}}), !dbg
; CHECK-NEXT:    call void @__kmpc_barrier(ptr {{.*}}), !dbg
; CHECK:       omp_region.end:
; CHECK-NEXT:    call i32 @__kmpc_global_thread_num(ptr {{.*}}), !dbg
; CHECK-NEXT:    call void @__kmpc_barrier(ptr {{.*}}), !dbg

define dso_local void @merge_seq(i32 %a) local_unnamed_addr !dbg !9 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4, !dbg !11
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @1, i32 1, ptr @.omp_outlined., ptr nonnull %a.addr), !dbg !12
  %0 = load i32, ptr %a.addr, align 4, !dbg !13
  %add = add nsw i32 %0, 1, !dbg !13
  store i32 %add, ptr %a.addr, align 4, !dbg !13
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @1, i32 1, ptr @.omp_outlined..1, ptr nonnull %a.addr), !dbg !14
  %1 = load i32, ptr %a.addr, align 4, !dbg !15
  call void @use(i32 %1), !dbg !15
  ret void, !dbg !15
}

define internal void @.omp_outlined.(ptr noalias nocapture readnone %.global_tid., ptr noalias nocapture readnone %.bound_tid., ptr nocapture nonnull readonly align 4 dereferenceable(4) %a) !dbg !16 {
entry:
  %0 = load i32, ptr %a, align 4, !dbg !17
  call void @use(i32 %0), !dbg !17
  ret void, !dbg !17
}

define internal void @.omp_outlined..1(ptr noalias nocapture readnone %.global_tid., ptr noalias nocapture readnone %.bound_tid., ptr nocapture nonnull readonly align 4 dereferenceable(4) %a) !dbg !18 {
entry:
  %0 = load i32, ptr %a, align 4, !dbg !19
  call void @use(i32 %0), !dbg !19
  ret void, !dbg !19
}

declare dso_local void @use(i32) local_unnamed_addr

declare !callback !1 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr

!llvm.module.flags = !{!0, !3, !4, !5}
!llvm.dbg.cu = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2}
!2 = !{i64 2, i64 -1, i64 -1, i1 true}
!3 = !{i32 7, !"openmp", i32 50}
!4 = !{i32 7, !"Debug Info Version", i32 3}
!5 = !{i32 2, !"Dwarf Version", i32 5}
!6 = distinct !DICompileUnit(language: DW_LANG_C11, file: !7, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!7 = !DIFile(filename: "merge.c", directory: "/tmp")
!8 = !DISubroutineType(types: !{null})
!9 = distinct !DISubprogram(name: "merge_seq", scope: !7, file: !7, line: 12, type: !8, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !10)
!10 = !{}
!11 = !DILocation(line: 12, column: 1, scope: !9)
!12 = !DILocation(line: 13, column: 1, scope: !9)
!13 = !DILocation(line: 16, column: 1, scope: !9)
!14 = !DILocation(line: 17, column: 1, scope: !9)
!15 = !DILocation(line: 20, column: 1, scope: !9)
!16 = distinct !DISubprogram(name: "outlined_1", scope: !7, file: !7, line: 13, type: !8, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !10)
!17 = !DILocation(line: 14, column: 1, scope: !16)
!18 = distinct !DISubprogram(name: "outlined_2", scope: !7, file: !7, line: 17, type: !8, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !10)
!19 = !DILocation(line: 18, column: 1, scope: !18)
