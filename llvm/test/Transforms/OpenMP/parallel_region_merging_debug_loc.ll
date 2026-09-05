; RUN: opt -S -aa-pipeline= -passes='attributor,cgscc(openmp-opt-cgscc)' -openmp-opt-enable-merging < %s | FileCheck %s

; The two parallel regions are merged and the code between them is
; sequentialized behind a master region. Each barrier emitted for that must
; carry the location of the code it is emitted for: line 13 for the one that
; follows the first outlined call, line 16 for the one that ends the
; sequentialized region.
;
; void merge_seq(int a) {
; #pragma omp parallel          // line 13
;   use(a);
;   ++a;                        // line 16, sequentialized
; #pragma omp parallel          // line 17
;   use(a);
; }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, ptr @0 }, align 8

; CHECK-LABEL: define internal void @merge_seq..omp_par
; CHECK:       omp.par.merged:
; CHECK-NEXT:    call void (ptr, ptr, ...) @.omp_outlined.(
; CHECK-NEXT:    call i32 @__kmpc_global_thread_num(ptr {{.*}}), !dbg ![[PAR:[0-9]+]]
; CHECK-NEXT:    call void @__kmpc_barrier(ptr {{.*}}), !dbg ![[PAR]]
; CHECK:       omp_region.end:
; CHECK-NEXT:    call i32 @__kmpc_global_thread_num(ptr {{.*}}), !dbg ![[SEQ:[0-9]+]]
; CHECK-NEXT:    call void @__kmpc_barrier(ptr {{.*}}), !dbg ![[SEQ]]

; CHECK-DAG:   ![[PAR]] = !DILocation(line: 13,
; CHECK-DAG:   ![[SEQ]] = !DILocation(line: 16,

define dso_local void @merge_seq(i32 %a) local_unnamed_addr !dbg !7 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @1, i32 1, ptr @.omp_outlined., ptr nonnull %a.addr), !dbg !8
  %0 = load i32, ptr %a.addr, align 4, !dbg !9
  %add = add nsw i32 %0, 1, !dbg !9
  store i32 %add, ptr %a.addr, align 4, !dbg !9
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @1, i32 1, ptr @.omp_outlined..1, ptr nonnull %a.addr), !dbg !10
  ret void
}

define internal void @.omp_outlined.(ptr noalias nocapture readnone %.global_tid., ptr noalias nocapture readnone %.bound_tid., ptr nocapture nonnull readonly align 4 dereferenceable(4) %a) {
entry:
  %0 = load i32, ptr %a, align 4
  call void @use(i32 %0)
  ret void
}

define internal void @.omp_outlined..1(ptr noalias nocapture readnone %.global_tid., ptr noalias nocapture readnone %.bound_tid., ptr nocapture nonnull readonly align 4 dereferenceable(4) %a) {
entry:
  %0 = load i32, ptr %a, align 4
  call void @use(i32 %0)
  ret void
}

declare dso_local void @use(i32) local_unnamed_addr

declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!4}

!0 = !{i32 7, !"openmp", i32 50}
!1 = !{i32 7, !"Debug Info Version", i32 3}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
!4 = distinct !DICompileUnit(language: DW_LANG_C11, file: !5, emissionKind: FullDebug)
!5 = !DIFile(filename: "merge.c", directory: "/tmp")
!6 = !DISubroutineType(types: !{null})
!7 = distinct !DISubprogram(name: "merge_seq", scope: !5, file: !5, line: 12, type: !6, spFlags: DISPFlagDefinition, unit: !4)
!8 = !DILocation(line: 13, column: 1, scope: !7)
!9 = !DILocation(line: 16, column: 1, scope: !7)
!10 = !DILocation(line: 17, column: 1, scope: !7)
