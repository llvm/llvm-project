; RUN: opt -S -passes=next-silicon-relocate-variadic %s | FileCheck %s

declare void @__kmpc_fork_teams(ptr, i32, ptr, ...)

declare void @__kmpc_fork_call(ptr, i32, ptr, ...)

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_for_static_fini(ptr, i32)

;; Confirm that __kmpc_fork_teams call is correctly transformed to __next_boundary_thunk.
; CHECK-LABEL: define i32 @main
; CHECK:    %[[TMP1:[[:alnum:]]+]] = alloca [1000 x i32], align 16
; CHECK:    call void @__next_boundary_thunk1(ptr %[[TMP1]], i32 2, ptr %[[TMP1]], i32 42)
; CHECK:    ret i32 0
define i32 @main() {
  %1 = alloca [1000 x i32], align 16
  call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr nonnull %1, i32 2, ptr nonnull @.omp_outlined., ptr nonnull %1, i32 42)
  ret i32 0
}

;; Confirm that __kmpc_fork_call call is correctly transformed to __next_boundary_thunk.
; CHECK-LABEL: define void @.omp_outlined.
; CHECK-SAME:(ptr %{{[[:alnum:]]}}, ptr %[[TMP2:[[:alnum:]]+]]
; CHECK:    call void @__next_boundary_thunk2(ptr %[[TMP2]], i32 2, ptr %[[TMP2]], i32 42)
define void @.omp_outlined.(ptr %0, ptr %1, ptr noalias nocapture %2, i32 noundef %3) {
  %5 = alloca i32, align 4
  %6 = load i32, ptr %0, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull %0, i32 %6, i32 34, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, i32 1, i32 1)
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull %1, i32 2, ptr nonnull @.omp_outlined.omp_outlined., ptr nonnull %1, i32 42)
  call void @__kmpc_for_static_fini(ptr nonnull %0, i32 %6)
  ret void
}

;; Check that the boundary thunk is generated correctly for __kmpc_fork_teams.
; CHECK-LABEL: define internal void @__next_boundary_thunk1
; CHECK-SAME: ptr %[[TMP0:[[:alnum:]]+]], i32 %[[TMP1:[[:alnum:]]+]], ptr noalias nocapture %[[TMP2:[[:alnum:]]+]], i32 noundef %[[TMP3:[[:alnum:]]+]])
; CHECK-SAME: #[[$BOUNDARY_ATTR:[[:alnum:]]+]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__next_variadic_callsite_thunk1(ptr %[[TMP0]], i32 %[[TMP1]], ptr @.omp_outlined., ptr %[[TMP2]], i32 %[[TMP3]], ptr @__kmpc_fork_teams)
; CHECK-NEXT:    ret void

; CHECK-LABEL: define void @.omp_outlined.omp_outlined.
define void @.omp_outlined.omp_outlined.(ptr %0, ptr %1, ptr noalias nocapture %2, i32 noundef %3) {
  %5 = alloca i32, align 4
  %6 = load i32, ptr %0, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull %0, i32 %6, i32 34, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, i32 1, i32 1)
  call void @__kmpc_for_static_fini(ptr nonnull %0, i32 %6)
  ret void
}

;; Check that the boundary thunk is generated correctly for __kmpc_fork_call.
; CHECK-LABEL: define internal void @__next_boundary_thunk2
; CHECK-SAME: ptr %[[TMP0:[[:alnum:]]+]], i32 %[[TMP1:[[:alnum:]]+]], ptr noalias nocapture %[[TMP2:[[:alnum:]]+]], i32 noundef %[[TMP3:[[:alnum:]]+]])
; CHECK-SAME: #[[$BOUNDARY_ATTR]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__next_variadic_callsite_thunk1(ptr %[[TMP0]], i32 %[[TMP1]], ptr @.omp_outlined.omp_outlined., ptr %[[TMP2]], i32 %[[TMP3]], ptr @__kmpc_fork_call)
; CHECK-NEXT:    ret void

; CHECK: attributes #[[$BOUNDARY_ATTR]] = { noinline "ns-boundary-function" }

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "omp_lit.c", directory: "/")
!2 = !{!"NextSilicon clang version 16.0.4 (git@github.com:nextsilicon/next-llvm-project.git 8f2830cab3d60993a39c6c67df49792cf2572c54)"}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
