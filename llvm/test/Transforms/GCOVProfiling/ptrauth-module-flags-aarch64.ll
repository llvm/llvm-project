; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling | FileCheck %s
; REQUIRES: aarch64-registered-target

; CHECK:      define internal void @__llvm_gcov_writeout() unnamed_addr #[[#T:]] {
; CHECK:      define internal void @__llvm_gcov_reset() unnamed_addr #[[#T]] {
; CHECK:      define internal void @__llvm_gcov_init() unnamed_addr #[[#T]] {

; CHECK:      attributes #[[#T]] =
; CHECK-SAME: "aarch64-jump-table-hardening"
; CHECK-SAME: "ptrauth-auth-traps"
; CHECK-SAME: "ptrauth-indirect-gotos"
; CHECK-SAME: "ptrauth-returns"

; CHECK:      !llvm.module.flags = !{!0, !1, !2, !3, !4}
; CHECK:      !1 = !{i32 7, !"ptrauth-returns", i32 1}
; CHECK:      !2 = !{i32 7, !"ptrauth-auth-traps", i32 1}
; CHECK:      !3 = !{i32 7, !"ptrauth-indirect-gotos", i32 1}
; CHECK:      !4 = !{i32 7, !"aarch64-jump-table-hardening", i32 1}

target triple = "aarch64-unknown-linux-gnu"

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 7, !"ptrauth-returns", i32 1}
!2 = !{i32 7, !"ptrauth-auth-traps", i32 1}
!3 = !{i32 7, !"ptrauth-indirect-gotos", i32 1}
!4 = !{i32 7, !"aarch64-jump-table-hardening", i32 1}

!llvm.dbg.cu = !{!5}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6)
!6 = !DIFile(filename: "a.c", directory: "")

define void @empty() {
  ret void
}
