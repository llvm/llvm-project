; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/funcsig.ll \
; RUN:   | FileCheck %s --check-prefix=FUNCSIG
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/unlabeled.ll \
; RUN:   | FileCheck %s --check-prefix=UNLABELED

; Mirrors cf-branch-ctor-default-label.ll, but for llvm.global_dtors instead
; of llvm.global_ctors, to exercise the other array checked by
; isListedInGlobalCtorOrDtorArray.

;--- funcsig.ll
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @dtor, ptr null }]

; FUNCSIG-LABEL: dtor:
; FUNCSIG:       lpad 308871
define internal void @dtor() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- unlabeled.ll
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @dtor, ptr null }]

; UNLABELED-LABEL: dtor:
; UNLABELED:       lpad 0
define internal void @dtor() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}