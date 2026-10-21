; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/funcsig.ll \
; RUN:   | FileCheck %s --check-prefix=FUNCSIG
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/unlabeled.ll \
; RUN:   | FileCheck %s --check-prefix=UNLABELED

; Functions registered in llvm.global_ctors/llvm.global_dtors have the fixed
; ABI signature void(), which mangles to "FvvE". Under func-sig, the backend
; supplies this label as a fallback when the frontend-generated cfi_type_hash
; attribute is absent.

;--- funcsig.ll
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]

; FUNCSIG-LABEL: ctor:
; FUNCSIG:       lpad 308871
define internal void @ctor() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- unlabeled.ll
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]

; UNLABELED-LABEL: ctor:
; UNLABELED:       lpad 0
define internal void @ctor() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
