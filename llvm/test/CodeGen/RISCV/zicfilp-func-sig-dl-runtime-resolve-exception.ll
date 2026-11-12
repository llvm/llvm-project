; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %t/external.ll \
; RUN:   | FileCheck %s --check-prefix=EXTERNAL
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %t/internal.ll \
; RUN:   | FileCheck %s --check-prefix=INTERNAL

;; _dl_runtime_resolve is the dynamic loader's PLT trampoline resolver. Per
;; psABI, it unconditionally receives landing-pad label 0 under func-sig.
;; This exemption is name-based, but is restricted to externally-visible 
;; (non-local-linkage) functions. It must not fire for an unrelated
;; internal/static function that merely happens to share the name.

;--- external.ll
; EXTERNAL-LABEL: _dl_runtime_resolve:
; EXTERNAL:       lpad 0
define void @_dl_runtime_resolve() {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- internal.ll
@funcptr = global ptr @_dl_runtime_resolve
; INTERNAL-LABEL: _dl_runtime_resolve:
; INTERNAL:       lpad 778899
define internal void @_dl_runtime_resolve() "cfi_type_hash"="778899" {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}