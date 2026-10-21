; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/missing-entry-funcsig.ll \
; RUN:   | FileCheck %s --check-prefix=ENTRY-ZERO
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/missing-callsite.ll \
; RUN:   | FileCheck %s --check-prefix=CALLSITE-ZERO
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/missing-ctor-attr.ll \
; RUN:   | FileCheck %s --check-prefix=CTOR-FALLBACK
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/missing-dtor-attr.ll \
; RUN:   | FileCheck %s --check-prefix=DTOR-FALLBACK
; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %t/missing-local-jump-attr.ll \
; RUN:   | FileCheck %s --check-prefix=LOCAL-JUMP-ZERO

;; tests behaviour of IR code with missing call / landing pad label.
;; current policy is to default to 0 in most cases.
;; ctors/dtors listed in the llvm.global_ctors/llvm.global_dtors arrays
;; receive a special fallback label because they have a constant FvvE signature.

;--- missing-entry-funcsig.ll
; ENTRY-ZERO-LABEL: external_no_attr:
; ENTRY-ZERO:       lpad 0
define void @external_no_attr() {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- missing-callsite.ll
; CALLSITE-ZERO-LABEL: call_no_metadata:
; CALLSITE-ZERO-NOT:   lui t2
; CALLSITE-ZERO:       jalr
; CALLSITE-ZERO-LABEL: tail_call_no_metadata:
; CALLSITE-ZERO-NOT:   lui t2
; CALLSITE-ZERO:       jr
define i32 @call_no_metadata(ptr %f, i32 %x) "cfi_type_hash"="778899" {
  %call = call i32 %f(i32 %x)
  ret i32 %call
}
define void @tail_call_no_metadata(ptr %f) "cfi_type_hash"="445566" {
  tail call void %f()
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- missing-ctor-attr.ll
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null }]
; CTOR-FALLBACK-LABEL: ctor:
; CTOR-FALLBACK:       lpad 308871
define internal void @ctor() {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- missing-dtor-attr.ll
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @dtor, ptr null }]
; DTOR-FALLBACK-LABEL: dtor:
; DTOR-FALLBACK:       lpad 308871
define internal void @dtor() {
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}

;--- missing-local-jump-attr.ll
@arr = internal unnamed_addr constant [1 x ptr] [ptr blockaddress(@internal_indirectbr, %target)]
; LOCAL-JUMP-ZERO-LABEL: internal_indirectbr:
; LOCAL-JUMP-ZERO:       %target
; LOCAL-JUMP-ZERO-NEXT:  lpad 0
define internal void @internal_indirectbr(i32 %i) {
entry:
  %p = getelementptr inbounds [1 x ptr], ptr @arr, i32 0, i32 %i
  %dest = load ptr, ptr %p
  indirectbr ptr %dest, [label %target]
target:
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}