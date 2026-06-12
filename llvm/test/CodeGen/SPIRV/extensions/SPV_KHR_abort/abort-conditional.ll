; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Conditional abort -- the assert() pattern where only some paths abort.
;; Verifies that:
;;   1. The abort BB ends with OpAbortKHR (no OpUnreachable after it).
;;   2. The non-abort BB is unaffected (still has OpReturn).
;;   3. No double terminators in the abort block.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturn
; CHECK-DAG: OpAbortKHR

;; Crucially, no OpUnreachable should appear: the trap block's `unreachable`
;; is consumed by OpAbortKHR.
; CHECK-NOT: OpUnreachable

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0

define spir_func void @assert_like(i32 %gid, i32 %N, i32 %msg) {
entry:
  %cmp = icmp slt i32 %gid, %N
  br i1 %cmp, label %ok, label %trap

ok:
  ret void

trap:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %msg)
  unreachable
}

attributes #0 = { noreturn }
