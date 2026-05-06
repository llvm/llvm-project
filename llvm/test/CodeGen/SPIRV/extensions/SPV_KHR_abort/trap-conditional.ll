; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Conditional trap: assert() pattern where only one path traps.
;; Verifies that the trap BB ends with OpAbortKHR (no OpUnreachable) and
;; the non-trap BB still has OpReturn.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK:     OpFunction
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturn
; CHECK-DAG: OpAbortKHR

;; The trap block's `unreachable` must be consumed by OpAbortKHR.
; CHECK-NOT: OpUnreachable

define spir_func void @assert_like(i32 %gid, i32 %N) {
entry:
  %cmp = icmp slt i32 %gid, %N
  br i1 %cmp, label %ok, label %trap

ok:
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() #0

attributes #0 = { cold noreturn nounwind }
