; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; The OpenCL builtin rewrite must not break IR when the abort call sits in a
;; conditional block whose successor has PHI nodes referring back to it. The
;; pass should drop the original successor edge from the abort block (along
;; with the trailing branch), so the merge block's PHI is left consistent.
;; Regression test for a verifier failure where a stale PHI predecessor was
;; left after the abort block was re-terminated with `unreachable`.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0

; The abort block must end with OpAbortKHR (no OpReturn / OpBranch / OpUnreachable
; in between).
; CHECK:     OpAbortKHR %[[#I32]] %{{[0-9]+}}
; CHECK-NOT: OpUnreachable

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0

define spir_kernel void @abort_in_conditional(i1 %c, i32 %x) {
entry:
  br i1 %c, label %then, label %else

then:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %x)
  br label %merge

else:
  br label %merge

merge:
  %v = phi i32 [ 1, %then ], [ 2, %else ]
  ret void
}

attributes #0 = { convergent nounwind }
