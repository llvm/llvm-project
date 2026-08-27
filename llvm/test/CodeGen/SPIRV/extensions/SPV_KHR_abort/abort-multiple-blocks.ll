; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Multiple basic blocks: some abort, some don't.
;; Verifies that:
;;   1. Non-abort blocks are unaffected (normal terminators preserved).
;;   2. Multiple abort blocks in the same function each get their own OpAbortKHR.
;;   3. No cross-contamination between blocks.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK:     OpFunction
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpBranchConditional
; CHECK-DAG: OpReturnValue
; CHECK-DAG: OpAbortKHR
; CHECK-DAG: OpAbortKHR
; CHECK:     OpFunctionEnd
; CHECK-NOT: OpUnreachable

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0

define spir_func i32 @multi_abort(i32 %x, i32 %m1, i32 %m2) {
entry:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %work, label %err1

work:
  %result = mul i32 %x, 42
  %cmp2 = icmp slt i32 %result, 1000
  br i1 %cmp2, label %ret, label %err2

ret:
  ret i32 %result

err1:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %m1)
  unreachable

err2:
  call spir_func void @_Z16__spirv_AbortKHRj(i32 %m2)
  unreachable
}

attributes #0 = { noreturn }
