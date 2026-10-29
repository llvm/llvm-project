; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Sanity check: enabling SPV_KHR_abort does not affect functions that don't
;; abort. Normal terminators (OpReturn, OpReturnValue, OpBranch,
;; OpUnreachable) must be preserved.

;; No abort instructions emitted anywhere.
; CHECK-NOT: OpAbortKHR

;; Normal void return preserved.
; CHECK:     OpFunction
; CHECK:     OpReturn
; CHECK:     OpFunctionEnd

;; Normal value return preserved.
; CHECK:     OpFunction
; CHECK:     OpReturnValue
; CHECK:     OpFunctionEnd

;; Plain unreachable preserved (no abort precedes it).
; CHECK:     OpFunction
; CHECK:     OpUnreachable
; CHECK:     OpFunctionEnd

;; Branch preserved.
; CHECK:     OpFunction
; CHECK:     OpBranchConditional
; CHECK:     OpReturnValue
; CHECK:     OpReturnValue
; CHECK:     OpFunctionEnd

define spir_func void @void_return() {
entry:
  ret void
}

define spir_func i32 @value_return(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_func void @plain_unreachable() {
entry:
  unreachable
}

define spir_func i32 @branched(i1 %cond, i32 %a, i32 %b) {
entry:
  br i1 %cond, label %t, label %f

t:
  ret i32 %a

f:
  ret i32 %b
}
