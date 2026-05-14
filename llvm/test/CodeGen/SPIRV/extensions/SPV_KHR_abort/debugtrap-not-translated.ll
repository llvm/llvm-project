; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s

;; Negative test: llvm.debugtrap must NOT lower to OpAbortKHR. Only llvm.trap
;; and llvm.ubsantrap are translated to OpAbortKHR; debugtrap is dropped (no
;; codegen) and the original `unreachable` terminator is preserved.

; CHECK-NOT: OpCapability AbortKHR
; CHECK-NOT: OpAbortKHR

; CHECK:     OpFunction
; CHECK:     OpLabel
; CHECK:     OpUnreachable
; CHECK:     OpFunctionEnd

define spir_func void @uses_debugtrap() {
entry:
  call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.debugtrap() #0

attributes #0 = { nounwind }
