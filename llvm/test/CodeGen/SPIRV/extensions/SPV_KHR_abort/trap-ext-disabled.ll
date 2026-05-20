; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; When SPV_KHR_abort is not enabled, llvm.trap is dropped (existing behavior),
;; and no AbortKHR capability/extension/instruction is emitted.

; CHECK-NOT: OpCapability AbortKHR
; CHECK-NOT: OpExtension "SPV_KHR_abort"
; CHECK-NOT: OpAbortKHR

define spir_func void @trap_with_ext_disabled() {
entry:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() #0

attributes #0 = { cold noreturn nounwind }
