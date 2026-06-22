; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Without the extension, llvm.trap is dropped silently (existing behavior).
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXT

;; llvm.trap lowers to OpAbortKHR with an all-ones (-1) i32 Message constant.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#MSG:]] = OpConstant %[[#I32]] 4294967295

; CHECK:     OpAbortKHR %[[#I32]] %[[#MSG]]
; CHECK-NOT: OpUnreachable
; CHECK-NEXT: OpFunctionEnd

; CHECK-NO-EXT-NOT: OpCapability AbortKHR
; CHECK-NO-EXT-NOT: OpAbortKHR

define spir_func void @trap_simple() {
entry:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() #0

attributes #0 = { cold noreturn nounwind }
