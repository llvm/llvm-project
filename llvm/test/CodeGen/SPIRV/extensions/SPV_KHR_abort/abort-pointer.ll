; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Pointers are concrete SPIR-V types and are valid Message Type operands
;; for OpAbortKHR.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]

; CHECK:     OpAbortKHR %[[#PTR]] %{{[0-9]+}}
; CHECK-NOT: OpUnreachable

declare void @llvm.spv.abort.p1(ptr addrspace(1)) #0

define spir_kernel void @abort_with_pointer(ptr addrspace(1) %p) {
entry:
  call void @llvm.spv.abort.p1(ptr addrspace(1) %p)
  unreachable
}

attributes #0 = { noreturn }
