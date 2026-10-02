; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: OpAbortKHR instruction requires the following SPIR-V extension: SPV_KHR_abort

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#MSG:]] = OpConstant %[[#I32]] 42

; CHECK: OpAbortKHR %[[#I32]] %[[#MSG]]
; CHECK-NOT: OpUnreachable

declare void @llvm.spv.abort(i32)

define void @abort_with_int() {
entry:
  call void @llvm.spv.abort(i32 42)
  unreachable
}
