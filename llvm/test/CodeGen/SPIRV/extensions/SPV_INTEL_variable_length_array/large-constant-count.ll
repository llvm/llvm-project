; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --implicit-check-not=OpTypeArray
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_INTEL_variable_length_array %s -filetype=obj -o - | spirv-val %}

; Counts wider than the backend's 32-bit array lengths retain VLA lowering.
; ERROR: LLVM ERROR: array allocation: this instruction requires the following SPIR-V extension: SPV_INTEL_variable_length_array

; CHECK: OpCapability VariableLengthArrayINTEL
; CHECK: OpExtension "SPV_INTEL_variable_length_array"
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Function %[[#INT]]
; CHECK-DAG: %[[#LONG:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#COUNT:]] = OpConstant %[[#LONG]] 4294967296
; CHECK: OpVariableLengthArrayINTEL %[[#PTR]] %[[#COUNT]]
define spir_func void @large_constant_count() {
entry:
  %array = alloca i32, i64 4294967296, align 4
  ret void
}
