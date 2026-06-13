; Declarations-only module must still reach AsmPrinter.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Addresses
; CHECK-DAG: OpCapability Kernel
; CHECK:     OpMemoryModel Physical64 OpenCL

declare spir_kernel void @fuzz_kernel(ptr addrspace(1), ptr addrspace(1), i32)
