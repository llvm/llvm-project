; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: [[#PtrT:]] = OpTypePointer Workgroup %[[#]]
; CHECK-SPIRV: %[[#]] = OpVariable %[[#PtrT]] Workgroup

@test_atomic_fn.L = internal addrspace(3) global [64 x i32] zeroinitializer, align 4

define spir_kernel void @test_atomic_fn() {
  ret void
}
