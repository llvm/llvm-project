; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpCapability Int8
; CHECK-DAG: OpCapability Int16
; CHECK-DAG: OpCapability Int64

; CHECK-DAG: %[[#]] = OpTypeInt 8 0
; CHECK-DAG: %[[#]] = OpTypeInt 16 0
; CHECK-DAG: %[[#]] = OpTypeInt 64 0

@a = addrspace(1) global i8 0, align 1
@b = addrspace(1) global i16 0, align 2
@c = addrspace(1) global i64 0, align 8

define spir_kernel void @test_atomic_fn() {
  ret void
}
