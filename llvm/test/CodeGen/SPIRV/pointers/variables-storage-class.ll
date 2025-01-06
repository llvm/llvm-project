; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#U8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0

; CHECK-DAG: %[[#TYPE:]] = OpTypePointer CrossWorkgroup %[[#U8]]
; CHECK-DAG: %[[#VAL:]] = OpConstantNull %[[#TYPE]]
; CHECK-DAG: %[[#VTYPE:]] = OpTypePointer CrossWorkgroup %[[#TYPE]]
; CHECK-DAG: %[[#PTR:]] = OpVariable %[[#VTYPE]] CrossWorkgroup %[[#VAL]]
@Ptr = addrspace(1) global ptr addrspace(1) null

; CHECK-DAG: %[[#VAL:]] = OpConstant %[[#U32]] 123
; CHECK-DAG: %[[#VTYPE:]] = OpTypePointer UniformConstant %[[#U32]]
; CHECK-DAG: %[[#INIT:]] = OpVariable %[[#VTYPE]] UniformConstant %[[#VAL]]
@Init = private addrspace(2) constant i32 123

; CHECK-DAG: %[[#VAL:]] = OpConstant %[[#U32]] 456
; CHECK-DAG: %[[#VTYPE:]] = OpTypePointer Private %[[#U32]]
; CHECK-DAG: %[[#]] = OpVariable %[[#VTYPE]] Private %[[#VAL]]
@PrivInternal = internal addrspace(10) global i32 456

define spir_kernel void @Foo() {
  ; CHECK: %[[#]] = OpLoad %[[#]] %[[#PTR]] Aligned 8
  %l = load ptr addrspace(1), ptr addrspace(1) @Ptr, align 8
  ; CHECK: OpCopyMemorySized %[[#]] %[[#INIT]] %[[#]] Aligned 4
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 4 %l, ptr addrspace(2) align 1 @Init, i64 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
