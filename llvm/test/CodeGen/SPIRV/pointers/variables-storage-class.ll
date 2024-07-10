; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@Ptr = addrspace(1) global ptr addrspace(1) null
@Init = private addrspace(2) constant i32 123

; CHECK-DAG: %[[#PTR:]] = OpVariable %[[#]] UniformConstant %[[#]]
; CHECK-DAG: %[[#INIT:]] = OpVariable %[[#]] CrossWorkgroup %[[#]]

; CHECK: %[[#]] = OpLoad %[[#]] %[[#INIT]] Aligned 8
; CHECK: OpCopyMemorySized %[[#]] %[[#PTR]] %[[#]] Aligned 4

define spir_kernel void @Foo() {
  %l = load ptr addrspace(1), ptr addrspace(1) @Ptr, align 8
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 4 %l, ptr addrspace(2) align 1 @Init, i64 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
