; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3%}

; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0

; CHECK-DAG: %[[#VAL:]] = OpConstant %[[#U32]] 456
; CHECK-DAG: %[[#VTYPE:]] = OpTypePointer Private %[[#U32]]
; CHECK-DAG: %[[#]] = OpVariable %[[#VTYPE]] Private %[[#VAL]]
@PrivInternal = internal addrspace(10) global i32 456

define hidden spir_func void @Foo() {
  %tmp = load i32, ptr addrspace(10) @PrivInternal
  ret void
}

define void @main() #1 {
  ret void
}

declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) noalias nocapture writeonly, ptr addrspace(2) noalias nocapture readonly, i64, i1 immarg)
attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
