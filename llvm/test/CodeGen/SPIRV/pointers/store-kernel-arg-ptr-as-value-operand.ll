; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define spir_kernel void @foo(ptr addrspace(1) %arg) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  %var = alloca ptr addrspace(1), align 8
; CHECK: %[[#VAR:]] = OpVariable %[[#]] Function
  store ptr addrspace(1) %arg, ptr %var, align 8
; The test itends to verify that OpStore uses OpVariable result directly (without a bitcast).
; Other type checking is done by spirv-val.
; CHECK: OpStore %[[#VAR]] %[[#]] Aligned 8
  %lod = load ptr addrspace(1), ptr %var, align 8
  %idx = getelementptr inbounds i64, ptr addrspace(1) %lod, i64 0
  ret void
}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"ulong*"}
!4 = !{!""}
