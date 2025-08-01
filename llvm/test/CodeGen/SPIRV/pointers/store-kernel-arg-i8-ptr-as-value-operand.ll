; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8
; CHECK-DAG: %[[#GLOBAL_PTR_CHAR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]

define spir_kernel void @foo(ptr addrspace(1) %arg) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
  %var = alloca ptr addrspace(1), align 8
; CHECK: %[[#]] = OpFunctionParameter %[[#GLOBAL_PTR_CHAR]]
; CHECK-NOT: %[[#]] = OpBitcast %[[#]] %[[#]]
  store ptr addrspace(1) %arg, ptr %var, align 8
  ret void
}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"char*"}
!4 = !{!""}
