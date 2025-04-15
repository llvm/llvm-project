; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:  %[[#FLOAT32:]] = OpTypeFloat 32
; CHECK:  %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#FLOAT32]]
; CHECK:  %[[#ARG:]] = OpFunctionParameter %[[#PTR]]
; CHECK:  %[[#GEP:]] = OpInBoundsPtrAccessChain %[[#PTR]] %[[#ARG]] %[[#]]
; CHECK:  %[[#]] = OpLoad %[[#FLOAT32]] %[[#GEP]] Aligned 4

define spir_kernel void @test1(ptr addrspace(1) %arg1) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_type_qual !4 {
  %a = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 1
  %b = load float, ptr addrspace(1) %a, align 4
  ret void
}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"float*"}
!4 = !{!""}
