; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:  %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG:  %[[#PTR1:]] = OpTypePointer Function %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR1]]

define spir_kernel void @test1(ptr addrspace(0) %arg1) !kernel_arg_addr_space !1 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(0) %arg1, i32 2
  ret void
}

!1 = !{i32 0}
!2 = !{!"int*"}

; CHECK-DAG:  %[[#PTR2:]] = OpTypePointer CrossWorkgroup %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR2]]

define spir_kernel void @test2(ptr addrspace(1) %arg1) !kernel_arg_addr_space !3 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 2
  ret void
}

!3 = !{i32 1}

; CHECK-DAG:  %[[#PTR3:]] = OpTypePointer UniformConstant %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR3]]

define spir_kernel void @test3(ptr addrspace(2) %arg1) !kernel_arg_addr_space !4 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(2) %arg1, i32 2
  ret void
}

!4 = !{i32 2}

; CHECK-DAG:  %[[#PTR4:]] = OpTypePointer Workgroup %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR4]]

define spir_kernel void @test4(ptr addrspace(3) %arg1) !kernel_arg_addr_space !5 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 2
  ret void
}

!5 = !{i32 3}

; CHECK-DAG:  %[[#PTR5:]] = OpTypePointer Generic %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR5]]

define spir_kernel void @test5(ptr addrspace(4) %arg1) !kernel_arg_addr_space !6 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(4) %arg1, i32 2
  ret void
}

!6 = !{i32 4}

; CHECK-DAG:  %[[#PTR6:]] = OpTypePointer Input %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR6]]

define spir_kernel void @test6(ptr addrspace(7) %arg1) !kernel_arg_addr_space !7 !kernel_arg_type !2 {
  %a = getelementptr inbounds i32, ptr addrspace(7) %arg1, i32 2
  ret void
}

!7 = !{i32 7}
