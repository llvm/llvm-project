; From Khronos Translator's test case: test/reqd_work_group_size_md.ll

; The purpose of this test is to check that the reqd_work_group_size metadata
; is correctly converted to the LocalSize execution mode for the kernels it is
; applied to.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel
; CHECK-DAG: OpEntryPoint Kernel %[[#ENTRY1:]] "test1"
; CHECK-DAG: OpEntryPoint Kernel %[[#ENTRY2:]] "test2"
; CHECK-DAG: OpEntryPoint Kernel %[[#ENTRY3:]] "test3"
; CHECK-DAG: OpExecutionMode %[[#ENTRY1]] LocalSize 1 2 3
; CHECK-DAG: OpExecutionMode %[[#ENTRY2]] LocalSize 2 3 1
; CHECK-DAG: OpExecutionMode %[[#ENTRY3]] LocalSize 3 1 1

define spir_kernel void @test1() !reqd_work_group_size !1 {
entry:
  ret void
}

define spir_kernel void @test2() !reqd_work_group_size !2 {
entry:
  ret void
}

define spir_kernel void @test3() !reqd_work_group_size !3 {
entry:
  ret void
}

!1 = !{i32 1, i32 2, i32 3}
!2 = !{i32 2, i32 3}
!3 = !{i32 3}
