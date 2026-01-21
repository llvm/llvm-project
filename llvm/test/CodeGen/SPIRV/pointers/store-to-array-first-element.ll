; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer Function %[[#Int]]
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Int]] %[[#]]
; CHECK-DAG: %[[#ArrayPtr:]] = OpTypePointer Function %[[#Array]]
; CHECK-DAG: %[[#Const:]] = OpConstant %[[#Int]] 123
; CHECK-DAG: %[[#Zero:]] = OpConstant %[[#Int]] 0

; CHECK: %[[#Var:]] = OpVariable %[[#ArrayPtr]] Function
; CHECK: %[[#GEP:]] = OpInBoundsAccessChain %[[#IntPtr]] %[[#Var]] %[[#Zero]]
; CHECK: OpStore %[[#GEP]] %[[#Const]]

define spir_func void @test_array_store() {
entry:
  %var = alloca [4 x i32]
  store i32 123, ptr %var
  ret void
}
