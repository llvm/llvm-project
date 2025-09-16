; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpEntryPoint Kernel %[[#Func:]] "test" %[[#Interface1:]] %[[#Interface2:]] %[[#Interface3:]] %[[#Interface4:]]
; CHECK-DAG: OpName %[[#Func]] "test"
; CHECK-DAG: OpName %[[#Interface1]] "var"
; CHECK-DAG: OpName %[[#Interface3]] "var2"
; CHECK-DAG: OpName %[[#Interface2]] "var.const"
; CHECK-DAG: OpName %[[#Interface4]] "var2.const"
; CHECK-DAG: %[[#TypeInt:]] = OpTypeInt  32 0
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#TypeInt]] 1
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#TypeInt]] 3

; CHECK: %[[#Interface1]] = OpVariable %[[#]] UniformConstant %[[#Const1]]
; CHECK: %[[#Interface3]] = OpVariable %[[#]] UniformConstant %[[#Const2]]
; CHECK: %[[#Interface2]] = OpVariable %[[#]] UniformConstant %[[#Const1]]
; CHECK: %[[#Interface4]] = OpVariable %[[#]] UniformConstant %[[#Const2]]

@var = dso_local addrspace(2) constant i32 1, align 4
@var2 = dso_local addrspace(2) constant i32 3, align 4
@var.const = private unnamed_addr addrspace(2) constant i32 1, align 4
@var2.const = private unnamed_addr addrspace(2) constant i32 3, align 4

define dso_local spir_kernel void @test() {
entry:
  %0 = load i32, ptr addrspace(2) @var.const, align 4
  %1 = load i32, ptr addrspace(2) @var2.const, align 4
  %mul = mul nsw i32 %0, %1
  %mul1 = mul nsw i32 %mul, 2
  ret void
}
