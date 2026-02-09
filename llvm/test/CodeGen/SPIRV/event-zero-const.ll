; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#EventTy:]] = OpTypeEvent
; CHECK-DAG: %[[#LongNull:]] = OpConstantNull %[[#LongTy]]
; CHECK-DAG: %[[#EventNull:]] = OpConstantNull %[[#EventTy]]
; CHECK-DAG: %[[#Scope:]] = OpConstant %[[#IntTy]] 2
; CHECK-DAG: %[[#One:]] = OpConstant %[[#IntTy]] 1
; CHECK: OpFunction
; CHECK: OpINotEqual %[[#]] %[[#]] %[[#LongNull]]
; CHECK: OpGroupAsyncCopy %[[#EventTy]] %[[#Scope]] %[[#]] %[[#]] %[[#One]] %[[#One]] %[[#EventNull]]

@G_r1 = global i1 0
@G_e1 = global target("spirv.Event") poison

define weak_odr dso_local spir_kernel void @foo(i64 %_arg_i, ptr addrspace(1) %_arg_ptr, ptr addrspace(3) %_arg_local) {
entry:
  %r1 = icmp ne i64 %_arg_i, 0
  store i1 %r1, ptr @G_r1
  %e1 = tail call spir_func target("spirv.Event") @__spirv_GroupAsyncCopy(i32 2, ptr addrspace(3) %_arg_local, ptr addrspace(1) %_arg_ptr, i32 1, i32 1, target("spirv.Event") zeroinitializer)
  store target("spirv.Event") %e1, ptr @G_e1
  ret void
}

declare dso_local spir_func target("spirv.Event") @__spirv_GroupAsyncCopy(i32, ptr addrspace(3), ptr addrspace(1), i32, i32, target("spirv.Event"))
