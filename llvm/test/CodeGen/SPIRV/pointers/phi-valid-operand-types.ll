; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PtrInt:]] = OpTypePointer Function %[[#Int]]
; CHECK: %[[#R1:]] = OpFunctionCall %[[#PtrChar]] %[[#]]
; CHECK: %[[#R2:]] = OpFunctionCall %[[#PtrInt]] %[[#]]
; CHECK: %[[#Casted:]] = OpBitcast %[[#PtrChar]] %[[#R2]]
; CHECK: OpPhi %[[#PtrChar]] %[[#R1]] %[[#]] %[[#Casted]] %[[#]]

define ptr @foo(i1 %arg) {
entry:
  %r1 = tail call ptr @f1()
  %r2 = tail call ptr @f2()
  br i1 %arg, label %l1, label %l2

l1:
  br label %exit

l2:
  br label %exit

exit:
  %ret = phi ptr [ %r1, %l1 ], [ %r2, %l2 ]
  %ret2 = phi ptr [ %r1, %l1 ], [ %r2, %l2 ]
  tail call void @f0(ptr %ret)
  ret ptr %ret2
}

define void @f0(ptr %arg) {
entry:
  ret void
}

define ptr @f1() {
entry:
  %p = alloca i8
  store i8 8, ptr %p
  ret ptr %p
}

define ptr @f2() {
entry:
  %p = alloca i32
  store i32 32, ptr %p
  ret ptr %p
}

; A call result phi'd with a local address keeps its OpFunctionCall type.
; CHECK-LABEL: Begin function phi_call_result
; CHECK: %[[#PhiVar:]] = OpVariable %[[#]]
; CHECK: %[[#PhiCall:]] = OpFunctionCall %[[#]]
; CHECK: %[[#PhiBCast1:]] = OpBitcast %[[#PhiPtr:]] %[[#PhiCall]]
; CHECK: %[[#PhiBCast2:]] = OpBitcast %[[#PhiPtr]] %[[#PhiCall]]
; CHECK: %[[#PhiCast:]] = OpPtrCastToGeneric %[[#PhiPtr]] %[[#PhiVar]]
; CHECK: %[[#Phi:]] = OpPhi %[[#PhiPtr]] %[[#PhiCast]] %[[#]] %[[#PhiBCast1]]
; CHECK: OpPtrNotEqual %[[#]] %[[#Phi]] %[[#PhiBCast2]]

declare ptr addrspace(4) @f()
define i1 @phi_call_result(i1 %c) {
entry:
  %a = alloca ptr addrspace(4), align 8
  %x = tail call ptr addrspace(4) @f()
  %ac = addrspacecast ptr %a to ptr addrspace(4)
  br i1 %c, label %t, label %j

t:
  br label %j

j:
  %p = phi ptr addrspace(4) [ %ac, %entry ], [ %x, %t ]
  %r = icmp ne ptr addrspace(4) %p, %x
  ret i1 %r
}
