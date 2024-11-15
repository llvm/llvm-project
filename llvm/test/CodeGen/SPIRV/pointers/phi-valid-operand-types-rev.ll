; The goal of the test case is to ensure that OpPhi is consistent with respect to operand types.
; -verify-machineinstrs is not available due to mutually exclusive requirements for G_BITCAST and G_PHI.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Char:]] = OpTypeInt 8 0
; CHECK: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]
; CHECK: %[[#Int:]] = OpTypeInt 32 0
; CHECK: %[[#PtrInt:]] = OpTypePointer Function %[[#Int]]
; CHECK: %[[#R1:]] = OpFunctionCall %[[#PtrChar]] %[[#]]
; CHECK: %[[#R2:]] = OpFunctionCall %[[#PtrInt]] %[[#]]
; CHECK-DAG: %[[#Casted1:]] = OpBitcast %[[#PtrChar]] %[[#R2]]
; CHECK-DAG: %[[#Casted2:]] = OpBitcast %[[#PtrChar]] %[[#R2]]
; CHECK: OpBranchConditional
; CHECK-DAG: OpPhi %[[#PtrChar]] %[[#R1]] %[[#]] %[[#Casted1]] %[[#]]
; CHECK-DAG: OpPhi %[[#PtrChar]] %[[#R1]] %[[#]] %[[#Casted2]] %[[#]]

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
