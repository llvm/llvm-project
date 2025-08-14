; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK-DAG: [[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: [[#VOID:]] = OpTypeVoid
; CHECK-DAG: [[#FN_BAR_TY:]] = OpTypeFunction [[#I32]] [[#I32]]
; CHECK-DAG: [[#FN_FOO_TY:]] = OpTypeFunction [[#VOID]] [[#I32]]

; CHECK-DAG: [[#BAR:]] = OpFunction [[#I32]] None [[#FN_BAR_TY]]
; CHECK-DAG: OpFunctionEnd

; CHECK-DAG: [[#FOO:]] = OpFunction [[#VOID]] None [[#FN_FOO_TY]]
; CHECK:      [[#PARAM_X:]] = OpFunctionParameter [[#I32]]
; CHECK:      [[#CALL1:]] = OpFunctionCall [[#I32]] [[#BAR]] [[#PARAM_X]]
; CHECK:      OpReturn
; CHECK:      OpFunctionEnd

define spir_kernel void @foo() {
entry:
  %iptr = alloca i32, align 4
  %fptr = alloca float, align 4
  br label %loop

loop:
  %ptr1 = phi ptr [%ptr2, %loop], [%iptr, %entry]
  %ptr2 = phi ptr [%ptr1, %loop], [%fptr, %entry]
  %cond = phi i32 [0, %entry], [%cond.next, %loop]
  %cond.next = add i32 %cond, 1
  %cmp = icmp slt i32 %cond.next, 150
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}
