; This test checks translation of function parameter which is untyped pointer.
; Lately, when we do support untyped variables, this one could be used to check
; "full" forward and reverse translation of opaque pointers.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK: Capability [[#]] = OpCapability UntypedPointersKHR
; CHECK: Extension [[#]] = OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: [[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: [[#Constant0:]] = OpConstant [[#IntTy]] 0
; CHECK-DAG: [[#Constant42:]] = OpConstant [[#IntTy]] 42
; CHECK-DAG: [[#UntypedPtrTy:]] = OpTypeUntypedPointerKHR 5
; CHECK-DAG: [[#UntypedPtrTyFunc:]] = OpTypeUntypedPointerKHR 7

; CHECK: [[#FuncParam:]] = OpFunctionParameter [[#UntypedPtrTy]]
; CHECK: [[#VarBId:]] = OpUntypedVariableKHR [[#UntypedPtrTyFunc]] 7 [[#UntypedPtrTy]]
; CHECK: OpStore [[#VarBId]] [[#FuncParam]] 2 4
; CHECK: [[#LoadId:]] = OpLoad [[#UntypedPtrTy]] [[#VarBId]] 2 4
; CHECK: OpStore [[#LoadId]] [[#Constant0]] 2 4

; CHECK: [[#FuncParam0:]] = OpFunctionParameter [[#UntypedPtrTy]]
; CHECK: [[#FuncParam1:]] = OpFunctionParameter [[#UntypedPtrTy]]
; CHECK: [[#VarCId:]] = OpUntypedVariableKHR [[#UntypedPtrTyFunc]] 7 [[#IntTy]]
; CHECK: OpStore [[#VarCId]] [[#Constant42]] 2 4
; CHECK: [[#LoadId:]] = OpLoad [[#IntTy]] [[#FuncParam1]] 2 4
; CHECK: OpStore [[#FuncParam0]] [[#LoadId]] 2 4

define spir_func void @foo(ptr addrspace(1) %a) {
entry:
  %b = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %a, ptr %b, align 4
  %0 = load ptr addrspace(1), ptr %b, align 4
  store i32 0, ptr addrspace(1) %0, align 4
  ret void
}

define dso_local void @boo(ptr addrspace(1) %0, ptr addrspace(1) %1) {
entry:
  %c = alloca i32, align 4
  store i32 42, ptr %c, align 4
  %2 = load i32, ptr addrspace(1) %1, align 4
  store i32 %2, ptr addrspace(1) %0, align 4
  ret void
}
