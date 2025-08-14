; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK: [[#cap_untyped_ptr:]] = OpCapability UntypedPointersKHR
; CHECK: [[#ext_untyped_ptr:]] = OpExtension "SPV_KHR_untyped_pointers"
; CHECK-DAG: [[#void_ty:]] = OpTypeVoid
; CHECK-DAG: [[#untyped_ptr_ty:]] = OpTypeUntypedPointerKHR 7
; CHECK-DAG: [[#func_ty:]] = OpTypeFunction [[#untyped_ptr_ty]] [[#untyped_ptr_ty]]

; CHECK: [[#process_func:]] = OpFunction [[#untyped_ptr_ty]] None [[#func_ty]]
; CHECK: OpFunctionParameter [[#untyped_ptr_ty]]
; CHECK: [[#main_func:]] = OpFunction [[#untyped_ptr_ty]] None [[#func_ty]]
; CHECK: [[#param:]] = OpFunctionParameter [[#untyped_ptr_ty]]
; CHECK: [[#call_res:]] = OpFunctionCall [[#untyped_ptr_ty]] [[#process_func]] [[#param]]
; CHECK: OpReturnValue [[#call_res]]

declare ptr @processPointer(ptr)

define ptr @example(ptr %arg) {
entry:
	%result = call ptr @processPointer(ptr %arg)
	ret ptr %result
}
