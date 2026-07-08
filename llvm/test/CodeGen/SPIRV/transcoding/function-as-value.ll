; Without SPV_INTEL_function_pointers, a Function referenced as a plain value
; (not a call target) must not leak its own FunctionType or address-space
; attribute into a pointer type. Two symptoms of the same root cause, kept as
; separate modules so each regresses on its own (a single module lets one
; case's incidental OpUndef mask the other's).

; RUN: split-file %s %t

; The reference lowers to a placeholder that must be OpTypePointer Function
; %uchar, not a pointer built from the function's own type or its
; address-space attribute.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %t/undef-placeholder.ll -o - | FileCheck %s --check-prefix=UNDEF
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/undef-placeholder.ll -o - -filetype=obj | spirv-val %}

; UNDEF-DAG: %[[#UCHAR:]] = OpTypeInt 8 0
; UNDEF-DAG: %[[#PTR:]] = OpTypePointer Function %[[#UCHAR]]
; UNDEF: OpUndef %[[#PTR]]

; Element-type deduction for the function's own parameters must ignore
; non-call users of F (such as the ptrcast wrapping an address-of-function
; load), so the parameter keeps its real pointee type instead of the
; function's FunctionType.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %t/param-deduction.ll -o - | FileCheck %s --check-prefix=PARAM
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %t/param-deduction.ll -o - -filetype=obj | spirv-val %}

; PARAM-DAG: %[[#CHAR:]] = OpTypeInt 8 0
; PARAM-DAG: %[[#PTRCHAR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]
; PARAM: OpFunctionParameter %[[#PTRCHAR]]

;--- undef-placeholder.ll
define spir_kernel void @foo() addrspace(5) {
entry:
  store i32 0, ptr addrspace(5) @foo, align 4
  ret void
}

;--- param-deduction.ll
define spir_kernel void @foo(ptr addrspace(1) %in) {
entry:
  %v = load i32, ptr @foo, align 4
  ret void
}
