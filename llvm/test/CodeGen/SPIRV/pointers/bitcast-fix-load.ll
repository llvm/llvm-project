; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck --check-prefix=CHECK-UNTYPED-PTRS %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TYLONG:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYSTRUCTLONG:]] = OpTypeStruct %[[#TYLONG]]
; CHECK-DAG: %[[#TYARRAY:]] = OpTypeArray %[[#TYSTRUCTLONG]] %[[#]]
; CHECK-DAG: %[[#TYSTRUCT:]] = OpTypeStruct %[[#TYARRAY]]
; CHECK-DAG: %[[#TYSTRUCTPTR:]] = OpTypePointer Function %[[#TYSTRUCT]]
; CHECK-DAG: %[[#TYLONGPTR:]] = OpTypePointer Function %[[#TYLONG]]
; CHECK: %[[#PTRTOSTRUCT:]] = OpFunctionParameter %[[#TYSTRUCTPTR]]
; CHECK: %[[#PTRTOLONG:]] = OpBitcast %[[#TYLONGPTR]] %[[#PTRTOSTRUCT]]
; CHECK-NEXT: OpLoad %[[#TYLONG]] %[[#PTRTOLONG]]

; CHECK-UNTYPED-PTRS-DAG: %[[#TYLONG:]] = OpTypeInt 32 0
; CHECK-UNTYPED-PTRS-DAG: %[[#TYSTRUCTLONG:]] = OpTypeStruct %[[#TYLONG]]
; CHECK-UNTYPED-PTRS-DAG: %[[#TYARRAY:]] = OpTypeArray %[[#TYSTRUCTLONG]] %[[#]]
; CHECK-UNTYPED-PTRS-DAG: %[[#TYSTRUCT:]] = OpTypeStruct %[[#TYARRAY]]
; CHECK-UNTYPED-PTRS-DAG: %[[#TYSTRUCTPTR:]] = OpTypePointer Function %[[#TYSTRUCT]]
; CHECK-UNTYPED-PTRS-DAG: %[[#TYPTR:]] = OpTypeUntypedPointerKHR Function
; CHECK-UNTYPED-PTRS: %[[#PTRTOSTRUCT:]] = OpFunctionParameter %[[#TYSTRUCTPTR]]
; CHECK-UNTYPED-PTRS: %[[#PTRTOLONG:]] = OpBitcast %[[#TYPTR]] %[[#PTRTOSTRUCT]]
; CHECK-UNTYPED-PTRS-NEXT: OpLoad %[[#TYLONG]] %[[#PTRTOLONG]]

%struct.S = type { i32 }
%struct.__wrapper_class = type { [7 x %struct.S] }

@G = global i32 0

define spir_kernel void @foo(ptr noundef byval(%struct.__wrapper_class) align 4 %arg) {
entry:
  %val = load i32, ptr %arg
  store i32 %val, ptr @G
  ret void
}
