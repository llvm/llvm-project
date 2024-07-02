; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TYCHAR:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#TYCHARPTR:]] = OpTypePointer Function %[[#TYCHAR]]
; CHECK-DAG: %[[#TYINT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYSTRUCTINT32:]] = OpTypeStruct %[[#TYINT32]]
; CHECK-DAG: %[[#TYARRAY:]] = OpTypeArray %[[#TYSTRUCTINT32]] %[[#]]
; CHECK-DAG: %[[#TYSTRUCT:]] = OpTypeStruct %[[#TYARRAY]]
; CHECK-DAG: %[[#TYSTRUCTPTR:]] = OpTypePointer Function %[[#TYSTRUCT]]
; CHECK-DAG: %[[#TYINT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#TYINT64PTR:]] = OpTypePointer Function %[[#TYINT64]]
; CHECK: OpFunction
; CHECK: %[[#PTRTOSTRUCT:]] = OpFunctionParameter %[[#TYSTRUCTPTR]]
; CHECK: %[[#PTRTOCHAR:]] = OpBitcast %[[#TYCHARPTR]] %[[#PTRTOSTRUCT]]
; CHECK-NEXT: OpInBoundsPtrAccessChain %[[#TYCHARPTR]] %[[#PTRTOCHAR]]
; CHECK: OpFunction
; CHECK: %[[#PTRTOSTRUCT2:]] = OpFunctionParameter %[[#TYSTRUCTPTR]]
; CHECK: %[[#ELEM:]] = OpInBoundsPtrAccessChain %[[#TYSTRUCTPTR]] %[[#PTRTOSTRUCT2]]
; CHECK-NEXT: %[[#TOLOAD:]] = OpBitcast %[[#TYINT64PTR]] %[[#ELEM]]
; CHECK-NEXT: OpLoad %[[#TYINT64]] %[[#TOLOAD]]

%struct.S = type { i32 }
%struct.__wrapper_class = type { [7 x %struct.S] }

define spir_kernel void @foo1(ptr noundef byval(%struct.__wrapper_class) align 4 %_arg_Arr) {
entry:
  %elem = getelementptr inbounds i8, ptr %_arg_Arr, i64 0
  ret void
}

define spir_kernel void @foo2(ptr noundef byval(%struct.__wrapper_class) align 4 %_arg_Arr) {
entry:
  %elem = getelementptr inbounds %struct.__wrapper_class, ptr %_arg_Arr, i64 0
  %data = load i64, ptr %elem
  ret void
}
