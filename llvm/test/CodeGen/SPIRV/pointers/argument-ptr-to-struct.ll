; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#UCHAR:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#STRUCT_S:]] = OpTypeStruct %[[#FLOAT]] %[[#UCHAR]] %[[#UINT]]
; CHECK-DAG: %[[#PTR_STRUCT_S:]] = OpTypePointer Function %[[#STRUCT_S]]
; CHECK-DAG: %[[#FUNC_TYPE_K:]] = OpTypeFunction %[[#VOID]] %[[#PTR_STRUCT_S]]
; CHECK-DAG: %[[#FUNC_TYPE_H:]] = OpTypeFunction %[[#UINT]] %[[#PTR_STRUCT_S]]

; CHECK: %[[#]] = OpFunction %[[#VOID]] None %[[#FUNC_TYPE_K]]
; CHECK: %[[#]] = OpFunctionParameter %[[#PTR_STRUCT_S]]

; CHECK: %[[#]] = OpFunction %[[#UINT]] None %[[#FUNC_TYPE_H]]
; CHECK: %[[#]] = OpFunctionParameter %[[#PTR_STRUCT_S]]

%struct.s = type { float, i8, i32 }

define spir_kernel void @k(ptr noundef byval(%struct.s) align 4 %x) {
entry:
  %c = getelementptr inbounds %struct.s, ptr %x, i32 0, i32 2
  %l = load i32, ptr %c, align 4
  %add = add nsw i32 %l, 1
  %c1 = getelementptr inbounds %struct.s, ptr %x, i32 0, i32 2
  store i32 %add, ptr %c1, align 4
  ret void
}

define spir_func i32 @h(ptr noundef byval(%struct.s) align 4 %x) {
entry:
  %c = getelementptr inbounds %struct.s, ptr %x, i32 0, i32 2
  %l = load i32, ptr %c, align 4
  %add = add nsw i32 %l, 1
  ret i32 %add
}
