; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#size:]] = OpConstant %[[#int32]] 8
; CHECK-DAG: %[[#arr:]] = OpTypeArray %[[#bool]] %[[#size]]
; CHECK-DAG: %[[#ptr:]] = OpTypePointer Function %[[#arr]]

; CHECK: OpFunction
; CHECK: OpVariable %[[#ptr]] Function

@global = internal global [8 x i1] zeroinitializer

define void @foo() {
entry:
  %gep = getelementptr inbounds i8, ptr @global, i32 1
  ret void
}
