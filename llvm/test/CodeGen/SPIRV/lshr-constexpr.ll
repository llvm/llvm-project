; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#type_int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#type_int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#type_vec:]] = OpTypeVector %[[#type_int32]] 2
; CHECK-DAG: %[[#const1:]] = OpConstant %[[#type_int32]] 1
; CHECK-DAG: %[[#vec_const:]] = OpConstantComposite %[[#type_vec]] %[[#const1]] %[[#const1]]
; CHECK-DAG: %[[#const32:]] = OpConstant %[[#type_int64]] 32

; CHECK:     %[[#bitcast_res:]] = OpBitcast %[[#type_int64]] %[[#vec_const]]
; CHECK:     %[[#shift_res:]] = OpShiftRightLogical %[[#type_int64]] %[[#bitcast_res]] %[[#const32]]

define void @foo(i64* %arg) {
entry:
  %0 = lshr i64 bitcast (<2 x i32> <i32 1, i32 1> to i64), 32
  store i64 %0, i64* %arg
  ret void
}
