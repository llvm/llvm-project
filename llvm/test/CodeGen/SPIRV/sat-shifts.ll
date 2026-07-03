; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Bar:]] "bar"

; CHECK: %[[#Foo]] = OpFunction
; CHECK: %[[#]] = OpShiftLeftLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftRightArithmetic %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSLessThan %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpINotEqual %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftLeftLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftRightLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpINotEqual %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]

; CHECK: %[[#Bar]] = OpFunction
; CHECK: %[[#]] = OpShiftLeftLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftRightArithmetic %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSLessThan %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpINotEqual %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftLeftLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpShiftRightLogical %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpINotEqual %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#]] %[[#]] %[[#]]

define spir_func void @foo(i16 %x, i16 %y) {
entry:
  %r1 = tail call i16 @llvm.sshl.sat.i16(i16 %x, i16 %y)
  store i16 %r1, ptr @G_r1_foo
  %r2 = tail call i16 @llvm.ushl.sat.i16(i16 %x, i16 %y)
  store i16 %r2, ptr @G_r2_foo
  ret void
}

define spir_func void @bar(<4 x i32> %x, <4 x i32> %y) {
entry:
  %r1 = tail call <4 x i32> @llvm.sshl.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r1, ptr @G_r1_bar
  %r2 = tail call <4 x i32> @llvm.ushl.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r2, ptr @G_r2_bar
  ret void
}

@G_r1_foo = global i16 0
@G_r2_foo = global i16 0
@G_r1_bar = global <4 x i32> zeroinitializer
@G_r2_bar = global <4 x i32> zeroinitializer

declare i16 @llvm.sshl.sat.i16(i16, i16)
declare i16 @llvm.ushl.sat.i16(i16, i16)
declare <4 x i32> @llvm.sshl.sat.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.ushl.sat.v4i32(<4 x i32>, <4 x i32>)
