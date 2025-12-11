; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "OpenCL.std"
; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK: %[[#Foo]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_add_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_sub_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_add_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_sub_sat
; CHECK: %[[#Bar]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_add_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_sub_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_add_sat
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] s_sub_sat

@G_r1_foo = global i16 0
@G_r2_foo = global i16 0
@G_r3_foo = global i16 0
@G_r4_foo = global i16 0
@G_r1_bar = global <4 x i32> zeroinitializer
@G_r2_bar = global <4 x i32> zeroinitializer
@G_r3_bar = global <4 x i32> zeroinitializer
@G_r4_bar = global <4 x i32> zeroinitializer

define spir_func void @foo(i16 %x, i16 %y) {
entry:
  %r1 = tail call i16 @llvm.uadd.sat.i16(i16 %x, i16 %y)
  store i16 %r1, ptr @G_r1_foo
  %r2 = tail call i16 @llvm.usub.sat.i16(i16 %x, i16 %y)
  store i16 %r2, ptr @G_r2_foo
  %r3 = tail call i16 @llvm.sadd.sat.i16(i16 %x, i16 %y)
  store i16 %r3, ptr @G_r3_foo
  %r4 = tail call i16 @llvm.ssub.sat.i16(i16 %x, i16 %y)
  store i16 %r4, ptr @G_r4_foo
  ret void
}

define spir_func void @bar(<4 x i32> %x, <4 x i32> %y) {
entry:
  %r1 = tail call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r1, ptr @G_r1_bar
  %r2 = tail call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r2, ptr @G_r2_bar
  %r3 = tail call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r3, ptr @G_r3_bar
  %r4 = tail call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  store <4 x i32> %r4, ptr @G_r4_bar
  ret void
}

declare i16 @llvm.uadd.sat.i16(i16, i16)
declare i16 @llvm.usub.sat.i16(i16, i16)
declare i16 @llvm.sadd.sat.i16(i16, i16)
declare i16 @llvm.ssub.sat.i16(i16, i16)
declare <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.usub.sat.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>)
