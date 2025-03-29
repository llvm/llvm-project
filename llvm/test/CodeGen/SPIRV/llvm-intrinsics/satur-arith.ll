; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "OpenCL.std"
; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK: %[[#Foo]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] u_sub_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_sub_sat
; CHECK: %[[#Bar]] = OpFunction
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] u_sub_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_sub_sat

define spir_func void @foo(i16 %x, i16 %y) {
entry:
  %r1 = tail call i16 @llvm.uadd.sat.i16(i16 %x, i16 %y)
  %r2 = tail call i16 @llvm.usub.sat.i16(i16 %x, i16 %y)
  %r3 = tail call i16 @llvm.sadd.sat.i16(i16 %x, i16 %y)
  %r4 = tail call i16 @llvm.ssub.sat.i16(i16 %x, i16 %y)
  ret void
}

define spir_func void @bar(<4 x i32> %x, <4 x i32> %y) {
entry:
  %r1 = tail call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  %r2 = tail call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  %r3 = tail call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  %r4 = tail call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  ret void
}
