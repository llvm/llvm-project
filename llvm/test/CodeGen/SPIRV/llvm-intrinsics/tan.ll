; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ext:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#type_f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#type_f64:]] = OpTypeFloat 64
; CHECK: %[[#extinst_f32:]] = OpExtInst %[[#type_f32]] %[[#ext]] tan %[[#]]
; CHECK: %[[#extinst_f64:]] = OpExtInst %[[#type_f64]] %[[#ext]] tan %[[#]]

define float @test_tan_f32(float %x) {
  %res = call float @llvm.tan.f32(float %x)
  ret float %res
}

define double @test_tan_f64(double %x) {
  %res = call double @llvm.tan.f64(double %x)
  ret double %res
}

declare float @llvm.tan.f32(float)
declare double @llvm.tan.f64(double)
