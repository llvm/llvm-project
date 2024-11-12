; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.5-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.5-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#bool:]] = OpTypeBool
; CHECK: %[[#uint:]] = OpTypeInt 32 0
; CHECK: %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_wave_aat
define i1 @test_wave_aat(i1 %p1) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#bool]]
; CHECK: %[[#ret:]] = OpGroupNonUniformAny %[[#bool]] %[[#scope]] %[[#param]]
  %ret = call i1 @llvm.spv.wave.activeanytrue(i1 %p1)
  ret i1 %ret
}

declare i1 @llvm.spv.wave.activeanytrue(i1)
