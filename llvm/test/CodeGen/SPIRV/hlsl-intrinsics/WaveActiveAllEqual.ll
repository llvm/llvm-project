; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend for various types and scalar/vector

; CHECK-DAG:   %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:   %[[#bool4:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#v4_half:]] = OpTypeVector %[[#f16]] 4
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_float
; CHECK:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
define i1 @test_float(float %fexpr) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#fexpr]]
  %0 = call i1 @llvm.spv.wave.all.equal.f32(float %fexpr)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_int
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i1 @test_int(i32 %iexpr) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#iexpr]]
  %0 = call i1 @llvm.spv.wave.all.equal.i32(i32 %iexpr)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_vhalf
; Here there's a vector, so we scalarize and then recombine the
; result back into one vector
define <4 x i1> @test_vhalf(<4 x half> %vbexpr) {
entry:
; CHECK: %[[#param:]] = OpFunctionParameter %[[#v4float:]]
; CHECK: %[[#ext1:]] = OpCompositeExtract %[[#f16]] %[[#param]] 0
; CHECK-NEXT: %[[#res1:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#ext1]]
; CHECK-NEXT: %[[#ext2:]] = OpCompositeExtract %[[#f16]] %[[#param]] 1
; CHECK-NEXT: %[[#res2:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#ext2]]
; CHECK-NEXT: %[[#ext3:]] = OpCompositeExtract %[[#f16]] %[[#param]] 2
; CHECK-NEXT: %[[#res3:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#ext3]]
; CHECK-NEXT: %[[#ext4:]] = OpCompositeExtract %[[#f16]] %[[#param]] 3
; CHECK-NEXT: %[[#res4:]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#scope]] %[[#ext4]]
; CHECK-NEXT: %[[#ret:]] = OpCompositeConstruct %[[#bool4]] %[[#res1:]] %[[#res2:]] %[[#res3:]] %[[#res4:]]
; CHECK-NEXT: OpReturnValue %[[#ret]]
  %0 = call <4 x i1> @llvm.spv.wave.all.equal.v4half(<4 x half> %vbexpr)
  ret <4 x i1> %0
}

declare i1 @llvm.spv.wave.all.equal.f32(float)
declare i1 @llvm.spv.wave.all.equal.i32(i32)
declare <4 x i1> @llvm.spv.wave.all.equal.v4half(<4 x half>)
