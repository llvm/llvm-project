; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --implicit-check-not=OpIsNormal --implicit-check-not="OpCapability Kernel"
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan -filetype=obj %s -o - | spirv-val %}

; CHECK-DAG: %[[I32:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[ABS32:[0-9]+]] = OpConstant %[[I32]] 2147483647
; CHECK-DAG: %[[LOW32:[0-9]+]] = OpConstant %[[I32]] 8388608
; CHECK-DAG: %[[SPAN32:[0-9]+]] = OpConstant %[[I32]] 2130706432
; CHECK-DAG: %[[I16:[0-9]+]] = OpTypeInt 16 0
; CHECK-DAG: %[[ABS16:[0-9]+]] = OpConstant %[[I16]] 32767
; CHECK-DAG: %[[LOW16:[0-9]+]] = OpConstant %[[I16]] 1024
; CHECK-DAG: %[[SPAN16:[0-9]+]] = OpConstant %[[I16]] 30720

; CHECK: %[[ARG:[0-9]+]] = OpFunctionParameter
; CHECK: %[[BITS:[0-9]+]] = OpBitcast %[[I32]] %[[ARG]]
; CHECK: %[[MAG:[0-9]+]] = OpBitwiseAnd %[[I32]] %[[BITS]] %[[ABS32]]
; CHECK: %[[DELTA:[0-9]+]] = OpISub %[[I32]] %[[MAG]] %[[LOW32]]
; CHECK: %[[NORMAL:[0-9]+]] = OpULessThan %{{[0-9]+}} %[[DELTA]] %[[SPAN32]]
; CHECK: OpReturnValue %[[NORMAL]]
define i1 @isnormal_float(float %x) {
  %normal = call i1 @llvm.is.fpclass.f32(float %x, i32 264)
  ret i1 %normal
}

; CHECK: %[[ARG:[0-9]+]] = OpFunctionParameter
; CHECK: %[[BITS:[0-9]+]] = OpBitcast %[[I16]] %[[ARG]]
; CHECK: %[[MAG:[0-9]+]] = OpBitwiseAnd %[[I16]] %[[BITS]] %[[ABS16]]
; CHECK: %[[DELTA:[0-9]+]] = OpISub %[[I16]] %[[MAG]] %[[LOW16]]
; CHECK: %[[NORMAL:[0-9]+]] = OpULessThan %{{[0-9]+}} %[[DELTA]] %[[SPAN16]]
; CHECK: OpReturnValue %[[NORMAL]]
define i1 @isnormal_half(half %x) {
  %normal = call i1 @llvm.is.fpclass.f16(half %x, i32 264)
  ret i1 %normal
}
