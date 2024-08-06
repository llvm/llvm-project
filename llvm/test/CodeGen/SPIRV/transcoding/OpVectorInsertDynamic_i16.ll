; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:     OpName %[[#v:]] "v"
; CHECK:     OpName %[[#index:]] "index"
; CHECK:     OpName %[[#res:]] "res"
; CHECK-DAG: %[[#int16:]] = OpTypeInt 16
; CHECK-DAG: %[[#int32:]] = OpTypeInt 32
; CHECK-DAG: %[[#int16_2:]] = OpTypeVector %[[#int16]] 2
; CHECK-DAG: %[[#undef:]] = OpUndef %[[#int16_2]]
; CHECK-DAG: %[[#const1:]] = OpConstant %[[#int16]] 4
; CHECK-DAG: %[[#const2:]] = OpConstant %[[#int16]] 8
; CHECK-NOT: %[[#idx1:]] = OpConstant %[[#int32]] 0
; CHECK-NOT: %[[#idx2:]] = OpConstant %[[#int32]] 1
; CHECK:     %[[#vec1:]] = OpCompositeInsert %[[#int16_2]] %[[#const1]] %[[#undef]] 0
; CHECK:     %[[#vec2:]] = OpCompositeInsert %[[#int16_2]] %[[#const2]] %[[#vec1]] 1
; CHECK:     %[[#res]] = OpVectorInsertDynamic %[[#int16_2]] %[[#vec2]] %[[#v]] %[[#index]]

define spir_kernel void @test(<2 x i16>* nocapture %out, i16 %v, i32 %index) {
entry:
  %vec1 = insertelement <2 x i16> undef, i16 4, i32 0
  %vec2 = insertelement <2 x i16> %vec1, i16 8, i32 1
  %res = insertelement <2 x i16> %vec2, i16 %v, i32 %index
  store <2 x i16> %res, <2 x i16>* %out, align 4
  ret void
}
