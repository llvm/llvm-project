; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     OpName %[[#v:]] "v"
; CHECK-SPIRV:     OpName %[[#index:]] "index"
; CHECK-SPIRV:     OpName %[[#res:]] "res"

; CHECK-SPIRV-DAG: %[[#int16:]] = OpTypeInt 16
; CHECK-SPIRV-DAG: %[[#int32:]] = OpTypeInt 32
; CHECK-SPIRV-DAG: %[[#int16_2:]] = OpTypeVector %[[#int16]] 2

; CHECK-SPIRV:     %[[#undef:]] = OpUndef %[[#int16_2]]

; CHECK-SPIRV-DAG: %[[#const1:]] = OpConstant %[[#int16]] 4
; CHECK-SPIRV-DAG: %[[#const2:]] = OpConstant %[[#int16]] 8
; CHECK-SPIRV-NOT: %[[#idx1:]] = OpConstant %[[#int32]] 0
; CHECK-SPIRV-NOT: %[[#idx2:]] = OpConstant %[[#int32]] 1

; CHECK-SPIRV:     %[[#vec1:]] = OpCompositeInsert %[[#int16_2]] %[[#const1]] %[[#undef]] 0
; CHECK-SPIRV:     %[[#vec2:]] = OpCompositeInsert %[[#int16_2]] %[[#const2]] %[[#vec1]] 1
; CHECK-SPIRV:     %[[#res]] = OpVectorInsertDynamic %[[#int16_2]] %[[#vec2]] %[[#v]] %[[#index]]

define spir_kernel void @test(<2 x i16>* nocapture %out, i16 %v, i32 %index) {
entry:
  %vec1 = insertelement <2 x i16> undef, i16 4, i32 0
  %vec2 = insertelement <2 x i16> %vec1, i16 8, i32 1
  %res = insertelement <2 x i16> %vec2, i16 %v, i32 %index
  store <2 x i16> %res, <2 x i16>* %out, align 4
  ret void
}
