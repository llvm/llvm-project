; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Float:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[One:.*]] = OpConstant %[[Int]] 1
; CHECK-DAG: %[[Seventeen:.*]] = OpConstant %[[Int]] 17
; CHECK-DAG: %[[FloatVec1:.*]] = OpTypeVectorIdEXT %[[Float]] %[[One]]
; CHECK-DAG: %[[FloatVec17:.*]] = OpTypeVectorIdEXT %[[Float]] %[[Seventeen]]

; CHECK: OpFunction
; CHECK: %[[X:.*]] = OpFunctionParameter %[[Float]]
; CHECK: %[[V:.*]] = OpFunctionParameter %[[FloatVec1]]
; CHECK: %[[Float1IntR:.*]] = OpCompositeExtract %[[Float]] %[[V]] 0
; CHECK: %[[Fmul:.*]] = OpFMul %[[Float]] %[[X]] %[[Float1IntR]]
; CHECK: OpReturnValue %[[Fmul]]
; CHECK: OpFunctionEnd
define spir_func float @test_vector_reduce_fmul_v1f32(float %x, <1 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v1f32(float %x, <1 x float> %v)
  ret float %res
}

; CHECK: OpFunction
; CHECK: %[[ParamX:.*]] = OpFunctionParameter %[[Float]]
; CHECK: %[[ParamVec17Float:.*]] = OpFunctionParameter %[[FloatVec17]]
; CHECK: %[[Vec17FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 0
; CHECK: %[[Vec17FloatR1:.*]] = OpFMul %[[Float]] %[[ParamX]] %[[Vec17FloatItem0]]
; CHECK: %[[Vec17FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 1
; CHECK: %[[Vec17FloatR2:.*]] = OpFMul %[[Float]] %[[Vec17FloatR1]] %[[Vec17FloatItem1]]
; CHECK: %[[Vec17FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 2
; CHECK: %[[Vec17FloatR3:.*]] = OpFMul %[[Float]] %[[Vec17FloatR2]] %[[Vec17FloatItem2]]
; CHECK: %[[Vec17FloatItem3:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 3
; CHECK: %[[Vec17FloatR4:.*]] = OpFMul %[[Float]] %[[Vec17FloatR3]] %[[Vec17FloatItem3]]
; CHECK: %[[Vec17FloatItem4:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 4
; CHECK: %[[Vec17FloatR5:.*]] = OpFMul %[[Float]] %[[Vec17FloatR4]] %[[Vec17FloatItem4]]
; CHECK: %[[Vec17FloatItem5:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 5
; CHECK: %[[Vec17FloatR6:.*]] = OpFMul %[[Float]] %[[Vec17FloatR5]] %[[Vec17FloatItem5]]
; CHECK: %[[Vec17FloatItem6:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 6
; CHECK: %[[Vec17FloatR7:.*]] = OpFMul %[[Float]] %[[Vec17FloatR6]] %[[Vec17FloatItem6]]
; CHECK: %[[Vec17FloatItem7:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 7
; CHECK: %[[Vec17FloatR8:.*]] = OpFMul %[[Float]] %[[Vec17FloatR7]] %[[Vec17FloatItem7]]
; CHECK: %[[Vec17FloatItem8:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 8
; CHECK: %[[Vec17FloatR9:.*]] = OpFMul %[[Float]] %[[Vec17FloatR8]] %[[Vec17FloatItem8]]
; CHECK: %[[Vec17FloatItem9:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 9
; CHECK: %[[Vec17FloatR10:.*]] = OpFMul %[[Float]] %[[Vec17FloatR9]] %[[Vec17FloatItem9]]
; CHECK: %[[Vec17FloatItem10:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 10
; CHECK: %[[Vec17FloatR11:.*]] = OpFMul %[[Float]] %[[Vec17FloatR10]] %[[Vec17FloatItem10]]
; CHECK: %[[Vec17FloatItem11:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 11
; CHECK: %[[Vec17FloatR12:.*]] = OpFMul %[[Float]] %[[Vec17FloatR11]] %[[Vec17FloatItem11]]
; CHECK: %[[Vec17FloatItem12:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 12
; CHECK: %[[Vec17FloatR13:.*]] = OpFMul %[[Float]] %[[Vec17FloatR12]] %[[Vec17FloatItem12]]
; CHECK: %[[Vec17FloatItem13:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 13
; CHECK: %[[Vec17FloatR14:.*]] = OpFMul %[[Float]] %[[Vec17FloatR13]] %[[Vec17FloatItem13]]
; CHECK: %[[Vec17FloatItem14:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 14
; CHECK: %[[Vec17FloatR15:.*]] = OpFMul %[[Float]] %[[Vec17FloatR14]] %[[Vec17FloatItem14]]
; CHECK: %[[Vec17FloatItem15:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 15
; CHECK: %[[Vec17FloatR16:.*]] = OpFMul %[[Float]] %[[Vec17FloatR15]] %[[Vec17FloatItem15]]
; CHECK: %[[Vec17FloatItem16:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 16
; CHECK: %[[Vec17FloatR:.*]] = OpFMul %[[Float]] %[[Vec17FloatR16]] %[[Vec17FloatItem16]]
; CHECK: OpReturnValue %[[Vec17FloatR]]
; CHECK: OpFunctionEnd
define spir_func float @test_vector_reduce_fmul_v17f32(float %x, <17 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v17i32(float %x, <17 x float> %v)
  ret float %res
}

declare float @llvm.vector.reduce.fmul.v1f32(float, <1 x float>)
declare float @llvm.vector.reduce.fmul.v17f32(float, <17 x float>)
