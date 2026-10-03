; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Float:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[One:.*]] = OpConstant %[[Int]] 1
; CHECK-DAG: %[[Seventeen:.*]] = OpConstant %[[Int]] 17
; CHECK-DAG: %[[FloatVec1:.*]] = OpTypeVectorIdEXT %[[Float]] %[[One]]
; CHECK-DAG: %[[FloatVec17:.*]] = OpTypeVectorIdEXT %[[Float]] %[[Seventeen]]

; CHECK: OpFunction
; CHECK: %[[V:.*]] = OpFunctionParameter %[[FloatVec1]]
; CHECK: %[[Float1IntR:.*]] = OpCompositeExtract %[[Float]] %[[V]] 0
; CHECK: OpReturnValue %[[Float1IntR]]
; CHECK: OpFunctionEnd
define spir_func float @test_vector_reduce_fmaximum_v1f32(<1 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v1f32(<1 x float> %v)
  ret float %res
}

; CHECK: OpFunction
; CHECK: %[[ParamVec17Float:.*]] = OpFunctionParameter %[[FloatVec17]]
; CHECK: %[[Vec17FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 0
; CHECK: %[[Vec17FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 1
; CHECK: %[[Vec17FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 2
; CHECK: %[[Vec17FloatItem3:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 3
; CHECK: %[[Vec17FloatItem4:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 4
; CHECK: %[[Vec17FloatItem5:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 5
; CHECK: %[[Vec17FloatItem6:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 6
; CHECK: %[[Vec17FloatItem7:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 7
; CHECK: %[[Vec17FloatItem8:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 8
; CHECK: %[[Vec17FloatItem9:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 9
; CHECK: %[[Vec17FloatItem10:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 10
; CHECK: %[[Vec17FloatItem11:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 11
; CHECK: %[[Vec17FloatItem12:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 12
; CHECK: %[[Vec17FloatItem13:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 13
; CHECK: %[[Vec17FloatItem14:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 14
; CHECK: %[[Vec17FloatItem15:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 15
; CHECK: %[[Vec17FloatItem16:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec17Float]] 16
; CHECK: %[[Vec17FloatR1MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatItem0]] %[[Vec17FloatItem1]]
; CHECK: %[[Vec17FloatR1Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatItem0]] %[[Vec17FloatItem1]]
; CHECK: %[[Vec17FloatR1NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR1Ord]] %[[Vec17FloatR1MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR1NaN]] %[[#]]
; CHECK: %[[Vec17FloatR1:.*]] = OpSelect %[[Float]] %[[Vec17FloatR1IsZero]] %[[#]] %[[Vec17FloatR1NaN]]
; CHECK: %[[Vec17FloatR2MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR1]] %[[Vec17FloatItem2]]
; CHECK: %[[Vec17FloatR2Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR1]] %[[Vec17FloatItem2]]
; CHECK: %[[Vec17FloatR2NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR2Ord]] %[[Vec17FloatR2MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR2IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR2NaN]] %[[#]]
; CHECK: %[[Vec17FloatR2:.*]] = OpSelect %[[Float]] %[[Vec17FloatR2IsZero]] %[[#]] %[[Vec17FloatR2NaN]]
; CHECK: %[[Vec17FloatR3MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR2]] %[[Vec17FloatItem3]]
; CHECK: %[[Vec17FloatR3Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR2]] %[[Vec17FloatItem3]]
; CHECK: %[[Vec17FloatR3NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR3Ord]] %[[Vec17FloatR3MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR3IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR3NaN]] %[[#]]
; CHECK: %[[Vec17FloatR3:.*]] = OpSelect %[[Float]] %[[Vec17FloatR3IsZero]] %[[#]] %[[Vec17FloatR3NaN]]
; CHECK: %[[Vec17FloatR4MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR3]] %[[Vec17FloatItem4]]
; CHECK: %[[Vec17FloatR4Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR3]] %[[Vec17FloatItem4]]
; CHECK: %[[Vec17FloatR4NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR4Ord]] %[[Vec17FloatR4MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR4IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR4NaN]] %[[#]]
; CHECK: %[[Vec17FloatR4:.*]] = OpSelect %[[Float]] %[[Vec17FloatR4IsZero]] %[[#]] %[[Vec17FloatR4NaN]]
; CHECK: %[[Vec17FloatR5MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR4]] %[[Vec17FloatItem5]]
; CHECK: %[[Vec17FloatR5Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR4]] %[[Vec17FloatItem5]]
; CHECK: %[[Vec17FloatR5NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR5Ord]] %[[Vec17FloatR5MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR5IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR5NaN]] %[[#]]
; CHECK: %[[Vec17FloatR5:.*]] = OpSelect %[[Float]] %[[Vec17FloatR5IsZero]] %[[#]] %[[Vec17FloatR5NaN]]
; CHECK: %[[Vec17FloatR6MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR5]] %[[Vec17FloatItem6]]
; CHECK: %[[Vec17FloatR6Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR5]] %[[Vec17FloatItem6]]
; CHECK: %[[Vec17FloatR6NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR6Ord]] %[[Vec17FloatR6MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR6IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR6NaN]] %[[#]]
; CHECK: %[[Vec17FloatR6:.*]] = OpSelect %[[Float]] %[[Vec17FloatR6IsZero]] %[[#]] %[[Vec17FloatR6NaN]]
; CHECK: %[[Vec17FloatR7MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR6]] %[[Vec17FloatItem7]]
; CHECK: %[[Vec17FloatR7Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR6]] %[[Vec17FloatItem7]]
; CHECK: %[[Vec17FloatR7NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR7Ord]] %[[Vec17FloatR7MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR7IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR7NaN]] %[[#]]
; CHECK: %[[Vec17FloatR7:.*]] = OpSelect %[[Float]] %[[Vec17FloatR7IsZero]] %[[#]] %[[Vec17FloatR7NaN]]
; CHECK: %[[Vec17FloatR8MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR7]] %[[Vec17FloatItem8]]
; CHECK: %[[Vec17FloatR8Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR7]] %[[Vec17FloatItem8]]
; CHECK: %[[Vec17FloatR8NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR8Ord]] %[[Vec17FloatR8MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR8IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR8NaN]] %[[#]]
; CHECK: %[[Vec17FloatR8:.*]] = OpSelect %[[Float]] %[[Vec17FloatR8IsZero]] %[[#]] %[[Vec17FloatR8NaN]]
; CHECK: %[[Vec17FloatR9MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR8]] %[[Vec17FloatItem9]]
; CHECK: %[[Vec17FloatR9Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR8]] %[[Vec17FloatItem9]]
; CHECK: %[[Vec17FloatR9NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR9Ord]] %[[Vec17FloatR9MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR9IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR9NaN]] %[[#]]
; CHECK: %[[Vec17FloatR9:.*]] = OpSelect %[[Float]] %[[Vec17FloatR9IsZero]] %[[#]] %[[Vec17FloatR9NaN]]
; CHECK: %[[Vec17FloatR10MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR9]] %[[Vec17FloatItem10]]
; CHECK: %[[Vec17FloatR10Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR9]] %[[Vec17FloatItem10]]
; CHECK: %[[Vec17FloatR10NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR10Ord]] %[[Vec17FloatR10MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR10IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR10NaN]] %[[#]]
; CHECK: %[[Vec17FloatR10:.*]] = OpSelect %[[Float]] %[[Vec17FloatR10IsZero]] %[[#]] %[[Vec17FloatR10NaN]]
; CHECK: %[[Vec17FloatR11MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR10]] %[[Vec17FloatItem11]]
; CHECK: %[[Vec17FloatR11Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR10]] %[[Vec17FloatItem11]]
; CHECK: %[[Vec17FloatR11NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR11Ord]] %[[Vec17FloatR11MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR11IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR11NaN]] %[[#]]
; CHECK: %[[Vec17FloatR11:.*]] = OpSelect %[[Float]] %[[Vec17FloatR11IsZero]] %[[#]] %[[Vec17FloatR11NaN]]
; CHECK: %[[Vec17FloatR12MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR11]] %[[Vec17FloatItem12]]
; CHECK: %[[Vec17FloatR12Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR11]] %[[Vec17FloatItem12]]
; CHECK: %[[Vec17FloatR12NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR12Ord]] %[[Vec17FloatR12MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR12IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR12NaN]] %[[#]]
; CHECK: %[[Vec17FloatR12:.*]] = OpSelect %[[Float]] %[[Vec17FloatR12IsZero]] %[[#]] %[[Vec17FloatR12NaN]]
; CHECK: %[[Vec17FloatR13MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR12]] %[[Vec17FloatItem13]]
; CHECK: %[[Vec17FloatR13Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR12]] %[[Vec17FloatItem13]]
; CHECK: %[[Vec17FloatR13NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR13Ord]] %[[Vec17FloatR13MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR13IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR13NaN]] %[[#]]
; CHECK: %[[Vec17FloatR13:.*]] = OpSelect %[[Float]] %[[Vec17FloatR13IsZero]] %[[#]] %[[Vec17FloatR13NaN]]
; CHECK: %[[Vec17FloatR14MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR13]] %[[Vec17FloatItem14]]
; CHECK: %[[Vec17FloatR14Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR13]] %[[Vec17FloatItem14]]
; CHECK: %[[Vec17FloatR14NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR14Ord]] %[[Vec17FloatR14MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR14IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR14NaN]] %[[#]]
; CHECK: %[[Vec17FloatR14:.*]] = OpSelect %[[Float]] %[[Vec17FloatR14IsZero]] %[[#]] %[[Vec17FloatR14NaN]]
; CHECK: %[[Vec17FloatR15MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR14]] %[[Vec17FloatItem15]]
; CHECK: %[[Vec17FloatR15Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR14]] %[[Vec17FloatItem15]]
; CHECK: %[[Vec17FloatR15NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR15Ord]] %[[Vec17FloatR15MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR15IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR15NaN]] %[[#]]
; CHECK: %[[Vec17FloatR15:.*]] = OpSelect %[[Float]] %[[Vec17FloatR15IsZero]] %[[#]] %[[Vec17FloatR15NaN]]
; CHECK: %[[Vec17FloatR16MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR15]] %[[Vec17FloatItem16]]
; CHECK: %[[Vec17FloatR16Ord:.*]] = OpOrdered %[[#]] %[[Vec17FloatR15]] %[[Vec17FloatItem16]]
; CHECK: %[[Vec17FloatR16NaN:.*]] = OpSelect %[[Float]] %[[Vec17FloatR16Ord]] %[[Vec17FloatR16MinMax]] %[[#]]
; CHECK: %[[Vec17FloatR16IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec17FloatR16NaN]] %[[#]]
; CHECK: %[[Vec17FloatR16:.*]] = OpSelect %[[Float]] %[[Vec17FloatR16IsZero]] %[[#]] %[[Vec17FloatR16NaN]]
; CHECK: OpReturnValue %[[Vec17FloatR16]]
; CHECK: OpFunctionEnd
define spir_func float @test_vector_reduce_fmaximum_v17f32(<17 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v17i32(<17 x float> %v)
  ret float %res
}

declare float @llvm.vector.reduce.fmaximum.v1f32(<1 x float>)
declare float @llvm.vector.reduce.fmaximum.v17f32(<17 x float>)
