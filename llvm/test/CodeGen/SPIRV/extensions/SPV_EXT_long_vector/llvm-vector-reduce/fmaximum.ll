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
; CHECK: %[[Vec17FloatR1XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatItem0]]
; CHECK: %[[Vec17FloatR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR1XSign]] %[[#]]
; CHECK: %[[Vec17FloatR1Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR1XNeg]] %[[Vec17FloatItem1]] %[[Vec17FloatItem0]]
; CHECK: %[[Vec17FloatR1Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR1MinMax]] %[[Vec17FloatR1Sign]]
; CHECK: %[[Vec17FloatR1Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatItem0]] %[[Vec17FloatItem1]]
; CHECK: %[[Vec17FloatR1:.*]] = OpSelect %[[Float]] %[[Vec17FloatR1Uno]] %[[#]] %[[Vec17FloatR1Signed]]
; CHECK: %[[Vec17FloatR2MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR1]] %[[Vec17FloatItem2]]
; CHECK: %[[Vec17FloatR2XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR1]]
; CHECK: %[[Vec17FloatR2XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR2XSign]] %[[#]]
; CHECK: %[[Vec17FloatR2Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR2XNeg]] %[[Vec17FloatItem2]] %[[Vec17FloatR1]]
; CHECK: %[[Vec17FloatR2Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR2MinMax]] %[[Vec17FloatR2Sign]]
; CHECK: %[[Vec17FloatR2Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR1]] %[[Vec17FloatItem2]]
; CHECK: %[[Vec17FloatR2:.*]] = OpSelect %[[Float]] %[[Vec17FloatR2Uno]] %[[#]] %[[Vec17FloatR2Signed]]
; CHECK: %[[Vec17FloatR3MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR2]] %[[Vec17FloatItem3]]
; CHECK: %[[Vec17FloatR3XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR2]]
; CHECK: %[[Vec17FloatR3XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR3XSign]] %[[#]]
; CHECK: %[[Vec17FloatR3Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR3XNeg]] %[[Vec17FloatItem3]] %[[Vec17FloatR2]]
; CHECK: %[[Vec17FloatR3Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR3MinMax]] %[[Vec17FloatR3Sign]]
; CHECK: %[[Vec17FloatR3Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR2]] %[[Vec17FloatItem3]]
; CHECK: %[[Vec17FloatR3:.*]] = OpSelect %[[Float]] %[[Vec17FloatR3Uno]] %[[#]] %[[Vec17FloatR3Signed]]
; CHECK: %[[Vec17FloatR4MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR3]] %[[Vec17FloatItem4]]
; CHECK: %[[Vec17FloatR4XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR3]]
; CHECK: %[[Vec17FloatR4XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR4XSign]] %[[#]]
; CHECK: %[[Vec17FloatR4Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR4XNeg]] %[[Vec17FloatItem4]] %[[Vec17FloatR3]]
; CHECK: %[[Vec17FloatR4Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR4MinMax]] %[[Vec17FloatR4Sign]]
; CHECK: %[[Vec17FloatR4Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR3]] %[[Vec17FloatItem4]]
; CHECK: %[[Vec17FloatR4:.*]] = OpSelect %[[Float]] %[[Vec17FloatR4Uno]] %[[#]] %[[Vec17FloatR4Signed]]
; CHECK: %[[Vec17FloatR5MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR4]] %[[Vec17FloatItem5]]
; CHECK: %[[Vec17FloatR5XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR4]]
; CHECK: %[[Vec17FloatR5XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR5XSign]] %[[#]]
; CHECK: %[[Vec17FloatR5Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR5XNeg]] %[[Vec17FloatItem5]] %[[Vec17FloatR4]]
; CHECK: %[[Vec17FloatR5Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR5MinMax]] %[[Vec17FloatR5Sign]]
; CHECK: %[[Vec17FloatR5Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR4]] %[[Vec17FloatItem5]]
; CHECK: %[[Vec17FloatR5:.*]] = OpSelect %[[Float]] %[[Vec17FloatR5Uno]] %[[#]] %[[Vec17FloatR5Signed]]
; CHECK: %[[Vec17FloatR6MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR5]] %[[Vec17FloatItem6]]
; CHECK: %[[Vec17FloatR6XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR5]]
; CHECK: %[[Vec17FloatR6XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR6XSign]] %[[#]]
; CHECK: %[[Vec17FloatR6Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR6XNeg]] %[[Vec17FloatItem6]] %[[Vec17FloatR5]]
; CHECK: %[[Vec17FloatR6Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR6MinMax]] %[[Vec17FloatR6Sign]]
; CHECK: %[[Vec17FloatR6Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR5]] %[[Vec17FloatItem6]]
; CHECK: %[[Vec17FloatR6:.*]] = OpSelect %[[Float]] %[[Vec17FloatR6Uno]] %[[#]] %[[Vec17FloatR6Signed]]
; CHECK: %[[Vec17FloatR7MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR6]] %[[Vec17FloatItem7]]
; CHECK: %[[Vec17FloatR7XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR6]]
; CHECK: %[[Vec17FloatR7XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR7XSign]] %[[#]]
; CHECK: %[[Vec17FloatR7Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR7XNeg]] %[[Vec17FloatItem7]] %[[Vec17FloatR6]]
; CHECK: %[[Vec17FloatR7Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR7MinMax]] %[[Vec17FloatR7Sign]]
; CHECK: %[[Vec17FloatR7Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR6]] %[[Vec17FloatItem7]]
; CHECK: %[[Vec17FloatR7:.*]] = OpSelect %[[Float]] %[[Vec17FloatR7Uno]] %[[#]] %[[Vec17FloatR7Signed]]
; CHECK: %[[Vec17FloatR8MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR7]] %[[Vec17FloatItem8]]
; CHECK: %[[Vec17FloatR8XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR7]]
; CHECK: %[[Vec17FloatR8XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR8XSign]] %[[#]]
; CHECK: %[[Vec17FloatR8Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR8XNeg]] %[[Vec17FloatItem8]] %[[Vec17FloatR7]]
; CHECK: %[[Vec17FloatR8Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR8MinMax]] %[[Vec17FloatR8Sign]]
; CHECK: %[[Vec17FloatR8Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR7]] %[[Vec17FloatItem8]]
; CHECK: %[[Vec17FloatR8:.*]] = OpSelect %[[Float]] %[[Vec17FloatR8Uno]] %[[#]] %[[Vec17FloatR8Signed]]
; CHECK: %[[Vec17FloatR9MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR8]] %[[Vec17FloatItem9]]
; CHECK: %[[Vec17FloatR9XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR8]]
; CHECK: %[[Vec17FloatR9XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR9XSign]] %[[#]]
; CHECK: %[[Vec17FloatR9Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR9XNeg]] %[[Vec17FloatItem9]] %[[Vec17FloatR8]]
; CHECK: %[[Vec17FloatR9Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR9MinMax]] %[[Vec17FloatR9Sign]]
; CHECK: %[[Vec17FloatR9Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR8]] %[[Vec17FloatItem9]]
; CHECK: %[[Vec17FloatR9:.*]] = OpSelect %[[Float]] %[[Vec17FloatR9Uno]] %[[#]] %[[Vec17FloatR9Signed]]
; CHECK: %[[Vec17FloatR10MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR9]] %[[Vec17FloatItem10]]
; CHECK: %[[Vec17FloatR10XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR9]]
; CHECK: %[[Vec17FloatR10XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR10XSign]] %[[#]]
; CHECK: %[[Vec17FloatR10Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR10XNeg]] %[[Vec17FloatItem10]] %[[Vec17FloatR9]]
; CHECK: %[[Vec17FloatR10Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR10MinMax]] %[[Vec17FloatR10Sign]]
; CHECK: %[[Vec17FloatR10Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR9]] %[[Vec17FloatItem10]]
; CHECK: %[[Vec17FloatR10:.*]] = OpSelect %[[Float]] %[[Vec17FloatR10Uno]] %[[#]] %[[Vec17FloatR10Signed]]
; CHECK: %[[Vec17FloatR11MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR10]] %[[Vec17FloatItem11]]
; CHECK: %[[Vec17FloatR11XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR10]]
; CHECK: %[[Vec17FloatR11XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR11XSign]] %[[#]]
; CHECK: %[[Vec17FloatR11Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR11XNeg]] %[[Vec17FloatItem11]] %[[Vec17FloatR10]]
; CHECK: %[[Vec17FloatR11Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR11MinMax]] %[[Vec17FloatR11Sign]]
; CHECK: %[[Vec17FloatR11Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR10]] %[[Vec17FloatItem11]]
; CHECK: %[[Vec17FloatR11:.*]] = OpSelect %[[Float]] %[[Vec17FloatR11Uno]] %[[#]] %[[Vec17FloatR11Signed]]
; CHECK: %[[Vec17FloatR12MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR11]] %[[Vec17FloatItem12]]
; CHECK: %[[Vec17FloatR12XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR11]]
; CHECK: %[[Vec17FloatR12XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR12XSign]] %[[#]]
; CHECK: %[[Vec17FloatR12Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR12XNeg]] %[[Vec17FloatItem12]] %[[Vec17FloatR11]]
; CHECK: %[[Vec17FloatR12Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR12MinMax]] %[[Vec17FloatR12Sign]]
; CHECK: %[[Vec17FloatR12Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR11]] %[[Vec17FloatItem12]]
; CHECK: %[[Vec17FloatR12:.*]] = OpSelect %[[Float]] %[[Vec17FloatR12Uno]] %[[#]] %[[Vec17FloatR12Signed]]
; CHECK: %[[Vec17FloatR13MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR12]] %[[Vec17FloatItem13]]
; CHECK: %[[Vec17FloatR13XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR12]]
; CHECK: %[[Vec17FloatR13XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR13XSign]] %[[#]]
; CHECK: %[[Vec17FloatR13Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR13XNeg]] %[[Vec17FloatItem13]] %[[Vec17FloatR12]]
; CHECK: %[[Vec17FloatR13Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR13MinMax]] %[[Vec17FloatR13Sign]]
; CHECK: %[[Vec17FloatR13Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR12]] %[[Vec17FloatItem13]]
; CHECK: %[[Vec17FloatR13:.*]] = OpSelect %[[Float]] %[[Vec17FloatR13Uno]] %[[#]] %[[Vec17FloatR13Signed]]
; CHECK: %[[Vec17FloatR14MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR13]] %[[Vec17FloatItem14]]
; CHECK: %[[Vec17FloatR14XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR13]]
; CHECK: %[[Vec17FloatR14XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR14XSign]] %[[#]]
; CHECK: %[[Vec17FloatR14Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR14XNeg]] %[[Vec17FloatItem14]] %[[Vec17FloatR13]]
; CHECK: %[[Vec17FloatR14Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR14MinMax]] %[[Vec17FloatR14Sign]]
; CHECK: %[[Vec17FloatR14Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR13]] %[[Vec17FloatItem14]]
; CHECK: %[[Vec17FloatR14:.*]] = OpSelect %[[Float]] %[[Vec17FloatR14Uno]] %[[#]] %[[Vec17FloatR14Signed]]
; CHECK: %[[Vec17FloatR15MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR14]] %[[Vec17FloatItem15]]
; CHECK: %[[Vec17FloatR15XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR14]]
; CHECK: %[[Vec17FloatR15XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR15XSign]] %[[#]]
; CHECK: %[[Vec17FloatR15Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR15XNeg]] %[[Vec17FloatItem15]] %[[Vec17FloatR14]]
; CHECK: %[[Vec17FloatR15Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR15MinMax]] %[[Vec17FloatR15Sign]]
; CHECK: %[[Vec17FloatR15Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR14]] %[[Vec17FloatItem15]]
; CHECK: %[[Vec17FloatR15:.*]] = OpSelect %[[Float]] %[[Vec17FloatR15Uno]] %[[#]] %[[Vec17FloatR15Signed]]
; CHECK: %[[Vec17FloatR16MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec17FloatR15]] %[[Vec17FloatItem16]]
; CHECK: %[[Vec17FloatR16XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec17FloatR15]]
; CHECK: %[[Vec17FloatR16XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec17FloatR16XSign]] %[[#]]
; CHECK: %[[Vec17FloatR16Sign:.*]] = OpSelect %[[Float]] %[[Vec17FloatR16XNeg]] %[[Vec17FloatItem16]] %[[Vec17FloatR15]]
; CHECK: %[[Vec17FloatR16Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec17FloatR16MinMax]] %[[Vec17FloatR16Sign]]
; CHECK: %[[Vec17FloatR16Uno:.*]] = OpUnordered %[[#]] %[[Vec17FloatR15]] %[[Vec17FloatItem16]]
; CHECK: %[[Vec17FloatR16:.*]] = OpSelect %[[Float]] %[[Vec17FloatR16Uno]] %[[#]] %[[Vec17FloatR16Signed]]
; CHECK: OpReturnValue %[[Vec17FloatR16]]
; CHECK: OpFunctionEnd
define spir_func float @test_vector_reduce_fmaximum_v17f32(<17 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v17i32(<17 x float> %v)
  ret float %res
}

declare float @llvm.vector.reduce.fmaximum.v1f32(<1 x float>)
declare float @llvm.vector.reduce.fmaximum.v17f32(<17 x float>)
