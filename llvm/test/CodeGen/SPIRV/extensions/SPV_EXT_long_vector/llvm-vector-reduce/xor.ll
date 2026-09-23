; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[One:.*]] = OpConstant %[[Int]] 1
; CHECK-DAG: %[[Seventeen:.*]] = OpConstant %[[Int]] 17
; CHECK-DAG: %[[IntVec1:.*]] = OpTypeVectorIdEXT %[[Int]] %[[One]]
; CHECK-DAG: %[[IntVec17:.*]] = OpTypeVectorIdEXT %[[Int]] %[[Seventeen]]

; CHECK: OpFunction
; CHECK: %[[V:.*]] = OpFunctionParameter %[[IntVec1]]
; CHECK: %[[Vec1IntR:.*]] = OpCompositeExtract %[[Int]] %[[V]] 0
; CHECK: OpReturnValue %[[Vec1IntR]]
; CHECK: OpFunctionEnd
define spir_func i32 @test_vector_reduce_xor_v1i32(<1 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.xor.v1i32(<1 x i32> %v)
  ret i32 %res
}

; CHECK: OpFunction
; CHECK: %[[ParamVec17Int:.*]] = OpFunctionParameter %[[IntVec17]]
; CHECK: %[[Vec17IntItem0:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 0
; CHECK: %[[Vec17IntItem1:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 1
; CHECK: %[[Vec17IntItem2:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 2
; CHECK: %[[Vec17IntItem3:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 3
; CHECK: %[[Vec17IntItem4:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 4
; CHECK: %[[Vec17IntItem5:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 5
; CHECK: %[[Vec17IntItem6:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 6
; CHECK: %[[Vec17IntItem7:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 7
; CHECK: %[[Vec17IntItem8:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 8
; CHECK: %[[Vec17IntItem9:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 9
; CHECK: %[[Vec17IntItem10:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 10
; CHECK: %[[Vec17IntItem11:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 11
; CHECK: %[[Vec17IntItem12:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 12
; CHECK: %[[Vec17IntItem13:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 13
; CHECK: %[[Vec17IntItem14:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 14
; CHECK: %[[Vec17IntItem15:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 15
; CHECK: %[[Vec17IntItem16:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec17Int]] 16
; CHECK: %[[Vec17IntR1:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntItem0]] %[[Vec17IntItem1]]
; CHECK: %[[Vec17IntR2:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR1]] %[[Vec17IntItem2]]
; CHECK: %[[Vec17IntR3:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR2]] %[[Vec17IntItem3]]
; CHECK: %[[Vec17IntR4:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR3]] %[[Vec17IntItem4]]
; CHECK: %[[Vec17IntR5:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR4]] %[[Vec17IntItem5]]
; CHECK: %[[Vec17IntR6:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR5]] %[[Vec17IntItem6]]
; CHECK: %[[Vec17IntR7:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR6]] %[[Vec17IntItem7]]
; CHECK: %[[Vec17IntR8:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR7]] %[[Vec17IntItem8]]
; CHECK: %[[Vec17IntR9:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR8]] %[[Vec17IntItem9]]
; CHECK: %[[Vec17IntR10:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR9]] %[[Vec17IntItem10]]
; CHECK: %[[Vec17IntR11:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR10]] %[[Vec17IntItem11]]
; CHECK: %[[Vec17IntR12:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR11]] %[[Vec17IntItem12]]
; CHECK: %[[Vec17IntR13:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR12]] %[[Vec17IntItem13]]
; CHECK: %[[Vec17IntR14:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR13]] %[[Vec17IntItem14]]
; CHECK: %[[Vec17IntR15:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR14]] %[[Vec17IntItem15]]
; CHECK: %[[Vec17IntR16:.*]] = OpBitwiseXor %[[Int]] %[[Vec17IntR15]] %[[Vec17IntItem16]]
; CHECK: OpReturnValue %[[Vec17IntR16]]
; CHECK: OpFunctionEnd
define spir_func i32 @test_vector_reduce_xor_v17i32(<17 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.xor.v17i32(<17 x i32> %v)
  ret i32 %res
}

declare i32 @llvm.vector.reduce.xor.v1i32(<1 x i32>)
declare i32 @llvm.vector.reduce.xor.v17i32(<17 x i32>)
