; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

target triple = "spir64-unknown-unknown"

; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[CharVec2:.*]] = OpTypeVector %[[Char]] 2
; CHECK-DAG: %[[CharVec3:.*]] = OpTypeVector %[[Char]] 3

; CHECK-DAG: %[[Short:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[ShortVec2:.*]] = OpTypeVector %[[Short]] 2
; CHECK-DAG: %[[ShortVec3:.*]] = OpTypeVector %[[Short]] 3

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[IntVec2:.*]] = OpTypeVector %[[Int]] 2
; CHECK-DAG: %[[IntVec3:.*]] = OpTypeVector %[[Int]] 3

; CHECK-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[LongVec2:.*]] = OpTypeVector %[[Long]] 2
; CHECK-DAG: %[[LongVec3:.*]] = OpTypeVector %[[Long]] 3

; CHECK: OpFunction
; CHECK: %[[Shuffle1:.*]] = OpVectorShuffle %[[CharVec2]] %[[#]] %[[#]] 1 -1
; CHECK: %[[Added1:.*]] = OpIMul %[[CharVec2]] %[[#]] %[[#]]
; CHECK: %[[Vec2CharR:.*]] = OpCompositeExtract %[[Char]] %[[Added1]] 0
; CHECK: OpReturnValue %[[Vec2CharR]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Char:.*]] = OpFunctionParameter %[[CharVec3]]
; CHECK: %[[Vec3CharItem0:.*]] = OpCompositeExtract %[[Char]] %[[ParamVec3Char]] 0
; CHECK: %[[Vec3CharItem1:.*]] = OpCompositeExtract %[[Char]] %[[ParamVec3Char]] 1
; CHECK: %[[Vec3CharItem2:.*]] = OpCompositeExtract %[[Char]] %[[ParamVec3Char]] 2
; CHECK: %[[Vec3CharR1:.*]] = OpIMul %[[Char]] %[[Vec3CharItem0]] %[[Vec3CharItem1]]
; CHECK: %[[Vec3CharR2:.*]] = OpIMul %[[Char]] %[[Vec3CharR1]] %[[Vec3CharItem2]]
; CHECK: OpReturnValue %[[Vec3CharR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Shuffle1:.*]] = OpVectorShuffle %[[ShortVec2]] %[[#]] %[[#]] 1 -1
; CHECK: %[[Added1:.*]] = OpIMul %[[ShortVec2]] %[[#]] %[[#]]
; CHECK: %[[Vec2ShortR:.*]] = OpCompositeExtract %[[Short]] %[[Added1]] 0
; CHECK: OpReturnValue %[[Vec2ShortR]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Short:.*]] = OpFunctionParameter %[[ShortVec3]]
; CHECK: %[[Vec3ShortItem0:.*]] = OpCompositeExtract %[[Short]] %[[ParamVec3Short]] 0
; CHECK: %[[Vec3ShortItem1:.*]] = OpCompositeExtract %[[Short]] %[[ParamVec3Short]] 1
; CHECK: %[[Vec3ShortItem2:.*]] = OpCompositeExtract %[[Short]] %[[ParamVec3Short]] 2
; CHECK: %[[Vec3ShortR1:.*]] = OpIMul %[[Short]] %[[Vec3ShortItem0]] %[[Vec3ShortItem1]]
; CHECK: %[[Vec3ShortR2:.*]] = OpIMul %[[Short]] %[[Vec3ShortR1]] %[[Vec3ShortItem2]]
; CHECK: OpReturnValue %[[Vec3ShortR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Shuffle1:.*]] = OpVectorShuffle %[[IntVec2]] %[[#]] %[[#]] 1 -1
; CHECK: %[[Added1:.*]] = OpIMul %[[IntVec2]] %[[#]] %[[#]]
; CHECK: %[[Vec2IntR:.*]] = OpCompositeExtract %[[Int]] %[[Added1]] 0
; CHECK: OpReturnValue %[[Vec2IntR]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Int:.*]] = OpFunctionParameter %[[IntVec3]]
; CHECK: %[[Vec3IntItem0:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec3Int]] 0
; CHECK: %[[Vec3IntItem1:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec3Int]] 1
; CHECK: %[[Vec3IntItem2:.*]] = OpCompositeExtract %[[Int]] %[[ParamVec3Int]] 2
; CHECK: %[[Vec3IntR1:.*]] = OpIMul %[[Int]] %[[Vec3IntItem0]] %[[Vec3IntItem1]]
; CHECK: %[[Vec3IntR2:.*]] = OpIMul %[[Int]] %[[Vec3IntR1]] %[[Vec3IntItem2]]
; CHECK: OpReturnValue %[[Vec3IntR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Shuffle1:.*]] = OpVectorShuffle %[[LongVec2]] %[[#]] %[[#]] 1 -1
; CHECK: %[[Added1:.*]] = OpIMul %[[LongVec2]] %[[#]] %[[#]]
; CHECK: %[[Vec2LongR:.*]] = OpCompositeExtract %[[Long]] %[[Added1]] 0
; CHECK: OpReturnValue %[[Vec2LongR]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Long:.*]] = OpFunctionParameter %[[LongVec3]]
; CHECK: %[[Vec3LongItem0:.*]] = OpCompositeExtract %[[Long]] %[[ParamVec3Long]] 0
; CHECK: %[[Vec3LongItem1:.*]] = OpCompositeExtract %[[Long]] %[[ParamVec3Long]] 1
; CHECK: %[[Vec3LongItem2:.*]] = OpCompositeExtract %[[Long]] %[[ParamVec3Long]] 2
; CHECK: %[[Vec3LongR1:.*]] = OpIMul %[[Long]] %[[Vec3LongItem0]] %[[Vec3LongItem1]]
; CHECK: %[[Vec3LongR2:.*]] = OpIMul %[[Long]] %[[Vec3LongR1]] %[[Vec3LongItem2]]
; CHECK: OpReturnValue %[[Vec3LongR2]]
; CHECK: OpFunctionEnd

define spir_func i8 @test_vector_reduce_mul_v2i8(<2 x i8> %v) {
entry:
  %res = call i8 @llvm.vector.reduce.mul.v2i8(<2 x i8> %v)
  ret i8 %res
}

define spir_func i8 @test_vector_reduce_mul_v3i8(<3 x i8> %v) {
entry:
  %res = call i8 @llvm.vector.reduce.mul.v3i8(<3 x i8> %v)
  ret i8 %res
}

define spir_func i8 @test_vector_reduce_mul_v4i8(<4 x i8> %v) {
entry:
  %res = call i8 @llvm.vector.reduce.mul.v4i8(<4 x i8> %v)
  ret i8 %res
}

define spir_func i8 @test_vector_reduce_mul_v8i8(<8 x i8> %v) {
entry:
  %res = call i8 @llvm.vector.reduce.mul.v8i8(<8 x i8> %v)
  ret i8 %res
}

define spir_func i8 @test_vector_reduce_mul_v16i8(<16 x i8> %v) {
entry:
  %res = call i8 @llvm.vector.reduce.mul.v16i8(<16 x i8> %v)
  ret i8 %res
}

define spir_func i16 @test_vector_reduce_mul_v2i16(<2 x i16> %v) {
entry:
  %res = call i16 @llvm.vector.reduce.mul.v2i16(<2 x i16> %v)
  ret i16 %res
}

define spir_func i16 @test_vector_reduce_mul_v3i16(<3 x i16> %v) {
entry:
  %res = call i16 @llvm.vector.reduce.mul.v3i16(<3 x i16> %v)
  ret i16 %res
}

define spir_func i16 @test_vector_reduce_mul_v4i16(<4 x i16> %v) {
entry:
  %res = call i16 @llvm.vector.reduce.mul.v4i16(<4 x i16> %v)
  ret i16 %res
}

define spir_func i16 @test_vector_reduce_mul_v8i16(<8 x i16> %v) {
entry:
  %res = call i16 @llvm.vector.reduce.mul.v8i16(<8 x i16> %v)
  ret i16 %res
}

define spir_func i16 @test_vector_reduce_mul_v16i16(<16 x i16> %v) {
entry:
  %res = call i16 @llvm.vector.reduce.mul.v16i16(<16 x i16> %v)
  ret i16 %res
}

define spir_func i32 @test_vector_reduce_mul_v2i32(<2 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.mul.v2i32(<2 x i32> %v)
  ret i32 %res
}

define spir_func i32 @test_vector_reduce_mul_v3i32(<3 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.mul.v3i32(<3 x i32> %v)
  ret i32 %res
}

define spir_func i32 @test_vector_reduce_mul_v4i32(<4 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> %v)
  ret i32 %res
}

define spir_func i32 @test_vector_reduce_mul_v8i32(<8 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.mul.v8i32(<8 x i32> %v)
  ret i32 %res
}

define spir_func i32 @test_vector_reduce_mul_v16i32(<16 x i32> %v) {
entry:
  %res = call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> %v)
  ret i32 %res
}

define spir_func i64 @test_vector_reduce_mul_v2i64(<2 x i64> %v) {
entry:
  %res = call i64 @llvm.vector.reduce.mul.v2i64(<2 x i64> %v)
  ret i64 %res
}

define spir_func i64 @test_vector_reduce_mul_v3i64(<3 x i64> %v) {
entry:
  %res = call i64 @llvm.vector.reduce.mul.v3i64(<3 x i64> %v)
  ret i64 %res
}

define spir_func i64 @test_vector_reduce_mul_v4i64(<4 x i64> %v) {
entry:
  %res = call i64 @llvm.vector.reduce.mul.v4i64(<4 x i64> %v)
  ret i64 %res
}

define spir_func i64 @test_vector_reduce_mul_v8i64(<8 x i64> %v) {
entry:
  %res = call i64 @llvm.vector.reduce.mul.v8i64(<8 x i64> %v)
  ret i64 %res
}

define spir_func i64 @test_vector_reduce_mul_v16i64(<16 x i64> %v) {
entry:
  %res = call i64 @llvm.vector.reduce.mul.v16i64(<16 x i64> %v)
  ret i64 %res
}

declare i8 @llvm.vector.reduce.mul.v2i8(<2 x i8>)
declare i8 @llvm.vector.reduce.mul.v3i8(<3 x i8>)
declare i8 @llvm.vector.reduce.mul.v4i8(<4 x i8>)
declare i8 @llvm.vector.reduce.mul.v8i8(<8 x i8>)
declare i8 @llvm.vector.reduce.mul.v16i8(<16 x i8>)

declare i16 @llvm.vector.reduce.mul.v2i16(<2 x i16>)
declare i16 @llvm.vector.reduce.mul.v3i16(<3 x i16>)
declare i16 @llvm.vector.reduce.mul.v4i16(<4 x i16>)
declare i16 @llvm.vector.reduce.mul.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.mul.v16i16(<16 x i16>)

declare i32 @llvm.vector.reduce.mul.v2i32(<2 x i32>)
declare i32 @llvm.vector.reduce.mul.v3i32(<3 x i32>)
declare i32 @llvm.vector.reduce.mul.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.mul.v8i32(<8 x i32>)
declare i32 @llvm.vector.reduce.mul.v16i32(<16 x i32>)

declare i64 @llvm.vector.reduce.mul.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.mul.v3i64(<3 x i64>)
declare i64 @llvm.vector.reduce.mul.v4i64(<4 x i64>)
declare i64 @llvm.vector.reduce.mul.v8i64(<8 x i64>)
declare i64 @llvm.vector.reduce.mul.v16i64(<16 x i64>)
