; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

target triple = "spir64-unknown-unknown"

; CHECK-DAG: %[[Half:.*]] = OpTypeFloat 16
; CHECK-DAG: %[[HalfVec2:.*]] = OpTypeVector %[[Half]] 2
; CHECK-DAG: %[[HalfVec3:.*]] = OpTypeVector %[[Half]] 3

; CHECK-DAG: %[[Float:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[FloatVec2:.*]] = OpTypeVector %[[Float]] 2
; CHECK-DAG: %[[FloatVec3:.*]] = OpTypeVector %[[Float]] 3

; CHECK-DAG: %[[Double:.*]] = OpTypeFloat 64
; CHECK-DAG: %[[DoubleVec2:.*]] = OpTypeVector %[[Double]] 2
; CHECK-DAG: %[[DoubleVec3:.*]] = OpTypeVector %[[Double]] 3

; CHECK: OpFunction
; CHECK: %[[ParamVec2Half:.*]] = OpFunctionParameter %[[HalfVec2]]
; CHECK: %[[Vec2HalfShuf:.*]] = OpVectorShuffle %[[HalfVec2]] %[[ParamVec2Half]] %[[#]] 1 0xFFFFFFFF
; CHECK: %[[Vec2HalfR1MinMax:.*]] = OpExtInst %[[HalfVec2]] %[[#]] fmax %[[ParamVec2Half]] %[[Vec2HalfShuf]]
; CHECK: %[[Vec2HalfR1Ord:.*]] = OpOrdered %[[#]] %[[ParamVec2Half]] %[[Vec2HalfShuf]]
; CHECK: %[[Vec2HalfR1NaN:.*]] = OpSelect %[[HalfVec2]] %[[Vec2HalfR1Ord]] %[[Vec2HalfR1MinMax]] %[[#]]
; CHECK: %[[Vec2HalfR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec2HalfR1NaN]] %[[#]]
; CHECK: %[[Vec2HalfR1:.*]] = OpSelect %[[HalfVec2]] %[[Vec2HalfR1IsZero]] %[[#]] %[[Vec2HalfR1NaN]]
; CHECK: %[[Vec2HalfR2:.*]] = OpCompositeExtract %[[Half]] %[[Vec2HalfR1]] 0
; CHECK: OpReturnValue %[[Vec2HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Half:.*]] = OpFunctionParameter %[[HalfVec3]]
; CHECK: %[[Vec3HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 0
; CHECK: %[[Vec3HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 1
; CHECK: %[[Vec3HalfItem2:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 2
; CHECK: %[[Vec3HalfR1MinMax:.*]] = OpExtInst %[[Half]] %[[#]] fmax %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR1Ord:.*]] = OpOrdered %[[#]] %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR1NaN:.*]] = OpSelect %[[Half]] %[[Vec3HalfR1Ord]] %[[Vec3HalfR1MinMax]] %[[#]]
; CHECK: %[[Vec3HalfR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3HalfR1NaN]] %[[#]]
; CHECK: %[[Vec3HalfR1:.*]] = OpSelect %[[Half]] %[[Vec3HalfR1IsZero]] %[[#]] %[[Vec3HalfR1NaN]]
; CHECK: %[[Vec3HalfR2MinMax:.*]] = OpExtInst %[[Half]] %[[#]] fmax %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: %[[Vec3HalfR2Ord:.*]] = OpOrdered %[[#]] %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: %[[Vec3HalfR2NaN:.*]] = OpSelect %[[Half]] %[[Vec3HalfR2Ord]] %[[Vec3HalfR2MinMax]] %[[#]]
; CHECK: %[[Vec3HalfR2IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3HalfR2NaN]] %[[#]]
; CHECK: %[[Vec3HalfR2:.*]] = OpSelect %[[Half]] %[[Vec3HalfR2IsZero]] %[[#]] %[[Vec3HalfR2NaN]]
; CHECK: OpReturnValue %[[Vec3HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Float:.*]] = OpFunctionParameter %[[FloatVec2]]
; CHECK: %[[Vec2FloatShuf:.*]] = OpVectorShuffle %[[FloatVec2]] %[[ParamVec2Float]] %[[#]] 1 0xFFFFFFFF
; CHECK: %[[Vec2FloatR1MinMax:.*]] = OpExtInst %[[FloatVec2]] %[[#]] fmax %[[ParamVec2Float]] %[[Vec2FloatShuf]]
; CHECK: %[[Vec2FloatR1Ord:.*]] = OpOrdered %[[#]] %[[ParamVec2Float]] %[[Vec2FloatShuf]]
; CHECK: %[[Vec2FloatR1NaN:.*]] = OpSelect %[[FloatVec2]] %[[Vec2FloatR1Ord]] %[[Vec2FloatR1MinMax]] %[[#]]
; CHECK: %[[Vec2FloatR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec2FloatR1NaN]] %[[#]]
; CHECK: %[[Vec2FloatR1:.*]] = OpSelect %[[FloatVec2]] %[[Vec2FloatR1IsZero]] %[[#]] %[[Vec2FloatR1NaN]]
; CHECK: %[[Vec2FloatR2:.*]] = OpCompositeExtract %[[Float]] %[[Vec2FloatR1]] 0
; CHECK: OpReturnValue %[[Vec2FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Float:.*]] = OpFunctionParameter %[[FloatVec3]]
; CHECK: %[[Vec3FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 0
; CHECK: %[[Vec3FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 1
; CHECK: %[[Vec3FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 2
; CHECK: %[[Vec3FloatR1MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR1Ord:.*]] = OpOrdered %[[#]] %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR1NaN:.*]] = OpSelect %[[Float]] %[[Vec3FloatR1Ord]] %[[Vec3FloatR1MinMax]] %[[#]]
; CHECK: %[[Vec3FloatR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3FloatR1NaN]] %[[#]]
; CHECK: %[[Vec3FloatR1:.*]] = OpSelect %[[Float]] %[[Vec3FloatR1IsZero]] %[[#]] %[[Vec3FloatR1NaN]]
; CHECK: %[[Vec3FloatR2MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmax %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: %[[Vec3FloatR2Ord:.*]] = OpOrdered %[[#]] %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: %[[Vec3FloatR2NaN:.*]] = OpSelect %[[Float]] %[[Vec3FloatR2Ord]] %[[Vec3FloatR2MinMax]] %[[#]]
; CHECK: %[[Vec3FloatR2IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3FloatR2NaN]] %[[#]]
; CHECK: %[[Vec3FloatR2:.*]] = OpSelect %[[Float]] %[[Vec3FloatR2IsZero]] %[[#]] %[[Vec3FloatR2NaN]]
; CHECK: OpReturnValue %[[Vec3FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Double:.*]] = OpFunctionParameter %[[DoubleVec2]]
; CHECK: %[[Vec2DoubleShuf:.*]] = OpVectorShuffle %[[DoubleVec2]] %[[ParamVec2Double]] %[[#]] 1 0xFFFFFFFF
; CHECK: %[[Vec2DoubleR1MinMax:.*]] = OpExtInst %[[DoubleVec2]] %[[#]] fmax %[[ParamVec2Double]] %[[Vec2DoubleShuf]]
; CHECK: %[[Vec2DoubleR1Ord:.*]] = OpOrdered %[[#]] %[[ParamVec2Double]] %[[Vec2DoubleShuf]]
; CHECK: %[[Vec2DoubleR1NaN:.*]] = OpSelect %[[DoubleVec2]] %[[Vec2DoubleR1Ord]] %[[Vec2DoubleR1MinMax]] %[[#]]
; CHECK: %[[Vec2DoubleR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec2DoubleR1NaN]] %[[#]]
; CHECK: %[[Vec2DoubleR1:.*]] = OpSelect %[[DoubleVec2]] %[[Vec2DoubleR1IsZero]] %[[#]] %[[Vec2DoubleR1NaN]]
; CHECK: %[[Vec2DoubleR2:.*]] = OpCompositeExtract %[[Double]] %[[Vec2DoubleR1]] 0
; CHECK: OpReturnValue %[[Vec2DoubleR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Double:.*]] = OpFunctionParameter %[[DoubleVec3]]
; CHECK: %[[Vec3DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 0
; CHECK: %[[Vec3DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 1
; CHECK: %[[Vec3DoubleItem2:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 2
; CHECK: %[[Vec3DoubleR1MinMax:.*]] = OpExtInst %[[Double]] %[[#]] fmax %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR1Ord:.*]] = OpOrdered %[[#]] %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR1NaN:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR1Ord]] %[[Vec3DoubleR1MinMax]] %[[#]]
; CHECK: %[[Vec3DoubleR1IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3DoubleR1NaN]] %[[#]]
; CHECK: %[[Vec3DoubleR1:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR1IsZero]] %[[#]] %[[Vec3DoubleR1NaN]]
; CHECK: %[[Vec3DoubleR2MinMax:.*]] = OpExtInst %[[Double]] %[[#]] fmax %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: %[[Vec3DoubleR2Ord:.*]] = OpOrdered %[[#]] %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: %[[Vec3DoubleR2NaN:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR2Ord]] %[[Vec3DoubleR2MinMax]] %[[#]]
; CHECK: %[[Vec3DoubleR2IsZero:.*]] = OpFOrdEqual %[[#]] %[[Vec3DoubleR2NaN]] %[[#]]
; CHECK: %[[Vec3DoubleR2:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR2IsZero]] %[[#]] %[[Vec3DoubleR2NaN]]
; CHECK: OpReturnValue %[[Vec3DoubleR2]]
; CHECK: OpFunctionEnd

define spir_func half @test_vector_reduce_fmaximum_v2half(<2 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmaximum.v2half(<2 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmaximum_v3half(<3 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmaximum.v3half(<3 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmaximum_v4half(<4 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmaximum.v4half(<4 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmaximum_v8half(<8 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmaximum.v8half(<8 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmaximum_v16half(<16 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmaximum.v16half(<16 x half> %v)
  ret half %res
}

define spir_func float @test_vector_reduce_fmaximum_v2float(<2 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v2float(<2 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmaximum_v3float(<3 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v3float(<3 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmaximum_v4float(<4 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v4float(<4 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmaximum_v8float(<8 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v8float(<8 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmaximum_v16float(<16 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmaximum.v16float(<16 x float> %v)
  ret float %res
}


define spir_func double @test_vector_reduce_fmaximum_v2double(<2 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmaximum.v2double(<2 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmaximum_v3double(<3 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmaximum.v3double(<3 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmaximum_v4double(<4 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmaximum.v4double(<4 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmaximum_v8double(<8 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmaximum.v8double(<8 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmaximum_v16double(<16 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmaximum.v16double(<16 x double> %v)
  ret double %res
}

declare half @llvm.vector.reduce.fmaximum.v2half(<2 x half>)
declare half @llvm.vector.reduce.fmaximum.v3half(<3 x half>)
declare half @llvm.vector.reduce.fmaximum.v4half(<4 x half>)
declare half @llvm.vector.reduce.fmaximum.v8half(<8 x half>)
declare half @llvm.vector.reduce.fmaximum.v16half(<16 x half>)
declare float @llvm.vector.reduce.fmaximum.v2float(<2 x float>)
declare float @llvm.vector.reduce.fmaximum.v3float(<3 x float>)
declare float @llvm.vector.reduce.fmaximum.v4float(<4 x float>)
declare float @llvm.vector.reduce.fmaximum.v8float(<8 x float>)
declare float @llvm.vector.reduce.fmaximum.v16float(<16 x float>)
declare double @llvm.vector.reduce.fmaximum.v2double(<2 x double>)
declare double @llvm.vector.reduce.fmaximum.v3double(<3 x double>)
declare double @llvm.vector.reduce.fmaximum.v4double(<4 x double>)
declare double @llvm.vector.reduce.fmaximum.v8double(<8 x double>)
declare double @llvm.vector.reduce.fmaximum.v16double(<16 x double>)
