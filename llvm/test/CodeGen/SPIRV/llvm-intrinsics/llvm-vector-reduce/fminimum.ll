; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
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
; CHECK: %[[Vec2HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec2Half]] 0
; CHECK: %[[Vec2HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec2Half]] 1
; CHECK: %[[Vec2HalfR1:.*]] = OpExtInst %[[Half]] %[[#]] fmin %[[Vec2HalfItem0]] %[[Vec2HalfItem1]]
; CHECK: OpReturnValue %[[Vec2HalfR1]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Half:.*]] = OpFunctionParameter %[[HalfVec3]]
; CHECK: %[[Vec3HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 0
; CHECK: %[[Vec3HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 1
; CHECK: %[[Vec3HalfItem2:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 2
; CHECK: %[[Vec3HalfR1:.*]] = OpExtInst %[[Half]] %[[#]] fmin %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR2:.*]] = OpExtInst %[[Half]] %[[#]] fmin %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: OpReturnValue %[[Vec3HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Float:.*]] = OpFunctionParameter %[[FloatVec2]]
; CHECK: %[[Vec2FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec2Float]] 0
; CHECK: %[[Vec2FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec2Float]] 1
; CHECK: %[[Vec2FloatR1:.*]] = OpExtInst %[[Float]] %[[#]] fmin %[[Vec2FloatItem0]] %[[Vec2FloatItem1]]
; CHECK: OpReturnValue %[[Vec2FloatR1]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Float:.*]] = OpFunctionParameter %[[FloatVec3]]
; CHECK: %[[Vec3FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 0
; CHECK: %[[Vec3FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 1
; CHECK: %[[Vec3FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 2
; CHECK: %[[Vec3FloatR1:.*]] = OpExtInst %[[Float]] %[[#]] fmin %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR2:.*]] = OpExtInst %[[Float]] %[[#]] fmin %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: OpReturnValue %[[Vec3FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Double:.*]] = OpFunctionParameter %[[DoubleVec2]]
; CHECK: %[[Vec2DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec2Double]] 0
; CHECK: %[[Vec2DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec2Double]] 1
; CHECK: %[[Vec2DoubleR1:.*]] = OpExtInst %[[Double]] %[[#]] fmin %[[Vec2DoubleItem0]] %[[Vec2DoubleItem1]]
; CHECK: OpReturnValue %[[Vec2DoubleR1]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Double:.*]] = OpFunctionParameter %[[DoubleVec3]]
; CHECK: %[[Vec3DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 0
; CHECK: %[[Vec3DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 1
; CHECK: %[[Vec3DoubleItem2:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 2
; CHECK: %[[Vec3DoubleR1:.*]] = OpExtInst %[[Double]] %[[#]] fmin %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR2:.*]] = OpExtInst %[[Double]] %[[#]] fmin %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: OpReturnValue %[[Vec3DoubleR2]]
; CHECK: OpFunctionEnd

define spir_func half @test_vector_reduce_fminimum_v2half(<2 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fminimum.v2half(<2 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fminimum_v3half(<3 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fminimum.v3half(<3 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fminimum_v4half(<4 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fminimum.v4half(<4 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fminimum_v8half(<8 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fminimum.v8half(<8 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fminimum_v16half(<16 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fminimum.v16half(<16 x half> %v)
  ret half %res
}

define spir_func float @test_vector_reduce_fminimum_v2float(<2 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fminimum.v2float(<2 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fminimum_v3float(<3 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fminimum.v3float(<3 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fminimum_v4float(<4 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fminimum.v4float(<4 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fminimum_v8float(<8 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fminimum.v8float(<8 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fminimum_v16float(<16 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fminimum.v16float(<16 x float> %v)
  ret float %res
}


define spir_func double @test_vector_reduce_fminimum_v2double(<2 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fminimum.v2double(<2 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fminimum_v3double(<3 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fminimum.v3double(<3 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fminimum_v4double(<4 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fminimum.v4double(<4 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fminimum_v8double(<8 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fminimum.v8double(<8 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fminimum_v16double(<16 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fminimum.v16double(<16 x double> %v)
  ret double %res
}

declare half @llvm.vector.reduce.fminimum.v2half(<2 x half>)
declare half @llvm.vector.reduce.fminimum.v3half(<3 x half>)
declare half @llvm.vector.reduce.fminimum.v4half(<4 x half>)
declare half @llvm.vector.reduce.fminimum.v8half(<8 x half>)
declare half @llvm.vector.reduce.fminimum.v16half(<16 x half>)
declare float @llvm.vector.reduce.fminimum.v2float(<2 x float>)
declare float @llvm.vector.reduce.fminimum.v3float(<3 x float>)
declare float @llvm.vector.reduce.fminimum.v4float(<4 x float>)
declare float @llvm.vector.reduce.fminimum.v8float(<8 x float>)
declare float @llvm.vector.reduce.fminimum.v16float(<16 x float>)
declare double @llvm.vector.reduce.fminimum.v2double(<2 x double>)
declare double @llvm.vector.reduce.fminimum.v3double(<3 x double>)
declare double @llvm.vector.reduce.fminimum.v4double(<4 x double>)
declare double @llvm.vector.reduce.fminimum.v8double(<8 x double>)
declare double @llvm.vector.reduce.fminimum.v16double(<16 x double>)
