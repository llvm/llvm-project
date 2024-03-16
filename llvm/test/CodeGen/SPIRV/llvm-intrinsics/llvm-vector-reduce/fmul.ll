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
; CHECK: %[[Param2Half:.*]] = OpFunctionParameter %[[Half]]
; CHECK: %[[ParamVec2Half:.*]] = OpFunctionParameter %[[HalfVec2]]
; CHECK: %[[Vec2HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec2Half]] 0
; CHECK: %[[Vec2HalfR1:.*]] = OpFMul %[[Half]] %[[Param2Half]] %[[Vec2HalfItem0]]
; CHECK: %[[Vec2HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec2Half]] 1
; CHECK: %[[Vec2HalfR2:.*]] = OpFMul %[[Half]] %[[Vec2HalfR1]] %[[Vec2HalfItem1]]
; CHECK: OpReturnValue %[[Vec2HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Param2Half:.*]] = OpFunctionParameter %[[Half]]
; CHECK: %[[ParamVec3Half:.*]] = OpFunctionParameter %[[HalfVec3]]
; CHECK: %[[Vec3HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 0
; CHECK: %[[Vec3HalfR1:.*]] = OpFMul %[[Half]] %[[Param2Half]] %[[Vec3HalfItem0]]
; CHECK: %[[Vec3HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 1
; CHECK: %[[Vec3HalfR2:.*]] = OpFMul %[[Half]] %[[Vec3HalfR1]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfItem2:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 2
; CHECK: %[[Vec3HalfR3:.*]] = OpFMul %[[Half]] %[[Vec3HalfR2]] %[[Vec3HalfItem2]]
; CHECK: OpReturnValue %[[Vec3HalfR3]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Param2Float:.*]] = OpFunctionParameter %[[Float]]
; CHECK: %[[ParamVec2Float:.*]] = OpFunctionParameter %[[FloatVec2]]
; CHECK: %[[Vec2FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec2Float]] 0
; CHECK: %[[Vec2FloatR1:.*]] = OpFMul %[[Float]] %[[Param2Float]] %[[Vec2FloatItem0]]
; CHECK: %[[Vec2FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec2Float]] 1
; CHECK: %[[Vec2FloatR2:.*]] = OpFMul %[[Float]] %[[Vec2FloatR1]] %[[Vec2FloatItem1]]
; CHECK: OpReturnValue %[[Vec2FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Param2Float:.*]] = OpFunctionParameter %[[Float]]
; CHECK: %[[ParamVec3Float:.*]] = OpFunctionParameter %[[FloatVec3]]
; CHECK: %[[Vec3FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 0
; CHECK: %[[Vec3FloatR1:.*]] = OpFMul %[[Float]] %[[Param2Float]] %[[Vec3FloatItem0]]
; CHECK: %[[Vec3FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 1
; CHECK: %[[Vec3FloatR2:.*]] = OpFMul %[[Float]] %[[Vec3FloatR1]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 2
; CHECK: %[[Vec3FloatR3:.*]] = OpFMul %[[Float]] %[[Vec3FloatR2]] %[[Vec3FloatItem2]]
; CHECK: OpReturnValue %[[Vec3FloatR3]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Param2Double:.*]] = OpFunctionParameter %[[Double]]
; CHECK: %[[ParamVec2Double:.*]] = OpFunctionParameter %[[DoubleVec2]]
; CHECK: %[[Vec2DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec2Double]] 0
; CHECK: %[[Vec2DoubleR1:.*]] = OpFMul %[[Double]] %[[Param2Double]] %[[Vec2DoubleItem0]]
; CHECK: %[[Vec2DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec2Double]] 1
; CHECK: %[[Vec2DoubleR2:.*]] = OpFMul %[[Double]] %[[Vec2DoubleR1]] %[[Vec2DoubleItem1]]
; CHECK: OpReturnValue %[[Vec2DoubleR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[Param2Double:.*]] = OpFunctionParameter %[[Double]]
; CHECK: %[[ParamVec3Double:.*]] = OpFunctionParameter %[[DoubleVec3]]
; CHECK: %[[Vec3DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 0
; CHECK: %[[Vec3DoubleR1:.*]] = OpFMul %[[Double]] %[[Param2Double]] %[[Vec3DoubleItem0]]
; CHECK: %[[Vec3DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 1
; CHECK: %[[Vec3DoubleR2:.*]] = OpFMul %[[Double]] %[[Vec3DoubleR1]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleItem2:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 2
; CHECK: %[[Vec3DoubleR3:.*]] = OpFMul %[[Double]] %[[Vec3DoubleR2]] %[[Vec3DoubleItem2]]
; CHECK: OpReturnValue %[[Vec3DoubleR3]]
; CHECK: OpFunctionEnd

define spir_func half @test_vector_reduce_fmul_v2half(half %sp, <2 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmul.v2half(half %sp, <2 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmul_v3half(half %sp, <3 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmul.v3half(half %sp, <3 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmul_v4half(half %sp, <4 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmul.v4half(half %sp, <4 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmul_v8half(half %sp, <8 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmul.v8half(half %sp, <8 x half> %v)
  ret half %res
}

define spir_func half @test_vector_reduce_fmul_v16half(half %sp, <16 x half> %v) {
entry:
  %res = call half @llvm.vector.reduce.fmul.v16half(half %sp, <16 x half> %v)
  ret half %res
}

define spir_func float @test_vector_reduce_fmul_v2float(float %sp, <2 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v2float(float %sp, <2 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmul_v3float(float %sp, <3 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v3float(float %sp, <3 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmul_v4float(float %sp, <4 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v4float(float %sp, <4 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmul_v8float(float %sp, <8 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v8float(float %sp, <8 x float> %v)
  ret float %res
}

define spir_func float @test_vector_reduce_fmul_v16float(float %sp, <16 x float> %v) {
entry:
  %res = call float @llvm.vector.reduce.fmul.v16float(float %sp, <16 x float> %v)
  ret float %res
}


define spir_func double @test_vector_reduce_fmul_v2double(double %sp, <2 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmul.v2double(double %sp, <2 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmul_v3double(double %sp, <3 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmul.v3double(double %sp, <3 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmul_v4double(double %sp, <4 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmul.v4double(double %sp, <4 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmul_v8double(double %sp, <8 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmul.v8double(double %sp, <8 x double> %v)
  ret double %res
}

define spir_func double @test_vector_reduce_fmul_v16double(double %sp, <16 x double> %v) {
entry:
  %res = call double @llvm.vector.reduce.fmul.v16double(double %sp, <16 x double> %v)
  ret double %res
}

declare half @llvm.vector.reduce.fmul.v2half(half, <2 x half>)
declare half @llvm.vector.reduce.fmul.v3half(half, <3 x half>)
declare half @llvm.vector.reduce.fmul.v4half(half, <4 x half>)
declare half @llvm.vector.reduce.fmul.v8half(half, <8 x half>)
declare half @llvm.vector.reduce.fmul.v16half(half, <16 x half>)
declare float @llvm.vector.reduce.fmul.v2float(float, <2 x float>)
declare float @llvm.vector.reduce.fmul.v3float(float, <3 x float>)
declare float @llvm.vector.reduce.fmul.v4float(float, <4 x float>)
declare float @llvm.vector.reduce.fmul.v8float(float, <8 x float>)
declare float @llvm.vector.reduce.fmul.v16float(float, <16 x float>)
declare double @llvm.vector.reduce.fmul.v2double(double, <2 x double>)
declare double @llvm.vector.reduce.fmul.v3double(double, <3 x double>)
declare double @llvm.vector.reduce.fmul.v4double(double, <4 x double>)
declare double @llvm.vector.reduce.fmul.v8double(double, <8 x double>)
declare double @llvm.vector.reduce.fmul.v16double(double, <16 x double>)
