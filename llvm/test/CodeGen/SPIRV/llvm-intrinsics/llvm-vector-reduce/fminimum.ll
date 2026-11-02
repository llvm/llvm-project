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
; CHECK: %[[Vec2HalfR1MinMax:.*]] = OpExtInst %[[HalfVec2]] %[[#]] fmin %[[ParamVec2Half]] %[[Vec2HalfShuf]]
; CHECK: %[[Vec2HalfR1XSign:.*]] = OpExtInst %[[HalfVec2]] %[[#]] copysign %[[#]] %[[ParamVec2Half]]
; CHECK: %[[Vec2HalfR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec2HalfR1XSign]] %[[#]]
; CHECK: %[[Vec2HalfR1Sign:.*]] = OpSelect %[[HalfVec2]] %[[Vec2HalfR1XNeg]] %[[ParamVec2Half]] %[[Vec2HalfShuf]]
; CHECK: %[[Vec2HalfR1Signed:.*]] = OpExtInst %[[HalfVec2]] %[[#]] copysign %[[Vec2HalfR1MinMax]] %[[Vec2HalfR1Sign]]
; CHECK: %[[Vec2HalfR1Uno:.*]] = OpUnordered %[[#]] %[[ParamVec2Half]] %[[Vec2HalfShuf]]
; CHECK: %[[Vec2HalfR1:.*]] = OpSelect %[[HalfVec2]] %[[Vec2HalfR1Uno]] %[[#]] %[[Vec2HalfR1Signed]]
; CHECK: %[[Vec2HalfR2:.*]] = OpCompositeExtract %[[Half]] %[[Vec2HalfR1]] 0
; CHECK: OpReturnValue %[[Vec2HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Half:.*]] = OpFunctionParameter %[[HalfVec3]]
; CHECK: %[[Vec3HalfItem0:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 0
; CHECK: %[[Vec3HalfItem1:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 1
; CHECK: %[[Vec3HalfItem2:.*]] = OpCompositeExtract %[[Half]] %[[ParamVec3Half]] 2
; CHECK: %[[Vec3HalfR1MinMax:.*]] = OpExtInst %[[Half]] %[[#]] fmin %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR1XSign:.*]] = OpExtInst %[[Half]] %[[#]] copysign %[[#]] %[[Vec3HalfItem0]]
; CHECK: %[[Vec3HalfR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3HalfR1XSign]] %[[#]]
; CHECK: %[[Vec3HalfR1Sign:.*]] = OpSelect %[[Half]] %[[Vec3HalfR1XNeg]] %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR1Signed:.*]] = OpExtInst %[[Half]] %[[#]] copysign %[[Vec3HalfR1MinMax]] %[[Vec3HalfR1Sign]]
; CHECK: %[[Vec3HalfR1Uno:.*]] = OpUnordered %[[#]] %[[Vec3HalfItem0]] %[[Vec3HalfItem1]]
; CHECK: %[[Vec3HalfR1:.*]] = OpSelect %[[Half]] %[[Vec3HalfR1Uno]] %[[#]] %[[Vec3HalfR1Signed]]
; CHECK: %[[Vec3HalfR2MinMax:.*]] = OpExtInst %[[Half]] %[[#]] fmin %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: %[[Vec3HalfR2XSign:.*]] = OpExtInst %[[Half]] %[[#]] copysign %[[#]] %[[Vec3HalfR1]]
; CHECK: %[[Vec3HalfR2XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3HalfR2XSign]] %[[#]]
; CHECK: %[[Vec3HalfR2Sign:.*]] = OpSelect %[[Half]] %[[Vec3HalfR2XNeg]] %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: %[[Vec3HalfR2Signed:.*]] = OpExtInst %[[Half]] %[[#]] copysign %[[Vec3HalfR2MinMax]] %[[Vec3HalfR2Sign]]
; CHECK: %[[Vec3HalfR2Uno:.*]] = OpUnordered %[[#]] %[[Vec3HalfR1]] %[[Vec3HalfItem2]]
; CHECK: %[[Vec3HalfR2:.*]] = OpSelect %[[Half]] %[[Vec3HalfR2Uno]] %[[#]] %[[Vec3HalfR2Signed]]
; CHECK: OpReturnValue %[[Vec3HalfR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Float:.*]] = OpFunctionParameter %[[FloatVec2]]
; CHECK: %[[Vec2FloatShuf:.*]] = OpVectorShuffle %[[FloatVec2]] %[[ParamVec2Float]] %[[#]] 1 0xFFFFFFFF
; CHECK: %[[Vec2FloatR1MinMax:.*]] = OpExtInst %[[FloatVec2]] %[[#]] fmin %[[ParamVec2Float]] %[[Vec2FloatShuf]]
; CHECK: %[[Vec2FloatR1XSign:.*]] = OpExtInst %[[FloatVec2]] %[[#]] copysign %[[#]] %[[ParamVec2Float]]
; CHECK: %[[Vec2FloatR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec2FloatR1XSign]] %[[#]]
; CHECK: %[[Vec2FloatR1Sign:.*]] = OpSelect %[[FloatVec2]] %[[Vec2FloatR1XNeg]] %[[ParamVec2Float]] %[[Vec2FloatShuf]]
; CHECK: %[[Vec2FloatR1Signed:.*]] = OpExtInst %[[FloatVec2]] %[[#]] copysign %[[Vec2FloatR1MinMax]] %[[Vec2FloatR1Sign]]
; CHECK: %[[Vec2FloatR1Uno:.*]] = OpUnordered %[[#]] %[[ParamVec2Float]] %[[Vec2FloatShuf]]
; CHECK: %[[Vec2FloatR1:.*]] = OpSelect %[[FloatVec2]] %[[Vec2FloatR1Uno]] %[[#]] %[[Vec2FloatR1Signed]]
; CHECK: %[[Vec2FloatR2:.*]] = OpCompositeExtract %[[Float]] %[[Vec2FloatR1]] 0
; CHECK: OpReturnValue %[[Vec2FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Float:.*]] = OpFunctionParameter %[[FloatVec3]]
; CHECK: %[[Vec3FloatItem0:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 0
; CHECK: %[[Vec3FloatItem1:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 1
; CHECK: %[[Vec3FloatItem2:.*]] = OpCompositeExtract %[[Float]] %[[ParamVec3Float]] 2
; CHECK: %[[Vec3FloatR1MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmin %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR1XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec3FloatItem0]]
; CHECK: %[[Vec3FloatR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3FloatR1XSign]] %[[#]]
; CHECK: %[[Vec3FloatR1Sign:.*]] = OpSelect %[[Float]] %[[Vec3FloatR1XNeg]] %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR1Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec3FloatR1MinMax]] %[[Vec3FloatR1Sign]]
; CHECK: %[[Vec3FloatR1Uno:.*]] = OpUnordered %[[#]] %[[Vec3FloatItem0]] %[[Vec3FloatItem1]]
; CHECK: %[[Vec3FloatR1:.*]] = OpSelect %[[Float]] %[[Vec3FloatR1Uno]] %[[#]] %[[Vec3FloatR1Signed]]
; CHECK: %[[Vec3FloatR2MinMax:.*]] = OpExtInst %[[Float]] %[[#]] fmin %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: %[[Vec3FloatR2XSign:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[#]] %[[Vec3FloatR1]]
; CHECK: %[[Vec3FloatR2XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3FloatR2XSign]] %[[#]]
; CHECK: %[[Vec3FloatR2Sign:.*]] = OpSelect %[[Float]] %[[Vec3FloatR2XNeg]] %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: %[[Vec3FloatR2Signed:.*]] = OpExtInst %[[Float]] %[[#]] copysign %[[Vec3FloatR2MinMax]] %[[Vec3FloatR2Sign]]
; CHECK: %[[Vec3FloatR2Uno:.*]] = OpUnordered %[[#]] %[[Vec3FloatR1]] %[[Vec3FloatItem2]]
; CHECK: %[[Vec3FloatR2:.*]] = OpSelect %[[Float]] %[[Vec3FloatR2Uno]] %[[#]] %[[Vec3FloatR2Signed]]
; CHECK: OpReturnValue %[[Vec3FloatR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec2Double:.*]] = OpFunctionParameter %[[DoubleVec2]]
; CHECK: %[[Vec2DoubleShuf:.*]] = OpVectorShuffle %[[DoubleVec2]] %[[ParamVec2Double]] %[[#]] 1 0xFFFFFFFF
; CHECK: %[[Vec2DoubleR1MinMax:.*]] = OpExtInst %[[DoubleVec2]] %[[#]] fmin %[[ParamVec2Double]] %[[Vec2DoubleShuf]]
; CHECK: %[[Vec2DoubleR1XSign:.*]] = OpExtInst %[[DoubleVec2]] %[[#]] copysign %[[#]] %[[ParamVec2Double]]
; CHECK: %[[Vec2DoubleR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec2DoubleR1XSign]] %[[#]]
; CHECK: %[[Vec2DoubleR1Sign:.*]] = OpSelect %[[DoubleVec2]] %[[Vec2DoubleR1XNeg]] %[[ParamVec2Double]] %[[Vec2DoubleShuf]]
; CHECK: %[[Vec2DoubleR1Signed:.*]] = OpExtInst %[[DoubleVec2]] %[[#]] copysign %[[Vec2DoubleR1MinMax]] %[[Vec2DoubleR1Sign]]
; CHECK: %[[Vec2DoubleR1Uno:.*]] = OpUnordered %[[#]] %[[ParamVec2Double]] %[[Vec2DoubleShuf]]
; CHECK: %[[Vec2DoubleR1:.*]] = OpSelect %[[DoubleVec2]] %[[Vec2DoubleR1Uno]] %[[#]] %[[Vec2DoubleR1Signed]]
; CHECK: %[[Vec2DoubleR2:.*]] = OpCompositeExtract %[[Double]] %[[Vec2DoubleR1]] 0
; CHECK: OpReturnValue %[[Vec2DoubleR2]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[ParamVec3Double:.*]] = OpFunctionParameter %[[DoubleVec3]]
; CHECK: %[[Vec3DoubleItem0:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 0
; CHECK: %[[Vec3DoubleItem1:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 1
; CHECK: %[[Vec3DoubleItem2:.*]] = OpCompositeExtract %[[Double]] %[[ParamVec3Double]] 2
; CHECK: %[[Vec3DoubleR1MinMax:.*]] = OpExtInst %[[Double]] %[[#]] fmin %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR1XSign:.*]] = OpExtInst %[[Double]] %[[#]] copysign %[[#]] %[[Vec3DoubleItem0]]
; CHECK: %[[Vec3DoubleR1XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3DoubleR1XSign]] %[[#]]
; CHECK: %[[Vec3DoubleR1Sign:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR1XNeg]] %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR1Signed:.*]] = OpExtInst %[[Double]] %[[#]] copysign %[[Vec3DoubleR1MinMax]] %[[Vec3DoubleR1Sign]]
; CHECK: %[[Vec3DoubleR1Uno:.*]] = OpUnordered %[[#]] %[[Vec3DoubleItem0]] %[[Vec3DoubleItem1]]
; CHECK: %[[Vec3DoubleR1:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR1Uno]] %[[#]] %[[Vec3DoubleR1Signed]]
; CHECK: %[[Vec3DoubleR2MinMax:.*]] = OpExtInst %[[Double]] %[[#]] fmin %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: %[[Vec3DoubleR2XSign:.*]] = OpExtInst %[[Double]] %[[#]] copysign %[[#]] %[[Vec3DoubleR1]]
; CHECK: %[[Vec3DoubleR2XNeg:.*]] = OpFOrdLessThan %[[#]] %[[Vec3DoubleR2XSign]] %[[#]]
; CHECK: %[[Vec3DoubleR2Sign:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR2XNeg]] %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: %[[Vec3DoubleR2Signed:.*]] = OpExtInst %[[Double]] %[[#]] copysign %[[Vec3DoubleR2MinMax]] %[[Vec3DoubleR2Sign]]
; CHECK: %[[Vec3DoubleR2Uno:.*]] = OpUnordered %[[#]] %[[Vec3DoubleR1]] %[[Vec3DoubleItem2]]
; CHECK: %[[Vec3DoubleR2:.*]] = OpSelect %[[Double]] %[[Vec3DoubleR2Uno]] %[[#]] %[[Vec3DoubleR2Signed]]
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
