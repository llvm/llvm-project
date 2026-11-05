; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; HLSL lowers ldexp(X, Exp) to exp2(Exp) * X, so an i32 exponent arrives as
; exp2(sitofp(Exp)) * X.

; CHECK-DAG: %[[#ExtInstSetId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#Int4:]] = OpTypeVector %[[#Int32]] 4

; CHECK-LABEL: Begin function test_exp2_mul{{$}}
; CHECK-NOT: Exp2
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Ldexp
define float @test_exp2_mul(float %x, i32 %k) {
  %e = sitofp i32 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  ret float %r
}

; CHECK-LABEL: Begin function test_mul_exp2{{$}}
; CHECK-NOT: Exp2
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Ldexp
define float @test_mul_exp2(float %x, i32 %k) {
  %e = sitofp i32 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %x, %t
  ret float %r
}

; CHECK-LABEL: Begin function test_exp2_mul_v4f32{{$}}
; CHECK-NOT: Exp2
; CHECK: %[[#Splat:]] = OpCompositeConstruct %[[#Int4]]
; CHECK: %[[#]] = OpExtInst %[[#Float4]] %[[#ExtInstSetId]] Ldexp %[[#]] %[[#Splat]]
define <4 x float> @test_exp2_mul_v4f32(<4 x float> %x, i32 %k) {
  %e = sitofp i32 %k to float
  %se = insertelement <4 x float> poison, float %e, i32 0
  %sp = shufflevector <4 x float> %se, <4 x float> poison, <4 x i32> zeroinitializer
  %t = call reassoc <4 x float> @llvm.exp2.v4f32(<4 x float> %sp)
  %r = fmul reassoc <4 x float> %t, %x
  ret <4 x float> %r
}

; CHECK-LABEL: Begin function test_no_reassoc{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2N:]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#]] = OpFMul %[[#Float]] %[[#Exp2N]] %[[#]]
define float @test_no_reassoc(float %x, i32 %k) {
  %e = sitofp i32 %k to float
  %t = call float @llvm.exp2.f32(float %e)
  %r = fmul float %t, %x
  ret float %r
}

; Ldexp truncates its exponent, so a float one must stay an exp2 and a multiply.

; CHECK-LABEL: Begin function test_float_exponent{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2:]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#]] = OpFMul %[[#Float]] %[[#Exp2]] %[[#]]
define float @test_float_exponent(float %x, float %e) {
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  ret float %r
}

; A per-lane exponent does not fit G_FLDEXP, whose exponent is scalar.

; CHECK-LABEL: Begin function test_vector_exponent{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2V:]] = OpExtInst %[[#Float4]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#]] = OpFMul %[[#Float4]] %[[#Exp2V]] %[[#]]
define <4 x float> @test_vector_exponent(<4 x float> %x, <4 x i32> %k) {
  %e = sitofp <4 x i32> %k to <4 x float>
  %t = call reassoc <4 x float> @llvm.exp2.v4f32(<4 x float> %e)
  %r = fmul reassoc <4 x float> %t, %x
  ret <4 x float> %r
}

; Only an i32 exponent folds: the fold is not environment gated, and OpenCL.std
; ldexp takes i32. Both of the next two bail out for that reason.

; CHECK-LABEL: Begin function test_i64_exponent{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2L:]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#]] = OpFMul %[[#Float]] %[[#Exp2L]] %[[#]]
define float @test_i64_exponent(float %x, i64 %k) {
  %e = sitofp i64 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  ret float %r
}

; CHECK-LABEL: Begin function test_i1_exponent{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2B:]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#]] = OpFMul %[[#Float]] %[[#Exp2B]] %[[#]]
define float @test_i1_exponent(float %x, i1 %k) {
  %e = sitofp i1 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  ret float %r
}

; CHECK-LABEL: Begin function test_exp2_multi_use{{$}}
; CHECK-NOT: Ldexp
; CHECK: %[[#Exp2M:]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Exp2
; CHECK: %[[#Mul:]] = OpFMul %[[#Float]] %[[#Exp2M]] %[[#]]
; CHECK: %[[#]] = OpFAdd %[[#Float]] %[[#Mul]] %[[#Exp2M]]
define float @test_exp2_multi_use(float %x, i32 %k) {
  %e = sitofp i32 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  %s = fadd float %r, %t
  ret float %s
}

declare float @llvm.exp2.f32(float)
declare <4 x float> @llvm.exp2.v4f32(<4 x float>)
