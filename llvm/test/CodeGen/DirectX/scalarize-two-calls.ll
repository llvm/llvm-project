; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

; CHECK: target triple = "dxilv1.3-pc-shadermodel6.3-library"
; CHECK-LABEL: cos_sin_float_test
define noundef <4 x float> @cos_sin_float_test(<4 x float> noundef %a) {
    ; CHECK: [[ee0:%.*]] = extractelement <4 x float> %a, i64 0
    ; CHECK: [[ie0:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee0]])
    ; CHECK: [[ee1:%.*]] = extractelement <4 x float> %a, i64 1
    ; CHECK: [[ie1:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee1]])
    ; CHECK: [[ee2:%.*]] = extractelement <4 x float> %a, i64 2
    ; CHECK: [[ie2:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee2]])
    ; CHECK: [[ee3:%.*]] = extractelement <4 x float> %a, i64 3
    ; CHECK: [[ie3:%.*]] = call float @dx.op.unary.f32(i32 13, float [[ee3]])
    ; CHECK: [[ie4:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie0]])
    ; CHECK: [[ie5:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie1]])
    ; CHECK: [[ie6:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie2]])
    ; CHECK: [[ie7:%.*]] = call float @dx.op.unary.f32(i32 12, float [[ie3]])
    ; CHECK: insertelement <4 x float> poison, float [[ie4]], i64 0
    ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie5]], i64 1
    ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie6]], i64 2
    ; CHECK: insertelement <4 x float> %{{.*}}, float [[ie7]], i64 3
    %2 = tail call <4 x float> @llvm.sin.v4f32(<4 x float> %a) 
    %3 = tail call <4 x float> @llvm.cos.v4f32(<4 x float> %2) 
    ret <4 x float> %3 
} 
