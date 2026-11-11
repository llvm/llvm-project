; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK

; CHECK-LABEL: dot_half3
define noundef half @dot_half3(<3 x half> noundef %a, <3 x half> noundef %b) {
entry:
; CHECK: extractelement <3 x half> %a, i32 0
; CHECK: extractelement <3 x half> %a, i32 1
; CHECK: extractelement <3 x half> %a, i32 2
; CHECK: extractelement <3 x half> %b, i32 0
; CHECK: extractelement <3 x half> %b, i32 1
; CHECK: extractelement <3 x half> %b, i32 2
; CHECK: call half @llvm.dx.dot3.f16(half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}})
  %dx.dot = call half @llvm.dx.fdot.v3f16(<3 x half> %a, <3 x half> %b)
  ret half %dx.dot
}

; CHECK-LABEL: dot_float4
define noundef float @dot_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
; CHECK: extractelement <4 x float> %a, i32 0
; CHECK: extractelement <4 x float> %a, i32 1
; CHECK: extractelement <4 x float> %a, i32 2
; CHECK: extractelement <4 x float> %a, i32 3
; CHECK: extractelement <4 x float> %b, i32 0
; CHECK: extractelement <4 x float> %b, i32 1
; CHECK: extractelement <4 x float> %b, i32 2
; CHECK: extractelement <4 x float> %b, i32 3
; CHECK: call float @llvm.dx.dot4.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  %dx.dot = call float @llvm.dx.fdot.v4f32(<4 x float> %a, <4 x float> %b)
  ret float %dx.dot
}

; CHECK-LABEL: define noundef float @dot_float5(
; EXPCHECK: [[A0:%.*]] = shufflevector <5 x float> %a, <5 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
; EXPCHECK: [[B0:%.*]] = shufflevector <5 x float> %b, <5 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
; EXPCHECK: [[DOT0:%.*]] = call float @llvm.dx.dot3.f32(
; EXPCHECK: [[A1:%.*]] = shufflevector <5 x float> %a, <5 x float> poison, <2 x i32> <i32 3, i32 4>
; EXPCHECK: [[B1:%.*]] = shufflevector <5 x float> %b, <5 x float> poison, <2 x i32> <i32 3, i32 4>
; EXPCHECK: [[DOT1:%.*]] = call float @llvm.dx.dot2.f32(
; EXPCHECK: [[RESULT:%.*]] = fadd float [[DOT0]], [[DOT1]]
; EXPCHECK: ret float [[RESULT]]
; SM69CHECK: [[DOT:%.*]] = call float @llvm.dx.fdot.v5f32(<5 x float> %a, <5 x float> %b)
; SM69CHECK-NEXT: ret float [[DOT]]
define noundef float @dot_float5(<5 x float> noundef %a, <5 x float> noundef %b) {
entry:
  %dx.dot = call float @llvm.dx.fdot.v5f32(<5 x float> %a, <5 x float> %b)
  ret float %dx.dot
}

; CHECK-LABEL: define noundef float @dot_float8(
; EXPCHECK: [[DOT0:%.*]] = call float @llvm.dx.dot4.f32(
; EXPCHECK: [[DOT1:%.*]] = call float @llvm.dx.dot4.f32(
; EXPCHECK: [[RESULT:%.*]] = fadd float [[DOT0]], [[DOT1]]
; EXPCHECK: ret float [[RESULT]]
; SM69CHECK: [[DOT:%.*]] = call float @llvm.dx.fdot.v8f32(<8 x float> %a, <8 x float> %b)
; SM69CHECK-NEXT: ret float [[DOT]]
define noundef float @dot_float8(<8 x float> noundef %a, <8 x float> noundef %b) {
entry:
  %dx.dot = call float @llvm.dx.fdot.v8f32(<8 x float> %a, <8 x float> %b)
  ret float %dx.dot
}

; CHECK-LABEL: define noundef float @dot_float9(
; EXPCHECK: [[DOT0:%.*]] = call float @llvm.dx.dot4.f32(
; EXPCHECK: [[DOT1:%.*]] = call float @llvm.dx.dot3.f32(
; EXPCHECK: [[SUM:%.*]] = fadd float [[DOT0]], [[DOT1]]
; EXPCHECK: [[DOT2:%.*]] = call float @llvm.dx.dot2.f32(
; EXPCHECK: [[RESULT:%.*]] = fadd float [[SUM]], [[DOT2]]
; EXPCHECK: ret float [[RESULT]]
; SM69CHECK: [[DOT:%.*]] = call float @llvm.dx.fdot.v9f32(<9 x float> %a, <9 x float> %b)
; SM69CHECK-NEXT: ret float [[DOT]]
define noundef float @dot_float9(<9 x float> noundef %a, <9 x float> noundef %b) {
entry:
  %dx.dot = call float @llvm.dx.fdot.v9f32(<9 x float> %a, <9 x float> %b)
  ret float %dx.dot
}

; CHECK-LABEL: define noundef float @dot_float10(
; EXPCHECK: [[DOT0:%.*]] = call float @llvm.dx.dot4.f32(
; EXPCHECK: [[DOT1:%.*]] = call float @llvm.dx.dot4.f32(
; EXPCHECK: [[SUM:%.*]] = fadd float [[DOT0]], [[DOT1]]
; EXPCHECK: [[DOT2:%.*]] = call float @llvm.dx.dot2.f32(
; EXPCHECK: [[RESULT:%.*]] = fadd float [[SUM]], [[DOT2]]
; EXPCHECK: ret float [[RESULT]]
; SM69CHECK: [[DOT:%.*]] = call float @llvm.dx.fdot.v10f32(<10 x float> %a, <10 x float> %b)
; SM69CHECK-NEXT: ret float [[DOT]]
define noundef float @dot_float10(<10 x float> noundef %a, <10 x float> noundef %b) {
entry:
  %dx.dot = call float @llvm.dx.fdot.v10f32(<10 x float> %a, <10 x float> %b)
  ret float %dx.dot
}

declare float @llvm.dx.fdot.v5f32(<5 x float>, <5 x float>)
declare float @llvm.dx.fdot.v8f32(<8 x float>, <8 x float>)
declare float @llvm.dx.fdot.v9f32(<9 x float>, <9 x float>)
declare float @llvm.dx.fdot.v10f32(<10 x float>, <10 x float>)
