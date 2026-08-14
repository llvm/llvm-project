; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK

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

declare float @llvm.dx.fdot.v5f32(<5 x float>, <5 x float>)
declare float @llvm.dx.fdot.v9f32(<9 x float>, <9 x float>)
