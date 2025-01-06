; RUN: opt -S -passes=dxil-intrinsic-expansion,dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library < %s | FileCheck %s

; Make sure dxil operation function calls for all are generated for float and half.

; CHECK-LABEL: all_bool
; CHECK: icmp ne i1 %{{.*}}, false
define noundef i1 @all_bool(i1 noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.i1(i1 %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_int64_t
; CHECK: icmp ne i64 %{{.*}}, 0
define noundef i1 @all_int64_t(i64 noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.i64(i64 %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_int
; CHECK: icmp ne i32 %{{.*}}, 0
define noundef i1 @all_int(i32 noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.i32(i32 %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_int16_t
; CHECK: icmp ne i16 %{{.*}}, 0
define noundef i1 @all_int16_t(i16 noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.i16(i16 %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_double
; CHECK: fcmp une double %{{.*}}, 0.000000e+00
define noundef i1 @all_double(double noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.f64(double %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_float
; CHECK: fcmp une float %{{.*}}, 0.000000e+00
define noundef i1 @all_float(float noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.f32(float %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_half
; CHECK: fcmp une half %{{.*}}, 0xH0000
define noundef i1 @all_half(half noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.f16(half %p0)
  ret i1 %dx.all
}

; CHECK-LABEL: all_bool4
; CHECK: icmp ne <4 x i1> %{{.*}}, zeroinitialize
; CHECK: extractelement <4 x i1> %{{.*}}, i64 0
; CHECK: extractelement <4 x i1> %{{.*}}, i64 1
; CHECK: and i1  %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i1> %{{.*}}, i64 2
; CHECK: and i1  %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i1> %{{.*}}, i64 3
; CHECK: and i1  %{{.*}}, %{{.*}}
define noundef i1 @all_bool4(<4 x i1> noundef %p0) {
entry:
  %dx.all = call i1 @llvm.dx.all.v4i1(<4 x i1> %p0)
  ret i1 %dx.all
}

declare i1 @llvm.dx.all.v4i1(<4 x i1>)
declare i1 @llvm.dx.all.i1(i1)
declare i1 @llvm.dx.all.i16(i16)
declare i1 @llvm.dx.all.i32(i32)
declare i1 @llvm.dx.all.i64(i64)
declare i1 @llvm.dx.all.f16(half)
declare i1 @llvm.dx.all.f32(float)
declare i1 @llvm.dx.all.f64(double)
