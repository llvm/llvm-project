; RUN: opt -S -passes=dxil-intrinsic-expansion,dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library < %s | FileCheck %s

; Make sure dxil operation function calls for any are generated for float and half.

; CHECK-LABEL: any_bool
; CHECK: icmp ne i1 %{{.*}}, false
define noundef i1 @any_bool(i1 noundef %p0) {
entry:
  %p0.addr = alloca i8, align 1
  %frombool = zext i1 %p0 to i8
  store i8 %frombool, ptr %p0.addr, align 1
  %0 = load i8, ptr %p0.addr, align 1
  %tobool = trunc i8 %0 to i1
  %dx.any = call i1 @llvm.dx.any.i1(i1 %tobool)
  ret i1 %dx.any
}

; CHECK-LABEL: any_int64_t
; CHECK: icmp ne i64 %{{.*}}, 0
define noundef i1 @any_int64_t(i64 noundef %p0) {
entry:
  %p0.addr = alloca i64, align 8
  store i64 %p0, ptr %p0.addr, align 8
  %0 = load i64, ptr %p0.addr, align 8
  %dx.any = call i1 @llvm.dx.any.i64(i64 %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_int
; CHECK: icmp ne i32 %{{.*}}, 0
define noundef i1 @any_int(i32 noundef %p0) {
entry:
  %p0.addr = alloca i32, align 4
  store i32 %p0, ptr %p0.addr, align 4
  %0 = load i32, ptr %p0.addr, align 4
  %dx.any = call i1 @llvm.dx.any.i32(i32 %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_int16_t
; CHECK: icmp ne i16 %{{.*}}, 0
define noundef i1 @any_int16_t(i16 noundef %p0) {
entry:
  %p0.addr = alloca i16, align 2
  store i16 %p0, ptr %p0.addr, align 2
  %0 = load i16, ptr %p0.addr, align 2
  %dx.any = call i1 @llvm.dx.any.i16(i16 %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_double
; CHECK: fcmp une double %{{.*}}, 0.000000e+00
define noundef i1 @any_double(double noundef %p0) {
entry:
  %p0.addr = alloca double, align 8
  store double %p0, ptr %p0.addr, align 8
  %0 = load double, ptr %p0.addr, align 8
  %dx.any = call i1 @llvm.dx.any.f64(double %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_float
; CHECK: fcmp une float %{{.*}}, 0.000000e+00
define noundef i1 @any_float(float noundef %p0) {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  %dx.any = call i1 @llvm.dx.any.f32(float %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_half
; CHECK: fcmp une half %{{.*}}, 0xH0000
define noundef i1 @any_half(half noundef %p0) {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %dx.any = call i1 @llvm.dx.any.f16(half %0)
  ret i1 %dx.any
}

; CHECK-LABEL: any_bool4
; CHECK: icmp ne <4 x i1> %extractvec, zeroinitialize
; CHECK: extractelement <4 x i1> %{{.*}}, i64 0
; CHECK: extractelement <4 x i1> %{{.*}}, i64 1
; CHECK: or i1  %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i1> %{{.*}}, i64 2
; CHECK: or i1  %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i1> %{{.*}}, i64 3
; CHECK: or i1  %{{.*}}, %{{.*}}
define noundef i1 @any_bool4(<4 x i1> noundef %p0) {
entry:
  %p0.addr = alloca i8, align 1
  %insertvec = shufflevector <4 x i1> %p0, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %0 = bitcast <8 x i1> %insertvec to i8
  store i8 %0, ptr %p0.addr, align 1
  %load_bits = load i8, ptr %p0.addr, align 1
  %1 = bitcast i8 %load_bits to <8 x i1>
  %extractvec = shufflevector <8 x i1> %1, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %dx.any = call i1 @llvm.dx.any.v4i1(<4 x i1> %extractvec)
  ret i1 %dx.any
}

declare i1 @llvm.dx.any.v4i1(<4 x i1>)
declare i1 @llvm.dx.any.i1(i1)
declare i1 @llvm.dx.any.i16(i16)
declare i1 @llvm.dx.any.i32(i32)
declare i1 @llvm.dx.any.i64(i64)
declare i1 @llvm.dx.any.f16(half)
declare i1 @llvm.dx.any.f32(float)
declare i1 @llvm.dx.any.f64(double)
