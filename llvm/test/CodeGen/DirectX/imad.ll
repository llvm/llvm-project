; RUN: opt -S -scalarizer -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for round are generated for float and half.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"
; Function Attrs: noinline nounwind optnone
define noundef i16 @imad_short(i16 noundef %p0, i16 noundef %p1, i16 noundef %p2) #0 {
entry:
  ; CHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR:]]
  %p2.addr = alloca i16, align 2
  %p1.addr = alloca i16, align 2
  %p0.addr = alloca i16, align 2
  store i16 %p2, ptr %p2.addr, align 2
  store i16 %p1, ptr %p1.addr, align 2
  store i16 %p0, ptr %p0.addr, align 2
  %0 = load i16, ptr %p0.addr, align 2
  %1 = load i16, ptr %p1.addr, align 2
  %2 = load i16, ptr %p2.addr, align 2
  %dx.imad = call i16 @llvm.dx.imad.i16(i16 %0, i16 %1, i16 %2)
  ret i16 %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i16 @llvm.dx.imad.i16(i16, i16, i16) #1

; Function Attrs: noinline nounwind optnone
define noundef i32 @imad_int(i32 noundef %p0, i32 noundef %p1, i32 noundef %p2) #0 {
entry:
  ; CHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  %p2.addr = alloca i32, align 4
  %p1.addr = alloca i32, align 4
  %p0.addr = alloca i32, align 4
  store i32 %p2, ptr %p2.addr, align 4
  store i32 %p1, ptr %p1.addr, align 4
  store i32 %p0, ptr %p0.addr, align 4
  %0 = load i32, ptr %p0.addr, align 4
  %1 = load i32, ptr %p1.addr, align 4
  %2 = load i32, ptr %p2.addr, align 4
  %dx.imad = call i32 @llvm.dx.imad.i32(i32 %0, i32 %1, i32 %2)
  ret i32 %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.dx.imad.i32(i32, i32, i32) #1

; Function Attrs: noinline nounwind optnone
define noundef i64 @imad_int64(i64 noundef %p0, i64 noundef %p1, i64 noundef %p2) #0 {
entry:
  ; CHECK: call i64 @dx.op.tertiary.i64(i32 48, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  %p2.addr = alloca i64, align 8
  %p1.addr = alloca i64, align 8
  %p0.addr = alloca i64, align 8
  store i64 %p2, ptr %p2.addr, align 8
  store i64 %p1, ptr %p1.addr, align 8
  store i64 %p0, ptr %p0.addr, align 8
  %0 = load i64, ptr %p0.addr, align 8
  %1 = load i64, ptr %p1.addr, align 8
  %2 = load i64, ptr %p2.addr, align 8
  %dx.imad = call i64 @llvm.dx.imad.i64(i64 %0, i64 %1, i64 %2)
  ret i64 %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i64 @llvm.dx.imad.i64(i64, i64, i64) #1

; Function Attrs: noinline nounwind optnone
define noundef <4 x i16> @imad_int16_t4(<4 x i16> noundef %p0, <4 x i16> noundef %p1, <4 x i16> noundef %p2) #0 {
entry:
  ; CHECK: extractelement <4 x i16> %p0, i64 0
  ; CHECK: extractelement <4 x i16> %p1, i64 0
  ; CHECK: extractelement <4 x i16> %p2, i64 0
  ; CHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i16> %p0, i64 1
  ; CHECK: extractelement <4 x i16> %p1, i64 1
  ; CHECK: extractelement <4 x i16> %p2, i64 1
  ; CHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i16> %p0, i64 2
  ; CHECK: extractelement <4 x i16> %p1, i64 2
  ; CHECK: extractelement <4 x i16> %p2, i64 2
  ; CHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i16> %p0, i64 3
  ; CHECK: extractelement <4 x i16> %p1, i64 3
  ; CHECK: extractelement <4 x i16> %p2, i64 3
  ; CHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}) #[[#ATTR]]
  ; CHECK: insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
  ; CHECK: insertelement <4 x i16> %{{.*}}, i16 %{{.*}}, i64 1
  ; CHECK: insertelement <4 x i16> %{{.*}}, i16 %{{.*}}, i64 2
  ; CHECK: insertelement <4 x i16> %{{.*}}, i16 %{{.*}}, i64 3
  %dx.imad = call <4 x i16> @llvm.dx.imad.v4i16(<4 x i16> %p0, <4 x i16> %p1, <4 x i16> %p2)
  ret <4 x i16> %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x i16> @llvm.dx.imad.v4i16(<4 x i16>, <4 x i16>, <4 x i16>) #1

; Function Attrs: noinline nounwind optnone
define noundef <4 x i32> @imad_int4(<4 x i32> noundef %p0, <4 x i32> noundef %p1, <4 x i32> noundef %p2) #0 {
entry:
  ; CHECK: extractelement <4 x i32> %p0, i64 0
  ; CHECK: extractelement <4 x i32> %p1, i64 0
  ; CHECK: extractelement <4 x i32> %p2, i64 0
  ; CHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i32> %p0, i64 1
  ; CHECK: extractelement <4 x i32> %p1, i64 1
  ; CHECK: extractelement <4 x i32> %p2, i64 1
  ; CHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i32> %p0, i64 2
  ; CHECK: extractelement <4 x i32> %p1, i64 2
  ; CHECK: extractelement <4 x i32> %p2, i64 2
  ; CHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i32> %p0, i64 3
  ; CHECK: extractelement <4 x i32> %p1, i64 3
  ; CHECK: extractelement <4 x i32> %p2, i64 3
  ; CHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}) #[[#ATTR]]
  ; CHECK: insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i64 1
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i64 2
  ; CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i64 3
  %dx.imad = call <4 x i32> @llvm.dx.imad.v4i32(<4 x i32> %p0, <4 x i32> %p1, <4 x i32> %p2)
  ret <4 x i32> %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x i32> @llvm.dx.imad.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #1

; Function Attrs: noinline nounwind optnone
define noundef <4 x i64> @imad_int64_t4(<4 x i64> noundef %p0, <4 x i64> noundef %p1, <4 x i64> noundef %p2) #0 {
entry:
  ; CHECK: extractelement <4 x i64> %p0, i64 0
  ; CHECK: extractelement <4 x i64> %p1, i64 0
  ; CHECK: extractelement <4 x i64> %p2, i64 0
  ; CHECK: call i64 @dx.op.tertiary.i64(i32 48, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i64> %p0, i64 1
  ; CHECK: extractelement <4 x i64> %p1, i64 1
  ; CHECK: extractelement <4 x i64> %p2, i64 1
  ; CHECK: call i64 @dx.op.tertiary.i64(i32 48, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i64> %p0, i64 2
  ; CHECK: extractelement <4 x i64> %p1, i64 2
  ; CHECK: extractelement <4 x i64> %p2, i64 2
  ; CHECK: call i64 @dx.op.tertiary.i64(i32 48, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  ; CHECK: extractelement <4 x i64> %p0, i64 3
  ; CHECK: extractelement <4 x i64> %p1, i64 3
  ; CHECK: extractelement <4 x i64> %p2, i64 3
  ; CHECK: call i64 @dx.op.tertiary.i64(i32 48, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) #[[#ATTR]]
  ; CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
  ; CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i64 1
  ; CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i64 2
  ; CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i64 3
  %dx.imad = call <4 x i64> @llvm.dx.imad.v4i64(<4 x i64> %p0, <4 x i64> %p1, <4 x i64> %p2)
  ret <4 x i64> %dx.imad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x i64> @llvm.dx.imad.v4i64(<4 x i64>, <4 x i64>, <4 x i64>) #1

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}
