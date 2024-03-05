; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for round are generated for float and half.
; CHECK:call i16 @dx.op.tertiary.i16(i32 49, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
; CHECK:call i32 @dx.op.tertiary.i32(i32 49, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK:call i64 @dx.op.tertiary.i64(i32 49, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"
; Function Attrs: noinline nounwind optnone
define noundef i16 @umad_ushort(i16 noundef %p0, i16 noundef %p1, i16 noundef %p2) #0 {
entry:
  %p2.addr = alloca i16, align 2
  %p1.addr = alloca i16, align 2
  %p0.addr = alloca i16, align 2
  store i16 %p2, ptr %p2.addr, align 2
  store i16 %p1, ptr %p1.addr, align 2
  store i16 %p0, ptr %p0.addr, align 2
  %0 = load i16, ptr %p0.addr, align 2
  %1 = load i16, ptr %p1.addr, align 2
  %2 = load i16, ptr %p2.addr, align 2
  %dx.umad = call i16 @llvm.dx.umad.i16(i16 %0, i16 %1, i16 %2)
  ret i16 %dx.umad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i16 @llvm.dx.umad.i16(i16, i16, i16) #1

; Function Attrs: noinline nounwind optnone
define noundef i32 @umad_uint(i32 noundef %p0, i32 noundef %p1, i32 noundef %p2) #0 {
entry:
  %p2.addr = alloca i32, align 4
  %p1.addr = alloca i32, align 4
  %p0.addr = alloca i32, align 4
  store i32 %p2, ptr %p2.addr, align 4
  store i32 %p1, ptr %p1.addr, align 4
  store i32 %p0, ptr %p0.addr, align 4
  %0 = load i32, ptr %p0.addr, align 4
  %1 = load i32, ptr %p1.addr, align 4
  %2 = load i32, ptr %p2.addr, align 4
  %dx.umad = call i32 @llvm.dx.umad.i32(i32 %0, i32 %1, i32 %2)
  ret i32 %dx.umad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.dx.umad.i32(i32, i32, i32) #1

; Function Attrs: noinline nounwind optnone
define noundef i64 @umad_uint64(i64 noundef %p0, i64 noundef %p1, i64 noundef %p2) #0 {
entry:
  %p2.addr = alloca i64, align 8
  %p1.addr = alloca i64, align 8
  %p0.addr = alloca i64, align 8
  store i64 %p2, ptr %p2.addr, align 8
  store i64 %p1, ptr %p1.addr, align 8
  store i64 %p0, ptr %p0.addr, align 8
  %0 = load i64, ptr %p0.addr, align 8
  %1 = load i64, ptr %p1.addr, align 8
  %2 = load i64, ptr %p2.addr, align 8
  %dx.umad = call i64 @llvm.dx.umad.i64(i64 %0, i64 %1, i64 %2)
  ret i64 %dx.umad
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i64 @llvm.dx.umad.i64(i64, i64, i64) #1
