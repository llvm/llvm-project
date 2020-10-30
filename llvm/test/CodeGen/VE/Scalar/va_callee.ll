; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i32 @va_func(i32, ...) {
; CHECK-LABEL: va_func:
; CHECK:       ldl.sx %s0, 184(, %s9)
; CHECK:       ld2b.sx %s18, 192(, %s9)
; CHECK:       ld1b.sx %s19, 200(, %s9)
; CHECK:       ldl.sx %s20, 208(, %s9)
; CHECK:       ld2b.zx %s21, 216(, %s9)
; CHECK:       ld1b.zx %s22, 224(, %s9)
; CHECK:       ldu %s23, 236(, %s9)
; CHECK:       ld %s24, 240(, %s9)
; CHECK:       ld %s25, 248(, %s9)

  %va = alloca i8*, align 8
  %va.i8 = bitcast i8** %va to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %va.i8)
  call void @llvm.va_start(i8* nonnull %va.i8)
  %p1 = va_arg i8** %va, i32
  %p2 = va_arg i8** %va, i16
  %p3 = va_arg i8** %va, i8
  %p4 = va_arg i8** %va, i32
  %p5 = va_arg i8** %va, i16
  %p6 = va_arg i8** %va, i8
  %p7 = va_arg i8** %va, float
  %p8 = va_arg i8** %va, i8*
  %p9 = va_arg i8** %va, i64
  %p10 = va_arg i8** %va, double
  call void @llvm.va_end(i8* nonnull %va.i8)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(i8* %p8)
  call void @use_i64(i64 %p9)
  call void @use_double(double %p10)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %va.i8)
  ret i32 0
}

define i32 @va_copy0(i32, ...) {
; CHECK-LABEL: va_copy0:
; CHECK:       ldl.sx %s0,
; CHECK:       ld2b.sx %s18,
; CHECK:       ld1b.sx %s19,
; CHECK:       ldl.sx %s20,
; CHECK:       ld2b.zx %s21,
; CHECK:       ld1b.zx %s22,
; CHECK:       ldu %s23,
; CHECK:       ld %s24,
; CHECK:       ld %s25,

  %va = alloca i8*, align 8
  %va.i8 = bitcast i8** %va to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %va.i8)
  call void @llvm.va_start(i8* nonnull %va.i8)
  %vb = alloca i8*, align 8
  %vb.i8 = bitcast i8** %vb to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %va.i8)
  call void @llvm.va_copy(i8* nonnull %vb.i8, i8* nonnull %va.i8)
  call void @llvm.va_end(i8* nonnull %va.i8)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %va.i8)
  %p1 = va_arg i8** %vb, i32
  %p2 = va_arg i8** %vb, i16
  %p3 = va_arg i8** %vb, i8
  %p4 = va_arg i8** %vb, i32
  %p5 = va_arg i8** %vb, i16
  %p6 = va_arg i8** %vb, i8
  %p7 = va_arg i8** %vb, float
  %p8 = va_arg i8** %vb, i8*
  %p9 = va_arg i8** %vb, i64
  %p10 = va_arg i8** %vb, double
  call void @llvm.va_end(i8* nonnull %vb.i8)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %vb.i8)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(i8* %p8)
  call void @use_i64(i64 %p9)
  call void @use_double(double %p10)
  ret i32 0
}

define i32 @va_copy8(i32, ...) {
; CHECK-LABEL: va_copy8:
; CHECK:       ldl.sx %s0,
; CHECK:       ld2b.sx %s18,
; CHECK:       ld1b.sx %s19,
; CHECK:       ldl.sx %s20,
; CHECK:       ld2b.zx %s21,
; CHECK:       ld1b.zx %s22,
; CHECK:       ldu %s23,
; CHECK:       ld %s24,
; CHECK:       ld %s25,

  %va = alloca i8*, align 8
  %va.i8 = bitcast i8** %va to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %va.i8)
  call void @llvm.va_start(i8* nonnull %va.i8)
  %p1 = va_arg i8** %va, i32
  %p2 = va_arg i8** %va, i16
  %p3 = va_arg i8** %va, i8
  %p4 = va_arg i8** %va, i32
  %p5 = va_arg i8** %va, i16
  %p6 = va_arg i8** %va, i8
  %p7 = va_arg i8** %va, float

  %vc = alloca i8*, align 8
  %vc.i8 = bitcast i8** %vc to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %va.i8)
  call void @llvm.va_copy(i8* nonnull %vc.i8, i8* nonnull %va.i8)
  call void @llvm.va_end(i8* nonnull %va.i8)
  %p8 = va_arg i8** %vc, i8*
  %p9 = va_arg i8** %vc, i64
  %p10 = va_arg i8** %vc, double
  call void @llvm.va_end(i8* nonnull %vc.i8)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(i8* %p8)
  call void @use_i64(i64 %p9)
  call void @use_double(double %p10)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %va.i8)
  ret i32 0
}

declare void @use_i64(i64)
declare void @use_i32(i32)
declare void @use_u16(i16 zeroext)
declare void @use_u8(i8 zeroext)
declare void @use_s16(i16 signext)
declare void @use_s8(i8 signext)
declare void @use_i8p(i8*)
declare void @use_float(float)
declare void @use_double(double)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
