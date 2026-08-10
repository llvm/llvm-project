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

  %va = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %va)
  call void @llvm.va_start(ptr nonnull %va)
  %p1 = va_arg ptr %va, i32
  %p2 = va_arg ptr %va, i16
  %p3 = va_arg ptr %va, i8
  %p4 = va_arg ptr %va, i32
  %p5 = va_arg ptr %va, i16
  %p6 = va_arg ptr %va, i8
  %p7 = va_arg ptr %va, float
  %p8 = va_arg ptr %va, ptr
  %p9 = va_arg ptr %va, i64
  %p10 = va_arg ptr %va, double
  call void @llvm.va_end(ptr nonnull %va)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(ptr %p8)
  call void @use_i64(i64 %p9)
  call void @use_double(double %p10)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %va)
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

  %va = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %va)
  call void @llvm.va_start(ptr nonnull %va)
  %vb = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %va)
  call void @llvm.va_copy(ptr nonnull %vb, ptr nonnull %va)
  call void @llvm.va_end(ptr nonnull %va)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %va)
  %p1 = va_arg ptr %vb, i32
  %p2 = va_arg ptr %vb, i16
  %p3 = va_arg ptr %vb, i8
  %p4 = va_arg ptr %vb, i32
  %p5 = va_arg ptr %vb, i16
  %p6 = va_arg ptr %vb, i8
  %p7 = va_arg ptr %vb, float
  %p8 = va_arg ptr %vb, ptr
  %p9 = va_arg ptr %vb, i64
  %p10 = va_arg ptr %vb, double
  call void @llvm.va_end(ptr nonnull %vb)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %vb)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(ptr %p8)
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

  %va = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %va)
  call void @llvm.va_start(ptr nonnull %va)
  %p1 = va_arg ptr %va, i32
  %p2 = va_arg ptr %va, i16
  %p3 = va_arg ptr %va, i8
  %p4 = va_arg ptr %va, i32
  %p5 = va_arg ptr %va, i16
  %p6 = va_arg ptr %va, i8
  %p7 = va_arg ptr %va, float

  %vc = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %va)
  call void @llvm.va_copy(ptr nonnull %vc, ptr nonnull %va)
  call void @llvm.va_end(ptr nonnull %va)
  %p8 = va_arg ptr %vc, ptr
  %p9 = va_arg ptr %vc, i64
  %p10 = va_arg ptr %vc, double
  call void @llvm.va_end(ptr nonnull %vc)
  call void @use_i32(i32 %p1)
  call void @use_s16(i16 %p2)
  call void @use_s8(i8 %p3)
  call void @use_i32(i32 %p4)
  call void @use_u16(i16 %p5)
  call void @use_u8(i8 %p6)
  call void @use_float(float %p7)
  call void @use_i8p(ptr %p8)
  call void @use_i64(i64 %p9)
  call void @use_double(double %p10)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %va)
  ret i32 0
}

declare void @use_i64(i64)
declare void @use_i32(i32)
declare void @use_u16(i16 zeroext)
declare void @use_u8(i8 zeroext)
declare void @use_s16(i16 signext)
declare void @use_s8(i8 signext)
declare void @use_i8p(ptr)
declare void @use_float(float)
declare void @use_double(double)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.va_start(ptr)
declare void @llvm.va_copy(ptr, ptr)
declare void @llvm.va_end(ptr)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
