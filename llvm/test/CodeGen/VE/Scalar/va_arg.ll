; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

@.str = private unnamed_addr constant [6 x i8] c"a=%d\0A\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"b=%d\0A\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"c=%d\0A\00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"d=%u\0A\00", align 1
@.str.4 = private unnamed_addr constant [6 x i8] c"e=%u\0A\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"f=%u\0A\00", align 1
@.str.6 = private unnamed_addr constant [6 x i8] c"g=%f\0A\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"h=%p\0A\00", align 1
@.str.8 = private unnamed_addr constant [7 x i8] c"i=%ld\0A\00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"j=%lf\0A\00", align 1
@.str.10 = private unnamed_addr constant [7 x i8] c"j=%Lf\0A\00", align 1

define i32 @func_vainout(i32, ...) {
; CHECK-LABEL: func_vainout:
; CHECK:         ldl.sx %s{{.*}}, 184(, %s9)
; CHECK:         ld2b.sx %s{{.*}}, 192(, %s9)
; CHECK:         ld1b.sx %s{{.*}}, 200(, %s9)
; CHECK:         ldl.sx %s{{.*}}, 208(, %s9)
; CHECK:         ld2b.zx %s{{.*}}, 216(, %s9)
; CHECK:         ld1b.zx %s{{.*}}, 224(, %s9)
; CHECK:         ldu %s{{.*}}, 236(, %s9)
; CHECK:         ld %s{{.*}}, 240(, %s9)
; CHECK:         ld %s{{.*}}, 248(, %s9)
; CHECK:         ld %s{{.*}}, 256(, %s9)
; CHECK:         lea %{{.*}}, 279(, %s9)
; CHECK:         and %{{.*}}, -16, %s0
; CHECK:         lea %{{.*}}, 16(, %s0)
; CHECK:         ld %s{{.*}}, 8(, %s0)
; CHECK:         ld %s{{.*}}, (, %s0)
; CHECK:         ld %s{{.*}}, 16(, %s0)
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic
; CHECK:         bsic

  %a = alloca i8*, align 8
  %a8 = bitcast i8** %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %a8)
  call void @llvm.va_start(i8* nonnull %a8)
  %p0 = va_arg i8** %a, i32
  %p1 = va_arg i8** %a, i16
  %p2 = va_arg i8** %a, i8
  %p3 = va_arg i8** %a, i32
  %p4 = va_arg i8** %a, i16
  %p5 = va_arg i8** %a, i8
  %p6 = va_arg i8** %a, float
  %p7 = va_arg i8** %a, i8*
  %p8 = va_arg i8** %a, i64
  %p9 = va_arg i8** %a, double
  %p10 = va_arg i8** %a, fp128
  %p11 = va_arg i8** %a, double
  call void @llvm.va_end(i8* nonnull %a8)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %a8)
  %pf0 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), i32 %p0)
  %p1.s32 = sext i16 %p1 to i32
  %pf1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), i32 %p1.s32)
  %p2.s32 = sext i8 %p2 to i32
  %pf2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.2, i64 0, i64 0), i32 %p2.s32)
  %pf3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.3, i64 0, i64 0), i32 %p3)
  %p4.z32 = zext i16 %p4 to i32
  %pf4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.4, i64 0, i64 0), i32 %p4.z32)
  %p5.z32 = zext i8 %p5 to i32
  %pf5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.5, i64 0, i64 0), i32 %p5.z32)
  %pf6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.6, i64 0, i64 0), float %p6)
  %pf7 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.7, i64 0, i64 0), i8* %p7)
  %pf8 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.8, i64 0, i64 0), i64 %p8)
  %pf9 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i64 0, i64 0), double %p9)
  %pf10 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.10, i64 0, i64 0), fp128 %p10)
  %pf11 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i64 0, i64 0), double %p11)
  ret i32 0
}
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare i32 @printf(i8* nocapture readonly, ...)
