; Test that -disable-simplify-libcalls is wired up correctly.
;
; RUN: opt < %s -instcombine -disable-simplify-libcalls -S | FileCheck %s
; RUN: opt < %s -passes=instcombine -disable-simplify-libcalls -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@.str  = constant [1 x i8] zeroinitializer, align 1
@.str1 = constant [13 x i8] c"hello, world\00", align 1
@.str2 = constant [4 x i8] c"foo\00", align 1
@.str3 = constant [4 x i8] c"bar\00", align 1
@.str4 = constant [6 x i8] c"123.4\00", align 1
@.str5 = constant [5 x i8] c"1234\00", align 1
@empty = constant [1 x i8] c"\00", align 1

declare double @ceil(double)
declare double @copysign(double, double)
declare double @cos(double)
declare double @fabs(double)
declare double @floor(double)
declare ptr @strcat(ptr, ptr)
declare ptr @strncat(ptr, ptr, i32)
declare ptr @strchr(ptr, i32)
declare ptr @strrchr(ptr, i32)
declare i32 @strcmp(ptr, ptr)
declare i32 @strncmp(ptr, ptr, i64)
declare ptr @strcpy(ptr, ptr)
declare ptr @stpcpy(ptr, ptr)
declare ptr @strncpy(ptr, ptr, i64)
declare i64 @strlen(ptr)
declare ptr @strpbrk(ptr, ptr)
declare i64 @strspn(ptr, ptr)
declare double @strtod(ptr, ptr)
declare float @strtof(ptr, ptr)
declare x86_fp80 @strtold(ptr, ptr)
declare i64 @strtol(ptr, ptr, i32)
declare i64 @strtoll(ptr, ptr, i32)
declare i64 @strtoul(ptr, ptr, i32)
declare i64 @strtoull(ptr, ptr, i32)
declare i64 @strcspn(ptr, ptr)
declare i32 @abs(i32)
declare i32 @ffs(i32)
declare i32 @ffsl(i64)
declare i32 @ffsll(i64)
declare i32 @fprintf(ptr, ptr)
declare i32 @isascii(i32)
declare i32 @isdigit(i32)
declare i32 @toascii(i32)
declare i64 @labs(i64)
declare i64 @llabs(i64)
declare i32 @printf(ptr)
declare i32 @sprintf(ptr, ptr)

define double @t1(double %x) {
; CHECK-LABEL: @t1(
  %ret = call double @ceil(double %x)
  ret double %ret
; CHECK: call double @ceil
}

define double @t2(double %x, double %y) {
; CHECK-LABEL: @t2(
  %ret = call double @copysign(double %x, double %y)
  ret double %ret
; CHECK: call double @copysign
}

define double @t3(double %x) {
; CHECK-LABEL: @t3(
  %call = call double @cos(double %x)
  ret double %call
; CHECK: call double @cos
}

define double @t4(double %x) {
; CHECK-LABEL: @t4(
  %ret = call double @fabs(double %x)
  ret double %ret
; CHECK: call double @fabs
}

define double @t5(double %x) {
; CHECK-LABEL: @t5(
  %ret = call double @floor(double %x)
  ret double %ret
; CHECK: call double @floor
}

define ptr @t6(ptr %x) {
; CHECK-LABEL: @t6(
  %ret = call ptr @strcat(ptr %x, ptr @empty)
  ret ptr %ret
; CHECK: call ptr @strcat
}

define ptr @t7(ptr %x) {
; CHECK-LABEL: @t7(
  %ret = call ptr @strncat(ptr %x, ptr @empty, i32 1)
  ret ptr %ret
; CHECK: call ptr @strncat
}

define ptr @t8() {
; CHECK-LABEL: @t8(
  %ret = call ptr @strchr(ptr @.str1, i32 119)
  ret ptr %ret
; CHECK: call ptr @strchr
}

define ptr @t9() {
; CHECK-LABEL: @t9(
  %ret = call ptr @strrchr(ptr @.str1, i32 119)
  ret ptr %ret
; CHECK: call ptr @strrchr
}

define i32 @t10() {
; CHECK-LABEL: @t10(
  %ret = call i32 @strcmp(ptr @.str2, ptr @.str3)
  ret i32 %ret
; CHECK: call i32 @strcmp
}

define i32 @t11() {
; CHECK-LABEL: @t11(
  %ret = call i32 @strncmp(ptr @.str2, ptr @.str3, i64 3)
  ret i32 %ret
; CHECK: call i32 @strncmp
}

define ptr @t12(ptr %x) {
; CHECK-LABEL: @t12(
  %ret = call ptr @strcpy(ptr %x, ptr @.str2)
  ret ptr %ret
; CHECK: call ptr @strcpy
}

define ptr @t13(ptr %x) {
; CHECK-LABEL: @t13(
  %ret = call ptr @stpcpy(ptr %x, ptr @.str2)
  ret ptr %ret
; CHECK: call ptr @stpcpy
}

define ptr @t14(ptr %x) {
; CHECK-LABEL: @t14(
  %ret = call ptr @strncpy(ptr %x, ptr @.str2, i64 3)
  ret ptr %ret
; CHECK: call ptr @strncpy
}

define i64 @t15() {
; CHECK-LABEL: @t15(
  %ret = call i64 @strlen(ptr @.str2)
  ret i64 %ret
; CHECK: call i64 @strlen
}

define ptr @t16(ptr %x) {
; CHECK-LABEL: @t16(
  %ret = call ptr @strpbrk(ptr %x, ptr @.str)
  ret ptr %ret
; CHECK: call ptr @strpbrk
}

define i64 @t17(ptr %x) {
; CHECK-LABEL: @t17(
  %ret = call i64 @strspn(ptr %x, ptr @.str)
  ret i64 %ret
; CHECK: call i64 @strspn
}

define double @t18(ptr %y) {
; CHECK-LABEL: @t18(
  %ret = call double @strtod(ptr @.str4, ptr %y)
  ret double %ret
; CHECK: call double @strtod
}

define float @t19(ptr %y) {
; CHECK-LABEL: @t19(
  %ret = call float @strtof(ptr @.str4, ptr %y)
  ret float %ret
; CHECK: call float @strtof
}

define x86_fp80 @t20(ptr %y) {
; CHECK-LABEL: @t20(
  %ret = call x86_fp80 @strtold(ptr @.str4, ptr %y)
  ret x86_fp80 %ret
; CHECK: call x86_fp80 @strtold
}

define i64 @t21(ptr %y) {
; CHECK-LABEL: @t21(
  %ret = call i64 @strtol(ptr @.str5, ptr %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtol
}

define i64 @t22(ptr %y) {
; CHECK-LABEL: @t22(
  %ret = call i64 @strtoll(ptr @.str5, ptr %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoll
}

define i64 @t23(ptr %y) {
; CHECK-LABEL: @t23(
  %ret = call i64 @strtoul(ptr @.str5, ptr %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoul
}

define i64 @t24(ptr %y) {
; CHECK-LABEL: @t24(
  %ret = call i64 @strtoull(ptr @.str5, ptr %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoull
}

define i64 @t25(ptr %y) {
; CHECK-LABEL: @t25(
  %ret = call i64 @strcspn(ptr @empty, ptr %y)
  ret i64 %ret
; CHECK: call i64 @strcspn
}

define i32 @t26(i32 %y) {
; CHECK-LABEL: @t26(
  %ret = call i32 @abs(i32 %y)
  ret i32 %ret
; CHECK: call i32 @abs
}

define i32 @t27(i32 %y) {
; CHECK-LABEL: @t27(
  %ret = call i32 @ffs(i32 %y)
  ret i32 %ret
; CHECK: call i32 @ffs
}

define i32 @t28(i64 %y) {
; CHECK-LABEL: @t28(
  %ret = call i32 @ffsl(i64 %y)
  ret i32 %ret
; CHECK: call i32 @ffsl
}

define i32 @t29(i64 %y) {
; CHECK-LABEL: @t29(
  %ret = call i32 @ffsll(i64 %y)
  ret i32 %ret
; CHECK: call i32 @ffsll
}

define void @t30() {
; CHECK-LABEL: @t30(
  call i32 @fprintf(ptr null, ptr @.str1)
  ret void
; CHECK: call i32 @fprintf
}

define i32 @t31(i32 %y) {
; CHECK-LABEL: @t31(
  %ret = call i32 @isascii(i32 %y)
  ret i32 %ret
; CHECK: call i32 @isascii
}

define i32 @t32(i32 %y) {
; CHECK-LABEL: @t32(
  %ret = call i32 @isdigit(i32 %y)
  ret i32 %ret
; CHECK: call i32 @isdigit
}

define i32 @t33(i32 %y) {
; CHECK-LABEL: @t33(
  %ret = call i32 @toascii(i32 %y)
  ret i32 %ret
; CHECK: call i32 @toascii
}

define i64 @t34(i64 %y) {
; CHECK-LABEL: @t34(
  %ret = call i64 @labs(i64 %y)
  ret i64 %ret
; CHECK: call i64 @labs
}

define i64 @t35(i64 %y) {
; CHECK-LABEL: @t35(
  %ret = call i64 @llabs(i64 %y)
  ret i64 %ret
; CHECK: call i64 @llabs
}

define void @t36() {
; CHECK-LABEL: @t36(
  call i32 @printf(ptr @empty)
  ret void
; CHECK: call i32 @printf
}

define void @t37(ptr %x) {
; CHECK-LABEL: @t37(
  call i32 @sprintf(ptr %x, ptr @.str1)
  ret void
; CHECK: call i32 @sprintf
}
