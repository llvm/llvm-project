; Test that -disable-simplify-libcalls is wired up correctly.
;
; RUN: opt < %s -instcombine -disable-simplify-libcalls -S | FileCheck %s

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
declare i8* @strcat(i8*, i8*)
declare i8* @strncat(i8*, i8*, i32)
declare i8* @strchr(i8*, i32)
declare i8* @strrchr(i8*, i32)
declare i32 @strcmp(i8*, i8*)
declare i32 @strncmp(i8*, i8*, i64)
declare i8* @strcpy(i8*, i8*)
declare i8* @stpcpy(i8*, i8*)
declare i8* @strncpy(i8*, i8*, i64)
declare i64 @strlen(i8*)
declare i8* @strpbrk(i8*, i8*)
declare i64 @strspn(i8*, i8*)
declare double @strtod(i8*, i8**)
declare float @strtof(i8*, i8**)
declare x86_fp80 @strtold(i8*, i8**)
declare i64 @strtol(i8*, i8**, i32)
declare i64 @strtoll(i8*, i8**, i32)
declare i64 @strtoul(i8*, i8**, i32)
declare i64 @strtoull(i8*, i8**, i32)
declare i64 @strcspn(i8*, i8*)

define double @t1(double %x) {
; CHECK: @t1
  %ret = call double @ceil(double %x)
  ret double %ret
; CHECK: call double @ceil
}

define double @t2(double %x, double %y) {
; CHECK: @t2
  %ret = call double @copysign(double %x, double %y)
  ret double %ret
; CHECK: call double @copysign
}

define double @t3(double %x) {
; CHECK: @t3
  %call = call double @cos(double %x)
  ret double %call
; CHECK: call double @cos
}

define double @t4(double %x) {
; CHECK: @t4
  %ret = call double @fabs(double %x)
  ret double %ret
; CHECK: call double @fabs
}

define double @t5(double %x) {
; CHECK: @t5
  %ret = call double @floor(double %x)
  ret double %ret
; CHECK: call double @floor
}

define i8* @t6(i8* %x) {
; CHECK: @t6
  %empty = getelementptr [1 x i8]* @empty, i32 0, i32 0
  %ret = call i8* @strcat(i8* %x, i8* %empty)
  ret i8* %ret
; CHECK: call i8* @strcat
}

define i8* @t7(i8* %x) {
; CHECK: @t7
  %empty = getelementptr [1 x i8]* @empty, i32 0, i32 0
  %ret = call i8* @strncat(i8* %x, i8* %empty, i32 1)
  ret i8* %ret
; CHECK: call i8* @strncat
}

define i8* @t8() {
; CHECK: @t8
  %x = getelementptr inbounds [13 x i8]* @.str1, i32 0, i32 0
  %ret = call i8* @strchr(i8* %x, i32 119)
  ret i8* %ret
; CHECK: call i8* @strchr
}

define i8* @t9() {
; CHECK: @t9
  %x = getelementptr inbounds [13 x i8]* @.str1, i32 0, i32 0
  %ret = call i8* @strrchr(i8* %x, i32 119)
  ret i8* %ret
; CHECK: call i8* @strrchr
}

define i32 @t10() {
; CHECK: @t10
  %x = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %y = getelementptr inbounds [4 x i8]* @.str3, i32 0, i32 0
  %ret = call i32 @strcmp(i8* %x, i8* %y)
  ret i32 %ret
; CHECK: call i32 @strcmp
}

define i32 @t11() {
; CHECK: @t11
  %x = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %y = getelementptr inbounds [4 x i8]* @.str3, i32 0, i32 0
  %ret = call i32 @strncmp(i8* %x, i8* %y, i64 3)
  ret i32 %ret
; CHECK: call i32 @strncmp
}

define i8* @t12(i8* %x) {
; CHECK: @t12
  %y = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %ret = call i8* @strcpy(i8* %x, i8* %y)
  ret i8* %ret
; CHECK: call i8* @strcpy
}

define i8* @t13(i8* %x) {
; CHECK: @t13
  %y = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %ret = call i8* @stpcpy(i8* %x, i8* %y)
  ret i8* %ret
; CHECK: call i8* @stpcpy
}

define i8* @t14(i8* %x) {
; CHECK: @t14
  %y = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %ret = call i8* @strncpy(i8* %x, i8* %y, i64 3)
  ret i8* %ret
; CHECK: call i8* @strncpy
}

define i64 @t15() {
; CHECK: @t15
  %x = getelementptr inbounds [4 x i8]* @.str2, i32 0, i32 0
  %ret = call i64 @strlen(i8* %x)
  ret i64 %ret
; CHECK: call i64 @strlen
}

define i8* @t16(i8* %x) {
; CHECK: @t16
  %y = getelementptr inbounds [1 x i8]* @.str, i32 0, i32 0
  %ret = call i8* @strpbrk(i8* %x, i8* %y)
  ret i8* %ret
; CHECK: call i8* @strpbrk
}

define i64 @t17(i8* %x) {
; CHECK: @t17
  %y = getelementptr inbounds [1 x i8]* @.str, i32 0, i32 0
  %ret = call i64 @strspn(i8* %x, i8* %y)
  ret i64 %ret
; CHECK: call i64 @strspn
}

define double @t18(i8** %y) {
; CHECK: @t18
  %x = getelementptr inbounds [6 x i8]* @.str4, i64 0, i64 0
  %ret = call double @strtod(i8* %x, i8** %y)
  ret double %ret
; CHECK: call double @strtod
}

define float @t19(i8** %y) {
; CHECK: @t19
  %x = getelementptr inbounds [6 x i8]* @.str4, i64 0, i64 0
  %ret = call float @strtof(i8* %x, i8** %y)
  ret float %ret
; CHECK: call float @strtof
}

define x86_fp80 @t20(i8** %y) {
; CHECK: @t20
  %x = getelementptr inbounds [6 x i8]* @.str4, i64 0, i64 0
  %ret = call x86_fp80 @strtold(i8* %x, i8** %y)
  ret x86_fp80 %ret
; CHECK: call x86_fp80 @strtold
}

define i64 @t21(i8** %y) {
; CHECK: @t21
  %x = getelementptr inbounds [5 x i8]* @.str5, i64 0, i64 0
  %ret = call i64 @strtol(i8* %x, i8** %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtol
}

define i64 @t22(i8** %y) {
; CHECK: @t22
  %x = getelementptr inbounds [5 x i8]* @.str5, i64 0, i64 0
  %ret = call i64 @strtoll(i8* %x, i8** %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoll
}

define i64 @t23(i8** %y) {
; CHECK: @t23
  %x = getelementptr inbounds [5 x i8]* @.str5, i64 0, i64 0
  %ret = call i64 @strtoul(i8* %x, i8** %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoul
}

define i64 @t24(i8** %y) {
; CHECK: @t24
  %x = getelementptr inbounds [5 x i8]* @.str5, i64 0, i64 0
  %ret = call i64 @strtoull(i8* %x, i8** %y, i32 10)
  ret i64 %ret
; CHECK: call i64 @strtoull
}

define i64 @t25(i8* %y) {
; CHECK: @t25
  %x = getelementptr [1 x i8]* @empty, i32 0, i32 0
  %ret = call i64 @strcspn(i8* %x, i8* %y)
  ret i64 %ret
; CHECK: call i64 @strcspn
}
