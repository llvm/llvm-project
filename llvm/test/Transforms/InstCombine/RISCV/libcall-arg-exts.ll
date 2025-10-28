; RUN: opt < %s -passes=instcombine -S -mtriple=riscv64 | FileCheck %s
;
; Check that i32 arguments to generated libcalls have the proper extension
; attributes.


declare double @exp2(double)
declare float @exp2f(float)
declare fp128 @exp2l(fp128)

define double @fun1(i32 %x) {
; CHECK-LABEL: @fun1
; CHECK: call double @ldexp
  %conv = sitofp i32 %x to double
  %ret = call double @exp2(double %conv)
  ret double %ret
}

define float @fun2(i32 %x) {
; CHECK-LABEL: @fun2
; CHECK: call float @ldexpf
  %conv = sitofp i32 %x to float
  %ret = call float @exp2f(float %conv)
  ret float %ret
}

define fp128 @fun3(i8 zeroext %x) {
; CHECK-LABEL: @fun3
; CHECK: call fp128 @ldexpl
  %conv = uitofp i8 %x to fp128
  %ret = call fp128 @exp2l(fp128 %conv)
  ret fp128 %ret
}

@a = common global [60 x i8] zeroinitializer, align 1
@b = common global [60 x i8] zeroinitializer, align 1
declare ptr @__memccpy_chk(ptr, ptr, i32, i64, i64)
define ptr @fun4() {
; CHECK-LABEL: @fun4
; CHECK: call ptr @memccpy
  %ret = call ptr @__memccpy_chk(ptr @a, ptr @b, i32 0, i64 60, i64 -1)
  ret ptr %ret
}

%FILE = type { }
@A = constant [2 x i8] c"A\00"
declare i32 @fputs(ptr, ptr)
define void @fun5(ptr %fp) {
; CHECK-LABEL: @fun5
; CHECK: call i32 @fputc
  call i32 @fputs(ptr @A, ptr %fp)
  ret void
}

@empty = constant [1 x i8] zeroinitializer
declare i32 @puts(ptr)
define void @fun6() {
; CHECK-LABEL: @fun6
; CHECK: call i32 @putchar
  call i32 @puts(ptr @empty)
  ret void
}

@.str1 = private constant [2 x i8] c"a\00"
declare ptr @strstr(ptr, ptr)
define ptr @fun7(ptr %str) {
; CHECK-LABEL: @fun7
; CHECK: call ptr @strchr
  %ret = call ptr @strstr(ptr %str, ptr @.str1)
  ret ptr %ret
}

; CHECK: declare ptr @strchr(ptr, i32 signext)

@hello = constant [14 x i8] c"hello world\5Cn\00"
@chp = global ptr zeroinitializer
declare ptr @strchr(ptr, i32)
define void @fun8(i32 %chr) {
; CHECK-LABEL: @fun8
; CHECK: call ptr @memchr
  %dst = call ptr @strchr(ptr @hello, i32 %chr)
  store ptr %dst, ptr @chp
  ret void
}

; CHECK: declare double @ldexp(double, i32 signext)
; CHECK: declare float @ldexpf(float, i32 signext)
; CHECK: declare fp128 @ldexpl(fp128, i32 signext)
; CHECK: declare ptr @memccpy(ptr noalias writeonly, ptr noalias readonly captures(none), i32 signext, i64)
; CHECK: declare noundef i32 @fputc(i32 noundef signext, ptr noundef captures(none))
; CHECK: declare noundef i32 @putchar(i32 noundef signext)
; CHECK: declare ptr @memchr(ptr, i32 signext, i64)
