; RUN: opt < %s -passes=instcombine -S -mtriple=systemz-unknown | FileCheck %s
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
declare i8* @__memccpy_chk(i8*, i8*, i32, i64, i64)
define i8* @fun4() {
; CHECK-LABEL: @fun4
; CHECK: call i8* @memccpy
  %dst = getelementptr inbounds [60 x i8], [60 x i8]* @a, i32 0, i32 0
  %src = getelementptr inbounds [60 x i8], [60 x i8]* @b, i32 0, i32 0
  %ret = call i8* @__memccpy_chk(i8* %dst, i8* %src, i32 0, i64 60, i64 -1)
  ret i8* %ret
}

%FILE = type { }
@A = constant [2 x i8] c"A\00"
declare i32 @fputs(i8*, %FILE*)
define void @fun5(%FILE* %fp) {
; CHECK-LABEL: @fun5
; CHECK: call i32 @fputc
  %str = getelementptr [2 x i8], [2 x i8]* @A, i32 0, i32 0
  call i32 @fputs(i8* %str, %FILE* %fp)
  ret void
}

@empty = constant [1 x i8] zeroinitializer
declare i32 @puts(i8*)
define void @fun6() {
; CHECK-LABEL: @fun6
; CHECK: call i32 @putchar
  %str = getelementptr [1 x i8], [1 x i8]* @empty, i32 0, i32 0
  call i32 @puts(i8* %str)
  ret void
}

@.str1 = private constant [2 x i8] c"a\00"
declare i8* @strstr(i8*, i8*)
define i8* @fun7(i8* %str) {
; CHECK-LABEL: @fun7
; CHECK: call i8* @strchr
  %pat = getelementptr inbounds [2 x i8], [2 x i8]* @.str1, i32 0, i32 0
  %ret = call i8* @strstr(i8* %str, i8* %pat)
  ret i8* %ret
}

; CHECK: declare i8* @strchr(i8*, i32 signext)

@hello = constant [14 x i8] c"hello world\5Cn\00"
@chp = global i8* zeroinitializer
declare i8* @strchr(i8*, i32)
define void @fun8(i32 %chr) {
; CHECK-LABEL: @fun8
; CHECK: call i8* @memchr
  %src = getelementptr [14 x i8], [14 x i8]* @hello, i32 0, i32 0
  %dst = call i8* @strchr(i8* %src, i32 %chr)
  store i8* %dst, i8** @chp
  ret void
}

; CHECK: declare double @ldexp(double, i32 signext)
; CHECK: declare float @ldexpf(float, i32 signext)
; CHECK: declare fp128 @ldexpl(fp128, i32 signext)
; CHECK: declare i8* @memccpy(i8* noalias writeonly, i8* noalias nocapture readonly, i32 signext, i64)
; CHECK: declare noundef i32 @fputc(i32 noundef signext, %FILE* nocapture noundef)
; CHECK: declare noundef i32 @putchar(i32 noundef signext)
; CHECK: declare i8* @memchr(i8*, i32 signext, i64)
