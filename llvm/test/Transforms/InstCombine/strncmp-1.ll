; Test that the strncmp library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@hell = constant [5 x i8] c"hell\00"
@bell = constant [5 x i8] c"bell\00"
@null = constant [1 x i8] zeroinitializer

declare i32 @strncmp(i8*, i8*, i32)

; strncmp("", x, n) -> -*x
define i32 @test1(i8* %str2) {
; CHECK: @test1
; CHECK: %strcmpload = load i8* %str
; CHECK: %1 = zext i8 %strcmpload to i32
; CHECK: %2 = sub i32 0, %1
; CHECK: ret i32 %2

  %str1 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 10)
  ret i32 %temp1
}

; strncmp(x, "", n) -> *x
define i32 @test2(i8* %str1) {
; CHECK: @test2
; CHECK: %strcmpload = load i8* %str1
; CHECK: %1 = zext i8 %strcmpload to i32
; CHECK: ret i32 %1

  %str2 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 10)
  ret i32 %temp1
}

; strncmp(x, y, n)  -> cnst
define i32 @test3() {
; CHECK: @test3
; CHECK: ret i32 -1

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [6 x i8]* @hello, i32 0, i32 0
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 10)
  ret i32 %temp1
}

define i32 @test4() {
; CHECK: @test4
; CHECK: ret i32 1

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 10)
  ret i32 %temp1
}

define i32 @test5() {
; CHECK: @test5
; CHECK: ret i32 0

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [6 x i8]* @hello, i32 0, i32 0
  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 4)
  ret i32 %temp1
}

; strncmp(x,y,1) -> memcmp(x,y,1)
define i32 @test6(i8* %str1, i8* %str2) {
; CHECK: @test6
; CHECK: [[LOAD1:%[a-z]+]] = load i8* %str1, align 1
; CHECK: [[ZEXT1:%[a-z]+]] = zext i8 [[LOAD1]] to i32
; CHECK: [[LOAD2:%[a-z]+]] = load i8* %str2, align 1
; CHECK: [[ZEXT2:%[a-z]+]] = zext i8 [[LOAD2]] to i32
; CHECK: [[RET:%[a-z]+]] = sub i32 [[ZEXT1]], [[ZEXT2]]
; CHECK: ret i32 [[RET]]

  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 1)
  ret i32 %temp1
}

; strncmp(x,y,0)   -> 0
define i32 @test7(i8* %str1, i8* %str2) {
; CHECK: @test7
; CHECK: ret i32 0

  %temp1 = call i32 @strncmp(i8* %str1, i8* %str2, i32 0)
  ret i32 %temp1
}

; strncmp(x,x,n)  -> 0
define i32 @test8(i8* %str, i32 %n) {
; CHECK: @test8
; CHECK: ret i32 0

  %temp1 = call i32 @strncmp(i8* %str, i8* %str, i32 %n)
  ret i32 %temp1
}
