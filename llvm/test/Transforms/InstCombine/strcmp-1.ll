; Test that the strcmp library call simplifier works correctly.
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@hello = constant [6 x i8] c"hello\00"
@hell = constant [5 x i8] c"hell\00"
@bell = constant [5 x i8] c"bell\00"
@null = constant [1 x i8] zeroinitializer

declare i32 @strcmp(i8*, i8*)

; strcmp("", x) -> -*x
define i32 @test1(i8* %str2) {
; CHECK: @test1
; CHECK: %strcmpload = load i8* %str
; CHECK: %1 = zext i8 %strcmpload to i32
; CHECK: %2 = sub i32 0, %1
; CHECK: ret i32 %2

  %str1 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strcmp(i8* %str1, i8* %str2)
  ret i32 %temp1

}

; strcmp(x, "") -> *x
define i32 @test2(i8* %str1) {
; CHECK: @test2
; CHECK: %strcmpload = load i8* %str
; CHECK: %1 = zext i8 %strcmpload to i32
; CHECK: ret i32 %1

  %str2 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strcmp(i8* %str1, i8* %str2)
  ret i32 %temp1
}

; strcmp(x, y)  -> cnst
define i32 @test3() {
; CHECK: @test3
; CHECK: ret i32 -1

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [6 x i8]* @hello, i32 0, i32 0
  %temp1 = call i32 @strcmp(i8* %str1, i8* %str2)
  ret i32 %temp1
}

define i32 @test4() {
; CHECK: @test4
; CHECK: ret i32 1

  %str1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %str2 = getelementptr inbounds [1 x i8]* @null, i32 0, i32 0
  %temp1 = call i32 @strcmp(i8* %str1, i8* %str2)
  ret i32 %temp1
}

; strcmp(x, y)   -> memcmp(x, y, <known length>)
; (This transform is rather difficult to trigger in a useful manner)
define i32 @test5(i1 %b) {
; CHECK: @test5
; CHECK: %memcmp = call i32 @memcmp(i8* getelementptr inbounds ([6 x i8]* @hello, i32 0, i32 0), i8* %str2, i32 5)
; CHECK: ret i32 %memcmp

  %str1 = getelementptr inbounds [6 x i8]* @hello, i32 0, i32 0
  %temp1 = getelementptr inbounds [5 x i8]* @hell, i32 0, i32 0
  %temp2 = getelementptr inbounds [5 x i8]* @bell, i32 0, i32 0
  %str2 = select i1 %b, i8* %temp1, i8* %temp2
  %temp3 = call i32 @strcmp(i8* %str1, i8* %str2)
  ret i32 %temp3
}

; strcmp(x,x)  -> 0
define i32 @test6(i8* %str) {
; CHECK: @test6
; CHECK: ret i32 0

  %temp1 = call i32 @strcmp(i8* %str, i8* %str)
  ret i32 %temp1
}
