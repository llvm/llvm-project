; Test that the memcmp library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@foo = constant [4 x i8] c"foo\00"
@hel = constant [4 x i8] c"hel\00"
@hello_u = constant [8 x i8] c"hello_u\00"

declare i32 @memcmp(i8*, i8*, i32)

; Check memcmp(mem, mem, size) -> 0.

define i32 @test_simplify1(i8* %mem, i32 %size) {
; CHECK: @test_simplify1
  %ret = call i32 @memcmp(i8* %mem, i8* %mem, i32 %size)
  ret i32 %ret
; CHECK: ret i32 0
}

; Check memcmp(mem1, mem2, 0) -> 0.

define i32 @test_simplify2(i8* %mem1, i8* %mem2) {
; CHECK: @test_simplify2
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 0)
  ret i32 %ret
; CHECK: ret i32 0
}

;; Check memcmp(mem1, mem2, 1) -> *(unsigned char*)mem1 - *(unsigned char*)mem2.

define i32 @test_simplify3(i8* %mem1, i8* %mem2) {
; CHECK: @test_simplify3
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 1)
; CHECK: [[LOAD1:%[a-z]+]] = load i8* %mem1, align 1
; CHECK: [[ZEXT1:%[a-z]+]] = zext i8 [[LOAD1]] to i32
; CHECK: [[LOAD2:%[a-z]+]] = load i8* %mem2, align 1
; CHECK: [[ZEXT2:%[a-z]+]] = zext i8 [[LOAD2]] to i32
; CHECK: [[RET:%[a-z]+]] = sub i32 [[ZEXT1]], [[ZEXT2]]
  ret i32 %ret
; CHECK: ret i32 [[RET]]
}

; Check memcmp(mem1, mem2, size) -> cnst, where all arguments are constants.

define i32 @test_simplify4() {
; CHECK: @test_simplify4
  %mem1 = getelementptr [4 x i8]* @hel, i32 0, i32 0
  %mem2 = getelementptr [8 x i8]* @hello_u, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
; CHECK: ret i32 0
}

define i32 @test_simplify5() {
; CHECK: @test_simplify5
  %mem1 = getelementptr [4 x i8]* @hel, i32 0, i32 0
  %mem2 = getelementptr [4 x i8]* @foo, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
; CHECK: ret i32 {{[0-9]+}}
}

define i32 @test_simplify6() {
; CHECK: @test_simplify6
  %mem1 = getelementptr [4 x i8]* @foo, i32 0, i32 0
  %mem2 = getelementptr [4 x i8]* @hel, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
; CHECK: ret i32 {{-[0-9]+}}
}
