; RUN: opt -instcombine -S < %s | FileCheck %s

; This test is to verify that the instruction combiner is able to fold
; a cttz/ctlz followed by a icmp + select into a single cttz/ctlz with
; the 'is_zero_undef' flag cleared.

define i16 @test1(i16 %x) {
; CHECK-LABEL: @test1(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %x, i1 false)
; CHECK-NEXT: ret i16 [[VAR]]
entry:
  %0 = tail call i16 @llvm.ctlz.i16(i16 %x, i1 true)
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i16 %0, i16 16
  ret i16 %cond
}

define i32 @test2(i32 %x) {
; CHECK-LABEL: @test2(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; CHECK-NEXT: ret i32 [[VAR]]
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i32 %0, i32 32
  ret i32 %cond
}

define i64 @test3(i64 %x) {
; CHECK-LABEL: @test3(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; CHECK-NEXT: ret i64 [[VAR]]
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i64 %0, i64 64
  ret i64 %cond
}

define i16 @test4(i16 %x) {
; CHECK-LABEL: @test4(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %x, i1 false)
; CHECK-NEXT: ret i16 [[VAR]]
entry:
  %0 = tail call i16 @llvm.ctlz.i16(i16 %x, i1 true)
  %tobool = icmp eq i16 %x, 0
  %cond = select i1 %tobool, i16 16, i16 %0
  ret i16 %cond
}

define i32 @test5(i32 %x) {
; CHECK-LABEL: @test5(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; CHECK-NEXT: ret i32 [[VAR]]
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %tobool = icmp eq i32 %x, 0
  %cond = select i1 %tobool, i32 32, i32 %0
  ret i32 %cond
}

define i64 @test6(i64 %x) {
; CHECK-LABEL: @test6(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; CHECK-NEXT: ret i64 [[VAR]]
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %tobool = icmp eq i64 %x, 0
  %cond = select i1 %tobool, i64 64, i64 %0
  ret i64 %cond
}

define i16 @test1b(i16 %x) {
; CHECK-LABEL: @test1b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i16 @llvm.cttz.i16(i16 %x, i1 false)
; CHECK-NEXT: ret i16 [[VAR]]
entry:
  %0 = tail call i16 @llvm.cttz.i16(i16 %x, i1 true)
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i16 %0, i16 16
  ret i16 %cond
}

define i32 @test2b(i32 %x) {
; CHECK-LABEL: @test2b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: ret i32 [[VAR]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i32 %0, i32 32
  ret i32 %cond
}

define i64 @test3b(i64 %x) {
; CHECK-LABEL: @test3b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %x, i1 false)
; CHECK-NEXT: ret i64 [[VAR]]
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i64 %0, i64 64
  ret i64 %cond
}

define i16 @test4b(i16 %x) {
; CHECK-LABEL: @test4b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i16 @llvm.cttz.i16(i16 %x, i1 false)
; CHECK-NEXT: ret i16 [[VAR]]
entry:
  %0 = tail call i16 @llvm.cttz.i16(i16 %x, i1 true)
  %tobool = icmp eq i16 %x, 0
  %cond = select i1 %tobool, i16 16, i16 %0
  ret i16 %cond
}

define i32 @test5b(i32 %x) {
; CHECK-LABEL: @test5b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: ret i32 [[VAR]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %tobool = icmp eq i32 %x, 0
  %cond = select i1 %tobool, i32 32, i32 %0
  ret i32 %cond
}

define i64 @test6b(i64 %x) {
; CHECK-LABEL: @test6b(
; CHECK: [[VAR:%[a-zA-Z0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %x, i1 false)
; CHECK-NEXT: ret i64 [[VAR]]
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %tobool = icmp eq i64 %x, 0
  %cond = select i1 %tobool, i64 64, i64 %0
  ret i64 %cond
}

define i32 @test1c(i16 %x) {
; CHECK-LABEL: @test1c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i16 @llvm.cttz.i16(i16 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i16 [[VAR1]] to i32
; CHECK-NEXT: ret i32 [[VAR2]]
entry:
  %0 = tail call i16 @llvm.cttz.i16(i16 %x, i1 true)
  %cast2 = zext i16 %0 to i32
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i32 %cast2, i32 16
  ret i32 %cond
}

define i64 @test2c(i16 %x) {
; CHECK-LABEL: @test2c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i16 @llvm.cttz.i16(i16 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i16 [[VAR1]] to i64
; CHECK-NEXT: ret i64 [[VAR2]]
entry:
  %0 = tail call i16 @llvm.cttz.i16(i16 %x, i1 true)
  %conv = zext i16 %0 to i64
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i64 %conv, i64 16
  ret i64 %cond
}

define i64 @test3c(i32 %x) {
; CHECK-LABEL: @test3c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i32 [[VAR1]] to i64
; CHECK-NEXT: ret i64 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %conv = zext i32 %0 to i64
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i64 %conv, i64 32
  ret i64 %cond
}

define i32 @test4c(i16 %x) {
; CHECK-LABEL: @test4c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i16 [[VAR1]] to i32
; CHECK-NEXT: ret i32 [[VAR2]]
entry:
  %0 = tail call i16 @llvm.ctlz.i16(i16 %x, i1 true)
  %cast = zext i16 %0 to i32
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i32 %cast, i32 16
  ret i32 %cond
}

define i64 @test5c(i16 %x) {
; CHECK-LABEL: @test5c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i16 @llvm.ctlz.i16(i16 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i16 [[VAR1]] to i64
; CHECK-NEXT: ret i64 [[VAR2]]
entry:
  %0 = tail call i16 @llvm.ctlz.i16(i16 %x, i1 true)
  %cast = zext i16 %0 to i64
  %tobool = icmp ne i16 %x, 0
  %cond = select i1 %tobool, i64 %cast, i64 16
  ret i64 %cond
}

define i64 @test6c(i32 %x) {
; CHECK-LABEL: @test6c(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i32 [[VAR1]] to i64
; CHECK-NEXT: ret i64 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %cast = zext i32 %0 to i64
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i64 %cast, i64 32
  ret i64 %cond
}

define i16 @test1d(i64 %x) {
; CHECK-LABEL: @test1d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i64 [[VAR1]] to i16
; CHECK-NEXT: ret i16 [[VAR2]]
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %conv = trunc i64 %0 to i16
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i16 %conv, i16 64
  ret i16 %cond
}

define i32 @test2d(i64 %x) {
; CHECK-LABEL: @test2d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i64 @llvm.cttz.i64(i64 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i64 [[VAR1]] to i32
; CHECK-NEXT: ret i32 [[VAR2]]
entry:
  %0 = tail call i64 @llvm.cttz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i32
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i32 %cast, i32 64
  ret i32 %cond
}

define i16 @test3d(i32 %x) {
; CHECK-LABEL: @test3d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i32 [[VAR1]] to i16
; CHECK-NEXT: ret i16 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %cast = trunc i32 %0 to i16
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i16 %cast, i16 32
  ret i16 %cond
}

define i16 @test4d(i64 %x) {
; CHECK-LABEL: @test4d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i64 [[VAR1]] to i16
; CHECK-NEXT: ret i16 [[VAR2]]
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i16
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i16 %cast, i16 64
  ret i16 %cond
}

define i32 @test5d(i64 %x) {
; CHECK-LABEL: @test5d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i64 @llvm.ctlz.i64(i64 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i64 [[VAR1]] to i32
; CHECK-NEXT: ret i32 [[VAR2]]
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %cast = trunc i64 %0 to i32
  %tobool = icmp ne i64 %x, 0
  %cond = select i1 %tobool, i32 %cast, i32 64
  ret i32 %cond
}

define i16 @test6d(i32 %x) {
; CHECK-LABEL: @test6d(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i32 [[VAR1]] to i16
; CHECK-NEXT: ret i16 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %x, i1 true)
  %cast = trunc i32 %0 to i16
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i16 %cast, i16 32
  ret i16 %cond
}

define i64 @select_bug1(i32 %x) {
; CHECK-LABEL: @select_bug1(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = zext i32 [[VAR1]] to i64
; CHECK-NEXT: ret i64 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
  %conv = zext i32 %0 to i64
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i64 %conv, i64 32
  ret i64 %cond
}

define i16 @select_bug2(i32 %x) {
; CHECK-LABEL: @select_bug2(
; CHECK: [[VAR1:%[a-zA-Z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
; CHECK-NEXT: [[VAR2:%[a-zA-Z0-9]+]] = trunc i32 [[VAR1]] to i16
; CHECK-NEXT: ret i16 [[VAR2]]
entry:
  %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
  %conv = trunc i32 %0 to i16
  %tobool = icmp ne i32 %x, 0
  %cond = select i1 %tobool, i16 %conv, i16 32
  ret i16 %cond
}

define i128 @test7(i128 %x) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i128 @llvm.ctlz.i128(i128 [[X:%.*]], i1 false), !range !3
; CHECK-NEXT:    ret i128 [[TMP1]]
;
  %1 = tail call i128 @llvm.ctlz.i128(i128 %x, i1 true)
  %tobool = icmp ne i128 %x, 0
  %cond = select i1 %tobool, i128 %1, i128 128
  ret i128 %cond
}

define i128 @test8(i128 %x) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i128 @llvm.cttz.i128(i128 [[X:%.*]], i1 false), !range !3
; CHECK-NEXT:    ret i128 [[TMP1]]
;
  %1 = tail call i128 @llvm.cttz.i128(i128 %x, i1 true)
  %tobool = icmp ne i128 %x, 0
  %cond = select i1 %tobool, i128 %1, i128 128
  ret i128 %cond
}

declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i128 @llvm.ctlz.i128(i128, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i128 @llvm.cttz.i128(i128, i1)
