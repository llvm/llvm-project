; Test that the memcmp library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32:64"

@foo = constant [4 x i8] c"foo\00"
@hel = constant [4 x i8] c"hel\00"
@hello_u = constant [8 x i8] c"hello_u\00"

declare i32 @memcmp(i8*, i8*, i32)

; Check memcmp(mem, mem, size) -> 0.

define i32 @test_simplify1(i8* %mem, i32 %size) {
; CHECK-LABEL: @test_simplify1(
; CHECK-NEXT:    ret i32 0
;
  %ret = call i32 @memcmp(i8* %mem, i8* %mem, i32 %size)
  ret i32 %ret
}

; Check memcmp(mem1, mem2, 0) -> 0.

define i32 @test_simplify2(i8* %mem1, i8* %mem2) {
; CHECK-LABEL: @test_simplify2(
; CHECK-NEXT:    ret i32 0
;
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 0)
  ret i32 %ret
}

;; Check memcmp(mem1, mem2, 1) -> *(unsigned char*)mem1 - *(unsigned char*)mem2.

define i32 @test_simplify3(i8* %mem1, i8* %mem2) {
; CHECK-LABEL: @test_simplify3(
; CHECK-NEXT:    [[LHSC:%.*]] = load i8, i8* %mem1, align 1
; CHECK-NEXT:    [[LHSV:%.*]] = zext i8 [[LHSC]] to i32
; CHECK-NEXT:    [[RHSC:%.*]] = load i8, i8* %mem2, align 1
; CHECK-NEXT:    [[RHSV:%.*]] = zext i8 [[RHSC]] to i32
; CHECK-NEXT:    [[CHARDIFF:%.*]] = sub nsw i32 [[LHSV]], [[RHSV]]
; CHECK-NEXT:    ret i32 [[CHARDIFF]]
;
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 1)
  ret i32 %ret
}

; Check memcmp(mem1, mem2, size) -> cnst, where all arguments are constants.

define i32 @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
; CHECK-NEXT:    ret i32 0
;
  %mem1 = getelementptr [4 x i8], [4 x i8]* @hel, i32 0, i32 0
  %mem2 = getelementptr [8 x i8], [8 x i8]* @hello_u, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
}

define i32 @test_simplify5() {
; CHECK-LABEL: @test_simplify5(
; CHECK-NEXT:    ret i32 1
;
  %mem1 = getelementptr [4 x i8], [4 x i8]* @hel, i32 0, i32 0
  %mem2 = getelementptr [4 x i8], [4 x i8]* @foo, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
}

define i32 @test_simplify6() {
; CHECK-LABEL: @test_simplify6(
; CHECK-NEXT:    ret i32 -1
;
  %mem1 = getelementptr [4 x i8], [4 x i8]* @foo, i32 0, i32 0
  %mem2 = getelementptr [4 x i8], [4 x i8]* @hel, i32 0, i32 0
  %ret = call i32 @memcmp(i8* %mem1, i8* %mem2, i32 3)
  ret i32 %ret
}

; Check memcmp(mem1, mem2, 8)==0 -> *(int64_t*)mem1 == *(int64_t*)mem2

define i1 @test_simplify7(i64 %x, i64 %y) {
; CHECK-LABEL: @test_simplify7(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 %x, %y
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %x.addr = alloca i64, align 8
  %y.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  store i64 %y, i64* %y.addr, align 8
  %xptr = bitcast i64* %x.addr to i8*
  %yptr = bitcast i64* %y.addr to i8*
  %call = call i32 @memcmp(i8* %xptr, i8* %yptr, i32 8)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}

; Check memcmp(mem1, mem2, 4)==0 -> *(int32_t*)mem1 == *(int32_t*)mem2

define i1 @test_simplify8(i32 %x, i32 %y) {
; CHECK-LABEL: @test_simplify8(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %x, %y
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %xptr = bitcast i32* %x.addr to i8*
  %yptr = bitcast i32* %y.addr to i8*
  %call = call i32 @memcmp(i8* %xptr, i8* %yptr, i32 4)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}

; Check memcmp(mem1, mem2, 2)==0 -> *(int16_t*)mem1 == *(int16_t*)mem2

define i1 @test_simplify9(i16 %x, i16 %y) {
; CHECK-LABEL: @test_simplify9(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 %x, %y
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %x.addr = alloca i16, align 2
  %y.addr = alloca i16, align 2
  store i16 %x, i16* %x.addr, align 2
  store i16 %y, i16* %y.addr, align 2
  %xptr = bitcast i16* %x.addr to i8*
  %yptr = bitcast i16* %y.addr to i8*
  %call = call i32 @memcmp(i8* %xptr, i8* %yptr, i32 2)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
