; RUN: opt < %s -simplifycfg -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -S | FileCheck %s
; Avoid if-conversion if there is a long dependence chain.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; The first several cases test FindLongDependenceChain returns true, so
; if-conversion is blocked.

define i64 @test1(i64** %pp, i64* %p) {
entry:
  %0 = load i64*, i64** %pp, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp slt i64 %1, 0
  %pint = ptrtoint i64* %p to i64
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = or i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %ptr = inttoptr i64 %p3 to i64*
  %val = load i64, i64* %ptr, align 8
  ret i64 %val

; CHECK-NOT: select
}

define i64 @test2(i64** %pp, i64* %p) {
entry:
  %0 = load i64*, i64** %pp, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp slt i64 %1, 0
  %pint = ptrtoint i64* %p to i64
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = add i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %ptr = inttoptr i64 %p3 to i64*
  %val = load i64, i64* %ptr, align 8
  ret i64 %val

; CHECK-LABEL: @test2
; CHECK-NOT: select
}

; The following cases test FindLongDependenceChain returns false, so
; if-conversion will proceed.

; Non trivial LatencyAdjustment.
define i64 @test3(i64** %pp, i64* %p) {
entry:
  %0 = load i64*, i64** %pp, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp slt i64 %1, 0
  %pint = ptrtoint i64* %p to i64
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = or i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %p4 = add i64 %p3, %1
  %ptr = inttoptr i64 %p4 to i64*
  %val = load i64, i64* %ptr, align 8
  ret i64 %val

; CHECK-LABEL: @test3
; CHECK: select
}

; Short dependence chain.
define i64 @test4(i64* %pp, i64* %p) {
entry:
  %0 = load i64, i64* %pp, align 8
  %cmp = icmp slt i64 %0, 0
  %pint = ptrtoint i64* %p to i64
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = or i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %ptr = inttoptr i64 %p3 to i64*
  %val = load i64, i64* %ptr, align 8
  ret i64 %val

; CHECK-LABEL: @test4
; CHECK: select
}

; High IPC.
define i64 @test5(i64** %pp, i64* %p) {
entry:
  %0 = load i64*, i64** %pp, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp slt i64 %1, 0
  %pint = ptrtoint i64* %p to i64
  %2 = add i64 %pint, 2
  %3 = add i64 %pint, 3
  %4 = or i64 %pint, 16
  %5 = and i64 %pint, 255

  %6 = or i64 %2, 9
  %7 = and i64 %3, 255
  %8 = add i64 %4, 4
  %9 = add i64 %5, 5

  %10 = add i64 %6, 2
  %11 = add i64 %7, 3
  %12 = add i64 %8, 4
  %13 = add i64 %9, 5

  %14 = add i64 %10, 6
  %15 = add i64 %11, 7
  %16 = add i64 %12, 8
  %17 = add i64 %13, 9

  %18 = add i64 %14, 10
  %19 = add i64 %15, 11
  %20 = add i64 %16, 12
  %21 = add i64 %17, 13

  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = or i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %ptr = inttoptr i64 %p3 to i64*
  %val = load i64, i64* %ptr, align 8

  ret i64 %val

; CHECK-LABEL: @test5
; CHECK: select
}

; Large BB size.
define i64 @test6(i64** %pp, i64* %p) {
entry:
  %0 = load i64*, i64** %pp, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp slt i64 %1, 0
  %pint = ptrtoint i64* %p to i64
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  %p1 = add i64 %pint, 8
  br label %cond.end

cond.false:
  %p2 = or i64 %pint, 16
  br label %cond.end

cond.end:
  %p3 = phi i64 [%p1, %cond.true], [%p2, %cond.false]
  %ptr = inttoptr i64 %p3 to i64*
  %val = load i64, i64* %ptr, align 8
  %2 = add i64 %pint, 2
  %3 = add i64 %pint, 3
  %4 = add i64 %2, 4
  %5 = add i64 %3, 5
  %6 = add i64 %4, 6
  %7 = add i64 %5, 7
  %8 = add i64 %6, 6
  %9 = add i64 %7, 7
  %10 = add i64 %8, 6
  %11 = add i64 %9, 7
  %12 = add i64 %10, 6
  %13 = add i64 %11, 7
  %14 = add i64 %12, 6
  %15 = add i64 %13, 7
  %16 = add i64 %14, 6
  %17 = add i64 %15, 7
  %18 = add i64 %16, 6
  %19 = add i64 %17, 7
  %20 = add i64 %18, 6
  %21 = add i64 %19, 7
  %22 = add i64 %20, 6
  %23 = add i64 %21, 7
  %24 = add i64 %22, 6
  %25 = add i64 %23, 7
  %26 = add i64 %24, 6
  %27 = add i64 %25, 7
  %28 = add i64 %26, 6
  %29 = add i64 %27, 7
  %30 = add i64 %28, 6
  %31 = add i64 %29, 7
  %32 = add i64 %30, 8
  %33 = add i64 %31, 9
  %34 = add i64 %32, %33
  %35 = and i64 %34, 255
  %res = add i64 %val, %35

  ret i64 %res

; CHECK-LABEL: @test6
; CHECK: select
}
