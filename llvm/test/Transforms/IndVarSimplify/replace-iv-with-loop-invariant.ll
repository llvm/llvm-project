; RUN: opt < %s -indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = external global i32

define void @test0(i64* %arg) {
bb:
  br label %bb2

bb2:
  %tmp = phi i64* [%arg, %bb ], [ %tmp7, %bb2 ]
  %tmp4 = call i32* @wobble(i64* nonnull %tmp, i32* null)
  %tmp5 = load i32, i32* %tmp4, align 8
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

; CHECK-LABEL: void @test0
; CHECK:         load i32, i32* null

define void @test1(i64* %arg) {
bb:
  br label %bb2

bb2:
  %tmp = phi i64* [%arg, %bb ], [ %tmp7, %bb2 ]
  %tmp4 = call i32* @wobble(i64* nonnull %tmp, i32* inttoptr (i64 4 to i32*))
  %tmp5 = load i32, i32* %tmp4
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

; CHECK-LABEL: void @test1
; CHECK:         load i32, i32* inttoptr (i64 4 to i32*)

define void @test2(i64* %arg) {
bb:
  br label %bb2

bb2:
  %tmp = phi i64* [%arg, %bb ], [ %tmp7, %bb2 ]
  %tmp4 = call i32* @wobble(i64* nonnull %tmp, i32* @G)
  %tmp5 = load i32, i32* %tmp4
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

; CHECK-LABEL: void @test2
; CHECK:         load i32, i32* @G


define void @test3(i64* %arg, i32* %loop.invariant) {
bb:
  br label %bb2

bb2:
  %tmp = phi i64* [%arg, %bb ], [ %tmp7, %bb2 ]
  %tmp4 = call i32* @wobble(i64* nonnull %tmp, i32* %loop.invariant)
  %tmp5 = load i32, i32* %tmp4
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

; CHECK-LABEL: void @test3
; CHECK:         load i32, i32* %loop.invariant

define void @test4(i64* %arg, i32* %loop.invariant, i64 %N) {
bb:
  br label %bb2

bb2:
  %tmp = phi i64* [%arg, %bb ], [ %tmp7, %bb2 ]
  %mul = mul nsw i64 %N, 64
  %ptr = getelementptr inbounds i32, i32* %loop.invariant, i64 %mul 
  %tmp4 = call i32* @wobble(i64* nonnull %tmp, i32* %ptr)
  %tmp5 = load i32, i32* %tmp4
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

; CHECK-LABEL: void @test4
; CHECK:         [[P:%[a-zA-Z$._0-9]+]] = getelementptr i32, i32* %loop.invariant
; CHECK:         phi
; CHECK:         load i32, i32* [[P]]

declare i32* @wobble(i64*, i32* returned)
