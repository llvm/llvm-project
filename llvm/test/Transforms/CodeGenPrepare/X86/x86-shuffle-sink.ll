; RUN: opt -S -codegenprepare -mcpu=core-avx2 %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AVX2
; RUN: opt -S -codegenprepare -mcpu=corei7 %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SSE2

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin10.9.0"

define <16 x i8> @test_8bit(<16 x i8> %lhs, <16 x i8> %tmp, i1 %tst) {
; CHECK-LABEL: @test_8bit
; CHECK: if_true:
; CHECK-NOT: shufflevector

; CHECK: if_false:
; CHECK-NOT: shufflevector
; CHECK: shl <16 x i8> %lhs, %mask
  %mask = shufflevector <16 x i8> %tmp, <16 x i8> undef, <16 x i32> zeroinitializer
  br i1 %tst, label %if_true, label %if_false

if_true:
  ret <16 x i8> %mask

if_false:
  %res = shl <16 x i8> %lhs, %mask
  ret <16 x i8> %res
}

define <8 x i16> @test_16bit(<8 x i16> %lhs, <8 x i16> %tmp, i1 %tst) {
; CHECK-LABEL: @test_16bit
; CHECK: if_true:
; CHECK-NOT: shufflevector

; CHECK: if_false:
; CHECK: [[SPLAT:%[0-9a-zA-Z_]+]] = shufflevector
; CHECK: shl <8 x i16> %lhs, [[SPLAT]]
  %mask = shufflevector <8 x i16> %tmp, <8 x i16> undef, <8 x i32> zeroinitializer
  br i1 %tst, label %if_true, label %if_false

if_true:
  ret <8 x i16> %mask

if_false:
  %res = shl <8 x i16> %lhs, %mask
  ret <8 x i16> %res
}

define <4 x i32> @test_notsplat(<4 x i32> %lhs, <4 x i32> %tmp, i1 %tst) {
; CHECK-LABEL: @test_notsplat
; CHECK: if_true:
; CHECK-NOT: shufflevector

; CHECK: if_false:
; CHECK-NOT: shufflevector
; CHECK: shl <4 x i32> %lhs, %mask
  %mask = shufflevector <4 x i32> %tmp, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 1, i32 0>
  br i1 %tst, label %if_true, label %if_false

if_true:
  ret <4 x i32> %mask

if_false:
  %res = shl <4 x i32> %lhs, %mask
  ret <4 x i32> %res
}

define <4 x i32> @test_32bit(<4 x i32> %lhs, <4 x i32> %tmp, i1 %tst) {
; CHECK-AVX2-LABEL: @test_32bit
; CHECK-AVX2: if_false:
; CHECK-AVX2-NOT: shufflevector
; CHECK-AVX2: ashr <4 x i32> %lhs, %mask

; CHECK-SSE2-LABEL: @test_32bit
; CHECK-SSE2: if_false:
; CHECK-SSE2: [[SPLAT:%[0-9a-zA-Z_]+]] = shufflevector
; CHECK-SSE2: ashr <4 x i32> %lhs, [[SPLAT]]
  %mask = shufflevector <4 x i32> %tmp, <4 x i32> undef, <4 x i32> <i32 0, i32 undef, i32 0, i32 0>
  br i1 %tst, label %if_true, label %if_false

if_true:
  ret <4 x i32> %mask

if_false:
  %res = ashr <4 x i32> %lhs, %mask
  ret <4 x i32> %res
}

define <2 x i64> @test_64bit(<2 x i64> %lhs, <2 x i64> %tmp, i1 %tst) {
; CHECK-AVX2-LABEL: @test_64bit
; CHECK-AVX2: if_false:
; CHECK-AVX2-NOT: shufflevector
; CHECK-AVX2: lshr <2 x i64> %lhs, %mask

; CHECK-SSE2-LABEL: @test_64bit
; CHECK-SSE2: if_false:
; CHECK-SSE2: [[SPLAT:%[0-9a-zA-Z_]+]] = shufflevector
; CHECK-SSE2: lshr <2 x i64> %lhs, [[SPLAT]]

  %mask = shufflevector <2 x i64> %tmp, <2 x i64> undef, <2 x i32> zeroinitializer
  br i1 %tst, label %if_true, label %if_false

if_true:
  ret <2 x i64> %mask

if_false:
  %res = lshr <2 x i64> %lhs, %mask
  ret <2 x i64> %res
}
