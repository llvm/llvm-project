; RUN: llc < %s -mattr=+simd128 -verify-machineinstrs | FileCheck %s --check-prefixes CHECK

; Check that shuffles maintain their type when being custom
; lowered. Regression test for bug 39275.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK: i8x16.shuffle
define <4 x i32> @foo(<4 x i32> %x) {
  %1 = shufflevector <4 x i32> %x, <4 x i32> undef,
    <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef,
    <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %3 = add <4 x i32> %2, %2
  ret <4 x i32> %3
}
