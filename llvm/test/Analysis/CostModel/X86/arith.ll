; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @add(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} add
  %A = add <4 x i32> undef, undef
  ;CHECK: cost of 4 {{.*}} add
  %B = add <8 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} add
  %C = add <2 x i64> undef, undef
  ;CHECK: cost of 4 {{.*}} add
  %D = add <4 x i64> undef, undef
  ;CHECK: cost of 8 {{.*}} add
  %E = add <8 x i64> undef, undef
  ;CHECK: cost of 1 {{.*}} ret
  ret i32 undef
}


define i32 @xor(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} xor
  %A = xor <4 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %B = xor <8 x i32> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %C = xor <2 x i64> undef, undef
  ;CHECK: cost of 1 {{.*}} xor
  %D = xor <4 x i64> undef, undef
  ;CHECK: cost of 1 {{.*}} ret
  ret i32 undef
}


define i32 @fmul(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} fmul
  %A = fmul <4 x float> undef, undef
  ;CHECK: cost of 1 {{.*}} fmul
  %B = fmul <8 x float> undef, undef
  ret i32 undef
}
