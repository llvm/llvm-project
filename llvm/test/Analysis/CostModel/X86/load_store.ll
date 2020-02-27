; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @stores(i32 %arg) {

  ;CHECK: cost of 1 {{.*}} store
  store i8 undef, i8* undef, align 4
  ;CHECK: cost of 1 {{.*}} store
  store i16 undef, i16* undef, align 4
  ;CHECK: cost of 1 {{.*}} store
  store i32 undef, i32* undef, align 4
  ;CHECK: cost of 1 {{.*}} store
  store i64 undef, i64* undef, align 4
  ;CHECK: cost of 2 {{.*}} store
  store i128 undef, i128* undef, align 4

  ;CHECK: cost of 1 {{.*}} store
  store <4 x i16> undef, <4 x i16>* undef, align 4
  ;CHECK: cost of 1 {{.*}} store
  store <4 x i32> undef, <4 x i32>* undef, align 4
  ;CHECK: cost of 2 {{.*}} store
  store <4 x i64> undef, <4 x i64>* undef, align 4

  ;CHECK: cost of 1 {{.*}} store
  store <8 x i16> undef, <8 x i16>* undef, align 4
  ;CHECK: cost of 2 {{.*}} store
  store <8 x i32> undef, <8 x i32>* undef, align 4
  ;CHECK: cost of 4 {{.*}} store
  store <8 x i64> undef, <8 x i64>* undef, align 4

  ret i32 undef
}
define i32 @loads(i32 %arg) {
  ;CHECK: cost of 1 {{.*}} load
  load i8, i8* undef, align 4
  ;CHECK: cost of 1 {{.*}} load
  load i16, i16* undef, align 4
  ;CHECK: cost of 1 {{.*}} load
  load i32, i32* undef, align 4
  ;CHECK: cost of 1 {{.*}} load
  load i64, i64* undef, align 4
  ;CHECK: cost of 2 {{.*}} load
  load i128, i128* undef, align 4

  ;CHECK: cost of 1 {{.*}} load
  load <2 x i32>, <2 x i32>* undef, align 4
  ;CHECK: cost of 1 {{.*}} load
  load <4 x i32>, <4 x i32>* undef, align 4
  ;CHECK: cost of 2 {{.*}} load
  load <8 x i32>, <8 x i32>* undef, align 4


  ;CHECK: cost of 1 {{.*}} load
  load <2 x i64>, <2 x i64>* undef, align 4
  ;CHECK: cost of 2 {{.*}} load
  load <4 x i64>, <4 x i64>* undef, align 4
  ;CHECK: cost of 4 {{.*}} load
  load <8 x i64>, <8 x i64>* undef, align 4


  ;CHECK: cost of 3 {{.*}} load
  load <3 x float>, <3 x float>* undef, align 4

  ;CHECK: cost of 3 {{.*}} load
  load <3 x double>, <3 x double>* undef, align 4

  ;CHECK: cost of 3 {{.*}} load
  load <3 x i32>, <3 x i32>* undef, align 4

  ;CHECK: cost of 3 {{.*}} load
  load <3 x i64>, <3 x i64>* undef, align 4

  ;CHECK: cost of 12 {{.*}} load
  load <5 x i32>, <5 x i32>* undef, align 4

  ;CHECK: cost of 14 {{.*}} load
  load <5 x i64>, <5 x i64>* undef, align 4

  ret i32 undef
}

