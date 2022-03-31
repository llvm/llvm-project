; RUN: opt < %s  -passes='print<cost-model>' 2>&1 -disable-output -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 -disable-ppc-unaligned | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @stores(i32 %arg) {

  ; CHECK: cost of 1 {{.*}} store
  store i8 undef, i8* undef, align 4
  ; CHECK: cost of 1 {{.*}} store
  store i16 undef, i16* undef, align 4
  ; CHECK: cost of 1 {{.*}} store
  store i32 undef, i32* undef, align 4
  ; CHECK: cost of 2 {{.*}} store
  store i64 undef, i64* undef, align 4
  ; CHECK: cost of 4 {{.*}} store
  store i128 undef, i128* undef, align 4

  ret i32 undef
}
define i32 @loads(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} load
  load i8, i8* undef, align 4
  ; CHECK: cost of 1 {{.*}} load
  load i16, i16* undef, align 4
  ; CHECK: cost of 1 {{.*}} load
  load i32, i32* undef, align 4
  ; CHECK: cost of 2 {{.*}} load
  load i64, i64* undef, align 4
  ; CHECK: cost of 4 {{.*}} load
  load i128, i128* undef, align 4

  ; FIXME: There actually are sub-vector Altivec loads, and so we could handle
  ; this with a small expense, but we don't currently.
  ; CHECK: cost of 42 {{.*}} load
  load <4 x i16>, <4 x i16>* undef, align 2

  ; CHECK: cost of 2 {{.*}} load
  load <4 x i32>, <4 x i32>* undef, align 4

  ; CHECK: cost of 46 {{.*}} load
  load <3 x float>, <3 x float>* undef, align 1

  ret i32 undef
}

