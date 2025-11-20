; RUN: opt < %s  -passes="print<cost-model>" 2>&1 -disable-output -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 -disable-ppc-unaligned | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @stores(i32 %arg) {

  ; CHECK: cost of 1 {{.*}} store
  store i8 undef, ptr undef, align 4
  ; CHECK: cost of 1 {{.*}} store
  store i16 undef, ptr undef, align 4
  ; CHECK: cost of 1 {{.*}} store
  store i32 undef, ptr undef, align 4
  ; CHECK: cost of 2 {{.*}} store
  store i64 undef, ptr undef, align 4
  ; CHECK: cost of 4 {{.*}} store
  store i128 undef, ptr undef, align 4

  ret i32 undef
}
define i32 @loads(i32 %arg) {
  ; CHECK: cost of 1 {{.*}} load
  load i8, ptr undef, align 4
  ; CHECK: cost of 1 {{.*}} load
  load i16, ptr undef, align 4
  ; CHECK: cost of 1 {{.*}} load
  load i32, ptr undef, align 4
  ; CHECK: cost of 2 {{.*}} load
  load i64, ptr undef, align 4
  ; CHECK: cost of 4 {{.*}} load
  load i128, ptr undef, align 4

  ; FIXME: There actually are sub-vector Altivec loads, and so we could handle
  ; this with a small expense, but we don't currently.
  ; CHECK: cost of 42 {{.*}} load
  load <4 x i16>, ptr undef, align 2

  ; CHECK: cost of 2 {{.*}} load
  load <4 x i32>, ptr undef, align 4

  ; CHECK: cost of 46 {{.*}} load
  load <3 x float>, ptr undef, align 1

  ret i32 undef
}

define i32 @partialvector32(i32 %arg) #0 {

  ; CHECK: cost of 1 {{.*}} store
  store <4 x i8> undef, ptr undef, align 16

  ret i32 undef
}

define i32 @partialvector64(i32 %arg) #1 {

  ; CHECK: cost of 1 {{.*}} store
  store <4 x i16> undef, ptr undef, align 16

  ret i32 undef
}

attributes #0 = { "target-features"="+power8-vector,+vsx" }

attributes #1 = { "target-features"="+vsx" }
