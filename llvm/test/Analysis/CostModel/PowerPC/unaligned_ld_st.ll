; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=+vsx | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test(i32 %arg) {

  ; CHECK: cost of 1 {{.*}} load
  load i8, i8* undef, align 1
  ; CHECK: cost of 1 {{.*}} load
  load i16, i16* undef, align 1
  ; CHECK: cost of 1 {{.*}} load
  load i32, i32* undef, align 1
  ; CHECK: cost of 1 {{.*}} load
  load i64, i64* undef, align 1

  ; CHECK: cost of 1 {{.*}} store
  store i8 undef, i8* undef, align 1
  ; CHECK: cost of 1 {{.*}} store
  store i16 undef, i16* undef, align 1
  ; CHECK: cost of 1 {{.*}} store
  store i32 undef, i32* undef, align 1
  ; CHECK: cost of 1 {{.*}} store
  store i64 undef, i64* undef, align 1

  ret i32 undef
}
