; RUN: opt < %s -cost-model -analyze -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @exts() {

  ; CHECK: cost of 1 {{.*}} sext
  %v1 = sext i16 undef to i32

  ; CHECK: cost of 1 {{.*}} sext
  %v2 = sext <2 x i16> undef to <2 x i32>

  ; CHECK: cost of 1 {{.*}} sext
  %v3 = sext <4 x i16> undef to <4 x i32>

  ; CHECK: cost of 3 {{.*}} sext
  %v4 = sext <8 x i16> undef to <8 x i32>

  ret void
}

