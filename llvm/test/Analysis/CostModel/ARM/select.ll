; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

; CHECK: casts
define void @casts() {
    ; Scalar values
  ; CHECK: cost of 1 {{.*}} select
  %v1 = select i1 undef, i8 undef, i8 undef
  ; CHECK: cost of 1 {{.*}} select
  %v2 = select i1 undef, i16 undef, i16 undef
  ; CHECK: cost of 1 {{.*}} select
  %v3 = select i1 undef, i32 undef, i32 undef
  ; CHECK: cost of 2 {{.*}} select
  %v4 = select i1 undef, i64 undef, i64 undef
  ; CHECK: cost of 1 {{.*}} select
  %v5 = select i1 undef, float undef, float undef
  ; CHECK: cost of 1 {{.*}} select
  %v6 = select i1 undef, double undef, double undef

    ; Vector values
  ; CHECK: cost of 1 {{.*}} select
  %v7 = select <2 x i1> undef, <2 x i8> undef, <2 x i8> undef
  ; CHECK: cost of 1 {{.*}} select
  %v8 = select <4 x i1>  undef, <4 x i8> undef, <4 x i8> undef
  ; CHECK: cost of 1 {{.*}} select
  %v9 = select <8 x i1>  undef, <8 x i8> undef, <8 x i8> undef
  ; CHECK: cost of 1 {{.*}} select
  %v10 = select <16 x i1>  undef, <16 x i8> undef, <16 x i8> undef

  ; CHECK: cost of 1 {{.*}} select
  %v11 = select <2 x i1> undef, <2 x i16> undef, <2 x i16> undef
  ; CHECK: cost of 1 {{.*}} select
  %v12 = select <4 x i1>  undef, <4 x i16> undef, <4 x i16> undef
  ; CHECK: cost of 1 {{.*}} select
  %v13 = select <8 x i1>  undef, <8 x i16> undef, <8 x i16> undef
  ; CHECK: cost of 2 {{.*}} select
  %v13b = select <16 x i1>  undef, <16 x i16> undef, <16 x i16> undef

  ; CHECK: cost of 1 {{.*}} select
  %v14 = select <2 x i1> undef, <2 x i32> undef, <2 x i32> undef
  ; CHECK: cost of 1 {{.*}} select
  %v15 = select <4 x i1>  undef, <4 x i32> undef, <4 x i32> undef
  ; CHECK: cost of 2 {{.*}} select
  %v15b = select <8 x i1>  undef, <8 x i32> undef, <8 x i32> undef
  ; CHECK: cost of 4 {{.*}} select
  %v15c = select <16 x i1>  undef, <16 x i32> undef, <16 x i32> undef

  ; CHECK: cost of 1 {{.*}} select
  %v16 = select <2 x i1> undef, <2 x i64> undef, <2 x i64> undef
  ; CHECK: cost of 19 {{.*}} select
  %v16a = select <4 x i1> undef, <4 x i64> undef, <4 x i64> undef
  ; CHECK: cost of 50 {{.*}} select
  %v16b = select <8 x i1> undef, <8 x i64> undef, <8 x i64> undef
  ; CHECK: cost of 100 {{.*}} select
  %v16c = select <16 x i1> undef, <16 x i64> undef, <16 x i64> undef

  ; CHECK: cost of 1 {{.*}} select
  %v17 = select <2 x i1> undef, <2 x float> undef, <2 x float> undef
  ; CHECK: cost of 1 {{.*}} select
  %v18 = select <4 x i1>  undef, <4 x float> undef, <4 x float> undef

  ; CHECK: cost of 1 {{.*}} select
  %v19 = select <2 x i1>  undef, <2 x double> undef, <2 x double> undef

  ; odd vectors get legalized and should have similar costs
  ; CHECK: cost of 1 {{.*}} select
  %v20 = select <1 x i1>  undef, <1 x i32> undef, <1 x i32> undef
  ; CHECK: cost of 1 {{.*}} select
  %v21 = select <3 x i1>  undef, <3 x float> undef, <3 x float> undef
  ; CHECK: cost of 4 {{.*}} select
  %v22 = select <5 x i1>  undef, <5 x double> undef, <5 x double> undef

  ret void
}
