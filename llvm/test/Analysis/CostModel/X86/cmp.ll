; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=pentium4 | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=yonah | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE3 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSSE3 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=penryn | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE41 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 | FileCheck --check-prefix=CHECK --check-prefix=SSE --check-prefix=SSE42 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core-avx2 | FileCheck --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=knl | FileCheck --check-prefix=CHECK --check-prefix=AVX --check-prefix=AVX512 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @cmp(i32 %arg) {
  ;  -- floats --
  ;SSE2:  cost of 3 {{.*}} fcmp
  ;SSE3:  cost of 3 {{.*}} fcmp
  ;SSSE3: cost of 3 {{.*}} fcmp
  ;SSE41: cost of 3 {{.*}} fcmp
  ;SSE42: cost of 1 {{.*}} fcmp
  ;AVX:   cost of 1 {{.*}} fcmp
  %A = fcmp olt <2 x float> undef, undef

  ;SSE2:  cost of 7 {{.*}} fcmp
  ;SSE3:  cost of 7 {{.*}} fcmp
  ;SSSE3: cost of 7 {{.*}} fcmp
  ;SSE41: cost of 7 {{.*}} fcmp
  ;SSE42: cost of 1 {{.*}} fcmp
  ;AVX:   cost of 1 {{.*}} fcmp
  %B = fcmp olt <4 x float> undef, undef

  ;SSE2:  cost of 14 {{.*}} fcmp
  ;SSE3:  cost of 14 {{.*}} fcmp
  ;SSSE3: cost of 14 {{.*}} fcmp
  ;SSE41: cost of 14 {{.*}} fcmp
  ;SSE42: cost of 2 {{.*}} fcmp
  ;AVX:   cost of 1 {{.*}} fcmp
  %C = fcmp olt <8 x float> undef, undef

  ;SSE2:  cost of 3 {{.*}} fcmp
  ;SSE3:  cost of 3 {{.*}} fcmp
  ;SSSE3: cost of 3 {{.*}} fcmp
  ;SSE41: cost of 3 {{.*}} fcmp
  ;SSE42: cost of 1 {{.*}} fcmp
  ;AVX:   cost of 1 {{.*}} fcmp
  %D = fcmp olt <2 x double> undef, undef

  ;SSE2:  cost of 6 {{.*}} fcmp
  ;SSE3:  cost of 6 {{.*}} fcmp
  ;SSSE3: cost of 6 {{.*}} fcmp
  ;SSE41: cost of 6 {{.*}} fcmp
  ;SSE42: cost of 2 {{.*}} fcmp
  ;AVX:   cost of 1 {{.*}} fcmp
  %E = fcmp olt <4 x double> undef, undef

  ; AVX512: cost of 1 {{.*}} %E1 = fcmp
  %E1 = fcmp olt <16 x float> undef, undef

  ; AVX512: cost of 1 {{.*}} %E2 = fcmp
  %E2 = fcmp olt <8 x double> undef, undef

  ; AVX512: cost of 2 {{.*}} %E3 = fcmp
  %E3 = fcmp olt <16 x double> undef, undef

  ;  -- integers --

  ;SSE2:  cost of 1 {{.*}} icmp
  ;SSE3:  cost of 1 {{.*}} icmp
  ;SSSE3: cost of 1 {{.*}} icmp
  ;SSE41: cost of 1 {{.*}} icmp
  ;SSE42: cost of 1 {{.*}} icmp
  ;AVX:   cost of 1 {{.*}} icmp
  %F = icmp eq <16 x i8> undef, undef

  ;SSE2:  cost of 1 {{.*}} icmp
  ;SSE3:  cost of 1 {{.*}} icmp
  ;SSSE3: cost of 1 {{.*}} icmp
  ;SSE41: cost of 1 {{.*}} icmp
  ;SSE42: cost of 1 {{.*}} icmp
  ;AVX:   cost of 1 {{.*}} icmp
  %G = icmp eq <8 x i16> undef, undef

  ;SSE2:  cost of 1 {{.*}} icmp
  ;SSE3:  cost of 1 {{.*}} icmp
  ;SSSE3: cost of 1 {{.*}} icmp
  ;SSE41: cost of 1 {{.*}} icmp
  ;SSE42: cost of 1 {{.*}} icmp
  ;AVX:   cost of 1 {{.*}} icmp
  %H = icmp eq <4 x i32> undef, undef

  ;SSE2:  cost of 8 {{.*}} icmp
  ;SSE3:  cost of 8 {{.*}} icmp
  ;SSSE3: cost of 8 {{.*}} icmp
  ;SSE41: cost of 8 {{.*}} icmp
  ;SSE42: cost of 1 {{.*}} icmp
  ;AVX:   cost of 1 {{.*}} icmp
  %I = icmp eq <2 x i64> undef, undef

  ;SSE2:  cost of 16 {{.*}} icmp
  ;SSE3:  cost of 16 {{.*}} icmp
  ;SSSE3: cost of 16 {{.*}} icmp
  ;SSE41: cost of 16 {{.*}} icmp
  ;SSE42: cost of 2 {{.*}} icmp
  ;AVX1:  cost of 4 {{.*}} icmp
  ;AVX2:  cost of 1 {{.*}} icmp
  %J = icmp eq <4 x i64> undef, undef

  ;SSE2:  cost of 2 {{.*}} icmp
  ;SSE3:  cost of 2 {{.*}} icmp
  ;SSSE3: cost of 2 {{.*}} icmp
  ;SSE41: cost of 2 {{.*}} icmp
  ;SSE42: cost of 2 {{.*}} icmp
  ;AVX1:  cost of 4 {{.*}} icmp
  ;AVX2:  cost of 1 {{.*}} icmp
  %K = icmp eq <8 x i32> undef, undef

  ;SSE2:  cost of 2 {{.*}} icmp
  ;SSE3:  cost of 2 {{.*}} icmp
  ;SSSE3: cost of 2 {{.*}} icmp
  ;SSE41: cost of 2 {{.*}} icmp
  ;SSE42: cost of 2 {{.*}} icmp
  ;AVX1:  cost of 4 {{.*}} icmp
  ;AVX2:  cost of 1 {{.*}} icmp
  %L = icmp eq <16 x i16> undef, undef

  ;SSE2:  cost of 2 {{.*}} icmp
  ;SSE3:  cost of 2 {{.*}} icmp
  ;SSSE3: cost of 2 {{.*}} icmp
  ;SSE41: cost of 2 {{.*}} icmp
  ;SSE42: cost of 2 {{.*}} icmp
  ;AVX1:  cost of 4 {{.*}} icmp
  ;AVX2:  cost of 1 {{.*}} icmp
  %M = icmp eq <32 x i8> undef, undef

  ; AVX512: cost of 1 {{.*}} %M1 = icmp
  %M1 = icmp eq <16 x i32> undef, undef

  ; AVX512: cost of 1 {{.*}} %M2 = icmp
  %M2 = icmp eq <8 x i64> undef, undef

  ; AVX512: cost of 2 {{.*}} %M3 = icmp
  %M3 = icmp eq <16 x i64> undef, undef

  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}


