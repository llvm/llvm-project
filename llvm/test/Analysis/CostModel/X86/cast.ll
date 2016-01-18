; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=knl | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AVX512
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core-avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AVX2
; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AVX

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @add(i32 %arg) {
; CHECK-LABEL: for function 'add'
  ; -- Same size registeres --
  ;CHECK: cost of 1 {{.*}} zext
  %A = zext <4 x i1> undef to <4 x i32>
  ;CHECK: cost of 2 {{.*}} sext
  %B = sext <4 x i1> undef to <4 x i32>
  ;CHECK: cost of 0 {{.*}} trunc
  %C = trunc <4 x i32> undef to <4 x i1>

  ; -- Different size registers --
  ;CHECK-NOT: cost of 1 {{.*}} zext
  %D = zext <8 x i1> undef to <8 x i32>
  ;CHECK-NOT: cost of 2 {{.*}} sext
  %E = sext <8 x i1> undef to <8 x i32>
  ;CHECK-NOT: cost of 2 {{.*}} trunc
  %F = trunc <8 x i32> undef to <8 x i1>

  ; -- scalars --

  ;CHECK: cost of 1 {{.*}} zext
  %G = zext i1 undef to i32
  ;CHECK: cost of 0 {{.*}} trunc
  %H = trunc i32 undef to i1

  ;CHECK: cost of 0 {{.*}} ret
  ret i32 undef
}

define i32 @zext_sext(<8 x i1> %in) {
; CHECK-AVX2-LABEL: for function 'zext_sext'
; CHECK-AVX-LABEL: for function 'zext_sext'
  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %Z = zext <8 x i1> %in to <8 x i32>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 7 {{.*}} sext
  %S = sext <8 x i1> %in to <8 x i32>

  ;CHECK-AVX2: cost of 1 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %A1 = zext <16 x i8> undef to <16 x i16>
  ;CHECK-AVX2: cost of 1 {{.*}} sext
  ;CHECK-AVX: cost of 4 {{.*}} sext
  %A2 = sext <16 x i8> undef to <16 x i16>
  ;CHECK-AVX2: cost of 1 {{.*}} sext
  ;CHECK-AVX: cost of 4 {{.*}} sext
  %A = sext <8 x i16> undef to <8 x i32>
  ;CHECK-AVX2: cost of 1 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %B = zext <8 x i16> undef to <8 x i32>
  ;CHECK-AVX2: cost of 1 {{.*}} sext
  ;CHECK-AVX: cost of 4 {{.*}} sext
  %C = sext <4 x i32> undef to <4 x i64>

  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %C.v8i8.z = zext <8 x i8> undef to <8 x i32>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 7 {{.*}} sext
  %C.v8i8.s = sext <8 x i8> undef to <8 x i32>
  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 3 {{.*}} zext
  %C.v4i16.z = zext <4 x i16> undef to <4 x i64>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 6 {{.*}} sext
  %C.v4i16.s = sext <4 x i16> undef to <4 x i64>

  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %C.v4i8.z = zext <4 x i8> undef to <4 x i64>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 6 {{.*}} sext
  %C.v4i8.s = sext <4 x i8> undef to <4 x i64>

  ;CHECK-AVX2: cost of 1 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %D = zext <4 x i32> undef to <4 x i64>

  ;CHECK-AVX512: cost of 3 {{.*}} %D1 = zext
  %D1 = zext <16 x i32> undef to <16 x i64>

  ;CHECK-AVX512: cost of 3 {{.*}} %D2 = sext
  %D2 = sext <16 x i32> undef to <16 x i64>

  ;CHECK-AVX512: cost of 1 {{.*}} %D3 = zext
  %D3 = zext <16 x i16> undef to <16 x i32>
  ;CHECK-AVX512: cost of 1 {{.*}} %D4 = zext
  %D4 = zext <16 x i8> undef to <16 x i32>
  ;CHECK-AVX512: cost of 2 {{.*}} %D5 = zext
  %D5 = zext <16 x i1> undef to <16 x i32>

  ;CHECK-AVX2: cost of 2 {{.*}} trunc
  ;CHECK-AVX: cost of 4 {{.*}} trunc
  %E = trunc <4 x i64> undef to <4 x i32>
  ;CHECK-AVX2: cost of 2 {{.*}} trunc
  ;CHECK-AVX: cost of 5 {{.*}} trunc
  %F = trunc <8 x i32> undef to <8 x i16>
  ;CHECK-AVX2: cost of 4 {{.*}} trunc
  ;CHECK-AVX: cost of 4 {{.*}} trunc
  %F1 = trunc <16 x i16> undef to <16 x i8>
  ;CHECK-AVX2: cost of 2 {{.*}} trunc
  ;CHECK-AVX: cost of 4 {{.*}} trunc
  %F2 = trunc <8 x i32> undef to <8 x i8>
  ;CHECK-AVX2: cost of 2 {{.*}} trunc
  ;CHECK-AVX: cost of 4 {{.*}} trunc
  %F3 = trunc <4 x i64> undef to <4 x i8>

  ;CHECK-AVX2: cost of 4 {{.*}} trunc
  ;CHECK-AVX: cost of 9 {{.*}} trunc
  ;CHECK_AVX512: cost of 1 {{.*}} G = trunc
  %G = trunc <8 x i64> undef to <8 x i32>

  ;CHECK-AVX512: cost of 4 {{.*}} %G1 = trunc
  %G1 = trunc <16 x i64> undef to <16 x i32>

  ret i32 undef
}

define i32 @masks8(<8 x i1> %in) {
; CHECK-AVX2-LABEL: for function 'masks8'
; CHECK-AVX-LABEL: for function 'masks8'

  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %Z = zext <8 x i1> %in to <8 x i32>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 7 {{.*}} sext
  %S = sext <8 x i1> %in to <8 x i32>
  ret i32 undef
}

define i32 @masks4(<4 x i1> %in) {
; CHECK-AVX2-LABEL: for function 'masks4'
; CHECK-AVX-LABEL: for function 'masks4'

  ;CHECK-AVX2: cost of 3 {{.*}} zext
  ;CHECK-AVX: cost of 4 {{.*}} zext
  %Z = zext <4 x i1> %in to <4 x i64>
  ;CHECK-AVX2: cost of 3 {{.*}} sext
  ;CHECK-AVX: cost of 6 {{.*}} sext
  %S = sext <4 x i1> %in to <4 x i64>
  ret i32 undef
}

define void @sitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
; CHECK-LABEL: for function 'sitofp4'
  ; CHECK: cost of 3 {{.*}} sitofp
  %A1 = sitofp <4 x i1> %a to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %A2 = sitofp <4 x i1> %a to <4 x double>

  ; CHECK: cost of 3 {{.*}} sitofp
  %B1 = sitofp <4 x i8> %b to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %B2 = sitofp <4 x i8> %b to <4 x double>

  ; CHECK: cost of 3 {{.*}} sitofp
  %C1 = sitofp <4 x i16> %c to <4 x float>
  ; CHECK: cost of 3 {{.*}} sitofp
  %C2 = sitofp <4 x i16> %c to <4 x double>

  ; CHECK: cost of 1 {{.*}} sitofp
  %D1 = sitofp <4 x i32> %d to <4 x float>
  ; CHECK: cost of 1 {{.*}} sitofp
  %D2 = sitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @sitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
; CHECK-LABEL: for function 'sitofp8'
  ; CHECK: cost of 8 {{.*}} sitofp
  %A1 = sitofp <8 x i1> %a to <8 x float>

  ; CHECK: cost of 8 {{.*}} sitofp
  %B1 = sitofp <8 x i8> %b to <8 x float>

  ; CHECK: cost of 5 {{.*}} sitofp
  %C1 = sitofp <8 x i16> %c to <8 x float>

  ; CHECK: cost of 1 {{.*}} sitofp
  %D1 = sitofp <8 x i32> %d to <8 x float>
  ret void
}

define void @uitofp4(<4 x i1> %a, <4 x i8> %b, <4 x i16> %c, <4 x i32> %d) {
; CHECK-LABEL: for function 'uitofp4'
  ; CHECK: cost of 7 {{.*}} uitofp
  %A1 = uitofp <4 x i1> %a to <4 x float>
  ; CHECK: cost of 7 {{.*}} uitofp
  %A2 = uitofp <4 x i1> %a to <4 x double>

  ; CHECK: cost of 2 {{.*}} uitofp
  %B1 = uitofp <4 x i8> %b to <4 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %B2 = uitofp <4 x i8> %b to <4 x double>

  ; CHECK: cost of 2 {{.*}} uitofp
  %C1 = uitofp <4 x i16> %c to <4 x float>
  ; CHECK: cost of 2 {{.*}} uitofp
  %C2 = uitofp <4 x i16> %c to <4 x double>

  ; CHECK: cost of 6 {{.*}} uitofp
  %D1 = uitofp <4 x i32> %d to <4 x float>
  ; CHECK: cost of 6 {{.*}} uitofp
  %D2 = uitofp <4 x i32> %d to <4 x double>
  ret void
}

define void @uitofp8(<8 x i1> %a, <8 x i8> %b, <8 x i16> %c, <8 x i32> %d) {
; CHECK-LABEL: for function 'uitofp8'
  ; CHECK: cost of 6 {{.*}} uitofp
  %A1 = uitofp <8 x i1> %a to <8 x float>

  ; CHECK: cost of 5 {{.*}} uitofp
  %B1 = uitofp <8 x i8> %b to <8 x float>

  ; CHECK: cost of 5 {{.*}} uitofp
  %C1 = uitofp <8 x i16> %c to <8 x float>

  ; CHECK-AVX2: cost of 8 {{.*}} uitofp
  ; CHECK-AVX512: cost of 8 {{.*}} uitofp
  ; CHECK-AVX: cost of 9 {{.*}} uitofp
  %D1 = uitofp <8 x i32> %d to <8 x float>
  ret void
}

define void @fp_conv(<8 x float> %a, <16 x float>%b) {
;CHECK-LABEL: for function 'fp_conv'
  ; CHECK-AVX512: cost of 1 {{.*}} fpext
  %A1 = fpext <8 x float> %a to <8 x double>

  ; CHECK-AVX512: cost of 3 {{.*}} fpext
  %A2 = fpext <16 x float> %b to <16 x double>

  ; CHECK-AVX2:   cost of 3 {{.*}} %A3 = fpext
  ; CHECK-AVX512: cost of 1 {{.*}} %A3 = fpext
  %A3 = fpext <8 x float> %a to <8 x double>

  ; CHECK-AVX2:   cost of 3 {{.*}} %A4 = fptrunc
  ; CHECK-AVX512: cost of 1 {{.*}} %A4 = fptrunc
  %A4 = fptrunc <8 x double> undef to <8 x float>

  ; CHECK-AVX512: cost of 3 {{.*}} %A5 = fptrunc
  %A5 = fptrunc <16 x double> undef to <16 x float>
  ret void
}
