; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512F %s

define <2 x double> @uitofpv2i8v2double(<2 x i8> %a) {
  ; SSE2: uitofpv2i8v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i8v2double
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i8v2double
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i8v2double
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i8v4double(<4 x i8> %a) {
  ; SSE2: uitofpv4i8v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i8v4double
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i8v4double
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i8v4double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i8> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i8v8double(<8 x i8> %a) {
  ; SSE2: uitofpv8i8v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i8v8double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i8v8double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i8v8double
  ; AVX512F: cost of 22 {{.*}} uitofp
  %1 = uitofp <8 x i8> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i8v16double(<16 x i8> %a) {
  ; SSE2: uitofpv16i8v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i8v16double
  ; AVX1: cost of 40 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i8v16double
  ; AVX2: cost of 40 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i8v16double
  ; AVX512F: cost of 44 {{.*}} uitofp
  %1 = uitofp <16 x i8> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i8v32double(<32 x i8> %a) {
  ; SSE2: uitofpv32i8v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i8v32double
  ; AVX1: cost of 80 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i8v32double
  ; AVX2: cost of 80 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i8v32double
  ; AVX512F: cost of 88 {{.*}} uitofp
  %1 = uitofp <32 x i8> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i16v2double(<2 x i16> %a) {
  ; SSE2: uitofpv2i16v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i16v2double
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i16v2double
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i16v2double
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i16v4double(<4 x i16> %a) {
  ; SSE2: uitofpv4i16v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i16v4double
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i16v4double
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i16v4double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i16> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i16v8double(<8 x i16> %a) {
  ; SSE2: uitofpv8i16v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i16v8double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i16v8double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i16v8double
  ; AVX512F: cost of 22 {{.*}} uitofp
  %1 = uitofp <8 x i16> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i16v16double(<16 x i16> %a) {
  ; SSE2: uitofpv16i16v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i16v16double
  ; AVX1: cost of 40 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i16v16double
  ; AVX2: cost of 40 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i16v16double
  ; AVX512F: cost of 44 {{.*}} uitofp
  %1 = uitofp <16 x i16> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i16v32double(<32 x i16> %a) {
  ; SSE2: uitofpv32i16v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i16v32double
  ; AVX1: cost of 80 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i16v32double
  ; AVX2: cost of 80 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i16v32double
  ; AVX512F: cost of 88 {{.*}} uitofp
  %1 = uitofp <32 x i16> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i32v2double(<2 x i32> %a) {
  ; SSE2: uitofpv2i32v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i32v2double
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i32v2double
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i32v2double
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i32v4double(<4 x i32> %a) {
  ; SSE2: uitofpv4i32v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i32v4double
  ; AVX1: cost of 6 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i32v4double
  ; AVX2: cost of 6 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i32v4double
  ; AVX512F: cost of 6 {{.*}} uitofp
  %1 = uitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i32v8double(<8 x i32> %a) {
  ; SSE2: uitofpv8i32v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i32v8double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i32v8double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i32v8double
  ; AVX512F: cost of 22 {{.*}} uitofp
  %1 = uitofp <8 x i32> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i32v16double(<16 x i32> %a) {
  ; SSE2: uitofpv16i32v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i32v16double
  ; AVX1: cost of 40 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i32v16double
  ; AVX2: cost of 40 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i32v16double
  ; AVX512F: cost of 44 {{.*}} uitofp
  %1 = uitofp <16 x i32> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i32v32double(<32 x i32> %a) {
  ; SSE2: uitofpv32i32v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i32v32double
  ; AVX1: cost of 80 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i32v32double
  ; AVX2: cost of 80 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i32v32double
  ; AVX512F: cost of 88 {{.*}} uitofp
  %1 = uitofp <32 x i32> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i64v2double(<2 x i64> %a) {
  ; SSE2: uitofpv2i64v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i64v2double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i64v2double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i64v2double
  ; AVX512F: cost of 20 {{.*}} uitofp
  %1 = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i64v4double(<4 x i64> %a) {
  ; SSE2: uitofpv4i64v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i64v4double
  ; AVX1: cost of 40 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i64v4double
  ; AVX2: cost of 40 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i64v4double
  ; AVX512F: cost of 40 {{.*}} uitofp
  %1 = uitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i64v8double(<8 x i64> %a) {
  ; SSE2: uitofpv8i64v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i64v8double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i64v8double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i64v8double
  ; AVX512F: cost of 22 {{.*}} uitofp
  %1 = uitofp <8 x i64> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i64v16double(<16 x i64> %a) {
  ; SSE2: uitofpv16i64v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i64v16double
  ; AVX1: cost of 40 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i64v16double
  ; AVX2: cost of 40 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i64v16double
  ; AVX512F: cost of 44 {{.*}} uitofp
  %1 = uitofp <16 x i64> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i64v32double(<32 x i64> %a) {
  ; SSE2: uitofpv32i64v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i64v32double
  ; AVX1: cost of 80 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i64v32double
  ; AVX2: cost of 80 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i64v32double
  ; AVX512F: cost of 88 {{.*}} uitofp
  %1 = uitofp <32 x i64> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x float> @uitofpv2i8v2float(<2 x i8> %a) {
  ; SSE2: uitofpv2i8v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i8v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i8v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i8v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i8> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i8v4float(<4 x i8> %a) {
  ; SSE2: uitofpv4i8v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i8v4float
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i8v4float
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i8v4float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i8> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i8v8float(<8 x i8> %a) {
  ; SSE2: uitofpv8i8v8float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i8v8float
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i8v8float
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i8v8float
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <8 x i8> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i8v16float(<16 x i8> %a) {
  ; SSE2: uitofpv16i8v16float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i8v16float
  ; AVX1: cost of 44 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i8v16float
  ; AVX2: cost of 44 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i8v16float
  ; AVX512F: cost of 46 {{.*}} uitofp
  %1 = uitofp <16 x i8> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i8v32float(<32 x i8> %a) {
  ; SSE2: uitofpv32i8v32float
  ; SSE2: cost of 16 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i8v32float
  ; AVX1: cost of 88 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i8v32float
  ; AVX2: cost of 88 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i8v32float
  ; AVX512F: cost of 92 {{.*}} uitofp
  %1 = uitofp <32 x i8> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i16v2float(<2 x i16> %a) {
  ; SSE2: uitofpv2i16v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i16v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i16v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i16v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i16> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i16v4float(<4 x i16> %a) {
  ; SSE2: uitofpv4i16v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i16v4float
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i16v4float
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i16v4float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i16v8float(<8 x i16> %a) {
  ; SSE2: uitofpv8i16v8float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i16v8float
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i16v8float
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i16v8float
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i16v16float(<16 x i16> %a) {
  ; SSE2: uitofpv16i16v16float
  ; SSE2: cost of 30 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i16v16float
  ; AVX1: cost of 44 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i16v16float
  ; AVX2: cost of 44 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i16v16float
  ; AVX512F: cost of 46 {{.*}} uitofp
  %1 = uitofp <16 x i16> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i16v32float(<32 x i16> %a) {
  ; SSE2: uitofpv32i16v32float
  ; SSE2: cost of 60 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i16v32float
  ; AVX1: cost of 88 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i16v32float
  ; AVX2: cost of 88 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i16v32float
  ; AVX512F: cost of 92 {{.*}} uitofp
  %1 = uitofp <32 x i16> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i32v2float(<2 x i32> %a) {
  ; SSE2: uitofpv2i32v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i32v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i32v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i32v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i32v4float(<4 x i32> %a) {
  ; SSE2: uitofpv4i32v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i32v4float
  ; AVX1: cost of 6 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i32v4float
  ; AVX2: cost of 6 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i32v4float
  ; AVX512F: cost of 6 {{.*}} uitofp
  %1 = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i32v8float(<8 x i32> %a) {
  ; SSE2: uitofpv8i32v8float
  ; SSE2: cost of 16 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i32v8float
  ; AVX1: cost of 9 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i32v8float
  ; AVX2: cost of 8 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i32v8float
  ; AVX512F: cost of 8 {{.*}} uitofp
  %1 = uitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i32v16float(<16 x i32> %a) {
  ; SSE2: uitofpv16i32v16float
  ; SSE2: cost of 32 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i32v16float
  ; AVX1: cost of 44 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i32v16float
  ; AVX2: cost of 44 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i32v16float
  ; AVX512F: cost of 46 {{.*}} uitofp
  %1 = uitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i32v32float(<32 x i32> %a) {
  ; SSE2: uitofpv32i32v32float
  ; SSE2: cost of 64 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i32v32float
  ; AVX1: cost of 88 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i32v32float
  ; AVX2: cost of 88 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i32v32float
  ; AVX512F: cost of 92 {{.*}} uitofp
  %1 = uitofp <32 x i32> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i64v2float(<2 x i64> %a) {
  ; SSE2: uitofpv2i64v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1: uitofpv2i64v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2: uitofpv2i64v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv2i64v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i64v4float(<4 x i64> %a) {
  ; SSE2: uitofpv4i64v4float
  ; SSE2: cost of 30 {{.*}} uitofp
  ;
  ; AVX1: uitofpv4i64v4float
  ; AVX1: cost of 10 {{.*}} uitofp
  ;
  ; AVX2: uitofpv4i64v4float
  ; AVX2: cost of 10 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv4i64v4float
  ; AVX512F: cost of 10 {{.*}} uitofp
  %1 = uitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i64v8float(<8 x i64> %a) {
  ; SSE2: uitofpv8i64v8float
  ; SSE2: cost of 60 {{.*}} uitofp
  ;
  ; AVX1: uitofpv8i64v8float
  ; AVX1: cost of 22 {{.*}} uitofp
  ;
  ; AVX2: uitofpv8i64v8float
  ; AVX2: cost of 22 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv8i64v8float
  ; AVX512F: cost of 22 {{.*}} uitofp
  %1 = uitofp <8 x i64> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i64v16float(<16 x i64> %a) {
  ; SSE2: uitofpv16i64v16float
  ; SSE2: cost of 120 {{.*}} uitofp
  ;
  ; AVX1: uitofpv16i64v16float
  ; AVX1: cost of 44 {{.*}} uitofp
  ;
  ; AVX2: uitofpv16i64v16float
  ; AVX2: cost of 44 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv16i64v16float
  ; AVX512F: cost of 46 {{.*}} uitofp
  %1 = uitofp <16 x i64> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i64v32float(<32 x i64> %a) {
  ; SSE2: uitofpv32i64v32float
  ; SSE2: cost of 240 {{.*}} uitofp
  ;
  ; AVX1: uitofpv32i64v32float
  ; AVX1: cost of 88 {{.*}} uitofp
  ;
  ; AVX2: uitofpv32i64v32float
  ; AVX2: cost of 88 {{.*}} uitofp
  ;
  ; AVX512F: uitofpv32i64v32float
  ; AVX512F: cost of 92 {{.*}} uitofp
  %1 = uitofp <32 x i64> %a to <32 x float>
  ret <32 x float> %1
}

