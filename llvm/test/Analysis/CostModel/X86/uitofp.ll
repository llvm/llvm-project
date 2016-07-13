; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512F %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512dq -cost-model -analyze < %s | FileCheck --check-prefix=AVX512DQ %s

define <2 x double> @uitofpv2i8v2double(<2 x i8> %a) {
  ; SSE2-LABEL: uitofpv2i8v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i8v2double
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i8v2double
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i8v2double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i8v4double(<4 x i8> %a) {
  ; SSE2-LABEL: uitofpv4i8v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i8v4double
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i8v4double
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i8v4double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i8> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i8v8double(<8 x i8> %a) {
  ; SSE2-LABEL: uitofpv8i8v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i8v8double
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i8v8double
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i8v8double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <8 x i8> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i8v16double(<16 x i8> %a) {
  ; SSE2-LABEL: uitofpv16i8v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i8v16double
  ; AVX1: cost of 11 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i8v16double
  ; AVX2: cost of 11 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i8v16double
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <16 x i8> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i8v32double(<32 x i8> %a) {
  ; SSE2-LABEL: uitofpv32i8v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i8v32double
  ; AVX1: cost of 23 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i8v32double
  ; AVX2: cost of 23 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i8v32double
  ; AVX512F: cost of 11 {{.*}} uitofp
  %1 = uitofp <32 x i8> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i16v2double(<2 x i16> %a) {
  ; SSE2-LABEL: uitofpv2i16v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i16v2double
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i16v2double
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i16v2double
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i16v4double(<4 x i16> %a) {
  ; SSE2-LABEL: uitofpv4i16v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i16v4double
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i16v4double
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i16v4double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i16> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i16v8double(<8 x i16> %a) {
  ; SSE2-LABEL: uitofpv8i16v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i16v8double
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i16v8double
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i16v8double
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <8 x i16> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i16v16double(<16 x i16> %a) {
  ; SSE2-LABEL: uitofpv16i16v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i16v16double
  ; AVX1: cost of 11 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i16v16double
  ; AVX2: cost of 11 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i16v16double
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <16 x i16> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i16v32double(<32 x i16> %a) {
  ; SSE2-LABEL: uitofpv32i16v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i16v32double
  ; AVX1: cost of 23 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i16v32double
  ; AVX2: cost of 23 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i16v32double
  ; AVX512F: cost of 11 {{.*}} uitofp
  %1 = uitofp <32 x i16> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i32v2double(<2 x i32> %a) {
  ; SSE2-LABEL: uitofpv2i32v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i32v2double
  ; AVX1: cost of 6 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i32v2double
  ; AVX2: cost of 6 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i32v2double
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i32v4double(<4 x i32> %a) {
  ; SSE2-LABEL: uitofpv4i32v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i32v4double
  ; AVX1: cost of 6 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i32v4double
  ; AVX2: cost of 6 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i32v4double
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i32v8double(<8 x i32> %a) {
  ; SSE2-LABEL: uitofpv8i32v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i32v8double
  ; AVX1: cost of 13 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i32v8double
  ; AVX2: cost of 13 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i32v8double
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <8 x i32> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i32v16double(<16 x i32> %a) {
  ; SSE2-LABEL: uitofpv16i32v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i32v16double
  ; AVX1: cost of 27 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i32v16double
  ; AVX2: cost of 27 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i32v16double
  ; AVX512F: cost of 3 {{.*}} uitofp
  %1 = uitofp <16 x i32> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i32v32double(<32 x i32> %a) {
  ; SSE2-LABEL: uitofpv32i32v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i32v32double
  ; AVX1: cost of 55 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i32v32double
  ; AVX2: cost of 55 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i32v32double
  ; AVX512F: cost of 7 {{.*}} uitofp
  %1 = uitofp <32 x i32> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @uitofpv2i64v2double(<2 x i64> %a) {
  ; SSE2-LABEL: uitofpv2i64v2double
  ; SSE2: cost of 20 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i64v2double
  ; AVX1: cost of 10 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i64v2double
  ; AVX2: cost of 10 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i64v2double
  ; AVX512F: cost of 5 {{.*}} uitofp
  ;
  ; AVX512DQ-LABEL: uitofpv2i64v2double
  ; AVX512DQ: cost of 1 {{.*}} uitofp
  %1 = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @uitofpv4i64v4double(<4 x i64> %a) {
  ; SSE2-LABEL: uitofpv4i64v4double
  ; SSE2: cost of 40 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i64v4double
  ; AVX1: cost of 20 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i64v4double
  ; AVX2: cost of 20 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i64v4double
  ; AVX512F: cost of 12 {{.*}} uitofp
  ;
  ; AVX512DQ-LABEL: uitofpv4i64v4double
  ; AVX512DQ: cost of 1 {{.*}} uitofp
  %1 = uitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @uitofpv8i64v8double(<8 x i64> %a) {
  ; SSE2-LABEL: uitofpv8i64v8double
  ; SSE2: cost of 80 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i64v8double
  ; AVX1: cost of 41 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i64v8double
  ; AVX2: cost of 41 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i64v8double
  ; AVX512F: cost of 26 {{.*}} uitofp
  ;
  ; AVX512DQ-LABEL: uitofpv8i64v8double
  ; AVX512DQ: cost of 1 {{.*}} uitofp
  %1 = uitofp <8 x i64> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @uitofpv16i64v16double(<16 x i64> %a) {
  ; SSE2-LABEL: uitofpv16i64v16double
  ; SSE2: cost of 160 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i64v16double
  ; AVX1: cost of 83 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i64v16double
  ; AVX2: cost of 83 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i64v16double
  ; AVX512F: cost of 53 {{.*}} uitofp
  ;
  ; AVX512DQ-LABEL: uitofpv16i64v16double
  ; AVX512DQ: cost of 3 {{.*}} uitofp
  %1 = uitofp <16 x i64> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @uitofpv32i64v32double(<32 x i64> %a) {
  ; SSE2-LABEL: uitofpv32i64v32double
  ; SSE2: cost of 320 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i64v32double
  ; AVX1: cost of 167 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i64v32double
  ; AVX2: cost of 167 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i64v32double
  ; AVX512F: cost of 107 {{.*}} uitofp
  ;
  ; AVX512DQ-LABEL: uitofpv32i64v32double
  ; AVX512DQ: cost of 2 {{.*}} uitofp
  %1 = uitofp <32 x i64> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x float> @uitofpv2i8v2float(<2 x i8> %a) {
  ; SSE2-LABEL: uitofpv2i8v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i8v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i8v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i8v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i8> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i8v4float(<4 x i8> %a) {
  ; SSE2-LABEL: uitofpv4i8v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i8v4float
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i8v4float
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i8v4float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i8> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i8v8float(<8 x i8> %a) {
  ; SSE2-LABEL: uitofpv8i8v8float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i8v8float
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i8v8float
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i8v8float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <8 x i8> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i8v16float(<16 x i8> %a) {
  ; SSE2-LABEL: uitofpv16i8v16float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i8v16float
  ; AVX1: cost of 11 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i8v16float
  ; AVX2: cost of 11 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i8v16float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <16 x i8> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i8v32float(<32 x i8> %a) {
  ; SSE2-LABEL: uitofpv32i8v32float
  ; SSE2: cost of 16 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i8v32float
  ; AVX1: cost of 23 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i8v32float
  ; AVX2: cost of 23 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i8v32float
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <32 x i8> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i16v2float(<2 x i16> %a) {
  ; SSE2-LABEL: uitofpv2i16v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i16v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i16v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i16v2float
  ; AVX512F: cost of 4 {{.*}} uitofp
  %1 = uitofp <2 x i16> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i16v4float(<4 x i16> %a) {
  ; SSE2-LABEL: uitofpv4i16v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i16v4float
  ; AVX1: cost of 2 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i16v4float
  ; AVX2: cost of 2 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i16v4float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i16v8float(<8 x i16> %a) {
  ; SSE2-LABEL: uitofpv8i16v8float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i16v8float
  ; AVX1: cost of 5 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i16v8float
  ; AVX2: cost of 5 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i16v8float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i16v16float(<16 x i16> %a) {
  ; SSE2-LABEL: uitofpv16i16v16float
  ; SSE2: cost of 30 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i16v16float
  ; AVX1: cost of 11 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i16v16float
  ; AVX2: cost of 11 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i16v16float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <16 x i16> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i16v32float(<32 x i16> %a) {
  ; SSE2-LABEL: uitofpv32i16v32float
  ; SSE2: cost of 60 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i16v32float
  ; AVX1: cost of 23 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i16v32float
  ; AVX2: cost of 23 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i16v32float
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <32 x i16> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i32v2float(<2 x i32> %a) {
  ; SSE2-LABEL: uitofpv2i32v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i32v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i32v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i32v2float
  ; AVX512F: cost of 2 {{.*}} uitofp
  %1 = uitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i32v4float(<4 x i32> %a) {
  ; SSE2-LABEL: uitofpv4i32v4float
  ; SSE2: cost of 8 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i32v4float
  ; AVX1: cost of 6 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i32v4float
  ; AVX2: cost of 6 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i32v4float
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i32v8float(<8 x i32> %a) {
  ; SSE2-LABEL: uitofpv8i32v8float
  ; SSE2: cost of 16 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i32v8float
  ; AVX1: cost of 9 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i32v8float
  ; AVX2: cost of 8 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i32v8float
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i32v16float(<16 x i32> %a) {
  ; SSE2-LABEL: uitofpv16i32v16float
  ; SSE2: cost of 32 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i32v16float
  ; AVX1: cost of 19 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i32v16float
  ; AVX2: cost of 17 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i32v16float
  ; AVX512F: cost of 1 {{.*}} uitofp
  %1 = uitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i32v32float(<32 x i32> %a) {
  ; SSE2-LABEL: uitofpv32i32v32float
  ; SSE2: cost of 64 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i32v32float
  ; AVX1: cost of 39 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i32v32float
  ; AVX2: cost of 35 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i32v32float
  ; AVX512F: cost of 3 {{.*}} uitofp
  %1 = uitofp <32 x i32> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @uitofpv2i64v2float(<2 x i64> %a) {
  ; SSE2-LABEL: uitofpv2i64v2float
  ; SSE2: cost of 15 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv2i64v2float
  ; AVX1: cost of 4 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv2i64v2float
  ; AVX2: cost of 4 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv2i64v2float
  ; AVX512F: cost of 5 {{.*}} uitofp
  %1 = uitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @uitofpv4i64v4float(<4 x i64> %a) {
  ; SSE2-LABEL: uitofpv4i64v4float
  ; SSE2: cost of 30 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv4i64v4float
  ; AVX1: cost of 10 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv4i64v4float
  ; AVX2: cost of 10 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv4i64v4float
  ; AVX512F: cost of 10 {{.*}} uitofp
  %1 = uitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @uitofpv8i64v8float(<8 x i64> %a) {
  ; SSE2-LABEL: uitofpv8i64v8float
  ; SSE2: cost of 60 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv8i64v8float
  ; AVX1: cost of 21 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv8i64v8float
  ; AVX2: cost of 21 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv8i64v8float
  ; AVX512F: cost of 26 {{.*}} uitofp
  %1 = uitofp <8 x i64> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @uitofpv16i64v16float(<16 x i64> %a) {
  ; SSE2-LABEL: uitofpv16i64v16float
  ; SSE2: cost of 120 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv16i64v16float
  ; AVX1: cost of 43 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv16i64v16float
  ; AVX2: cost of 43 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv16i64v16float
  ; AVX512F: cost of 53 {{.*}} uitofp
  %1 = uitofp <16 x i64> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @uitofpv32i64v32float(<32 x i64> %a) {
  ; SSE2-LABEL: uitofpv32i64v32float
  ; SSE2: cost of 240 {{.*}} uitofp
  ;
  ; AVX1-LABEL: uitofpv32i64v32float
  ; AVX1: cost of 87 {{.*}} uitofp
  ;
  ; AVX2-LABEL: uitofpv32i64v32float
  ; AVX2: cost of 87 {{.*}} uitofp
  ;
  ; AVX512F-LABEL: uitofpv32i64v32float
  ; AVX512F: cost of 107 {{.*}} uitofp
  %1 = uitofp <32 x i64> %a to <32 x float>
  ret <32 x float> %1
}

define <8 x i32> @fptouiv8f32v8i32(<8 x float> %a) {
  ; AVX512F-LABEL: fptouiv8f32v8i32
  ; AVX512F: cost of 1 {{.*}} fptoui
  %1 = fptoui <8 x float> %a to <8 x i32>
  ret <8 x i32> %1
}

define <4 x i32> @fptouiv4f32v4i32(<4 x float> %a) {
  ; AVX512F-LABEL: fptouiv4f32v4i32
  ; AVX512F: cost of 1 {{.*}} fptoui
  %1 = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %1
}

define <2 x i32> @fptouiv2f32v2i32(<2 x float> %a) {
  ; AVX512F-LABEL: fptouiv2f32v2i32
  ; AVX512F: cost of 1 {{.*}} fptoui
  %1 = fptoui <2 x float> %a to <2 x i32>
  ret <2 x i32> %1
}

define <16 x i32> @fptouiv16f32v16i32(<16 x float> %a) {
  ; AVX512F-LABEL: fptouiv16f32v16i32
  ; AVX512F: cost of 1 {{.*}} fptoui
  %1 = fptoui <16 x float> %a to <16 x i32>
  ret <16 x i32> %1
}

define <8 x i64> @fptouiv8f32v8i64(<8 x float> %a) {
  ; AVX512DQ-LABEL: fptouiv8f32v8i64
  ; AVX512DQ: cost of 1 {{.*}} fptoui
  %1 = fptoui <8 x float> %a to <8 x i64>
  ret <8 x i64> %1
}

define <4 x i64> @fptouiv4f32v4i64(<4 x float> %a) {
  ; AVX512DQ-LABEL: fptouiv4f32v4i64
  ; AVX512DQ: cost of 1 {{.*}} fptoui
  %1 = fptoui <4 x float> %a to <4 x i64>
  ret <4 x i64> %1
}

define <2 x i64> @fptouiv2f32v2i64(<2 x float> %a) {
  ; AVX512DQ-LABEL: fptouiv2f32v2i64
  ; AVX512DQ: cost of 1 {{.*}} fptoui
  %1 = fptoui <2 x float> %a to <2 x i64>
  ret <2 x i64> %1
}
