; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx  -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX1 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx2 -cost-model -analyze < %s | FileCheck --check-prefix=AVX --check-prefix=AVX2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+avx512f -cost-model -analyze < %s | FileCheck --check-prefix=AVX512F %s

define <2 x double> @sitofpv2i8v2double(<2 x i8> %a) {
  ; SSE2-LABEL: sitofpv2i8v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i8v2double
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i8v2double
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i8v2double
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i8v4double(<4 x i8> %a) {
  ; SSE2-LABEL: sitofpv4i8v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i8v4double
  ; AVX1: cost of 3 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i8v4double
  ; AVX2: cost of 3 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i8v4double
  ; AVX512F: cost of 3 {{.*}} sitofp
  %1 = sitofp <4 x i8> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i8v8double(<8 x i8> %a) {
  ; SSE2-LABEL: sitofpv8i8v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i8v8double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i8v8double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i8v8double
  ; AVX512F: cost of 2 {{.*}} sitofp
  %1 = sitofp <8 x i8> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i8v16double(<16 x i8> %a) {
  ; SSE2-LABEL: sitofpv16i8v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i8v16double
  ; AVX1: cost of 40 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i8v16double
  ; AVX2: cost of 40 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i8v16double
  ; AVX512F: cost of 44 {{.*}} sitofp
  %1 = sitofp <16 x i8> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i8v32double(<32 x i8> %a) {
  ; SSE2-LABEL: sitofpv32i8v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i8v32double
  ; AVX1: cost of 80 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i8v32double
  ; AVX2: cost of 80 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i8v32double
  ; AVX512F: cost of 88 {{.*}} sitofp
  %1 = sitofp <32 x i8> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i16v2double(<2 x i16> %a) {
  ; SSE2-LABEL: sitofpv2i16v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i16v2double
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i16v2double
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i16v2double
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i16v4double(<4 x i16> %a) {
  ; SSE2-LABEL: sitofpv4i16v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i16v4double
  ; AVX1: cost of 3 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i16v4double
  ; AVX2: cost of 3 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i16v4double
  ; AVX512F: cost of 3 {{.*}} sitofp
  %1 = sitofp <4 x i16> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i16v8double(<8 x i16> %a) {
  ; SSE2-LABEL: sitofpv8i16v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i16v8double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i16v8double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i16v8double
  ; AVX512F: cost of 2 {{.*}} sitofp
  %1 = sitofp <8 x i16> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i16v16double(<16 x i16> %a) {
  ; SSE2-LABEL: sitofpv16i16v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i16v16double
  ; AVX1: cost of 40 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i16v16double
  ; AVX2: cost of 40 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i16v16double
  ; AVX512F: cost of 44 {{.*}} sitofp
  %1 = sitofp <16 x i16> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i16v32double(<32 x i16> %a) {
  ; SSE2-LABEL: sitofpv32i16v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i16v32double
  ; AVX1: cost of 80 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i16v32double
  ; AVX2: cost of 80 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i16v32double
  ; AVX512F: cost of 88 {{.*}} sitofp
  %1 = sitofp <32 x i16> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i32v2double(<2 x i32> %a) {
  ; SSE2-LABEL: sitofpv2i32v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i32v2double
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i32v2double
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i32v2double
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i32v4double(<4 x i32> %a) {
  ; SSE2-LABEL: sitofpv4i32v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i32v4double
  ; AVX1: cost of 1 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i32v4double
  ; AVX2: cost of 1 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i32v4double
  ; AVX512F: cost of 1 {{.*}} sitofp
  %1 = sitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i32v8double(<8 x i32> %a) {
  ; SSE2-LABEL: sitofpv8i32v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i32v8double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i32v8double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i32v8double
  ; AVX512F: cost of 1 {{.*}} sitofp
  %1 = sitofp <8 x i32> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i32v16double(<16 x i32> %a) {
  ; SSE2-LABEL: sitofpv16i32v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i32v16double
  ; AVX1: cost of 40 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i32v16double
  ; AVX2: cost of 40 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i32v16double
  ; AVX512F: cost of 44 {{.*}} sitofp
  %1 = sitofp <16 x i32> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i32v32double(<32 x i32> %a) {
  ; SSE2-LABEL: sitofpv32i32v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i32v32double
  ; AVX1: cost of 80 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i32v32double
  ; AVX2: cost of 80 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i32v32double
  ; AVX512F: cost of 88 {{.*}} sitofp
  %1 = sitofp <32 x i32> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i64v2double(<2 x i64> %a) {
  ; SSE2-LABEL: sitofpv2i64v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i64v2double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i64v2double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i64v2double
  ; AVX512F: cost of 20 {{.*}} sitofp
  %1 = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i64v4double(<4 x i64> %a) {
  ; SSE2-LABEL: sitofpv4i64v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i64v4double
  ; AVX1: cost of 10 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i64v4double
  ; AVX2: cost of 10 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i64v4double
  ; AVX512F: cost of 10 {{.*}} sitofp
  %1 = sitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i64v8double(<8 x i64> %a) {
  ; SSE2-LABEL: sitofpv8i64v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i64v8double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i64v8double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i64v8double
  ; AVX512F: cost of 22 {{.*}} sitofp
  %1 = sitofp <8 x i64> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i64v16double(<16 x i64> %a) {
  ; SSE2-LABEL: sitofpv16i64v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i64v16double
  ; AVX1: cost of 40 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i64v16double
  ; AVX2: cost of 40 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i64v16double
  ; AVX512F: cost of 44 {{.*}} sitofp
  %1 = sitofp <16 x i64> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i64v32double(<32 x i64> %a) {
  ; SSE2-LABEL: sitofpv32i64v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i64v32double
  ; AVX1: cost of 80 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i64v32double
  ; AVX2: cost of 80 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i64v32double
  ; AVX512F: cost of 88 {{.*}} sitofp
  %1 = sitofp <32 x i64> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x float> @sitofpv2i8v2float(<2 x i8> %a) {
  ; SSE2-LABEL: sitofpv2i8v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i8v2float
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i8v2float
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i8v2float
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i8> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i8v4float(<4 x i8> %a) {
  ; SSE2-LABEL: sitofpv4i8v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i8v4float
  ; AVX1: cost of 3 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i8v4float
  ; AVX2: cost of 3 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i8v4float
  ; AVX512F: cost of 3 {{.*}} sitofp
  %1 = sitofp <4 x i8> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i8v8float(<8 x i8> %a) {
  ; SSE2-LABEL: sitofpv8i8v8float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i8v8float
  ; AVX1: cost of 8 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i8v8float
  ; AVX2: cost of 8 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i8v8float
  ; AVX512F: cost of 8 {{.*}} sitofp
  %1 = sitofp <8 x i8> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i8v16float(<16 x i8> %a) {
  ; SSE2-LABEL: sitofpv16i8v16float
  ; SSE2: cost of 8 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i8v16float
  ; AVX1: cost of 44 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i8v16float
  ; AVX2: cost of 44 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i8v16float
  ; AVX512F: cost of 2 {{.*}} sitofp
  %1 = sitofp <16 x i8> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i8v32float(<32 x i8> %a) {
  ; SSE2-LABEL: sitofpv32i8v32float
  ; SSE2: cost of 16 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i8v32float
  ; AVX1: cost of 88 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i8v32float
  ; AVX2: cost of 88 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i8v32float
  ; AVX512F: cost of 92 {{.*}} sitofp
  %1 = sitofp <32 x i8> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i16v2float(<2 x i16> %a) {
  ; SSE2-LABEL: sitofpv2i16v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i16v2float
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i16v2float
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i16v2float
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i16> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i16v4float(<4 x i16> %a) {
  ; SSE2-LABEL: sitofpv4i16v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i16v4float
  ; AVX1: cost of 3 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i16v4float
  ; AVX2: cost of 3 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i16v4float
  ; AVX512F: cost of 3 {{.*}} sitofp
  %1 = sitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i16v8float(<8 x i16> %a) {
  ; SSE2-LABEL: sitofpv8i16v8float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i16v8float
  ; AVX1: cost of 5 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i16v8float
  ; AVX2: cost of 5 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i16v8float
  ; AVX512F: cost of 5 {{.*}} sitofp
  %1 = sitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i16v16float(<16 x i16> %a) {
  ; SSE2-LABEL: sitofpv16i16v16float
  ; SSE2: cost of 30 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i16v16float
  ; AVX1: cost of 44 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i16v16float
  ; AVX2: cost of 44 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i16v16float
  ; AVX512F: cost of 2 {{.*}} sitofp
  %1 = sitofp <16 x i16> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i16v32float(<32 x i16> %a) {
  ; SSE2-LABEL: sitofpv32i16v32float
  ; SSE2: cost of 60 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i16v32float
  ; AVX1: cost of 88 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i16v32float
  ; AVX2: cost of 88 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i16v32float
  ; AVX512F: cost of 92 {{.*}} sitofp
  %1 = sitofp <32 x i16> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i32v2float(<2 x i32> %a) {
  ; SSE2-LABEL: sitofpv2i32v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i32v2float
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i32v2float
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i32v2float
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i32v4float(<4 x i32> %a) {
  ; SSE2-LABEL: sitofpv4i32v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i32v4float
  ; AVX1: cost of 1 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i32v4float
  ; AVX2: cost of 1 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i32v4float
  ; AVX512F: cost of 1 {{.*}} sitofp
  %1 = sitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i32v8float(<8 x i32> %a) {
  ; SSE2-LABEL: sitofpv8i32v8float
  ; SSE2: cost of 30 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i32v8float
  ; AVX1: cost of 1 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i32v8float
  ; AVX2: cost of 1 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i32v8float
  ; AVX512F: cost of 1 {{.*}} sitofp
  %1 = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i32v16float(<16 x i32> %a) {
  ; SSE2-LABEL: sitofpv16i32v16float
  ; SSE2: cost of 60 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i32v16float
  ; AVX1: cost of 44 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i32v16float
  ; AVX2: cost of 44 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i32v16float
  ; AVX512F: cost of 1 {{.*}} sitofp
  %1 = sitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i32v32float(<32 x i32> %a) {
  ; SSE2-LABEL: sitofpv32i32v32float
  ; SSE2: cost of 120 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i32v32float
  ; AVX1: cost of 88 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i32v32float
  ; AVX2: cost of 88 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i32v32float
  ; AVX512F: cost of 92 {{.*}} sitofp
  %1 = sitofp <32 x i32> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i64v2float(<2 x i64> %a) {
  ; SSE2-LABEL: sitofpv2i64v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv2i64v2float
  ; AVX1: cost of 4 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv2i64v2float
  ; AVX2: cost of 4 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv2i64v2float
  ; AVX512F: cost of 4 {{.*}} sitofp
  %1 = sitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i64v4float(<4 x i64> %a) {
  ; SSE2-LABEL: sitofpv4i64v4float
  ; SSE2: cost of 30 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv4i64v4float
  ; AVX1: cost of 10 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv4i64v4float
  ; AVX2: cost of 10 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv4i64v4float
  ; AVX512F: cost of 10 {{.*}} sitofp
  %1 = sitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i64v8float(<8 x i64> %a) {
  ; SSE2-LABEL: sitofpv8i64v8float
  ; SSE2: cost of 60 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i64v8float
  ; AVX1: cost of 22 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i64v8float
  ; AVX2: cost of 22 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i64v8float
  ; AVX512F: cost of 22 {{.*}} sitofp
  %1 = sitofp <8 x i64> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i64v16float(<16 x i64> %a) {
  ; SSE2-LABEL: sitofpv16i64v16float
  ; SSE2: cost of 120 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i64v16float
  ; AVX1: cost of 44 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i64v16float
  ; AVX2: cost of 44 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i64v16float
  ; AVX512F: cost of 46 {{.*}} sitofp
  %1 = sitofp <16 x i64> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i64v32float(<32 x i64> %a) {
  ; SSE2-LABEL: sitofpv32i64v32float
  ; SSE2: cost of 240 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv32i64v32float
  ; AVX1: cost of 88 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv32i64v32float
  ; AVX2: cost of 88 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv32i64v32float
  ; AVX512F: cost of 92 {{.*}} sitofp
  %1 = sitofp <32 x i64> %a to <32 x float>
  ret <32 x float> %1
}

define <8 x double> @sitofpv8i1v8double(<8 x double> %a) {
  ; SSE2-LABEL: sitofpv8i1v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv8i1v8double
  ; AVX1: cost of 20 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv8i1v8double
  ; AVX2: cost of 20 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv8i1v8double
  ; AVX512F: cost of 4 {{.*}} sitofp
  %cmpres = fcmp ogt <8 x double> %a, zeroinitializer
  %1 = sitofp <8 x i1> %cmpres to <8 x double>
  ret <8 x double> %1
}

define <16 x float> @sitofpv16i1v16float(<16 x float> %a) {
  ; SSE2-LABEL: sitofpv16i1v16float
  ; SSE2: cost of 8 {{.*}} sitofp
  ;
  ; AVX1-LABEL: sitofpv16i1v16float
  ; AVX1: cost of 44 {{.*}} sitofp
  ;
  ; AVX2-LABEL: sitofpv16i1v16float
  ; AVX2: cost of 44 {{.*}} sitofp
  ;
  ; AVX512F-LABEL: sitofpv16i1v16float
  ; AVX512F: cost of 3 {{.*}} sitofp
  %cmpres = fcmp ogt <16 x float> %a, zeroinitializer
  %1 = sitofp <16 x i1> %cmpres to <16 x float>
  ret <16 x float> %1
}
