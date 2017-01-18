; RUN: opt < %s -cost-model -analyze -mcpu=slm | FileCheck %s --check-prefix=SLM

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; 8bit mul
define i8 @slm-costs_8_scalar_mul(i8 %a, i8 %b)  {
entry:
; SLM:  cost of 1 {{.*}} mul nsw i8
  %res = mul nsw i8 %a, %b
  ret i8 %res
}

define <2 x i8> @slm-costs_8_v2_mul(<2 x i8> %a, <2 x i8> %b)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <2 x i8>
  %res = mul nsw <2 x i8> %a, %b
  ret <2 x i8> %res
}

define <4 x i8> @slm-costs_8_v4_mul(<4 x i8> %a, <4 x i8> %b)  {
entry:
; SLM:  cost of 3 {{.*}} mul nsw <4 x i8>
  %res = mul nsw <4 x i8> %a, %b
  ret <4 x i8> %res
}

define <4 x i32> @slm-costs_8_v4_zext_mul(<4 x i8> %a)  {
entry:
; SLM:  cost of 3 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i8> %a to <4 x i32> 
  %res = mul nsw <4 x i32> %zext, <i32 255, i32 255, i32 255, i32 255>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_8_v4_zext_mul_fail(<4 x i8> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i8> %a to <4 x i32>
  %res = mul nsw <4 x i32> %zext, <i32 255, i32 255, i32 -1, i32 255>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_8_v4_zext_mul_fail_2(<4 x i8> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i8> %a to <4 x i32>
  %res = mul nsw <4 x i32> %zext, <i32 255, i32 256, i32 255, i32 255>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_8_v4_sext_mul(<4 x i8> %a)  {
entry:
; SLM:  cost of 3 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i8> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 127, i32 -128, i32 127, i32 -128>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_8_v4_sext_mul_fail(<4 x i8> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i8> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 127, i32 -128, i32 128, i32 -128>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_8_v4_sext_mul_fail_2(<4 x i8> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i8> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 127, i32 -129, i32 127, i32 -128>
  ret <4 x i32> %res
}

define <8 x i8> @slm-costs_8_v8_mul(<8 x i8> %a, <8 x i8> %b)  {
entry:
; SLM:  cost of 2 {{.*}} mul nsw <8 x i8>
  %res = mul nsw <8 x i8> %a, %b
  ret <8 x i8> %res
}

define <16 x i8> @slm-costs_8_v16_mul(<16 x i8> %a, <16 x i8> %b)  {
entry:
; SLM:  cost of 14 {{.*}} mul nsw <16 x i8>
  %res = mul nsw <16 x i8> %a, %b
  ret <16 x i8> %res
}

; 16bit mul
define i16 @slm-costs_16_scalar_mul(i16 %a, i16 %b)  {
entry:
; SLM:  cost of 1 {{.*}} mul nsw i16
  %res = mul nsw i16 %a, %b
  ret i16 %res
}

define <2 x i16> @slm-costs_16_v2_mul(<2 x i16> %a, <2 x i16> %b)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <2 x i16>
  %res = mul nsw <2 x i16> %a, %b
  ret <2 x i16> %res
}

define <4 x i16> @slm-costs_16_v4_mul(<4 x i16> %a, <4 x i16> %b)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i16>
  %res = mul nsw <4 x i16> %a, %b
  ret <4 x i16> %res
}

define <4 x i32> @slm-costs_16_v4_zext_mul(<4 x i16> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %zext, <i32 65535, i32 65535, i32 65535, i32 65535>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_16_v4_zext_mul_fail(<4 x i16> %a)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %zext, <i32 -1, i32 65535, i32 65535, i32 65535>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_16_v4_zext_mul_fail_2(<4 x i16> %a)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <4 x i32>
  %zext = zext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %zext, <i32 65536, i32 65535, i32 65535, i32 65535>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_16_v4_sext_mul(<4 x i16> %a)  {
entry:
; SLM:  cost of 5 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 32767, i32 -32768, i32 32767, i32 -32768>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_16_v4_sext_mul_fail(<4 x i16> %a)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 32767, i32 -32768, i32 32768, i32 -32768>
  ret <4 x i32> %res
}

define <4 x i32> @slm-costs_16_v4_sext_mul_fail_2(<4 x i16> %a)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <4 x i32>
  %sext = sext <4 x i16> %a to <4 x i32>
  %res = mul nsw <4 x i32> %sext, <i32 32767, i32 -32768, i32 32767, i32 -32769>
  ret <4 x i32> %res
}

define <8 x i16> @slm-costs_16_v8_mul(<8 x i16> %a, <8 x i16> %b)  {
entry:
; SLM:  cost of 2 {{.*}} mul nsw <8 x i16>
  %res = mul nsw <8 x i16> %a, %b
  ret <8 x i16> %res
}

define <16 x i16> @slm-costs_16_v16_mul(<16 x i16> %a, <16 x i16> %b)  {
entry:
; SLM:  cost of 4 {{.*}} mul nsw <16 x i16>
  %res = mul nsw <16 x i16> %a, %b
  ret <16 x i16> %res
}

; 32bit mul
define i32 @slm-costs_32_scalar_mul(i32 %a, i32 %b)  {
entry:
; SLM:  cost of 1 {{.*}} mul nsw i32
  %res = mul nsw i32 %a, %b
  ret i32 %res 
}

define <2 x i32> @slm-costs_32_v2_mul(<2 x i32> %a, <2 x i32> %b)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <2 x i32>
  %res = mul nsw <2 x i32> %a, %b
  ret <2 x i32> %res
}

define <4 x i32> @slm-costs_32_v4_mul(<4 x i32> %a, <4 x i32> %b)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <4 x i32>
  %res = mul nsw <4 x i32> %a, %b
  ret <4 x i32> %res
}

define <8 x i32> @slm-costs_32_v8_mul(<8 x i32> %a, <8 x i32> %b)  {
entry:
; SLM:  cost of 22 {{.*}} mul nsw <8 x i32>
  %res = mul nsw <8 x i32> %a, %b
  ret <8 x i32> %res
}

define <16 x i32> @slm-costs_32_v16_mul(<16 x i32> %a, <16 x i32> %b)  {
entry:
; SLM:  cost of 44 {{.*}} mul nsw <16 x i32>
  %res = mul nsw <16 x i32> %a, %b
  ret <16 x i32> %res
}

; 64bit mul
define i64 @slm-costs_64_scalar_mul(i64 %a, i64 %b)  {
entry:
; SLM:  cost of 1 {{.*}} mul nsw i64
  %res = mul nsw i64 %a, %b
  ret i64 %res
}

define <2 x i64> @slm-costs_64_v2_mul(<2 x i64> %a, <2 x i64> %b)  {
entry:
; SLM:  cost of 11 {{.*}} mul nsw <2 x i64>
  %res = mul nsw <2 x i64> %a, %b
  ret <2 x i64> %res
}

define <4 x i64> @slm-costs_64_v4_mul(<4 x i64> %a, <4 x i64> %b)  {
entry:
; SLM:  cost of 22 {{.*}} mul nsw <4 x i64>
  %res = mul nsw <4 x i64> %a, %b
  ret <4 x i64> %res
}

define <8 x i64> @slm-costs_64_v8_mul(<8 x i64> %a, <8 x i64> %b)  {
entry:
; SLM:  cost of 44 {{.*}} mul nsw <8 x i64>
  %res = mul nsw <8 x i64> %a, %b
  ret <8 x i64> %res
}

define <16 x i64> @slm-costs_64_v16_mul(<16 x i64> %a, <16 x i64> %b)  {
entry:
; SLM:  cost of 88 {{.*}} mul nsw <16 x i64>
  %res = mul nsw <16 x i64> %a, %b
  ret <16 x i64> %res
}

; mulsd
define double @slm-costs_mulsd(double %a, double %b)  {
entry:
; SLM:  cost of 2 {{.*}} fmul double
  %res = fmul double %a, %b
  ret double %res
}

; mulpd
define <2 x double> @slm-costs_mulpd(<2 x double> %a, <2 x double> %b)  {
entry:
; SLM:  cost of 4 {{.*}} fmul <2 x double>
  %res = fmul <2 x double> %a, %b
  ret <2 x double> %res
}

; mulps
define <4 x float> @slm-costs_mulps(<4 x float> %a, <4 x float> %b)  {
entry:
; SLM:  cost of 2 {{.*}} fmul <4 x float>
  %res = fmul <4 x float> %a, %b
  ret <4 x float> %res
}

; divss
define float @slm-costs_divss(float %a, float %b)  {
entry:
; SLM:  cost of 17 {{.*}} fdiv float
  %res = fdiv float %a, %b
  ret float %res
}

; divps
define <4 x float> @slm-costs_divps(<4 x float> %a, <4 x float> %b)  {
entry:
; SLM:  cost of 39 {{.*}} fdiv <4 x float>
  %res = fdiv <4 x float> %a, %b
  ret <4 x float> %res
}

; divsd
define double @slm-costs_divsd(double %a, double %b)  {
entry:
; SLM:  cost of 32 {{.*}} fdiv double
  %res = fdiv double %a, %b
  ret double %res
}

; divpd
define <2 x double> @slm-costs_divpd(<2 x double> %a, <2 x double> %b)  {
entry:
; SLM:  cost of 69 {{.*}} fdiv <2 x double>
  %res = fdiv <2 x double> %a, %b
  ret <2 x double> %res
}

; addpd
define <2 x double> @slm-costs_addpd(<2 x double> %a, <2 x double> %b)  {
entry:
; SLM:  cost of 2 {{.*}} fadd <2 x double>
  %res = fadd <2 x double> %a, %b
  ret <2 x double> %res
}

; subpd
define <2 x double> @slm-costs_subpd(<2 x double> %a, <2 x double> %b)  {
entry:
; SLM:  cost of 2 {{.*}} fsub <2 x double>
  %res = fsub <2 x double> %a, %b
  ret <2 x double> %res
}

