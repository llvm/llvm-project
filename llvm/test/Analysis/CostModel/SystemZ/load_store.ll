; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

define void @store() {
  store i8 undef, i8* undef
  store i16 undef, i16* undef
  store i32 undef, i32* undef
  store i64 undef, i64* undef
  store float undef, float* undef
  store double undef, double* undef
  store fp128 undef, fp128* undef
  store <2 x i8> undef, <2 x i8>* undef
  store <2 x i16> undef, <2 x i16>* undef
  store <2 x i32> undef, <2 x i32>* undef
  store <2 x i64> undef, <2 x i64>* undef
  store <2 x float> undef, <2 x float>* undef
  store <2 x double> undef, <2 x double>* undef
  store <4 x i8> undef, <4 x i8>* undef
  store <4 x i16> undef, <4 x i16>* undef
  store <4 x i32> undef, <4 x i32>* undef
  store <4 x i64> undef, <4 x i64>* undef
  store <4 x float> undef, <4 x float>* undef
  store <4 x double> undef, <4 x double>* undef
  store <8 x i8> undef, <8 x i8>* undef
  store <8 x i16> undef, <8 x i16>* undef
  store <8 x i32> undef, <8 x i32>* undef
  store <8 x i64> undef, <8 x i64>* undef
  store <8 x float> undef, <8 x float>* undef
  store <8 x double> undef, <8 x double>* undef
  store <16 x i8> undef, <16 x i8>* undef
  store <16 x i16> undef, <16 x i16>* undef
  store <16 x i32> undef, <16 x i32>* undef
  store <16 x i64> undef, <16 x i64>* undef
  store <16 x float> undef, <16 x float>* undef
  store <16 x double> undef, <16 x double>* undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i8 undef, i8* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i16 undef, i16* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i32 undef, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i64 undef, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store float undef, float* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store double undef, double* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store fp128 undef, fp128* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i8> undef, <2 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i16> undef, <2 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i32> undef, <2 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i64> undef, <2 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x float> undef, <2 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x double> undef, <2 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i8> undef, <4 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i16> undef, <4 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i32> undef, <4 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <4 x i64> undef, <4 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x float> undef, <4 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <4 x double> undef, <4 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <8 x i8> undef, <8 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <8 x i16> undef, <8 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <8 x i32> undef, <8 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <8 x i64> undef, <8 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <8 x float> undef, <8 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <8 x double> undef, <8 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <16 x i8> undef, <16 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <16 x i16> undef, <16 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <16 x i32> undef, <16 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   store <16 x i64> undef, <16 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <16 x float> undef, <16 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   store <16 x double> undef, <16 x double>* undef

  ret void;
}

define void @load() {
  load i8, i8* undef
  load i16, i16* undef
  load i32, i32* undef
  load i64, i64* undef
  load float, float* undef
  load double, double* undef
  load fp128, fp128* undef
  load <2 x i8>, <2 x i8>* undef
  load <2 x i16>, <2 x i16>* undef
  load <2 x i32>, <2 x i32>* undef
  load <2 x i64>, <2 x i64>* undef
  load <2 x float>, <2 x float>* undef
  load <2 x double>, <2 x double>* undef
  load <4 x i8>, <4 x i8>* undef
  load <4 x i16>, <4 x i16>* undef
  load <4 x i32>, <4 x i32>* undef
  load <4 x i64>, <4 x i64>* undef
  load <4 x float>, <4 x float>* undef
  load <4 x double>, <4 x double>* undef
  load <8 x i8>, <8 x i8>* undef
  load <8 x i16>, <8 x i16>* undef
  load <8 x i32>, <8 x i32>* undef
  load <8 x i64>, <8 x i64>* undef
  load <8 x float>, <8 x float>* undef
  load <8 x double>, <8 x double>* undef
  load <16 x i8>, <16 x i8>* undef
  load <16 x i16>, <16 x i16>* undef
  load <16 x i32>, <16 x i32>* undef
  load <16 x i64>, <16 x i64>* undef
  load <16 x float>, <16 x float>* undef
  load <16 x double>, <16 x double>* undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = load i8, i8* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = load float, float* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = load double, double* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %7 = load fp128, fp128* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = load <2 x i8>, <2 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = load <2 x i16>, <2 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = load <2 x i32>, <2 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %11 = load <2 x i64>, <2 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %12 = load <2 x float>, <2 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %13 = load <2 x double>, <2 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %14 = load <4 x i8>, <4 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %15 = load <4 x i16>, <4 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %16 = load <4 x i32>, <4 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %17 = load <4 x i64>, <4 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %18 = load <4 x float>, <4 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %19 = load <4 x double>, <4 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %20 = load <8 x i8>, <8 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %21 = load <8 x i16>, <8 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %22 = load <8 x i32>, <8 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %23 = load <8 x i64>, <8 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %24 = load <8 x float>, <8 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %25 = load <8 x double>, <8 x double>* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %26 = load <16 x i8>, <16 x i8>* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %27 = load <16 x i16>, <16 x i16>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %28 = load <16 x i32>, <16 x i32>* undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %29 = load <16 x i64>, <16 x i64>* undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %30 = load <16 x float>, <16 x float>* undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %31 = load <16 x double>, <16 x double>* undef

  ret void;
}
