; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=zEC12 | FileCheck %s --check-prefixes=CHECK,ZEC12
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s --check-prefixes=CHECK,Z13
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z14 | FileCheck %s --check-prefixes=CHECK,Z14

define void @store() {
  store i8 undef, ptr undef
  store i16 undef, ptr undef
  store i32 undef, ptr undef
  store i64 undef, ptr undef
  store i128 undef, ptr undef
  store float undef, ptr undef
  store double undef, ptr undef
  store fp128 undef, ptr undef
  store <2 x i8> undef, ptr undef
  store <2 x i16> undef, ptr undef
  store <2 x i32> undef, ptr undef
  store <2 x i64> undef, ptr undef
  store <2 x float> undef, ptr undef
  store <2 x double> undef, ptr undef
  store <4 x i8> undef, ptr undef
  store <4 x i16> undef, ptr undef
  store <4 x i32> undef, ptr undef
  store <4 x i64> undef, ptr undef
  store <4 x float> undef, ptr undef
  store <4 x double> undef, ptr undef
  store <8 x i8> undef, ptr undef
  store <8 x i16> undef, ptr undef
  store <8 x i32> undef, ptr undef
  store <8 x i64> undef, ptr undef
  store <8 x float> undef, ptr undef
  store <8 x double> undef, ptr undef
  store <16 x i8> undef, ptr undef
  store <16 x i16> undef, ptr undef
  store <16 x i32> undef, ptr undef
  store <16 x i64> undef, ptr undef
  store <16 x float> undef, ptr undef
  store <16 x double> undef, ptr undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i8 undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i16 undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i32 undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store i64 undef, ptr undef
; ZEC12: Cost Model: Found an estimated cost of 2 for instruction:   store i128 undef, ptr undef
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   store i128 undef, ptr undef
; Z14:   Cost Model: Found an estimated cost of 1 for instruction:   store i128 undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store float undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store double undef, ptr undef
; ZEC12: Cost Model: Found an estimated cost of 2 for instruction:   store fp128 undef, ptr undef
; Z13:   Cost Model: Found an estimated cost of 2 for instruction:   store fp128 undef, ptr undef
; Z14:   Cost Model: Found an estimated cost of 1 for instruction:   store fp128 undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i8> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i16> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i32> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i64> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x float> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x double> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i8> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i16> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x i32> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <4 x i64> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <4 x float> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <4 x double> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <8 x i8> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <8 x i16> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <8 x i32> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <8 x i64> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <8 x float> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <8 x double> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store <16 x i8> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   store <16 x i16> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <16 x i32> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   store <16 x i64> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   store <16 x float> undef, ptr undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   store <16 x double> undef, ptr undef

  ret void;
}

define void @load() {
  load i8, ptr undef
  load i16, ptr undef
  load i32, ptr undef
  load i64, ptr undef
  load i128, ptr undef
  load float, ptr undef
  load double, ptr undef
  load fp128, ptr undef
  load <2 x i8>, ptr undef
  load <2 x i16>, ptr undef
  load <2 x i32>, ptr undef
  load <2 x i64>, ptr undef
  load <2 x float>, ptr undef
  load <2 x double>, ptr undef
  load <4 x i8>, ptr undef
  load <4 x i16>, ptr undef
  load <4 x i32>, ptr undef
  load <4 x i64>, ptr undef
  load <4 x float>, ptr undef
  load <4 x double>, ptr undef
  load <8 x i8>, ptr undef
  load <8 x i16>, ptr undef
  load <8 x i32>, ptr undef
  load <8 x i64>, ptr undef
  load <8 x float>, ptr undef
  load <8 x double>, ptr undef
  load <16 x i8>, ptr undef
  load <16 x i16>, ptr undef
  load <16 x i32>, ptr undef
  load <16 x i64>, ptr undef
  load <16 x float>, ptr undef
  load <16 x double>, ptr undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = load i8, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = load i16, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = load i32, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = load i64, ptr undef
; ZEC12: Cost Model: Found an estimated cost of 2 for instruction:   %5 = load i128, ptr undef
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %5 = load i128, ptr undef
; Z14:   Cost Model: Found an estimated cost of 1 for instruction:   %5 = load i128, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = load float, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = load double, ptr undef
; ZEC12: Cost Model: Found an estimated cost of 2 for instruction:   %8 = load fp128, ptr undef
; Z13:   Cost Model: Found an estimated cost of 2 for instruction:   %8 = load fp128, ptr undef
; Z14:   Cost Model: Found an estimated cost of 1 for instruction:   %8 = load fp128, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = load <2 x i8>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = load <2 x i16>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %11 = load <2 x i32>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %12 = load <2 x i64>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %13 = load <2 x float>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %14 = load <2 x double>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %15 = load <4 x i8>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %16 = load <4 x i16>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %17 = load <4 x i32>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %18 = load <4 x i64>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %19 = load <4 x float>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %20 = load <4 x double>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %21 = load <8 x i8>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %22 = load <8 x i16>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %23 = load <8 x i32>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %24 = load <8 x i64>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %25 = load <8 x float>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %26 = load <8 x double>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %27 = load <16 x i8>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %28 = load <16 x i16>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %29 = load <16 x i32>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %30 = load <16 x i64>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %31 = load <16 x float>, ptr undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %32 = load <16 x double>, ptr undef

  ret void;
}
