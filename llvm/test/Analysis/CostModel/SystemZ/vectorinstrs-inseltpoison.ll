; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; CHECK: vecinstrs
define void @vecinstrs() {

  ;; Extract element is penalized somewhat with a cost of 2 for index 0.
  extractelement <16 x i8> undef, i32 0
  extractelement <16 x i8> undef, i32 1

  extractelement <8 x i16> undef, i32 0
  extractelement <8 x i16> undef, i32 1

  extractelement <4 x i32> undef, i32 0
  extractelement <4 x i32> undef, i32 1

  extractelement <2 x i64> undef, i32 0
  extractelement <2 x i64> undef, i32 1

  extractelement <2 x double> undef, i32 0
  extractelement <2 x double> undef, i32 1

  ; Extraction of i1 means extract + test under mask before branch.
  extractelement <2 x i1> undef, i32 0
  extractelement <4 x i1> undef, i32 1
  extractelement <8 x i1> undef, i32 2

  ;; Insert element
  insertelement <16 x i8> poison, i8 undef, i32 0
  insertelement <8 x i16> poison, i16 undef, i32 0
  insertelement <4 x i32> poison, i32 undef, i32 0

  ; vlvgp will do two grs into a vector register: only add cost half of the time.
  insertelement <2 x i64> poison, i64 undef, i32 0
  insertelement <2 x i64> poison, i64 undef, i32 1

  ret void

; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %1 = extractelement <16 x i8> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = extractelement <16 x i8> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %3 = extractelement <8 x i16> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = extractelement <8 x i16> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %5 = extractelement <4 x i32> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = extractelement <4 x i32> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %7 = extractelement <2 x i64> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = extractelement <2 x i64> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = extractelement <2 x double> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = extractelement <2 x double> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %11 = extractelement <2 x i1> undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %12 = extractelement <4 x i1> undef, i32 1
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %13 = extractelement <8 x i1> undef, i32 2
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %14 = insertelement <16 x i8> poison, i8 undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %15 = insertelement <8 x i16> poison, i16 undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %16 = insertelement <4 x i32> poison, i32 undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %17 = insertelement <2 x i64> poison, i64 undef, i32 0
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %18 = insertelement <2 x i64> poison, i64 undef, i32 1
}
