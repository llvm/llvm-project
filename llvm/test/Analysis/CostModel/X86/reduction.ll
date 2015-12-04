; RUN: opt < %s -cost-model -costmodel-reduxcost=true -analyze -mcpu=core2 -mtriple=x86_64-apple-darwin | FileCheck %s
; RUN: opt < %s -cost-model -costmodel-reduxcost=true -analyze -mcpu=corei7 -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefix=SSE3
; RUN: opt < %s -cost-model -costmodel-reduxcost=true -analyze -mcpu=corei7-avx -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefix=AVX
; RUN: opt < %s -cost-model -costmodel-reduxcost=true -analyze -mcpu=core-avx2 -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefix=AVX2

define fastcc float @reduction_cost_float(<4 x float> %rdx) {
  %rdx.shuf = shufflevector <4 x float> %rdx, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x float> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x float> %bin.rdx, %rdx.shuf7

; Check that we recognize the tree starting at the extractelement as a
; reduction.
; CHECK-LABEL: reduction_cost
; CHECK:  cost of 9 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx8, i32 0
  ret float %r
}

define fastcc i32 @reduction_cost_int(<8 x i32> %rdx) {
  %rdx.shuf = shufflevector <8 x i32> %rdx, <8 x i32> undef,
   <8 x i32> <i32 4    , i32     5, i32     6, i32     7,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %rdx, %rdx.shuf
  %rdx.shuf.2 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef,
   <8 x i32> <i32 2    , i32 3,     i32 undef, i32 undef,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.2 = add <8 x i32> %bin.rdx, %rdx.shuf.2
  %rdx.shuf.3 = shufflevector <8 x i32> %bin.rdx.2, <8 x i32> undef,
   <8 x i32> <i32 1    , i32 undef, i32 undef, i32 undef,
              i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.3 = add <8 x i32> %bin.rdx.2, %rdx.shuf.3

; CHECK-LABEL: reduction_cost_int
; CHECK:  cost of 17 {{.*}} extractelement

  %r = extractelement <8 x i32> %bin.rdx.3, i32 0
  ret i32 %r
}

define fastcc float @pairwise_hadd(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}

define fastcc float @pairwise_hadd_assoc(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.1, %rdx.shuf.0.0
  %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd_assoc
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}

define fastcc float @pairwise_hadd_skip_first(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
        <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
        <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = fadd <4 x float> %bin.rdx.0, %rdx.shuf.1.1

; CHECK-LABEL: pairwise_hadd_skip_first
; CHECK: cost of 11 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx.1, i32 0
  %r2 = fadd float %r, %f1
  ret float %r2
}

define fastcc double @no_pairwise_reduction2double(<2 x double> %rdx, double %f1) {
  %rdx.shuf = shufflevector <2 x double> %rdx, <2 x double> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx = fadd <2 x double> %rdx, %rdx.shuf

; SSE3:  cost of 2 {{.*}} extractelement
; AVX:  cost of 2 {{.*}} extractelement
; AVX2:  cost of 2 {{.*}} extractelement

  %r = extractelement <2 x double> %bin.rdx, i32 0
  ret double %r
}

define fastcc float @no_pairwise_reduction4float(<4 x float> %rdx, float %f1) {
  %rdx.shuf = shufflevector <4 x float> %rdx, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x float> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x float> %bin.rdx, %rdx.shuf7

; SSE3:  cost of 4 {{.*}} extractelement
; AVX:  cost of 3 {{.*}} extractelement
; AVX2:  cost of 3 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx8, i32 0
  ret float %r
}

define fastcc double @no_pairwise_reduction4double(<4 x double> %rdx, double %f1) {
  %rdx.shuf = shufflevector <4 x double> %rdx, <4 x double> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x double> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x double> %bin.rdx, <4 x double> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x double> %bin.rdx, %rdx.shuf7

; AVX:  cost of 3 {{.*}} extractelement
; AVX2:  cost of 3 {{.*}} extractelement

  %r = extractelement <4 x double> %bin.rdx8, i32 0
  ret double %r
}

define fastcc float @no_pairwise_reduction8float(<8 x float> %rdx, float %f1) {
  %rdx.shuf3 = shufflevector <8 x float> %rdx, <8 x float> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = fadd <8 x float> %rdx, %rdx.shuf3
  %rdx.shuf = shufflevector <8 x float> %bin.rdx4, <8 x float> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd <8 x float> %bin.rdx4, %rdx.shuf
  %rdx.shuf7 = shufflevector <8 x float> %bin.rdx, <8 x float> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <8 x float> %bin.rdx, %rdx.shuf7

; AVX:  cost of 4 {{.*}} extractelement
; AVX2:  cost of 4 {{.*}} extractelement

  %r = extractelement <8 x float> %bin.rdx8, i32 0
  ret float %r
}

define fastcc i64 @no_pairwise_reduction2i64(<2 x i64> %rdx, i64 %f1) {
  %rdx.shuf = shufflevector <2 x i64> %rdx, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx = add <2 x i64> %rdx, %rdx.shuf

; SSE3:  cost of 2 {{.*}} extractelement
; AVX:  cost of 1 {{.*}} extractelement
; AVX2:  cost of 1 {{.*}} extractelement

  %r = extractelement <2 x i64> %bin.rdx, i32 0
  ret i64 %r
}

define fastcc i32 @no_pairwise_reduction4i32(<4 x i32> %rdx, i32 %f1) {
  %rdx.shuf = shufflevector <4 x i32> %rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i32> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <4 x i32> %bin.rdx, %rdx.shuf7

; SSE3:  cost of 3 {{.*}} extractelement
; AVX:  cost of 3 {{.*}} extractelement
; AVX2:  cost of 3 {{.*}} extractelement

  %r = extractelement <4 x i32> %bin.rdx8, i32 0
  ret i32 %r
}

define fastcc i64 @no_pairwise_reduction4i64(<4 x i64> %rdx, i64 %f1) {
  %rdx.shuf = shufflevector <4 x i64> %rdx, <4 x i64> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i64> %rdx, %rdx.shuf
  %rdx.shuf7 = shufflevector <4 x i64> %bin.rdx, <4 x i64> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <4 x i64> %bin.rdx, %rdx.shuf7

; AVX:  cost of 3 {{.*}} extractelement
; AVX2:  cost of 3 {{.*}} extractelement

  %r = extractelement <4 x i64> %bin.rdx8, i32 0
  ret i64 %r
}

define fastcc i16 @no_pairwise_reduction8i16(<8 x i16> %rdx, i16 %f1) {
  %rdx.shuf3 = shufflevector <8 x i16> %rdx, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = add <8 x i16> %rdx, %rdx.shuf3
  %rdx.shuf = shufflevector <8 x i16> %bin.rdx4, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %bin.rdx4, %rdx.shuf
  %rdx.shuf7 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i16> %bin.rdx, %rdx.shuf7

; SSE3:  cost of 4 {{.*}} extractelement
; AVX:  cost of 4 {{.*}} extractelement
; AVX2:  cost of 4 {{.*}} extractelement

  %r = extractelement <8 x i16> %bin.rdx8, i32 0
  ret i16 %r
}

define fastcc i32 @no_pairwise_reduction8i32(<8 x i32> %rdx, i32 %f1) {
  %rdx.shuf3 = shufflevector <8 x i32> %rdx, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = add <8 x i32> %rdx, %rdx.shuf3
  %rdx.shuf = shufflevector <8 x i32> %bin.rdx4, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %bin.rdx4, %rdx.shuf
  %rdx.shuf7 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i32> %bin.rdx, %rdx.shuf7

; AVX:  cost of 5 {{.*}} extractelement
; AVX2:  cost of 5 {{.*}} extractelement

  %r = extractelement <8 x i32> %bin.rdx8, i32 0
  ret i32 %r
}

define fastcc double @pairwise_reduction2double(<2 x double> %rdx, double %f1) {
  %rdx.shuf.1.0 = shufflevector <2 x double> %rdx, <2 x double> undef, <2 x i32> <i32 0, i32 undef>
  %rdx.shuf.1.1 = shufflevector <2 x double> %rdx, <2 x double> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx8 = fadd <2 x double> %rdx.shuf.1.0, %rdx.shuf.1.1

; SSE3:  cost of 2 {{.*}} extractelement
; AVX:  cost of 2 {{.*}} extractelement
; AVX2:  cost of 2 {{.*}} extractelement

  %r = extractelement <2 x double> %bin.rdx8, i32 0
  ret double %r
}

define fastcc float @pairwise_reduction4float(<4 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx, <4 x float> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1

; SSE3:  cost of 4 {{.*}} extractelement
; AVX:  cost of 4 {{.*}} extractelement
; AVX2:  cost of 4 {{.*}} extractelement

  %r = extractelement <4 x float> %bin.rdx8, i32 0
  ret float %r
}

define fastcc double @pairwise_reduction4double(<4 x double> %rdx, double %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x double> %rdx, <4 x double> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x double> %rdx, <4 x double> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x double> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x double> %bin.rdx, <4 x double> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x double> %bin.rdx, <4 x double> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <4 x double> %rdx.shuf.1.0, %rdx.shuf.1.1

; AVX:  cost of 5 {{.*}} extractelement
; AVX2:  cost of 5 {{.*}} extractelement

  %r = extractelement <4 x double> %bin.rdx8, i32 0
  ret double %r
}

define fastcc float @pairwise_reduction8float(<8 x float> %rdx, float %f1) {
  %rdx.shuf.0.0 = shufflevector <8 x float> %rdx, <8 x float> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6,i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <8 x float> %rdx, <8 x float> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd <8 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <8 x float> %bin.rdx, <8 x float> undef,<8 x i32> <i32 0, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <8 x float> %bin.rdx, <8 x float> undef,<8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = fadd <8 x float> %rdx.shuf.1.0, %rdx.shuf.1.1
  %rdx.shuf.2.0 = shufflevector <8 x float> %bin.rdx8, <8 x float> undef,<8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.2.1 = shufflevector <8 x float> %bin.rdx8, <8 x float> undef,<8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx9 = fadd <8 x float> %rdx.shuf.2.0, %rdx.shuf.2.1

; AVX:  cost of 7 {{.*}} extractelement
; AVX2:  cost of 7 {{.*}} extractelement

  %r = extractelement <8 x float> %bin.rdx9, i32 0
  ret float %r
}

define fastcc i64 @pairwise_reduction2i64(<2 x i64> %rdx, i64 %f1) {
  %rdx.shuf.1.0 = shufflevector <2 x i64> %rdx, <2 x i64> undef, <2 x i32> <i32 0, i32 undef>
  %rdx.shuf.1.1 = shufflevector <2 x i64> %rdx, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx8 = add <2 x i64> %rdx.shuf.1.0, %rdx.shuf.1.1

; SSE3:  cost of 2 {{.*}} extractelement
; AVX:  cost of 1 {{.*}} extractelement
; AVX2:  cost of 1 {{.*}} extractelement

  %r = extractelement <2 x i64> %bin.rdx8, i32 0
  ret i64 %r
}

define fastcc i32 @pairwise_reduction4i32(<4 x i32> %rdx, i32 %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x i32> %rdx, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x i32> %rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i32> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <4 x i32> %rdx.shuf.1.0, %rdx.shuf.1.1

; SSE3:  cost of 3 {{.*}} extractelement
; AVX:  cost of 3 {{.*}} extractelement
; AVX2:  cost of 3 {{.*}} extractelement

  %r = extractelement <4 x i32> %bin.rdx8, i32 0
  ret i32 %r
}

define fastcc i64 @pairwise_reduction4i64(<4 x i64> %rdx, i64 %f1) {
  %rdx.shuf.0.0 = shufflevector <4 x i64> %rdx, <4 x i64> undef, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <4 x i64> %rdx, <4 x i64> undef, <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i64> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <4 x i64> %bin.rdx, <4 x i64> undef, <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <4 x i64> %bin.rdx, <4 x i64> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <4 x i64> %rdx.shuf.1.0, %rdx.shuf.1.1

; AVX:  cost of 5 {{.*}} extractelement
; AVX2:  cost of 5 {{.*}} extractelement

  %r = extractelement <4 x i64> %bin.rdx8, i32 0
  ret i64 %r
}

define fastcc i16 @pairwise_reduction8i16(<8 x i16> %rdx, i16 %f1) {
  %rdx.shuf.0.0 = shufflevector <8 x i16> %rdx, <8 x i16> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6,i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <8 x i16> %rdx, <8 x i16> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef,<8 x i32> <i32 0, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef,<8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i16> %rdx.shuf.1.0, %rdx.shuf.1.1
  %rdx.shuf.2.0 = shufflevector <8 x i16> %bin.rdx8, <8 x i16> undef,<8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.2.1 = shufflevector <8 x i16> %bin.rdx8, <8 x i16> undef,<8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx9 = add <8 x i16> %rdx.shuf.2.0, %rdx.shuf.2.1

; SSE3:  cost of 5 {{.*}} extractelement
; AVX:  cost of 5 {{.*}} extractelement
; AVX2:  cost of 5 {{.*}} extractelement

  %r = extractelement <8 x i16> %bin.rdx9, i32 0
  ret i16 %r
}

define fastcc i32 @pairwise_reduction8i32(<8 x i32> %rdx, i32 %f1) {
  %rdx.shuf.0.0 = shufflevector <8 x i32> %rdx, <8 x i32> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6,i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.0.1 = shufflevector <8 x i32> %rdx, <8 x i32> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7,i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i32> %rdx.shuf.0.0, %rdx.shuf.0.1
  %rdx.shuf.1.0 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef,<8 x i32> <i32 0, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.1.1 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef,<8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i32> %rdx.shuf.1.0, %rdx.shuf.1.1
  %rdx.shuf.2.0 = shufflevector <8 x i32> %bin.rdx8, <8 x i32> undef,<8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.shuf.2.1 = shufflevector <8 x i32> %bin.rdx8, <8 x i32> undef,<8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx9 = add <8 x i32> %rdx.shuf.2.0, %rdx.shuf.2.1

; AVX:  cost of 5 {{.*}} extractelement
; AVX2:  cost of 5 {{.*}} extractelement

  %r = extractelement <8 x i32> %bin.rdx9, i32 0
  ret i32 %r
}
