; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-Z13 %s
; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z14 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-Z14 %s
;
; Note: The scalarized vector instructions cost is not including any
; extracts, due to the undef operands
;
; Note: FRem is implemented with libcall, so not included here.

define void @fadd() {
  %res0 = fadd float undef, undef
  %res1 = fadd double undef, undef
  %res2 = fadd fp128 undef, undef
  %res3 = fadd <2 x float> undef, undef
  %res4 = fadd <2 x double> undef, undef
  %res5 = fadd <4 x float> undef, undef
  %res6 = fadd <4 x double> undef, undef
  %res7 = fadd <8 x float> undef, undef
  %res8 = fadd <8 x double> undef, undef
  %res9 = fadd <16 x float> undef, undef
  %res10 = fadd <16 x double> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = fadd float undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = fadd double undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = fadd fp128 undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res3 = fadd <2 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = fadd <2 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = fadd <2 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res5 = fadd <4 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = fadd <4 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res6 = fadd <4 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 16 for instruction:   %res7 = fadd <8 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 2 for instruction:   %res7 = fadd <8 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res8 = fadd <8 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 32 for instruction:   %res9 = fadd <16 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 4 for instruction:   %res9 = fadd <16 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res10 = fadd <16 x double> undef, undef

  ret void;
}

define void @fsub() {
  %res0 = fsub float undef, undef
  %res1 = fsub double undef, undef
  %res2 = fsub fp128 undef, undef
  %res3 = fsub <2 x float> undef, undef
  %res4 = fsub <2 x double> undef, undef
  %res5 = fsub <4 x float> undef, undef
  %res6 = fsub <4 x double> undef, undef
  %res7 = fsub <8 x float> undef, undef
  %res8 = fsub <8 x double> undef, undef
  %res9 = fsub <16 x float> undef, undef
  %res10 = fsub <16 x double> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = fsub float undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = fsub double undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = fsub fp128 undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res3 = fsub <2 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = fsub <2 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = fsub <2 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res5 = fsub <4 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = fsub <4 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res6 = fsub <4 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 16 for instruction:   %res7 = fsub <8 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 2 for instruction:   %res7 = fsub <8 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res8 = fsub <8 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 32 for instruction:   %res9 = fsub <16 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 4 for instruction:   %res9 = fsub <16 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res10 = fsub <16 x double> undef, undef

  ret void;
}

define void @fmul() {
  %res0 = fmul float undef, undef
  %res1 = fmul double undef, undef
  %res2 = fmul fp128 undef, undef
  %res3 = fmul <2 x float> undef, undef
  %res4 = fmul <2 x double> undef, undef
  %res5 = fmul <4 x float> undef, undef
  %res6 = fmul <4 x double> undef, undef
  %res7 = fmul <8 x float> undef, undef
  %res8 = fmul <8 x double> undef, undef
  %res9 = fmul <16 x float> undef, undef
  %res10 = fmul <16 x double> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = fmul float undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = fmul double undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = fmul fp128 undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res3 = fmul <2 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = fmul <2 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = fmul <2 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res5 = fmul <4 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = fmul <4 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res6 = fmul <4 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 16 for instruction:   %res7 = fmul <8 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 2 for instruction:   %res7 = fmul <8 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res8 = fmul <8 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 32 for instruction:   %res9 = fmul <16 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 4 for instruction:   %res9 = fmul <16 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res10 = fmul <16 x double> undef, undef

  ret void;
}

define void @fdiv() {
  %res0 = fdiv float undef, undef
  %res1 = fdiv double undef, undef
  %res2 = fdiv fp128 undef, undef
  %res3 = fdiv <2 x float> undef, undef
  %res4 = fdiv <2 x double> undef, undef
  %res5 = fdiv <4 x float> undef, undef
  %res6 = fdiv <4 x double> undef, undef
  %res7 = fdiv <8 x float> undef, undef
  %res8 = fdiv <8 x double> undef, undef
  %res9 = fdiv <16 x float> undef, undef
  %res10 = fdiv <16 x double> undef, undef

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res0 = fdiv float undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res1 = fdiv double undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res2 = fdiv fp128 undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res3 = fdiv <2 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res3 = fdiv <2 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %res4 = fdiv <2 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 8 for instruction:   %res5 = fdiv <4 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 1 for instruction:   %res5 = fdiv <4 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %res6 = fdiv <4 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 16 for instruction:   %res7 = fdiv <8 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 2 for instruction:   %res7 = fdiv <8 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %res8 = fdiv <8 x double> undef, undef
; CHECK-Z13: Cost Model: Found an estimated cost of 32 for instruction:   %res9 = fdiv <16 x float> undef, undef
; CHECK-Z14: Cost Model: Found an estimated cost of 4 for instruction:   %res9 = fdiv <16 x float> undef, undef
; CHECK: Cost Model: Found an estimated cost of 8 for instruction:   %res10 = fdiv <16 x double> undef, undef

  ret void;
}

