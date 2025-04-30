; RUN: llc -mtriple=next32 -mcpu=next32gen1 < %s | FileCheck %s --check-prefix=NO-FMA
; RUN: llc -mtriple=next32 -mcpu=next32gen2 < %s | FileCheck %s --check-prefix=PREFER-FMA

; Original C source:
; #include <math.h>
;
; float test_fma(float a, float b, float c) {
;   return fmaf(b, c, a);
; }
;
; // This function is compiled with different -ffp-contract options and
; // -ffast-math option.
; float test_*(float a, float b, float c) {
;   return a + b * c;
; }

; Test that we are not generating call to fma.
define float @test_contract_off(float noundef %0, float noundef %1, float noundef %2) {
; NO-FMA-LABEL: test_contract_off:
; NO-FMA:    movl call_addr, __mulsf3
; NO-FMA:    movl call_addr, __addsf3
;
; PREFER-FMA-LABEL: test_contract_off:
; PREFER-FMA:    movl call_addr, __mulsf3
; PREFER-FMA:    movl call_addr, __addsf3
  %4 = fmul float %1, %2
  %5 = fadd float %4, %0
  ret float %5
}

define float @test_contract_fast(float noundef %0, float noundef %1, float noundef %2) {
; NO-FMA-LABEL: test_contract_fast:
; NO-FMA:    movl call_addr, __mulsf3
; NO-FMA:    movl call_addr, __addsf3
;
; PREFER-FMA-LABEL: test_contract_fast:
; PREFER-FMA:    movl call_addr, fmaf
  %4 = fmul contract float %1, %2
  %5 = fadd contract float %4, %0
  ret float %5
}

define float @test_ffast_math(float noundef %0, float noundef %1, float noundef %2) #0 {
; NO-FMA-LABEL: test_ffast_math:
; NO-FMA:    movl call_addr, __mulsf3
; NO-FMA:    movl call_addr, __addsf3
;
; PREFER-FMA-LABEL: test_ffast_math:
; PREFER-FMA:    movl call_addr, fmaf
  %4 = fmul fast float %2, %1
  %5 = fadd fast float %4, %0
  ret float %5
}

define float @test_contract_on(float noundef %0, float noundef %1, float noundef %2) {
; NO-FMA-LABEL: test_contract_on:
; NO-FMA:    movl call_addr, __mulsf3
; NO-FMA:    movl call_addr, __addsf3
;
; PREFER-FMA-LABEL: test_contract_on:
; PREFER-FMA:    movl call_addr, fmaf
  %4 = tail call float @llvm.fmuladd.f32(float %1, float %2, float %0)
  ret float %4
}

; Test that we are generating call to fma.
define float @test_fma(float noundef %0, float noundef %1, float noundef %2) {
; NO-FMA-LABEL: test_fma:
; NO-FMA:    movl call_addr, fmaf
;
; PREFER-FMA-LABEL: test_fma:
; PREFER-FMA:    movl call_addr, fmaf
  %4 = tail call fast float @llvm.fma.f32(float %1, float %2, float %0)
  ret float %4
}

declare float @llvm.fma.f32(float, float, float)
declare float @llvm.fmuladd.f32(float, float, float)

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
