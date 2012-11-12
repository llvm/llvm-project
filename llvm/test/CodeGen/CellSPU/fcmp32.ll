; RUN: llc --mtriple=cellspu-unknown-elf %s -o - | FileCheck %s

; Exercise the floating point comparison operators for f32:

declare double @fabs(double)
declare float @fabsf(float)

define i1 @fcmp_eq(float %arg1, float %arg2) {
; CHECK: fceq
; CHECK: bi $lr
        %A = fcmp oeq float %arg1,  %arg2
        ret i1 %A
}

define i1 @fcmp_mag_eq(float %arg1, float %arg2) {
; CHECK: fcmeq
; CHECK: bi $lr
        %1 = call float @fabsf(float %arg1) readnone
        %2 = call float @fabsf(float %arg2) readnone
        %3 = fcmp oeq float %1, %2
        ret i1 %3
}

define i1 @test_ogt(float %a, float %b) {
; CHECK: fcgt
; CHECK: bi $lr
	%cmp = fcmp ogt float %a, %b
	ret i1 %cmp
}

define i1 @test_ugt(float %a, float %b) {
; CHECK: fcgt
; CHECK: bi $lr
	%cmp = fcmp ugt float %a, %b
	ret i1 %cmp
}
