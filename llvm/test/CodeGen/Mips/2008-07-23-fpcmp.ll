; RUN: llc < %s -mtriple=mips -o %t
; RUN: grep "c\..*\.s" %t | count 3
; RUN: grep "bc1[tf]" %t | count 3

; FIXME: Disabled because branch instructions are generated where
; conditional move instructions are expected.
; REQUIRES: disabled

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-unknown-psp-elf"

define float @A(float %a, float %b) nounwind {
entry:
	fcmp ogt float %a, 1.000000e+00		; <i1>:0 [#uses=1]
	br i1 %0, label %bb, label %bb2

bb:		; preds = %entry
	fadd float %a, 1.000000e+00		; <float>:1 [#uses=1]
	ret float %1

bb2:		; preds = %entry
	ret float %b
}

define float @B(float %a, float %b) nounwind {
entry:
  fcmp ogt float %a, 1.000000e+00   ; <i1>:0 [#uses=1]
  %.0 = select i1 %0, float %a, float %b    ; <float> [#uses=1]
  ret float %.0
}

define i32 @C(i32 %a, i32 %b, float %j) nounwind {
entry:
  fcmp ogt float %j, 1.000000e+00   ; <i1>:0 [#uses=1]
  %.0 = select i1 %0, i32 %a, i32 %b    ; <i32> [#uses=1]
  ret i32 %.0
}

