; RUN: llc -mtriple=mips < %s
; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -mtriple=mipsel < %s
; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a "Cannot select ..." error.
; This happened because the legalizer treated undef's in the <4 x float>
; constant as equivalent to the defined elements when checking if it a constant
; splat, but then proceeded to legalize the undef's to zero, leaving it as a
; non-splat that cannot be selected. It should have eliminated the undef's by
; rewriting the splat constant.

; It should at least successfully build.

define void @autogen_SD2501752154() {
BB:
  %BC = bitcast <4 x i32> <i32 -1, i32 -1, i32 undef, i32 undef> to <4 x float>
  br label %CF74

CF74:                                             ; preds = %CF74, %CF
  %E54 = extractelement <1 x i1> undef, i32 0
  br i1 %E54, label %CF74, label %CF79

CF79:                                             ; preds = %CF75
  %I63 = insertelement <4 x float> %BC, float undef, i32 0
  ret void
}
