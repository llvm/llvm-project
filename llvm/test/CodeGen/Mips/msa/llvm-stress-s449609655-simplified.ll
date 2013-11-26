; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s

; This test is based on an llvm-stress generated test case with seed=449609655

; This test originally failed for MSA with a
; "Comparison requires equal bit widths" assertion.
; The legalizer legalized ; the <4 x i8>'s into <4 x i32>'s, then a call to
; isVSplat() returned the splat value for <i8 -1, i8 -1, ...> as a 32-bit APInt
; (255), but the zeroinitializer splat value as an 8-bit APInt (0). The
; assertion occured when trying to check the values were bitwise inverses of
; each-other.
;
; It should at least successfully build.

define void @autogen_SD449609655(i8) {
BB:
  %Cmp = icmp ult i8 -3, %0
  br label %CF78

CF78:                                             ; preds = %CF81, %CF78, %BB
  %Sl31 = select i1 %Cmp, <4 x i8> <i8 -1, i8 -1, i8 -1, i8 -1>, <4 x i8> zeroinitializer
  br i1 undef, label %CF78, label %CF81

CF81:                                             ; preds = %CF78
  br i1 undef, label %CF78, label %CF80

CF80:                                             ; preds = %CF81
  %I59 = insertelement <4 x i8> %Sl31, i8 undef, i32 1
  ret void
}
