; RUN: llvm-as < %s | opt -instcombine | llvm-dis
; rdar://5771353

define float @test(float %x, x86_fp80 %y) nounwind readonly  {
entry:
	%tmp67 = fcmp uno x86_fp80 %y, 0xK00000000000000000000		; <i1> [#uses=1]
	%tmp71 = fcmp uno float %x, 0.000000e+00		; <i1> [#uses=1]
	%bothcond = or i1 %tmp67, %tmp71		; <i1> [#uses=1]
	br i1 %bothcond, label %bb74, label %bb80

bb74:		; preds = %entry
	ret float 0.000000e+00

bb80:		; preds = %entry
	ret float 0.000000e+00
}
