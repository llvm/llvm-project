; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu
; PR4280

define i32 @__fixunssfsi(float %a) nounwind readnone {
entry:
	%0 = fcmp ult float %a, 0x41E0000000000000		; <i1> [#uses=1]
	br i1 %0, label %bb1, label %bb

bb:		; preds = %entry
	ret i32 1

bb1:		; preds = %entry
	ret i32 0
}

