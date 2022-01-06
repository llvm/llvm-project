; RUN: llc < %s -mtriple=i386-apple-darwin -no-integrated-as
; rdar://6781755
; PR3934

	%0 = type { i32, i32 }		; type %0

define void @bn_sqr_comba8(i32* nocapture %r, i32* %a) nounwind {
entry:
	%asmtmp23 = tail call %0 asm "mulq $3", "={ax},={dx},{ax},*m,~{dirflag},~{fpsr},~{flags},~{cc}"(i32 0, i32* elementtype(i32) %a) nounwind		; <%0> [#uses=1]
	%asmresult25 = extractvalue %0 %asmtmp23, 1		; <i32> [#uses=1]
	%asmtmp26 = tail call %0 asm "addq $0,$0; adcq $2,$1", "={dx},=r,imr,0,1,~{dirflag},~{fpsr},~{flags},~{cc}"(i32 0, i32 %asmresult25, i32 0) nounwind		; <%0> [#uses=1]
	%asmresult27 = extractvalue %0 %asmtmp26, 0		; <i32> [#uses=1]
	%asmtmp29 = tail call %0 asm "addq $0,$0; adcq $2,$1", "={ax},={dx},imr,0,1,~{dirflag},~{fpsr},~{flags},~{cc}"(i32 0, i32 0, i32 %asmresult27) nounwind		; <%0> [#uses=0]
	ret void
}
