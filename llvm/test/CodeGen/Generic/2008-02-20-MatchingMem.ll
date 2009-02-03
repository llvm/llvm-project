; RUN: llvm-as < %s | llc
; PR1133
define void @test(i32* %X) nounwind  {
entry:
	%tmp1 = getelementptr i32* %X, i32 10		; <i32*> [#uses=2]
	tail call void asm sideeffect " $0 $1 ", "=*im,*im,~{memory}"( i32* %tmp1, i32* %tmp1 ) nounwind 
	ret void
}

