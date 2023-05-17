; RUN: llc -no-integrated-as < %s
; PR1133
define void @test(ptr %X) nounwind  {
entry:
	%tmp1 = getelementptr i32, ptr %X, i32 10		; <ptr> [#uses=2]
	tail call void asm sideeffect " $0 $1 ", "=*im,*im,~{memory}"( ptr elementtype( i32) %tmp1, ptr elementtype(i32) %tmp1 ) nounwind 
	ret void
}

