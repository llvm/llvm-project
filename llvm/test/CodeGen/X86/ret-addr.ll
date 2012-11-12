; RUN: llc < %s -disable-fp-elim -march=x86 | not grep xor
; RUN: llc < %s -disable-fp-elim -march=x86-64 | not grep xor

define i8* @h() nounwind readnone optsize {
entry:
	%0 = tail call i8* @llvm.returnaddress(i32 2)		; <i8*> [#uses=1]
	ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

define i8* @g() nounwind readnone optsize {
entry:
	%0 = tail call i8* @llvm.returnaddress(i32 1)		; <i8*> [#uses=1]
	ret i8* %0
}

define i8* @f() nounwind readnone optsize {
entry:
	%0 = tail call i8* @llvm.returnaddress(i32 0)		; <i8*> [#uses=1]
	ret i8* %0
}
