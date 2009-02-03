; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep tmp5
; PR2599

define void @foo({ i32, i32 }* %x) nounwind  {
entry:
	%tmp4 = getelementptr { i32, i32 }* %x, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp5 = load i32* %tmp4, align 4		; <i32> [#uses=1]
	%tmp7 = getelementptr { i32, i32 }* %x, i32 0, i32 1		; <i32*> [#uses=2]
	%tmp8 = load i32* %tmp7, align 4		; <i32> [#uses=1]
	%tmp17 = sub i32 0, %tmp8		; <i32> [#uses=1]
	store i32 %tmp5, i32* %tmp4, align 4
	store i32 %tmp17, i32* %tmp7, align 4
	ret void
}
