; RUN: opt -tailcallelim %s -S | FileCheck %s
; PR615

declare void @bar(i32*)

define i32 @foo() {
; CHECK: i32 @foo()
; CHECK-NEXT: alloca
	%A = alloca i32		; <i32*> [#uses=2]
	store i32 17, i32* %A
	call void @bar( i32* %A )
	%X = tail call i32 @foo( )		; <i32> [#uses=1]
	ret i32 %X
}

