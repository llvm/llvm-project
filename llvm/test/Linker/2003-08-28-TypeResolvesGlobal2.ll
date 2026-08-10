; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%M = type i32" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out2.bc %t.out1.bc

%M = type opaque

define void @foo(ptr %V) {
	ret void
}

declare void @foo.upgrd.1(ptr)

define void @other() {
	call void @foo.upgrd.1( ptr null )
	call void @foo( ptr null )
	ret void
}

