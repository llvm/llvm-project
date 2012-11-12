; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%M = type { i32} " | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out2.bc %t.out1.bc

%M = type opaque

define void @foo(i32* %V) {
	ret void
}

declare void @foo.upgrd.1(%M*)

