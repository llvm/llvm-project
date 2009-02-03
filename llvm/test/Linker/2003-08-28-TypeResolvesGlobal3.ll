; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%S = type i32" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out2.bc %t.out1.bc

%S = type opaque

; GLobal using the resolved function prototype
global void (%S*)* @foo		; <void (%S*)**>:0 [#uses=0]

define void @foo.upgrd.1(i32* %V) {
	ret void
}

declare void @foo(%S*)

