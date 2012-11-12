; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-link -o %t2.bc %t1.bc

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin7.7.0"
deplibs = [ "c", "crtend" ]
@source = global <4 x i32> < i32 0, i32 1, i32 2, i32 3 >		; <<4 x i32>*> [#uses=0]

define i32 @main() {
entry:
	ret i32 0
}

