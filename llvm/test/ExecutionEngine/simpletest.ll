; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}

