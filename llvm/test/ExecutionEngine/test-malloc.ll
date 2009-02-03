; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

define i32 @main() {
	%X = malloc i32		; <i32*> [#uses=1]
	%Y = malloc i32, i32 100		; <i32*> [#uses=1]
	%u = add i32 1, 2		; <i32> [#uses=1]
	%Z = malloc i32, i32 %u		; <i32*> [#uses=1]
	free i32* %X
	free i32* %Y
	free i32* %Z
	ret i32 0
}

