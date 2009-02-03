; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

define i32 @main() {
	ret i32 0
}

