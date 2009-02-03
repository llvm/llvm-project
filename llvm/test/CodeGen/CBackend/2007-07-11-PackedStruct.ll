; RUN: llvm-as < %s | llc -march=c | grep {packed}

	%struct.p = type <{ i16 }>

define i32 @main() {
entry:
        %t = alloca %struct.p, align 2
	ret i32 5
}
