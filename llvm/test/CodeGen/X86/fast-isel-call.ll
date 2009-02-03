; RUN: llvm-as < %s | llc -fast-isel -march=x86 | grep and

define i32 @t() nounwind {
tak:
	%tmp = call i1 @foo()
	br i1 %tmp, label %BB1, label %BB2
BB1:
	ret i32 1
BB2:
	ret i32 0
}

declare i1 @foo() zeroext nounwind
