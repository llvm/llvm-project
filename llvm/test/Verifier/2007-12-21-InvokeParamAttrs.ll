; RUN: not llvm-as < %s > /dev/null 2>&1

declare void @foo(i8*)

define void @bar() {
	invoke void @foo(i8* signext null)
			to label %r unwind label %r
r:
	ret void
}
