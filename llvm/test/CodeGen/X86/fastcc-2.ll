; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -mattr=+sse2 | grep movsd
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -mattr=+sse2 | grep mov | count 1

define i32 @foo() nounwind {
entry:
	tail call fastcc void @bar( double 1.000000e+00 ) nounwind
	ret i32 0
}

declare fastcc void @bar(double)
