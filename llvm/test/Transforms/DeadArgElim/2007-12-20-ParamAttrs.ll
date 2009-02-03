; RUN: llvm-as < %s | opt -deadargelim | llvm-dis > %t
; RUN: cat %t | grep nounwind | count 2
; RUN: cat %t | grep signext | count 2
; RUN: cat %t | not grep inreg
; RUN: cat %t | not grep zeroext
; RUN: cat %t | not grep byval

	%struct = type { }

@g = global i8 0

define internal i8 @foo(i8* inreg %p, i8 signext %y, ... ) zeroext nounwind {
	store i8 %y, i8* @g
	ret i8 0
}

define i32 @bar() {
	%A = call i8(i8*, i8, ...)* @foo(i8* inreg null, i8 signext 1, %struct* byval null ) zeroext nounwind
	ret i32 0
}
