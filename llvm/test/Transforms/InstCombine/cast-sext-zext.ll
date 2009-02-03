; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep sext
; XFAIL: *

define zeroext i16 @t(i8 zeroext %on_off, i16* nocapture %puls) nounwind readonly {
entry:
	%0 = zext i8 %on_off to i32
	%1 = add i32 %0, -1
	%2 = sext i32 %1 to i64
	%3 = getelementptr i16* %puls, i64 %2
	%4 = load i16* %3, align 2
	ret i16 %4
}
