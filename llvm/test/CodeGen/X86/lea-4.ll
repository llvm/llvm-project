; RUN: llvm-as < %s | llc -march=x86-64 | grep lea | count 2

define zeroext i16 @t1(i32 %on_off) nounwind {
entry:
	%0 = sub i32 %on_off, 1
	%1 = mul i32 %0, 2
	%2 = trunc i32 %1 to i16
	%3 = zext i16 %2 to i32
	%4 = trunc i32 %3 to i16
	ret i16 %4
}

define i32 @t2(i32 %on_off) nounwind {
entry:
	%0 = sub i32 %on_off, 1
	%1 = mul i32 %0, 2
        %2 = and i32 %1, 65535
	ret i32 %2
}
