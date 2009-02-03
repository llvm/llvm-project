; RUN: llvm-as < %s | llc -march=x86-64 | grep rol | count 2

define i64 @test1(i64 %x) nounwind  {
entry:
	%tmp2 = lshr i64 %x, 55		; <i64> [#uses=1]
	%tmp4 = shl i64 %x, 9		; <i64> [#uses=1]
	%tmp5 = or i64 %tmp2, %tmp4		; <i64> [#uses=1]
	ret i64 %tmp5
}

define i64 @test2(i32 %x) nounwind  {
entry:
	%tmp2 = lshr i32 %x, 22		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 10		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp2, %tmp4		; <i32> [#uses=1]
	%tmp56 = zext i32 %tmp5 to i64		; <i64> [#uses=1]
	ret i64 %tmp56
}

