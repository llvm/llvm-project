; RUN: llc < %s -mtriple=i686-- | grep sar | count 1
; RUN: llc < %s -mtriple=x86_64-- | not grep sar

define i32 @test(i32 %f12) nounwind {
	%tmp7.25 = lshr i32 %f12, 16		
	%tmp7.26 = trunc i32 %tmp7.25 to i8
	%tmp78.2 = sext i8 %tmp7.26 to i32
	ret i32 %tmp78.2
}

define i32 @test2(i32 %f12) nounwind {
	%f11 = shl i32 %f12, 8
	%tmp7.25 = ashr i32 %f11, 24
	ret i32 %tmp7.25
}

define i32 @test3(i32 %f12) nounwind {
	%f11 = shl i32 %f12, 13
	%tmp7.25 = ashr i32 %f11, 24
	ret i32 %tmp7.25
}

define i64 @test4(i64 %f12) nounwind {
	%f11 = shl i64 %f12, 32
	%tmp7.25 = ashr i64 %f11, 32
	ret i64 %tmp7.25
}

define i16 @test5(i16 %f12) nounwind {
	%f11 = shl i16 %f12, 2
	%tmp7.25 = ashr i16 %f11, 8
	ret i16 %tmp7.25
}

define i16 @test6(i16 %f12) nounwind {
	%f11 = shl i16 %f12, 8
	%tmp7.25 = ashr i16 %f11, 8
	ret i16 %tmp7.25
}
