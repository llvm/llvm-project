; RUN: llvm-as < %s | llc

define void @test_add(i64 %AL, i64 %AH, i64 %BL, i64 %BH, i64* %RL, i64* %RH) {
entry:
	%tmp1 = zext i64 %AL to i128		; <i128> [#uses=1]
	%tmp23 = zext i64 %AH to i128		; <i128> [#uses=1]
	%tmp4 = shl i128 %tmp23, 64		; <i128> [#uses=1]
	%tmp5 = or i128 %tmp4, %tmp1		; <i128> [#uses=1]
	%tmp67 = zext i64 %BL to i128		; <i128> [#uses=1]
	%tmp89 = zext i64 %BH to i128		; <i128> [#uses=1]
	%tmp11 = shl i128 %tmp89, 64		; <i128> [#uses=1]
	%tmp12 = or i128 %tmp11, %tmp67		; <i128> [#uses=1]
	%tmp15 = add i128 %tmp12, %tmp5		; <i128> [#uses=2]
	%tmp1617 = trunc i128 %tmp15 to i64		; <i64> [#uses=1]
	store i64 %tmp1617, i64* %RL
	%tmp21 = lshr i128 %tmp15, 64		; <i128> [#uses=1]
	%tmp2122 = trunc i128 %tmp21 to i64		; <i64> [#uses=1]
	store i64 %tmp2122, i64* %RH
	ret void
}

define void @test_sub(i64 %AL, i64 %AH, i64 %BL, i64 %BH, i64* %RL, i64* %RH) {
entry:
	%tmp1 = zext i64 %AL to i128		; <i128> [#uses=1]
	%tmp23 = zext i64 %AH to i128		; <i128> [#uses=1]
	%tmp4 = shl i128 %tmp23, 64		; <i128> [#uses=1]
	%tmp5 = or i128 %tmp4, %tmp1		; <i128> [#uses=1]
	%tmp67 = zext i64 %BL to i128		; <i128> [#uses=1]
	%tmp89 = zext i64 %BH to i128		; <i128> [#uses=1]
	%tmp11 = shl i128 %tmp89, 64		; <i128> [#uses=1]
	%tmp12 = or i128 %tmp11, %tmp67		; <i128> [#uses=1]
	%tmp15 = sub i128 %tmp5, %tmp12		; <i128> [#uses=2]
	%tmp1617 = trunc i128 %tmp15 to i64		; <i64> [#uses=1]
	store i64 %tmp1617, i64* %RL
	%tmp21 = lshr i128 %tmp15, 64		; <i128> [#uses=1]
	%tmp2122 = trunc i128 %tmp21 to i64		; <i64> [#uses=1]
	store i64 %tmp2122, i64* %RH
	ret void
}
