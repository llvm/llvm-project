; RUN: llvm-as < %s | llc -march=arm -mattr=+v6

define i32 @test1(i32 %tmp54) {
	%tmp56 = tail call i32 asm "uxtb16 $0,$1", "=r,r"( i32 %tmp54 )		; <i32> [#uses=1]
	ret i32 %tmp56
}

define void @test2() {
	%tmp1 = call i64 asm "ldmia $1!, {$0, ${0:H}}", "=r,=*r,1"( i32** null, i32* null )		; <i64> [#uses=2]
	%tmp2 = lshr i64 %tmp1, 32		; <i64> [#uses=1]
	%tmp3 = trunc i64 %tmp2 to i32		; <i32> [#uses=1]
	%tmp4 = call i32 asm "pkhbt $0, $1, $2, lsl #16", "=r,r,r"( i32 0, i32 %tmp3 )		; <i32> [#uses=0]
	ret void
}

define void @test3() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret void
}
