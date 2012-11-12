; RUN: llc < %s -march=arm -mattr=+v6

define i32 @test1(i32 %tmp54) {
	%tmp56 = tail call i32 asm "uxtb16 $0,$1", "=r,r"( i32 %tmp54 )		; <i32> [#uses=1]
	ret i32 %tmp56
}

define void @test2() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret void
}
