; RUN: llc < %s -march=arm -mattr=+v6

define i32 @test3() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret i32 11
}
