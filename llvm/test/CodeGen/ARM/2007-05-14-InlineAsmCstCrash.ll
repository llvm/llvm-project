; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o /dev/null

define i32 @test3() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret i32 11
}
