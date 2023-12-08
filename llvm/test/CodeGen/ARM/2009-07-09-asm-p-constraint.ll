; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o /dev/null

define void @test(ptr %x) nounwind {
entry:
	call void asm sideeffect "pld\09${0:a}", "r,~{cc}"(ptr %x) nounwind
	ret void
}
