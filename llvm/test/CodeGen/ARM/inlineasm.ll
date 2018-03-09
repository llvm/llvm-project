; RUN: llc -mtriple=armv8-eabi -mattr=+neon %s -o - | FileCheck %s

define i32 @test1(i32 %tmp54) {
	%tmp56 = tail call i32 asm "uxtb16 $0,$1", "=r,r"( i32 %tmp54 )		; <i32> [#uses=1]
	ret i32 %tmp56
}

define void @test2() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret void
}

define float @t-constraint-int(i32 %i) {
	; CHECK-LABEL: t-constraint-int
	; CHECK: vcvt.f32.s32 {{s[0-9]+}}, {{s[0-9]+}}
	%ret = call float asm "vcvt.f32.s32 $0, $1\0A", "=t,t"(i32 %i)
	ret float %ret
}
