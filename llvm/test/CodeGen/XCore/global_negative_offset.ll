; RUN: llc < %s -march=xcore | FileCheck %s

; Don't fold negative offsets into cp / dp accesses to avoid a relocation
; error if the address + addend is less than the start of the cp / dp.

@a = external constant [0 x i32], section ".cp.rodata"
@b = external global [0 x i32]

define i32 *@f() nounwind {
entry:
; CHECK: f:
; CHECK: ldaw r11, cp[a]
; CHECK: sub r0, r11, 4
	%0 = getelementptr [0 x i32]* @a, i32 0, i32 -1
	ret i32* %0
}

define i32 *@g() nounwind {
entry:
; CHECK: g:
; CHECK: ldaw [[REG:r[0-9]+]], dp[b]
; CHECK: sub r0, [[REG]], 4
	%0 = getelementptr [0 x i32]* @b, i32 0, i32 -1
	ret i32* %0
}
