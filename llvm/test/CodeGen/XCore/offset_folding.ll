; RUN: llc < %s -march=xcore | FileCheck %s

@a = external dso_local constant [0 x i32], section ".cp.rodata"
@b = external dso_local global [0 x i32]

define i32 *@f1() nounwind {
entry:
; CHECK-LABEL: f1:
; CHECK: ldaw r11, cp[a+4]
; CHECK: mov r0, r11
	%0 = getelementptr [0 x i32], [0 x i32]* @a, i32 0, i32 1
	ret i32* %0
}

define i32 *@f2() nounwind {
entry:
; CHECK-LABEL: f2:
; CHECK: ldaw r0, dp[b+4]
	%0 = getelementptr [0 x i32], [0 x i32]* @b, i32 0, i32 1
	ret i32* %0
}

; Don't fold negative offsets into cp / dp accesses to avoid a relocation
; error if the address + addend is less than the start of the cp / dp.

define i32 *@f3() nounwind {
entry:
; CHECK-LABEL: f3:
; CHECK: ldaw r11, cp[a]
; CHECK: sub r0, r11, 4
	%0 = getelementptr [0 x i32], [0 x i32]* @a, i32 0, i32 -1
	ret i32* %0
}

define i32 *@f4() nounwind {
entry:
; CHECK-LABEL: f4:
; CHECK: ldaw [[REG:r[0-9]+]], dp[b]
; CHECK: sub r0, [[REG]], 4
	%0 = getelementptr [0 x i32], [0 x i32]* @b, i32 0, i32 -1
	ret i32* %0
}
