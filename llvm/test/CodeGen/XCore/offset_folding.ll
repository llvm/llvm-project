; RUN: llc < %s -mtriple=xcore | FileCheck %s

@a = external dso_local constant [0 x i32], section ".cp.rodata"
@b = external dso_local global [0 x i32]

define ptr @f1() nounwind {
entry:
; CHECK-LABEL: f1:
; CHECK: ldaw r11, cp[a+4]
; CHECK: mov r0, r11
	%0 = getelementptr [0 x i32], ptr @a, i32 0, i32 1
	ret ptr %0
}

define ptr @f2() nounwind {
entry:
; CHECK-LABEL: f2:
; CHECK: ldaw r0, dp[b+4]
	%0 = getelementptr [0 x i32], ptr @b, i32 0, i32 1
	ret ptr %0
}

; Don't fold negative offsets into cp / dp accesses to avoid a relocation
; error if the address + addend is less than the start of the cp / dp.

define ptr @f3() nounwind {
entry:
; CHECK-LABEL: f3:
; CHECK: ldaw r11, cp[a]
; CHECK: sub r0, r11, 4
	%0 = getelementptr [0 x i32], ptr @a, i32 0, i32 -1
	ret ptr %0
}

define ptr @f4() nounwind {
entry:
; CHECK-LABEL: f4:
; CHECK: ldaw [[REG:r[0-9]+]], dp[b]
; CHECK: sub r0, [[REG]], 4
	%0 = getelementptr [0 x i32], ptr @b, i32 0, i32 -1
	ret ptr %0
}
