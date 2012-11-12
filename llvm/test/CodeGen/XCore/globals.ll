; RUN: llc < %s -march=xcore -mcpu=xs1b-generic | FileCheck %s

define i32 *@addr_G1() {
entry:
; CHECK: addr_G1:
; CHECK: ldaw r0, dp[G1]
	ret i32* @G1
}

define i32 *@addr_G2() {
entry:
; CHECK: addr_G2:
; CHECK: ldaw r0, dp[G2]
	ret i32* @G2
}

define i32 *@addr_G3() {
entry:
; CHECK: addr_G3:
; CHECK: ldaw r11, cp[G3]
; CHECK: mov r0, r11
	ret i32* @G3
}

define i32 **@addr_G4() {
entry:
; CHECK: addr_G4:
; CHECK: ldaw r0, dp[G4]
	ret i32** @G4
}

define i32 **@addr_G5() {
entry:
; CHECK: addr_G5:
; CHECK: ldaw r11, cp[G5]
; CHECK: mov r0, r11
	ret i32** @G5
}

define i32 **@addr_G6() {
entry:
; CHECK: addr_G6:
; CHECK: ldaw r0, dp[G6]
	ret i32** @G6
}

define i32 **@addr_G7() {
entry:
; CHECK: addr_G7:
; CHECK: ldaw r11, cp[G7]
; CHECK: mov r0, r11
	ret i32** @G7
}

define i32 *@addr_G8() {
entry:
; CHECK: addr_G8:
; CHECK: ldaw r0, dp[G8]
	ret i32* @G8
}

@G1 = global i32 4712
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G1:

@G2 = global i32 0
; CHECK: .section .dp.bss,"awd",@nobits
; CHECK: G2:

@G3 = unnamed_addr constant i32 9401
; CHECK: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: G3:

@G4 = global i32* @G1
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G4:

@G5 = unnamed_addr constant i32* @G1
; CHECK: .section .cp.rodata,"ac",@progbits
; CHECK: G5:

@G6 = global i32* @G8
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G6:

@G7 = unnamed_addr constant i32* @G8
; CHECK: .section .cp.rodata,"ac",@progbits
; CHECK: G7:

@G8 = internal global i32 9312
; CHECK: .section .dp.data,"awd",@progbits
; CHECK: G8:
