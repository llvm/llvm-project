@ Integrated test for ARM unwind directive parser and assembler.

@ This is a simplified real world test case generated from this C++ code
@ (with and without -fomit-frame-pointer)
@
@   extern void print(int, int, int, int, int);
@   extern void print(double, double, double, double, double);
@
@   void test(int a, int b, int c, int d, int e,
@             double m, double n, double p, double q, double r) {
@     try {
@       print(a, b, c, d, e);
@     } catch (...) {
@       print(m, n, p, q, r);
@     }
@   }
@
@ This test case should check the unwind opcode to adjust the opcode and
@ restore the general-purpose and VFP registers.


@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd | FileCheck %s


@-------------------------------------------------------------------------------
@ Assembly without frame pointer elimination
@-------------------------------------------------------------------------------
	.syntax unified
	.section	.TEST1
	.globl	func1
	.align	2
	.type	func1,%function
func1:
	.fnstart
	.save	{r4, r11, lr}
	push	{r4, r11, lr}
	.setfp	r11, sp, #4
	add	r11, sp, #4
	.vsave	{d8, d9, d10, d11, d12}
	vpush	{d8, d9, d10, d11, d12}
	.pad	#28
	sub	sp, sp, #28
	sub	sp, r11, #44
	vpop	{d8, d9, d10, d11, d12}
	pop	{r4, r11, pc}
.Ltmp1:
	.size	func1, .Ltmp1-func1
	.globl	__gxx_personality_v0
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 C94A9B01 B0818484           |.....J......|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ Assembly with frame pointer elimination
@-------------------------------------------------------------------------------
	.section	.TEST2
	.globl	func2
	.align	2
	.type	func2,%function
func2:
	.fnstart
	.save	{r4, lr}
	push	{r4, lr}
	.vsave	{d8, d9, d10, d11, d12}
	vpush	{d8, d9, d10, d11, d12}
	.pad	#24
	sub	sp, sp, #24
	add	sp, sp, #24
	vpop	{d8, d9, d10, d11, d12}
	pop	{r4, pc}
.Ltmp2:
	.size	func2, .Ltmp2-func2
	.globl	__gxx_personality_v0
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 84C90501 B0B0B0A8           |............|
@ CHECK:   )
@ CHECK: }
