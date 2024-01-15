// RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 -abs printf=0x76bbe880 -show-entry-es %t.o | FileCheck %s

// Check that main is a thumb symbol (with LSB set) and printf is arm (with LSB clear)
//
// CHECK-LABEL: JITDylib "main"
// CHECK-NEXT: Link order: [ ("main", MatchAllSymbols), ("Process", MatchExportedSymbolsOnly) ]
// CHECK-NEXT: Symbol table:
// CHECK-NEXT:    "main":   0x{{[0-9a-f]+[13579bdf]}} [Callable] Ready
// CHECK-NEXT:    "printf": 0x76bbe880 [Data] Ready

	.globl	main
	.p2align	2
	.type	main,%function
	.code	16
	.thumb_func
main:
	.fnstart
	.save	{r7, lr}
	push	{r7, lr}
	.setfp	r7, sp
	mov	r7, sp
	.pad	#8
	sub	sp, #8
	movs	r0, #0
	str	r0, [sp]
	str	r0, [sp, #4]
	ldr	r0, .LCPI0_0
.LPC0_0:
	add	r0, pc
	bl	printf
	ldr	r0, [sp]
	add	sp, #8
	pop	{r7, pc}

	.p2align	2
.LCPI0_0:
	.long	.L.str-(.LPC0_0+4)

	.size	main, .-main
	.cantunwind
	.fnend

	.type	.L.str,%object
	.section	.rodata.str1.1,"aMS",%progbits,1
.L.str:
	.asciz	"Hello AArch32!\n"
	.size	.L.str, 12
