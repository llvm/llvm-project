# RUN: rm -rf %t && mkdir -p %t/armv7 && mkdir -p %t/thumbv7
# RUN: llvm-mc -triple=armv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t/armv7/out.o %s
# RUN: llvm-objdump -r %t/armv7/out.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs target=0x76bbe88f -check %s %t/armv7/out.o

# RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t/thumbv7/out.o %s
# RUN: llvm-objdump -r %t/thumbv7/out.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs target=0x76bbe88f -check %s %t/thumbv7/out.o

	.data
	.global target

	.text
	.syntax unified

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_ABS32 target
# jitlink-check: *{4}(abs32) = target
	.global abs32
abs32:
	.word target
	.size abs32, .-abs32

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_REL32 target
# jitlink-check: (rel32 + *{4}(rel32))[31:0] = target
	.global rel32
rel32:
	.word target - .
	.size rel32, .-rel32

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_TARGET1 target
# jitlink-check: *{4}(target1_abs32) = target
	.global target1_abs32
target1_abs32:
	.word target(target1)
	.size	target1_abs32, .-target1_abs32

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_GOT_PREL target
#
# The GOT entry contains the absolute address of the external:
# jitlink-check: *{4}(got_addr(out.o, target)) = target
#
# The embedded offset value contains the offset to the GOT entry relative to pc.
# The +12 accounts for the ARM branch offset (8) and the .LPC offset (4), which
# is stored as initial addend inline.
# FIXME: We shouldn't need to substract the 64-bit sign-extension manually.
# jitlink-check: *{4}got_prel_offset = got_addr(out.o, target) - (got_prel + 12) - 0xffffffff00000000
	.globl	got_prel
	.type	got_prel,%function
	.p2align	2
	.code	32
got_prel:
	ldr	r0, .LCPI
.LPC:
	ldr	r0, [pc, r0]
	ldr	r0, [r0]
	bx	lr
# Actual relocation site is on the embedded offset value:
	.globl	got_prel_offset
got_prel_offset:
.LCPI:
	.long	target(GOT_PREL)-((.LPC+8)-.LCPI)
	.size	got_prel_offset, .-got_prel_offset
	.size	got_prel, .-got_prel

# EH personality routine
# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_NONE __aeabi_unwind_cpp_pr0
	.globl __aeabi_unwind_cpp_pr0
	.type __aeabi_unwind_cpp_pr0,%function
	.align 2
__aeabi_unwind_cpp_pr0:
	bx lr

# Generate reference to EH personality (right now we ignore the resulting
# R_ARM_PREL31 relocation since it's in .ARM.exidx)
	.globl  prel31
	.type   prel31,%function
	.align  2
prel31:
	.fnstart
	.save   {r11, lr}
	push    {r11, lr}
	.setfp  r11, sp
	mov     r11, sp
	pop     {r11, lr}
	mov     pc, lr
	.size   prel31,.-prel31
	.fnend

# This test is executable with any 4-byte external target:
#  > echo "unsigned target = 42;" | clang -target armv7-linux-gnueabihf -o target.o -c -xc -
#  > llvm-jitlink target.o armv7/out.o
#
	.globl  main
	.type main, %function
	.p2align  2
main:
	push	{lr}
	bl	got_prel
	pop	{pc}
	.size   main, .-main
