# RUN: rm -rf %t && mkdir -p %t/armv4t && mkdir -p %t/armv6 && mkdir -p %t/armv7
#
# RUN: llvm-mc -triple=armv4t-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/armv4t/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/armv4t/out.o
#
# RUN: llvm-mc -triple=armv6-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/armv6/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/armv6/out.o
#
# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/armv7/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/armv7/out.o

	.text
	.syntax unified

# Check that calls/jumps to external functions trigger the generation of
# branch-range extension stubs. These stubs don't follow the default PLT model
# where the branch-target address is loaded from a GOT entry. Instead, they
# hard-code it in the immediate field.

# The external function ext will return to the caller directly.
# jitlink-check: decode_operand(test_arm_jump, 0) = stub_addr(out.o, ext) - next_pc(test_arm_jump)
	.globl	test_arm_jump
	.type	test_arm_jump,%function
	.p2align	2
test_arm_jump:
	b	ext
	.size	test_arm_jump, .-test_arm_jump

# The branch-with-link sets the LR register so that the external function ext
# returns to us. We have to save the register (push) and return to main manually
# (pop). This adds the +4 offset for the bl instruction we decode:
# jitlink-check: decode_operand(test_arm_call + 4, 0) = stub_addr(out.o, ext) - next_pc(test_arm_call) - 4
	.globl  test_arm_call
	.type	test_arm_call,%function
	.p2align	2
test_arm_call:
	push	{lr}
	bl	ext
	pop	{pc}
	.size	test_arm_call, .-test_arm_call

# This test is executable with any Arm (and for v7+ also Thumb) `ext` functions.
# It only has to return with `bx lr`. For example:
#   > echo "void ext() {}" | clang -target armv7-linux-gnueabihf -o ext.o -c -xc -
#   > llvm-jitlink ext.o out.o
#
	.globl	main
	.type	main,%function
	.p2align	2
main:
	push	{lr}
	bl	test_arm_call
	bl	test_arm_jump
	mov	r0, #0
	pop	{pc}
	.size	main, .-main
