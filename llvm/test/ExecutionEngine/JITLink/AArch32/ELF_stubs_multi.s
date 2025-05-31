# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate=10Kb -slab-page-size=4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/out.o

	.text
	.syntax unified

# Check that a single external symbol can have multiple stubs. We access them
# with the extra stub-index argument to stub_addr(). Stubs are sorted by
# ascending size (because the default memory manager lays out blocks by size).

# Thumb relocation site emits thumb stub
# jitlink-check: decode_operand(test_stub_thumb, 0) = stub_addr(out.o, ext, thumb) - next_pc(test_stub_thumb)
	.globl  test_stub_thumb
	.type	test_stub_thumb,%function
	.p2align	1
	.code	16
	.thumb_func
test_stub_thumb:
	b	ext
	.size	test_stub_thumb, .-test_stub_thumb

# Arm relocation site emits arm stub
# jitlink-check: decode_operand(test_stub_arm, 0) = stub_addr(out.o, ext, arm) - next_pc(test_stub_arm)
	.globl  test_stub_arm
	.type	test_stub_arm,%function
	.p2align	2
	.code	32
test_stub_arm:
	b	ext
	.size	test_stub_arm, .-test_stub_arm

# This test is executable with both, Arm and Thumb `ext` functions. It only has
# to return (directly to main) with `bx lr`. For example:
#   > echo "void ext() {}" | clang -target armv7-linux-gnueabihf -o ext-arm.o -c -xc -
#   > llvm-jitlink ext-arm.o out.o
#
	.globl	main
	.type	main,%function
	.p2align	2
main:
	push	{lr}
	bl	test_stub_arm
	bl	test_stub_thumb
	movw	r0, #0
	pop	{pc}
	.size	main, .-main
