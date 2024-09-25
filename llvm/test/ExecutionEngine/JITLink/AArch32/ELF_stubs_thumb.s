# RUN: rm -rf %t && mkdir -p %t/thumbv7m && mkdir -p %t/thumbv7
#
# RUN: llvm-mc -triple=thumbv7m-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/thumbv7m/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/thumbv7m/out.o
#
# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes \
# RUN:         -filetype=obj -o %t/thumbv7/out.o %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 \
# RUN:              -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs ext=0x76bbe880 -check %s %t/thumbv7/out.o


	.text
	.syntax unified

# Check that calls/jumps to external functions trigger the generation of
# branch-range extension stubs. These stubs don't follow the default PLT model
# where the branch-target address is loaded from a GOT entry. Instead, they
# hard-code it in the immediate field.
#
# jitlink-check: decode_operand(test_external_call, 2) = stub_addr(out.o, ext) - next_pc(test_external_call)
# jitlink-check: decode_operand(test_external_jump, 0) = stub_addr(out.o, ext) - next_pc(test_external_jump)
	.globl  test_external_call
	.type	test_external_call,%function
	.p2align	1
	.code	16
	.thumb_func
test_external_call:
	bl	ext
	.size test_external_call, .-test_external_call

	.globl  test_external_jump
	.type	test_external_jump,%function
	.p2align	1
	.code	16
	.thumb_func
test_external_jump:
	b	ext
	.size test_external_jump, .-test_external_jump

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	1
	.code	16
	.thumb_func
main:
	bx	lr
	.size	main,	.-main
