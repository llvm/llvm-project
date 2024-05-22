# RUN: llvm-mc -triple=armv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv7.o %s
# RUN: llvm-objdump -r %t_armv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs target=0x76bbe88f -check %s %t_armv7.o

# RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7.o %s
# RUN: llvm-objdump -r %t_thumbv7.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:              -abs target=0x76bbe88f -check %s %t_thumbv7.o

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

# Empty main function for jitlink to be happy
	.globl  main
	.type main, %function
	.p2align  2
main:
	bx lr
	.size   main, .-main
