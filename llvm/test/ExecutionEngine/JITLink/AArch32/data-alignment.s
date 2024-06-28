# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv7.o %s
# RUN: llvm-objdump -s --section=.rodata %t_armv7.o | FileCheck --check-prefix=CHECK-DATA %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -check %s %t_armv7.o

# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7.o %s
# RUN: llvm-objdump -s --section=.rodata %t_thumbv7.o | FileCheck --check-prefix=CHECK-DATA %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 -check %s %t_armv7.o

# The strings of "H1\00", "H2\00" and "H3\00" are encoded as
#               0x483100, 0x483200 and 0x483300 .rodata section.
# CHECK-DATA: Contents of section .rodata:
# CHECK-DATA: 0000 48310048 32004833 00

# FIXME: The expression we want is either *{3}(Lstr.H1) = ...
#                                  or *{4}(Lstr.H1) & 0x00ffffff = ...
#        The first is not supported and the latter segfaults.
#        Also, whitespaces are not recognized and not consumed by the checker.

# jitlink-check: 0x00ffffff&*{4}(Lstr.H1) = 0x003148
	.globl	Lstr.H1
	.type	Lstr.H1,%object
	.section	.rodata,"a",%progbits
Lstr.H1:
	.asciz	"H1"
	.size	Lstr.H1, 3

# Not 0x324800
# jitlink-check: 0x00ffffff&*{4}(Lstr.H2) = 0x003248
	.globl	Lstr.H2
	.type	Lstr.H2,%object
Lstr.H2:
	.asciz	"H2"
	.size	Lstr.H2, 3

# jitlink-check: 0x00ffffff&*{4}(Lstr.H3) = 0x003348
	.globl	Lstr.H3
	.type	Lstr.H3,%object
Lstr.H3:
	.asciz	"H3"
	.size	Lstr.H3, 3

# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	2
main:
	bx	lr
	.size	main,.-main
