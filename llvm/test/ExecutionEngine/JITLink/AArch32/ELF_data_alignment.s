# REQUIRES: asserts
# RUN: llvm-mc -triple=armv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_armv7.o %s
# RUN: llvm-objdump -s --section=.rodata %t_armv7.o | FileCheck --check-prefix=CHECK-OBJ %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 %t_armv7.o -debug-only=jitlink 2>&1 \
# RUN:              | FileCheck --check-prefix=CHECK-LG %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 %t_armv7.o -check %s

# RUN: llvm-mc -triple=thumbv7-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t_thumbv7.o %s
# RUN: llvm-objdump -s --section=.rodata %t_thumbv7.o | FileCheck --check-prefix=CHECK-OBJ %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 %t_thumbv7.o -debug-only=jitlink 2>&1 \
# RUN:              | FileCheck --check-prefix=CHECK-LG %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb \
# RUN:              -slab-page-size 4096 %t_thumbv7.o -check %s

# The strings of "H1\00", "H2\00" and "H3\00" are encoded as
#               0x483100, 0x483200 and 0x483300 in the .rodata section.
# CHECK-OBJ: Contents of section .rodata:
# CHECK-OBJ: 0000 48310048 32004833 00                 H1.H2.H3.

# CHECK-LG: Starting link phase 1 for graph
# CHECK-LG: section .rodata:

# CHECK-LG:       block 0x0 size = 0x00000009, align = 1, alignment-offset = 0
# CHECK-LG-NEXT:    symbols:
# CHECK-LG-NEXT:      0x0 (block + 0x00000000): size: 0x00000003, linkage: strong, scope: default, live  -   Lstr.H1
# CHECK-LG-NEXT:      0x3 (block + 0x00000003): size: 0x00000003, linkage: strong, scope: default, live  -   Lstr.H2
# CHECK-LG-NOT:       0x2 (block + 0x00000002): size: 0x00000003, linkage: strong, scope: default, live  -   Lstr.H2
# CHECK-LG-NEXT:      0x6 (block + 0x00000006): size: 0x00000003, linkage: strong, scope: default, live  -   Lstr.H3

# jitlink-check: Lstr.H1 = 0x76ff0000
# jitlink-check: (*{4}(Lstr.H1))[23:0] = 0x003148
	.globl	Lstr.H1
	.type	Lstr.H1,%object
	.section	.rodata,"a",%progbits
Lstr.H1:
	.asciz	"H1"
	.size	Lstr.H1, 3

# H2 is unaligned as its beginning address is base address + 0x3
# Make sure the string we get is 0x003248 and not 0x324800
# jitlink-check: Lstr.H2 = 0x76ff0003
# jitlink-check: (*{4}(Lstr.H2))[23:0] = 0x003248
	.globl	Lstr.H2
	.type	Lstr.H2,%object
Lstr.H2:
	.asciz	"H2"
	.size	Lstr.H2, 3

# jitlink-check: Lstr.H3 = 0x76ff0006
# jitlink-check: (*{4}(Lstr.H3))[23:0] = 0x003348
	.globl	Lstr.H3
	.type	Lstr.H3,%object
Lstr.H3:
	.asciz	"H3"
	.size	Lstr.H3, 3

	.text
	.syntax unified
# Empty main function for jitlink to be happy
	.globl	main
	.type	main,%function
	.p2align	2
main:
	bx	lr
	.size	main,.-main
