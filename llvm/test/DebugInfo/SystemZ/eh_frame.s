# RUN: llvm-mc -triple=s390x-linux-gnu -filetype=obj %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

	.text
	.globl	check_largest_class
	.align	4
	.type	check_largest_class,@function
check_largest_class:
	.cfi_startproc
	stmg	%r13, %r15, 104(%r15)
	.cfi_offset %r13, -56
	.cfi_offset %r14, -48
	.cfi_offset %r15, -40
	aghi	%r15, -224
	.cfi_def_cfa_offset 384
	std	%f8, 160(%r15)
	std	%f9, 168(%r15)
	std	%f10, 176(%r15)
	std	%f11, 184(%r15)
	std	%f12, 192(%r15)
	std	%f13, 200(%r15)
	std	%f14, 208(%r15)
	std	%f15, 216(%r15)
	.cfi_offset %f8, -224
	.cfi_offset %f9, -216
	.cfi_offset %f10, -208
	.cfi_offset %f11, -200
	.cfi_offset %f12, -192
	.cfi_offset %f13, -184
	.cfi_offset %f14, -176
	.cfi_offset %f15, -168
	lmg	%r13, %r15, 328(%r15)
	br	%r14
	.size	check_largest_class, .-check_largest_class
	.cfi_endproc

# The readelf rendering is:
#
# Contents of the .eh_frame section:
#
# 00000000 0000000000000014 00000000 CIE
#   Version:               3
#   Augmentation:          "zR"
#   Code alignment factor: 1
#   Data alignment factor: -8
#   Return address column: 14
#   Augmentation data:     1b
#
#   DW_CFA_def_cfa: r15 ofs 160
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#
# 000000.. 000000000000002c 0000001c FDE cie=00000000 pc=0000000000000000..0000000000000032
#   DW_CFA_advance_loc: 6 to 0000000000000006
#   DW_CFA_offset: r13 at cfa-56
#   DW_CFA_offset: r14 at cfa-48
#   DW_CFA_offset: r15 at cfa-40
#   DW_CFA_advance_loc: 4 to 000000000000000a
#   DW_CFA_def_cfa_offset: 384
#   DW_CFA_advance_loc: 32 to 000000000000002a
#   DW_CFA_offset: r24 at cfa-224
#   DW_CFA_offset: r28 at cfa-216
#   DW_CFA_offset: r25 at cfa-208
#   DW_CFA_offset: r29 at cfa-200
#   DW_CFA_offset: r26 at cfa-192
#   DW_CFA_offset: r30 at cfa-184
#   DW_CFA_offset: r27 at cfa-176
#   DW_CFA_offset: r31 at cfa-168
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#
# CHECK: Contents of section .eh_frame:
# CHECK-NEXT: 0000 00000014 00000000 017a5200 01780e01  {{.*}}
# CHECK-NEXT: 0010 1b0c0fa0 01000000 0000002c 0000001c  {{.*}}
# CHECK-NEXT: 0020 00000000 00000032 00468d07 8e068f05  {{.*}}
# CHECK-NEXT: 0030 440e8003 60981c9c 1b991a9d 199a189e  {{.*}}
# CHECK-NEXT: 0040 179b169f 15000000                    {{.*}}
