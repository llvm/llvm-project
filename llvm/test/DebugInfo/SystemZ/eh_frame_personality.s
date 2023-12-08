# RUN: llvm-mc -triple=s390x-linux-gnu -filetype=obj %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

	.text
	.globl	foo
	.align	4
	.type	foo,@function
foo:                                    # @foo
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
	stmg	%r14, %r15, 112(%r15)
	.cfi_offset %r14, -48
	.cfi_offset %r15, -40
	aghi	%r15, -160
	.cfi_def_cfa_offset 320
	lmg	%r14, %r15, 272(%r15)
	br	%r14
	.size	foo, .-foo
	.cfi_endproc

	.section	.gcc_except_table,"a",@progbits
	.align	4
.Lexception0:

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align	8
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0

# The readelf rendering is:
#
# Contents of the .eh_frame section:
#
# 00000000 0000001c 00000000 CIE
#   Version:               3
#   Augmentation:          "zPLR"
#   Code alignment factor: 1
#   Data alignment factor: -8
#   Return address column: 14
#   Augmentation data:     9b ff ff ff ed 1b 1b
#
#   DW_CFA_def_cfa: r15 ofs 160
#   DW_CFA_nop
#   DW_CFA_nop
#   DW_CFA_nop
#
# 00000020 0000001c 00000024 FDE cie=00000000 pc=00000000..00000012
#   Augmentation data:     ff ff ff cf
#
#   DW_CFA_advance_loc: 6 to 00000006
#   DW_CFA_offset: r14 at cfa-48
#   DW_CFA_offset: r15 at cfa-40
#   DW_CFA_advance_loc: 4 to 0000000a
#   DW_CFA_def_cfa_offset: 320
#   DW_CFA_nop
#   DW_CFA_nop
#
# CHECK: Contents of section .eh_frame:
# CHECK-NEXT: 0000 0000001c 00000000 017a504c 52000178  .........zPLR..x
# CHECK-NEXT: 0010 0e079b00 0000001b 1b0c0fa0 01000000  ................
# CHECK-NEXT: 0020 0000001c 00000024 00000000 00000012  .......$........
# CHECK-NEXT: 0030 04000000 00468e06 8f05440e c0020000  .....F....D.....
