# RUN: llvm-mc -filetype=obj -triple=ppc %s -o %t
# RUN: llvm-readelf -r %t | FileCheck %s 

# CHECK: Relocation section '.rela.debug_info' at offset 0xf8 contains 1 entries:
# CHECK-NEXT: Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# CHECK-NEXT: 00000000  0000024e R_PPC_DTPREL32         00000000   tls_rtp_var + 8000

	.text
	.globl	foo                             # -- Begin function foo
	.p2align	2
	.type	foo,@function
foo:                                    # @foo
# %bb.0:                                # %entry
	stwu 1, -16(1)
	stw 31, 12(1)
	mr	31, 1
	addis 3, 2, tls_rtp_var@tprel@ha
	addi 4, 3, tls_rtp_var@tprel@l
	lwz 3, 0(4)
	addi 3, 3, 1
	stw 3, 0(4)
	lis 4, my_global_var@ha
	lwz 3, my_global_var@l(4)
	addi 3, 3, 1
	stw 3, my_global_var@l(4)
	lwz 31, 12(1)
	addi 1, 1, 16
	blr

	.type	tls_rtp_var,@object             # @tls_rtp_var
	.section	.tdata,"awT",@progbits
	.globl	tls_rtp_var
	.p2align	2, 0x0
tls_rtp_var:
	.long	5                               # 0x5
	.size	tls_rtp_var, 4

	.type	my_global_var,@object           # @my_global_var
	.data
	.globl	my_global_var
	.p2align	2, 0x0
my_global_var:
	.long	7                               # 0x7
	.size	my_global_var, 4

	.section	.debug_info,"",@progbits
.Lcu_begin0:
.Ldebug_info_start0:
	.long	tls_rtp_var@DTPREL+32768
.Ldebug_info_end0:
