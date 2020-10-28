// RUN: llvm-mc -triple=powerpc64-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r - | FileCheck %s

// Test correct relocation generation for thread-local storage using
// the local dynamic model.

	.file	"/home/espindola/llvm/llvm/test/CodeGen/PowerPC/tls-ld-obj.ll"
	.text
	.globl	main
	.align	2
	.type	main,@function
	.section	.opd,"aw",@progbits
main:                                   # @main
	.align	3
	.quad	.L.main
	.quad	.TOC.@tocbase
	.quad	0
	.text
.L.main:
# %bb.0:                                # %entry
	addis 3, 2, a@got@tlsld@ha
	addi 3, 3, a@got@tlsld@l
	li 4, 0
	bl __tls_get_addr(a@tlsld)
	nop
	stw 4, -4(1)
	addis 3, 3, a@dtprel@ha
	addi 3, 3, a@dtprel@l
	lwz 4, 0(3)
	extsw 3, 4
	blr
	.long	0
	.quad	0
.Ltmp0:
	.size	main, .Ltmp0-.L.main

	.hidden	a                       # @a
	.type	a,@object
	.section	.tbss,"awT",@nobits
	.globl	a
	.align	2
a:
	.long	0                       # 0x0
	.size	a, 4


// Verify generation of R_PPC64_GOT_TLSLD16_HA, R_PPC64_GOT_TLSLD16_LO,
// R_PPC64_TLSLD, R_PPC64_DTPREL16_HA, and R_PPC64_DTPREL16_LO for
// accessing external variable a, and R_PPC64_REL24 for the call to
// __tls_get_addr.
//
// CHECK: Relocations [
// CHECK:   Section {{.*}} .rela.text {
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TLSLD16_HA a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TLSLD16_LO a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_TLSLD          a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_REL24          __tls_get_addr
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_DTPREL16_HA    a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_DTPREL16_LO    a
// CHECK:   }
// CHECK: ]
