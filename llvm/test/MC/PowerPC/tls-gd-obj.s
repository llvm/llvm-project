// RUN: llvm-mc -triple=powerpc64-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r - | FileCheck %s

// Test correct relocation generation for thread-local storage using
// the general dynamic model and integrated assembly.


	.file	"/home/espindola/llvm/llvm/test/CodeGen/PowerPC/tls-gd-obj.ll"
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
	addis 3, 2, a@got@tlsgd@ha
	addi 3, 3, a@got@tlsgd@l
	li 4, 0
	bl __tls_get_addr(a@tlsgd)
	nop
	stw 4, -4(1)
	lwz 4, 0(3)
	extsw 3, 4
	blr
	.long	0
	.quad	0
.Ltmp0:
	.size	main, .Ltmp0-.L.main

	.type	a,@object               # @a
	.section	.tbss,"awT",@nobits
	.globl	a
	.align	2
a:
	.long	0                       # 0x0
	.size	a, 4


// Verify generation of R_PPC64_GOT_TLSGD16_HA, R_PPC64_GOT_TLSGD16_LO,
// and R_PPC64_TLSGD for accessing external variable a, and R_PPC64_REL24
// for the call to __tls_get_addr.
//
// CHECK: Relocations [
// CHECK:   Section {{.*}} .rela.text {
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TLSGD16_HA a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TLSGD16_LO a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_TLSGD          a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_REL24          __tls_get_addr
// CHECK:   }
// CHECK: ]
