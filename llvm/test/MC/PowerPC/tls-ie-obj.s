// RUN: llvm-mc -triple=powerpc64-pc-linux -filetype=obj %s -o - | \
// RUN: llvm-readobj -r - | FileCheck %s

// Test correct relocation generation for thread-local storage
// using the initial-exec model and integrated assembly.

	.file	"/home/espindola/llvm/llvm/test/CodeGen/PowerPC/tls-ie-obj.ll"
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
	li 3, 0
	addis 4, 2, a@got@tprel@ha
	ld 4, a@got@tprel@l(4)
	add 4, 4, a@tls
	stw 3, -4(1)
	lwz 3, 0(4)
	extsw 3, 3
	blr
	.long	0
	.quad	0
.Ltmp0:
	.size	main, .Ltmp0-.L.main


// Verify generation of R_PPC64_GOT_TPREL16_DS and R_PPC64_TLS for
// accessing external variable a.
//
// CHECK: Relocations [
// CHECK:   Section {{.*}} .rela.text {
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_HA    a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_GOT_TPREL16_LO_DS a
// CHECK:     0x{{[0-9,A-F]+}} R_PPC64_TLS               a
// CHECK:   }
// CHECK: ]
