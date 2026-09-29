# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s
	.section .text

	.globl foo
	.type foo,@function
	.align 2
foo:
	bl printf@plt
	bl _GLOBAL_OFFSET_TABLE_@local-4
.LC1:
	.size foo, . - foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section {{.*}} .rela.text {
# CHECK-NEXT:     0x0 R_PPC_PLTREL24 printf 0x0
# CHECK-NEXT:     0x4 R_PPC_LOCAL24PC _GLOBAL_OFFSET_TABLE_ 0xFFFFFFFC
# CHECK-NEXT:   }
# CHECK-NEXT: ]
