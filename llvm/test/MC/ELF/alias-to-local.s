// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu < %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .rela.text {
// CHECK-NEXT:     0x1 R_X86_64_32 zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

foo:
	movl	$zed, %eax


	.section	.data.bar,"aGw",@progbits,zed,comdat
bar:
	.byte	42

	.globl	zed
zed = bar
