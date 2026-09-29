// RUN: llvm-mc -filetype=obj -triple x86_64-pc-windows-msvc < %s | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT:   Section {{.*}} .text {
// CHECK-NEXT:     0x3 IMAGE_REL_AMD64_REL32 zed
// CHECK-NEXT:     0xA IMAGE_REL_AMD64_REL32 zed2
// CHECK-NEXT:   }
// CHECK-NEXT: ]

foo:
	leaq	zed(%rip), %rax
	leaq	zed2(%rip), %rax
	retq

	.section	.rdata,"dr",discard,zed
Lbar:
	.zero	2

	.globl	zed
zed = Lbar+1

	.globl	zed2
zed2 = Lbar
