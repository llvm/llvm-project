// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.7 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.6 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=x86_64-apple-ios7.0.0 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=x86_64-apple-macosx10.5 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.6 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.5 -filetype=obj | llvm-readobj -r | FileCheck %s
// RUN: llvm-mc < %s -triple=i686-apple-macosx10.4 -filetype=obj | llvm-readobj -r | FileCheck %s

	.globl	_bar
	.align	4, 0x90
_bar:
	.cfi_startproc
	.cfi_endproc

// CHECK:      Relocations [
// CHECK-NEXT: ]
