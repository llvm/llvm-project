// RUN: llvm-mc -triple x86_64-apple-macosx10.6 -filetype obj -o - %s | llvm-readobj --sections - | FileCheck %s

_label:
	.cfi_startproc
	.cfi_def_cfa_register rsp
	.cfi_endproc

// CHECK: Section {
// CHECK:   Name: __eh_frame
// CHECK: }

