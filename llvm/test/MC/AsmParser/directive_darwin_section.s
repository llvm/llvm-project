# RUN: llvm-mc -triple i386-apple-darwin9 %s | FileCheck %s

# CHECK: .section __DWARF,__debug_frame,regular,debug
	.section	__DWARF,__debug_frame,regular,debug
