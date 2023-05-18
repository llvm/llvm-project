# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s
# RUN: llvm-mc -triple x86_64-pc-win32 %s | FileCheck %s
	.cg_profile a, b, 32
	.cg_profile freq, a, 11
	.cg_profile freq, b, 20

# CHECK: .cg_profile a, b, 32
# CHECK: .cg_profile freq, a, 11
# CHECK: .cg_profile freq, b, 20
