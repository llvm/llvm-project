# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

	.cg_profile a, b, 32
	.cg_profile freq, a, 11
	.cg_profile freq, b, 20

# CHECK: .cg_profile a, b, 32
# CHECK: .cg_profile freq, a, 11
# CHECK: .cg_profile freq, b, 20
