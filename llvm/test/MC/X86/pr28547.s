// RUN: llvm-mc -triple x86_64-unknown-unknown %s
	.intel_syntax
ones:
	.float +1.0, +1.0, +1.0, +1.0
	vmovaps xmm15, xmmword ptr [rip+ones] # +1.0, +1.0, +1.0, +1.0
