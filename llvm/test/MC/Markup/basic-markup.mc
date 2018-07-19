// RUN: llvm-mcmarkup %s | FileCheck %s

	push	{<reg:r1>, <reg:r2>, <reg:r7>}
	sub	<reg:sp>, <imm:#132>
	ldr	<reg:r0>, <mem:[<reg:r0>, <imm:#4>]>


// CHECK: reg
// CHECK: reg
// CHECK: reg
// CHECK: reg
// CHECK: imm
// CHECK: reg
// CHECK: mem
// CHECK: reg
// CHECK: imm
