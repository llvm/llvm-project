# RUN: llvm-mc -triple mips-unknown-unknown %s
#this test produces no output so there isS no FileCheck call
$BB0_2:
  .ent directives_test
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
	.set	noat
$JTI0_0:
	.gpword	($BB0_2)
	.set  at=$12
	.set macro
	.set reorder
	.end directives_test
