# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-objdump - -h -t | FileCheck %s

# PR38607: We assemble this, and make .CRT$XCU comdat with .rdata even though
# .rdata isn't comdat. It's not clear what the semantics are, but we assemble
# it anyway.

# CHECK: Sections:
# CHECK: Idx Name          Size     VMA              Type
# CHECK:   3 .rdata        00000004 0000000000000000 DATA
# CHECK:   4 .CRT$XCU      00000008 0000000000000000 DATA
# CHECK: SYMBOL TABLE:
# CHECK: [ 6](sec  4)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .rdata
# CHECK: AUX scnlen 0x4 nreloc 0 nlnno 0 checksum 0x0 assoc 4 comdat 0
# CHECK: [ 8](sec  5)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .CRT$XCU
# CHECK: AUX scnlen 0x8 nreloc 1 nlnno 0 checksum 0x0 assoc 4 comdat 5
# CHECK: [10](sec  0)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 register_x

	.section	.rdata,"dr"
	.p2align	2               # @x
.Lprivate:
	.long	0                       # 0x0

	.section	.CRT$XCU,"dr",associative,.Lprivate
	.p2align	3
	.quad	register_x

