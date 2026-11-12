// RUN: llvm-mc -triple x86_64-unknown-unknown -filetype=obj %s | llvm-objdump -r - | FileCheck %s
// CHECK-NOT: RELOCATION RECORDS

// ensure that a .loc directive at the end of a section doesn't bleed into the
// following section previously this would produce a relocation for
// .other_section in the line table. But it should actually produce no line
// table entries at all.
	.text
	.file	1 "fail.cpp"
	.loc	1 7 3 prologue_end      # fail.cpp:7:3
	# addss   %xmm0, %xmm1

	.section	.other_section,"",@progbits
	.long	46                      # Length of Unit

