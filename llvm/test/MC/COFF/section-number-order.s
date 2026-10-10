# RUN: llvm-mc -triple x86_64-pc-windows-msvc -filetype=obj %s -o %t.obj
# RUN: llvm-objdump -h -r -t %t.obj | FileCheck %s

# Sections are numbered in the order in which they are created. Linkers order
# the .CRT$XCU contributions of an object file by section number, so the entry
# for an inline variable's initializer, which is associative with the
# variable, must stay ahead of the entry for the ordered initializers that
# follow it.

	.section	.bss,"bw",discard,options
	.globl	options
options:
	.long	0

	.section	.CRT$XCU,"dr",associative,options
	.quad	init_options

	.section	.CRT$XCU,"dr"
	.quad	init_ordered

# An associative section is never numbered before the section it is
# associated with, even if it is created first.

	.section	.xdata,"dr",associative,func
	.long	0

	.section	.text,"xr",discard,func
	.globl	func
func:
	retq

# CHECK:      Sections:
# CHECK:        3 .bss
# CHECK-NEXT:   4 .CRT$XCU
# CHECK-NEXT:   5 .CRT$XCU
# CHECK-NEXT:   6 .text
# CHECK-NEXT:   7 .xdata

# CHECK:      SYMBOL TABLE:
# CHECK:      (sec  5){{.*}} .CRT$XCU
# CHECK-NEXT: AUX {{.*}} assoc 4 comdat 5
# CHECK:      (sec  8){{.*}} .xdata
# CHECK-NEXT: AUX {{.*}} assoc 7 comdat 5

# CHECK:      RELOCATION RECORDS FOR [.CRT$XCU]:
# CHECK-NEXT: OFFSET TYPE VALUE
# CHECK-NEXT: IMAGE_REL_AMD64_ADDR64 init_options
# CHECK:      RELOCATION RECORDS FOR [.CRT$XCU]:
# CHECK-NEXT: OFFSET TYPE VALUE
# CHECK-NEXT: IMAGE_REL_AMD64_ADDR64 init_ordered
