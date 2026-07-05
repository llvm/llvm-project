// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | llvm-readobj -S --sd - | FileCheck %s

// rdar://15526046

.text
.globl _main
_main:
	.space 0x01020f55, 0x90
bug:
	movl  $0, _key64b_9+4
.section __TEXT, __padding
	.space 0x515b91, 0
.data
	.space 0xa70, 0
.globl _key64b_9
_key64b_9:
	.long 1
	.long 2

// The movl instruction above should produce this encoding where the address
// of _key64b_9 is at 0x01537560.  This is testing falling back from using a
// scattered relocation to a normal relocation because the offset from the
// start of the section is more than 24-bits.  But need to get the item to
// be relocated, in this case _key64b_9+4, value correct in the instruction.
// 01020f55	c7056475530100000000	movl	$0x0, 0x1537564

// CHECK: SectionData (
// CHECK: F75530: 90909090 90909090 90909090 90909090  |................|
// CHECK: 1020F50: 90909090 90C70564 75530100 000000    |.......duS.....|
// CHECK: 75530: 00000000 00000000 00000000 00000000  |................|
// CHECK: )
