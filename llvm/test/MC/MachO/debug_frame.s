// RUN: llvm-mc -triple i386-apple-darwin %s -filetype=obj -o - | macho-dump | FileCheck %s

// Make sure MC can handle file level .cfi_startproc and .cfi_endproc that creates
// an empty frame.
// rdar://10017184
.cfi_startproc
.cfi_endproc

// Check that we don't produce a relocation for the CIE pointer and therefore
// we have only one relocation in __debug_frame.

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_f
	.align	4, 0x90
_f:                                     ## @f
Ltmp0:
	.cfi_startproc
## BB#0:                                ## %entry
	movl	$42, %eax
	ret
Ltmp1:
	.cfi_endproc
Leh_func_end0:

	.cfi_sections .debug_frame
Ltext_end:

// CHECK:       (('section_name', '__debug_frame\x00\x00\x00')
// CHECK-NEXT:   ('segment_name', '__DWARF\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:   ('address', 8)
// CHECK-NEXT:   ('size', 52)
// CHECK-NEXT:   ('offset', 332)
// CHECK-NEXT:   ('alignment', 2)
// CHECK-NEXT:   ('reloc_offset', 384)
// CHECK-NEXT:   ('num_reloc', 2)
// CHECK-NEXT:   ('flags', 0x2000000)
// CHECK-NEXT:   ('reserved1', 0)
// CHECK-NEXT:   ('reserved2', 0)
// CHECK-NEXT:  ),
// CHECK-NEXT: ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:   (('word-0', 0x2c),
// CHECK-NEXT:    ('word-1', 0x4000001)),
// CHECK-NEXT:   # Relocation 1
// CHECK-NEXT:   (('word-0', 0x1c),
// CHECK-NEXT:    ('word-1', 0x4000001)),
// CHECK-NEXT: ])
