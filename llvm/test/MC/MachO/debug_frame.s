// RUN: llvm-mc -triple i386-apple-darwin %s -filetype=obj -o - | llvm-readobj -S --sd -r | FileCheck %s

// Make sure MC can handle file level .cfi_startproc and .cfi_endproc that creates
// an empty frame.
// rdar://10017184
_proc:
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
## %bb.0:                               ## %entry
	movl	$42, %eax
	ret
Ltmp1:
	.cfi_endproc
Leh_func_end0:

	.cfi_sections .debug_frame
Ltext_end:

// CHECK: Section {
// CHECK:     Index: 1
// CHECK:     Name: __debug_frame (5F 5F 64 65 62 75 67 5F 66 72 61 6D 65 00 00 00)
// CHECK:     Segment: __DWARF (5F 5F 44 57 41 52 46 00 00 00 00 00 00 00 00 00)
// CHECK:     Address: 0x8
// CHECK:     Size: 0x34
// CHECK:     Offset: 332
// CHECK:     Alignment: 2
// CHECK:     RelocationOffset: 0x180
// CHECK:     RelocationCount: 2
// CHECK:     Type: 0x0
// CHECK:     Attributes [ (0x20000)
// CHECK:       Debug (0x20000)
// CHECK:     ]
// CHECK:     Reserved1: 0x0
// CHECK:     Reserved2: 0x0
// CHECK:     SectionData (
// CHECK:       0000: 10000000 FFFFFFFF 04000400 017C080C  |.............|..|
// CHECK:       0010: 04048801 0C000000 00000000 00000000  |................|
// CHECK:       0020: 00000000 0C000000 00000000 00000000  |................|
// CHECK:       0030: 06000000                             |....|
// CHECK:     )
// CHECK:   }
// CHECK: ]
// CHECK: Relocations [
// CHECK:   Section __debug_frame {
// CHECK:     0x2C 0 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:     0x1C 0 2 0 GENERIC_RELOC_VANILLA 0 __text
// CHECK:   }
// CHECK: ]
