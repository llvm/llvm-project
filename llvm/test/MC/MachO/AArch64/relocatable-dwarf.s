// With -macho-relocatable-dwarf, a reference from one DWARF section into
// another is a relocation, and every __DWARF section is numbered from address 0
// so that the relocated field holds the DWARF offset. The section data keeps
// its place in the file.

// RUN: llvm-mc -triple arm64-apple-macos14 -filetype=obj %s -o %t.o
// RUN: llvm-readobj --sections %t.o | FileCheck %s --check-prefix=PLAIN-SECT
// RUN: llvm-objdump -s --section=__debug_info %t.o | FileCheck %s --check-prefix=PLAIN-FIELD

// RUN: llvm-mc -triple arm64-apple-macos14 -filetype=obj -macho-relocatable-dwarf %s -o %t.reloc.o
// RUN: llvm-readobj --sections %t.reloc.o | FileCheck %s --check-prefix=RELOC-SECT
// RUN: llvm-objdump -r -s --section=__debug_info %t.reloc.o | FileCheck %s --check-prefix=RELOC-FIELD

// PLAIN-SECT:      Name: __debug_str
// PLAIN-SECT-NEXT: Segment: __DWARF
// PLAIN-SECT-NEXT: Address: 0x4
// PLAIN-SECT:      Name: __debug_info
// PLAIN-SECT-NEXT: Segment: __DWARF
// PLAIN-SECT-NEXT: Address: 0x11

// Without relocatable DWARF, the field holds the address of "second".
// PLAIN-FIELD: 0011 0a000000

// RELOC-SECT:      Name: __text
// RELOC-SECT-NEXT: Segment: __TEXT
// RELOC-SECT-NEXT: Address: 0x0
// RELOC-SECT-NEXT: Size: 0x4
// RELOC-SECT-NEXT: Offset: [[#TEXT:]]
// RELOC-SECT:      Name: __debug_str
// RELOC-SECT-NEXT: Segment: __DWARF
// RELOC-SECT-NEXT: Address: 0x0
// RELOC-SECT-NEXT: Size: 0xD
// RELOC-SECT-NEXT: Offset: [[#TEXT + 4]]
// RELOC-SECT:      Name: __debug_info
// RELOC-SECT-NEXT: Segment: __DWARF
// RELOC-SECT-NEXT: Address: 0x0
// RELOC-SECT-NEXT: Size: 0x4
// RELOC-SECT-NEXT: Offset: [[#TEXT + 17]]

// With relocatable DWARF, it holds the offset of "second" in __debug_str.
// RELOC-FIELD:      RELOCATION RECORDS FOR [__debug_info]:
// RELOC-FIELD-NEXT: OFFSET           TYPE                     VALUE
// RELOC-FIELD-NEXT: 0000000000000000 ARM64_RELOC_UNSIGNED     __debug_str
// RELOC-FIELD:      0000 06000000

	.text
	.globl	_foo
_foo:
	ret

	.section	__DWARF,__debug_str,regular,debug
	.asciz	"first"
Lsecond:
	.asciz	"second"

	.section	__DWARF,__debug_info,regular,debug
	.long	Lsecond
