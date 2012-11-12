// RUN: llvm-mc -n -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// Test case for rdar://10062261

// Must be no base, non-temporary, symbol before the reference to Lbar at the
// start of the section.  What we are testing for is that the reference does not
// create a relocation entry.
.text
Ladd:
	nop
	jmp Lbar
	.byte 0x0f,0x1f,0x40,0x00
	.byte 0x0f,0x1f,0x40,0x00
Lbar:	
	mov $1, %eax
	ret

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 1)
// CHECK: ('load_commands_size', 152)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 152)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 17)
// CHECK:   ('file_offset', 184)
// CHECK:   ('file_size', 17)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 17)
// CHECK:     ('offset', 184)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '90eb080f 1f40000f 1f4000b8 01000000 c3')
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
