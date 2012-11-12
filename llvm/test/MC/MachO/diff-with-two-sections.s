// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

	.section	__TEXT,__text,regular,pure_instructions
Leh_func_begin0:
	.section	__TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
Ltmp3:
Ltmp4 = Leh_func_begin0-Ltmp3
	.long	Ltmp4

// CHECK:      ('cputype', 7)
// CHECK-NEXT: ('cpusubtype', 3)
// CHECK-NEXT: ('filetype', 1)
// CHECK-NEXT: ('num_load_commands', 1)
// CHECK-NEXT: ('load_commands_size', 192)
// CHECK-NEXT: ('flag', 0)
// CHECK-NEXT: ('load_commands', [
// CHECK-NEXT:   # Load Command 0
// CHECK-NEXT:  (('command', 1)
// CHECK-NEXT:   ('size', 192)
// CHECK-NEXT:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:   ('vm_addr', 0)
// CHECK-NEXT:   ('vm_size', 4)
// CHECK-NEXT:   ('file_offset', 220)
// CHECK-NEXT:   ('file_size', 4)
// CHECK-NEXT:   ('maxprot', 7)
// CHECK-NEXT:   ('initprot', 7)
// CHECK-NEXT:   ('num_sections', 2)
// CHECK-NEXT:   ('flags', 0)
// CHECK-NEXT:   ('sections', [
// CHECK-NEXT:     # Section 0
// CHECK-NEXT:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 0)
// CHECK-NEXT:     ('size', 0)
// CHECK-NEXT:     ('offset', 220)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 0)
// CHECK-NEXT:     ('num_reloc', 0)
// CHECK-NEXT:     ('flags', 0x80000000)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '')
// CHECK-NEXT:     # Section 1
// CHECK-NEXT:    (('section_name', '__eh_frame\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 0)
// CHECK-NEXT:     ('size', 4)
// CHECK-NEXT:     ('offset', 220)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 0)
// CHECK-NEXT:     ('num_reloc', 0)
// CHECK-NEXT:     ('flags', 0x6800000b)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '00000000')
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
