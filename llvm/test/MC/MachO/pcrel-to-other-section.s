// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

nop
	.section	__TEXT,__StaticInit,regular,pure_instructions
	calll	foo

// CHECK:      ('cputype', 7)
// CHECK-NEXT: ('cpusubtype', 3)
// CHECK-NEXT: ('filetype', 1)
// CHECK-NEXT: ('num_load_commands', 3)
// CHECK-NEXT: ('load_commands_size', 296)
// CHECK-NEXT: ('flag', 0)
// CHECK-NEXT: ('load_commands', [
// CHECK-NEXT:   # Load Command 0
// CHECK-NEXT:  (('command', 1)
// CHECK-NEXT:   ('size', 192)
// CHECK-NEXT:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:   ('vm_addr', 0)
// CHECK-NEXT:   ('vm_size', 6)
// CHECK-NEXT:   ('file_offset', 324)
// CHECK-NEXT:   ('file_size', 6)
// CHECK-NEXT:   ('maxprot', 7)
// CHECK-NEXT:   ('initprot', 7)
// CHECK-NEXT:   ('num_sections', 2)
// CHECK-NEXT:   ('flags', 0)
// CHECK-NEXT:   ('sections', [
// CHECK-NEXT:     # Section 0
// CHECK-NEXT:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 0)
// CHECK-NEXT:     ('size', 1)
// CHECK-NEXT:     ('offset', 324)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 0)
// CHECK-NEXT:     ('num_reloc', 0)
// CHECK-NEXT:     ('flags', 0x80000400)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '90')
// CHECK-NEXT:     # Section 1
// CHECK-NEXT:    (('section_name', '__StaticInit\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 1)
// CHECK-NEXT:     ('size', 5)
// CHECK-NEXT:     ('offset', 325)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 332)
// CHECK-NEXT:     ('num_reloc', 1)
// CHECK-NEXT:     ('flags', 0x80000400)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:     # Relocation 0
// CHECK-NEXT:     (('word-0', 0x1),
// CHECK-NEXT:      ('word-1', 0xd000000)),
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', 'e8faffff ff')
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT:   # Load Command 1
// CHECK-NEXT:  (('command', 2)
// CHECK-NEXT:   ('size', 24)
// CHECK-NEXT:   ('symoff', 340)
// CHECK-NEXT:   ('nsyms', 1)
// CHECK-NEXT:   ('stroff', 352)
// CHECK-NEXT:   ('strsize', 8)
// CHECK-NEXT:   ('_string_data', '\x00foo\x00\x00\x00\x00')
// CHECK-NEXT:   ('_symbols', [
// CHECK-NEXT:     # Symbol 0
// CHECK-NEXT:    (('n_strx', 1)
// CHECK-NEXT:     ('n_type', 0x1)
// CHECK-NEXT:     ('n_sect', 0)
// CHECK-NEXT:     ('n_desc', 0)
// CHECK-NEXT:     ('n_value', 0)
// CHECK-NEXT:     ('_string', 'foo')
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT:   # Load Command 2
// CHECK-NEXT:  (('command', 11)
// CHECK-NEXT:   ('size', 80)
// CHECK-NEXT:   ('ilocalsym', 0)
// CHECK-NEXT:   ('nlocalsym', 0)
// CHECK-NEXT:   ('iextdefsym', 0)
// CHECK-NEXT:   ('nextdefsym', 0)
// CHECK-NEXT:   ('iundefsym', 0)
// CHECK-NEXT:   ('nundefsym', 1)
// CHECK-NEXT:   ('tocoff', 0)
// CHECK-NEXT:   ('ntoc', 0)
// CHECK-NEXT:   ('modtaboff', 0)
// CHECK-NEXT:   ('nmodtab', 0)
// CHECK-NEXT:   ('extrefsymoff', 0)
// CHECK-NEXT:   ('nextrefsyms', 0)
// CHECK-NEXT:   ('indirectsymoff', 0)
// CHECK-NEXT:   ('nindirectsyms', 0)
// CHECK-NEXT:   ('extreloff', 0)
// CHECK-NEXT:   ('nextrel', 0)
// CHECK-NEXT:   ('locreloff', 0)
// CHECK-NEXT:   ('nlocrel', 0)
// CHECK-NEXT:   ('_indirect_symbols', [
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
