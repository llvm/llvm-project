// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s
_g:
LFB2:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
_g.eh:
	.quad	LFB2-.

// CHECK:      ('cputype', 16777223)
// CHECK-NEXT: ('cpusubtype', 3)
// CHECK-NEXT: ('filetype', 1)
// CHECK-NEXT: ('num_load_commands', 3)
// CHECK-NEXT: ('load_commands_size', 336)
// CHECK-NEXT: ('flag', 0)
// CHECK-NEXT: ('reserved', 0)
// CHECK-NEXT: ('load_commands', [
// CHECK-NEXT:   # Load Command 0
// CHECK-NEXT:  (('command', 25)
// CHECK-NEXT:   ('size', 232)
// CHECK-NEXT:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:   ('vm_addr', 0)
// CHECK-NEXT:   ('vm_size', 8)
// CHECK-NEXT:   ('file_offset', 368)
// CHECK-NEXT:   ('file_size', 8)
// CHECK-NEXT:   ('maxprot', 7)
// CHECK-NEXT:   ('initprot', 7)
// CHECK-NEXT:   ('num_sections', 2)
// CHECK-NEXT:   ('flags', 0)
// CHECK-NEXT:   ('sections', [
// CHECK-NEXT:    # Section 0
// CHECK-NEXT:   (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:    ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:    ('address', 0)
// CHECK-NEXT:    ('size', 0)
// CHECK-NEXT:    ('offset', 368)
// CHECK-NEXT:    ('alignment', 0)
// CHECK-NEXT:    ('reloc_offset', 0)
// CHECK-NEXT:    ('num_reloc', 0)
// CHECK-NEXT:    ('flags', 0x80000000)
// CHECK-NEXT:    ('reserved1', 0)
// CHECK-NEXT:    ('reserved2', 0)
// CHECK-NEXT:    ('reserved3', 0)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:  ])
// CHECK-NEXT:  ('_section_data', '')
// CHECK-NEXT:    # Section 1
// CHECK-NEXT:   (('section_name', '__eh_frame\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:    ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:    ('address', 0)
// CHECK-NEXT:    ('size', 8)
// CHECK-NEXT:    ('offset', 368)
// CHECK-NEXT:    ('alignment', 0)
// CHECK-NEXT:    ('reloc_offset', 376)
// CHECK-NEXT:    ('num_reloc', 2)
// CHECK-NEXT:    ('flags', 0x6800000b)
// CHECK-NEXT:    ('reserved1', 0)
// CHECK-NEXT:    ('reserved2', 0)
// CHECK-NEXT:    ('reserved3', 0)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('word-0', 0x0),
// CHECK-NEXT:     ('word-1', 0x5e000001)),
// CHECK-NEXT:    # Relocation 1
// CHECK-NEXT:    (('word-0', 0x0),
// CHECK-NEXT:     ('word-1', 0xe000000)),
// CHECK-NEXT:  ])
// CHECK-NEXT:  ('_section_data', '00000000 00000000')
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
// CHECK-NEXT:  # Load Command 1
// CHECK-NEXT: (('command', 2)
// CHECK-NEXT:  ('size', 24)
// CHECK-NEXT:  ('symoff', 392)
// CHECK-NEXT:  ('nsyms', 2)
// CHECK-NEXT:  ('stroff', 424)
// CHECK-NEXT:  ('strsize', 12)
// CHECK-NEXT:  ('_string_data', '\x00_g\x00_g.eh\x00\x00\x00')
// CHECK-NEXT:  ('_symbols', [
// CHECK-NEXT:    # Symbol 0
// CHECK-NEXT:   (('n_strx', 1)
// CHECK-NEXT:    ('n_type', 0xe)
// CHECK-NEXT:    ('n_sect', 1)
// CHECK-NEXT:    ('n_desc', 0)
// CHECK-NEXT:    ('n_value', 0)
// CHECK-NEXT:    ('_string', '_g')
// CHECK-NEXT:   ),
// CHECK-NEXT:    # Symbol 1
// CHECK-NEXT:   (('n_strx', 4)
// CHECK-NEXT:    ('n_type', 0xe)
// CHECK-NEXT:    ('n_sect', 2)
// CHECK-NEXT:    ('n_desc', 0)
// CHECK-NEXT:    ('n_value', 0)
// CHECK-NEXT:    ('_string', '_g.eh')
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
// CHECK-NEXT:  # Load Command 2
// CHECK-NEXT: (('command', 11)
// CHECK-NEXT:  ('size', 80)
// CHECK-NEXT:  ('ilocalsym', 0)
// CHECK-NEXT:  ('nlocalsym', 2)
// CHECK-NEXT:  ('iextdefsym', 2)
// CHECK-NEXT:  ('nextdefsym', 0)
// CHECK-NEXT:  ('iundefsym', 2)
// CHECK-NEXT:  ('nundefsym', 0)
// CHECK-NEXT:  ('tocoff', 0)
// CHECK-NEXT:  ('ntoc', 0)
// CHECK-NEXT:  ('modtaboff', 0)
// CHECK-NEXT:  ('nmodtab', 0)
// CHECK-NEXT:  ('extrefsymoff', 0)
// CHECK-NEXT:  ('nextrefsyms', 0)
// CHECK-NEXT:  ('indirectsymoff', 0)
// CHECK-NEXT:  ('nindirectsyms', 0)
// CHECK-NEXT:  ('extreloff', 0)
// CHECK-NEXT:  ('nextrel', 0)
// CHECK-NEXT:  ('locreloff', 0)
// CHECK-NEXT:  ('nlocrel', 0)
// CHECK-NEXT:  ('_indirect_symbols', [
// CHECK-NEXT:  ])
// CHECK-NEXT: ),
// CHECK-NEXT:])
