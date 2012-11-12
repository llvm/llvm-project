// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: macho-dump --dump-section-data < %t.o > %t.dump
// RUN: FileCheck < %t.dump %s
        
_a:
L0:     
        .long 1
L1:     
        .long 2
        .long _c - _d + 4
        .long (_c - L0) - (_d - L1) // == (_c - _d) + (L1 - L0)
                                    // == (_c - _d + 4)
_c:
        .long 0
_d:
        .long 0

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 256)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 152)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 24)
// CHECK:   ('file_offset', 288)
// CHECK:   ('file_size', 24)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 24)
// CHECK:     ('offset', 288)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 312)
// CHECK:     ('num_reloc', 4)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0xc),
// CHECK:      ('word-1', 0x5c000002)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0xc),
// CHECK:      ('word-1', 0xc000001)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0x5c000002)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0xc000001)),
// CHECK:   ])
// CHECK:   ('_section_data', '01000000 02000000 04000000 04000000 00000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 344)
// CHECK:   ('nsyms', 3)
// CHECK:   ('stroff', 392)
// CHECK:   ('strsize', 12)
// CHECK:   ('_string_data', '\x00_a\x00_c\x00_d\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_a')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 4)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 16)
// CHECK:     ('_string', '_c')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 7)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 20)
// CHECK:     ('_string', '_d')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 3)
// CHECK:   ('iextdefsym', 3)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 3)
// CHECK:   ('nundefsym', 0)
// CHECK:   ('tocoff', 0)
// CHECK:   ('ntoc', 0)
// CHECK:   ('modtaboff', 0)
// CHECK:   ('nmodtab', 0)
// CHECK:   ('extrefsymoff', 0)
// CHECK:   ('nextrefsyms', 0)
// CHECK:   ('indirectsymoff', 0)
// CHECK:   ('nindirectsyms', 0)
// CHECK:   ('extreloff', 0)
// CHECK:   ('nextrel', 0)
// CHECK:   ('locreloff', 0)
// CHECK:   ('nlocrel', 0)
// CHECK:   ('_indirect_symbols', [
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
