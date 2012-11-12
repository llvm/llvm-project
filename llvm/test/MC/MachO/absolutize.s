// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

_text_a:
        xorl %eax,%eax
_text_b:
        xorl %eax,%eax
Ltext_c:
        xorl %eax,%eax
Ltext_d:        
        xorl %eax,%eax
        
        movl $(_text_a - _text_b), %eax
Ltext_expr_0 = _text_a - _text_b
        movl $(Ltext_expr_0), %eax

        movl $(Ltext_c - _text_b), %eax
Ltext_expr_1 = Ltext_c - _text_b
        movl $(Ltext_expr_1), %eax

        movl $(Ltext_d - Ltext_c), %eax
Ltext_expr_2 = Ltext_d - Ltext_c
        movl $(Ltext_expr_2), %eax

        movl $(_text_a + Ltext_expr_0), %eax

        .data
_data_a:
        .long 0
_data_b:
        .long 0
Ldata_c:
        .long 0
Ldata_d:        
        .long 0
        
        .long _data_a - _data_b
Ldata_expr_0 = _data_a - _data_b
        .long Ldata_expr_0

        .long Ldata_c - _data_b
Ldata_expr_1 = Ldata_c - _data_b
        .long Ldata_expr_1

        .long Ldata_d - Ldata_c
Ldata_expr_2 = Ldata_d - Ldata_c
        .long Ldata_expr_2

        .long _data_a + Ldata_expr_0

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 296)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 192)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 87)
// CHECK:   ('file_offset', 324)
// CHECK:   ('file_size', 87)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 43)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 412)
// CHECK:     ('num_reloc', 7)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0xa0000027),
// CHECK:      ('word-1', 0x0)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0xa400001d),
// CHECK:      ('word-1', 0x6)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x4)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0xa4000013),
// CHECK:      ('word-1', 0x4)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x2)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0xa4000009),
// CHECK:      ('word-1', 0x0)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x2)),
// CHECK:   ])
// CHECK:   ('_section_data', '31c031c0 31c031c0 b8feffff ffb8feff ffffb802 000000b8 02000000 b8020000 00b80200 0000b8fe ffffff')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 43)
// CHECK:     ('size', 44)
// CHECK:     ('offset', 367)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 468)
// CHECK:     ('num_reloc', 7)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0xa0000028),
// CHECK:      ('word-1', 0x2b)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0xa4000020),
// CHECK:      ('word-1', 0x37)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x33)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0xa4000018),
// CHECK:      ('word-1', 0x33)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x2f)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0xa4000010),
// CHECK:      ('word-1', 0x2b)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x2f)),
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000 00000000 fcffffff fcffffff 04000000 04000000 04000000 04000000 27000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 524)
// CHECK:   ('nsyms', 4)
// CHECK:   ('stroff', 572)
// CHECK:   ('strsize', 36)
// CHECK:   ('_string_data', '\x00_text_a\x00_text_b\x00_data_a\x00_data_b\x00\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_text_a')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 9)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 2)
// CHECK:     ('_string', '_text_b')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 17)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 43)
// CHECK:     ('_string', '_data_a')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 25)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 47)
// CHECK:     ('_string', '_data_b')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 4)
// CHECK:   ('iextdefsym', 4)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 4)
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
