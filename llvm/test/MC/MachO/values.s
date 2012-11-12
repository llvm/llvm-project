// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

        .long 0
text_def_int:
        .long 0

        .globl text_def_ext
text_def_ext:
        .long 0

        .data
        .long 0
data_def_int:
        .long 0

        .globl data_def_ext
data_def_ext:
        .long 0

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
// CHECK:   ('vm_size', 24)
// CHECK:   ('file_offset', 324)
// CHECK:   ('file_size', 24)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 12)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 12)
// CHECK:     ('size', 12)
// CHECK:     ('offset', 336)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 348)
// CHECK:   ('nsyms', 4)
// CHECK:   ('stroff', 396)
// CHECK:   ('strsize', 56)
// CHECK:   ('_string_data', '\x00text_def_ext\x00data_def_ext\x00text_def_int\x00data_def_int\x00\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 27)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4)
// CHECK:     ('_string', 'text_def_int')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 40)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 16)
// CHECK:     ('_string', 'data_def_int')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 14)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 20)
// CHECK:     ('_string', 'data_def_ext')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 8)
// CHECK:     ('_string', 'text_def_ext')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 2)
// CHECK:   ('iextdefsym', 2)
// CHECK:   ('nextdefsym', 2)
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
