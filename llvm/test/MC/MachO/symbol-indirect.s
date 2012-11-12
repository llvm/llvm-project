// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

// FIXME: We are missing a lot of diagnostics on this kind of stuff which the
// assembler has.
        
        .lazy_symbol_pointer
        .indirect_symbol sym_lsp_B
        .long 0
        
        .globl sym_lsp_A
        .indirect_symbol sym_lsp_A
        .long 0
        
sym_lsp_C:      
        .indirect_symbol sym_lsp_C
        .long 0

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
        .indirect_symbol sym_lsp_D
        .long sym_lsp_D
.endif

        .indirect_symbol sym_lsp_E
        .long 0xFA

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
sym_lsp_F = 10
        .indirect_symbol sym_lsp_F
        .long 0
.endif

        .globl sym_lsp_G
sym_lsp_G:
        .indirect_symbol sym_lsp_G
        .long 0
        
        .non_lazy_symbol_pointer
        .indirect_symbol sym_nlp_B
        .long 0

        .globl sym_nlp_A
        .indirect_symbol sym_nlp_A
        .long 0

sym_nlp_C:      
        .indirect_symbol sym_nlp_C
        .long 0

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
        .indirect_symbol sym_nlp_D
        .long sym_nlp_D
.endif

        .indirect_symbol sym_nlp_E
        .long 0xAF

// FIXME: Enable this test once missing llvm-mc support is in place.
.if 0
sym_nlp_F = 10
        .indirect_symbol sym_nlp_F
        .long 0
.endif

        .globl sym_nlp_G
sym_nlp_G:
        .indirect_symbol sym_nlp_G
        .long 0

// CHECK: ('cputype', 7)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 364)
// CHECK: ('flag', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 1)
// CHECK:   ('size', 260)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 40)
// CHECK:   ('file_offset', 392)
// CHECK:   ('file_size', 40)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 3)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 392)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:     # Section 1
// CHECK:    (('section_name', '__la_symbol_ptr\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 20)
// CHECK:     ('offset', 392)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x7)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:     # Section 2
// CHECK:    (('section_name', '__nl_symbol_ptr\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 20)
// CHECK:     ('size', 20)
// CHECK:     ('offset', 412)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x6)
        // FIXME: Enable this when fixed!
// CHECX:     ('reserved1', 5)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 472)
// CHECK:   ('nsyms', 10)
// CHECK:   ('stroff', 592)
// CHECK:   ('strsize', 104)
// CHECK:   ('_string_data', '\x00sym_lsp_A\x00sym_lsp_G\x00sym_nlp_A\x00sym_nlp_G\x00sym_nlp_B\x00sym_nlp_E\x00sym_lsp_B\x00sym_lsp_E\x00sym_lsp_C\x00sym_nlp_C\x00\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 81)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 8)
// CHECK:     ('_string', 'sym_lsp_C')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 91)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 28)
// CHECK:     ('_string', 'sym_nlp_C')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 11)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 16)
// CHECK:     ('_string', 'sym_lsp_G')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 31)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 36)
// CHECK:     ('_string', 'sym_nlp_G')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lsp_A')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 61)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 1)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lsp_B')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 71)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 1)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lsp_E')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 21)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_nlp_A')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 41)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_nlp_B')
// CHECK:    ),
// CHECK:     # Symbol 9
// CHECK:    (('n_strx', 51)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_nlp_E')
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
// CHECK:   ('nundefsym', 6)
// CHECK:   ('tocoff', 0)
// CHECK:   ('ntoc', 0)
// CHECK:   ('modtaboff', 0)
// CHECK:   ('nmodtab', 0)
// CHECK:   ('extrefsymoff', 0)
// CHECK:   ('nextrefsyms', 0)
// CHECK:   ('indirectsymoff', 432)
// CHECK:   ('nindirectsyms', 10)
// CHECK:   ('extreloff', 0)
// CHECK:   ('nextrel', 0)
// CHECK:   ('locreloff', 0)
// CHECK:   ('nlocrel', 0)
// CHECK:   ('_indirect_symbols', [
// CHECK:     # Indirect Symbol 0
// CHECK:     (('symbol_index', 0x5),),
// CHECK:     # Indirect Symbol 1
// CHECK:     (('symbol_index', 0x4),),
// CHECK:     # Indirect Symbol 2
// CHECK:     (('symbol_index', 0x0),),
// CHECK:     # Indirect Symbol 3
// CHECK:     (('symbol_index', 0x6),),
// CHECK:     # Indirect Symbol 4
// CHECK:     (('symbol_index', 0x2),),
// CHECK:     # Indirect Symbol 5
// CHECK:     (('symbol_index', 0x8),),
// CHECK:     # Indirect Symbol 6
// CHECK:     (('symbol_index', 0x7),),
// CHECK:     # Indirect Symbol 7
// CHECK:     (('symbol_index', 0x80000000),),
// CHECK:     # Indirect Symbol 8
// CHECK:     (('symbol_index', 0x9),),
// CHECK:     # Indirect Symbol 9
// CHECK:     (('symbol_index', 0x3),),
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
