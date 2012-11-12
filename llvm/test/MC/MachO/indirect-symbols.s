// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

_b:
        _c = 0
_e:
        _f = 0
        
	.section	__IMPORT,__jump_table,symbol_stubs,pure_instructions+self_modifying_code,5
.indirect_symbol _a
	.ascii	 "\364\364\364\364\364"        
.indirect_symbol _b
	.ascii	 "\364\364\364\364\364"        
.indirect_symbol _c
	.ascii	 "\364\364\364\364\364"        
	.section	__IMPORT,__pointers,non_lazy_symbol_pointers
.indirect_symbol _d
	.long	0
.indirect_symbol _e
	.long	0
.indirect_symbol _f
	.long	0

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
// CHECK:   ('vm_size', 27)
// CHECK:   ('file_offset', 392)
// CHECK:   ('file_size', 27)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 3)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
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
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__jump_table\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__IMPORT\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 15)
// CHECK:     ('offset', 392)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x84000008)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 5)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', 'f4f4f4f4 f4f4f4f4 f4f4f4f4 f4f4f4')
// CHECK:     # Section 2
// CHECK:    (('section_name', '__pointers\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__IMPORT\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 15)
// CHECK:     ('size', 12)
// CHECK:     ('offset', 407)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x6)
// CHECK:     ('reserved1', 3)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 444)
// CHECK:   ('nsyms', 6)
// CHECK:   ('stroff', 516)
// CHECK:   ('strsize', 20)
// CHECK:   ('_string_data', '\x00_d\x00_a\x00_b\x00_c\x00_e\x00_f\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 7)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_b')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 10)
// CHECK:     ('n_type', 0x2)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_c')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 13)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_e')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 16)
// CHECK:     ('n_type', 0x2)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_f')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 4)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 1)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_a')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_d')
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
// CHECK:   ('nundefsym', 2)
// CHECK:   ('tocoff', 0)
// CHECK:   ('ntoc', 0)
// CHECK:   ('modtaboff', 0)
// CHECK:   ('nmodtab', 0)
// CHECK:   ('extrefsymoff', 0)
// CHECK:   ('nextrefsyms', 0)
// CHECK:   ('indirectsymoff', 420)
// CHECK:   ('nindirectsyms', 6)
// CHECK:   ('extreloff', 0)
// CHECK:   ('nextrel', 0)
// CHECK:   ('locreloff', 0)
// CHECK:   ('nlocrel', 0)
// CHECK:   ('_indirect_symbols', [
// CHECK:     # Indirect Symbol 0
// CHECK:     (('symbol_index', 0x4),),
// CHECK:     # Indirect Symbol 1
// CHECK:     (('symbol_index', 0x0),),
// CHECK:     # Indirect Symbol 2
// CHECK:     (('symbol_index', 0x1),),
// CHECK:     # Indirect Symbol 3
// CHECK:     (('symbol_index', 0x5),),
// CHECK:     # Indirect Symbol 4
// CHECK:     (('symbol_index', 0x80000000),),
// CHECK:     # Indirect Symbol 5
// CHECK:     (('symbol_index', 0xc0000000),),
// CHECK:   ])
// CHECK:  ),
// CHECK: ])
