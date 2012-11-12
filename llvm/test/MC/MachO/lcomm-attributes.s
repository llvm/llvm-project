// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck %s

        // Note, this test intentionally mismatches Darwin 'as', which loses the
	// following global marker.
        //
        // FIXME: We should probably warn about our interpretation of this.
        .globl sym_lcomm_ext_A
        .lcomm sym_lcomm_ext_A, 4
        .lcomm sym_lcomm_ext_B, 4
        .globl sym_lcomm_ext_B

        .globl sym_zfill_ext_A
        .zerofill __DATA, __bss, sym_zfill_ext_A, 4
        .zerofill __DATA, __bss, sym_zfill_ext_B, 4
        .globl sym_zfill_ext_B

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
// CHECK:   ('vm_size', 16)
// CHECK:   ('file_offset', 324)
// CHECK:   ('file_size', 0)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:     # Section 1
// CHECK:    (('section_name', '__bss\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 16)
// CHECK:     ('offset', 0)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x1)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 324)
// CHECK:   ('nsyms', 4)
// CHECK:   ('stroff', 372)
// CHECK:   ('strsize', 68)
// CHECK:   ('_string_data', '\x00sym_lcomm_ext_A\x00sym_lcomm_ext_B\x00sym_zfill_ext_A\x00sym_zfill_ext_B\x00\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'sym_lcomm_ext_A')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 17)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4)
// CHECK:     ('_string', 'sym_lcomm_ext_B')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 33)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 8)
// CHECK:     ('_string', 'sym_zfill_ext_A')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 49)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 12)
// CHECK:     ('_string', 'sym_zfill_ext_B')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 0)
// CHECK:   ('iextdefsym', 0)
// CHECK:   ('nextdefsym', 4)
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
