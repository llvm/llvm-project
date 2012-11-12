// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump | FileCheck -check-prefix CHECK-X86_32 %s
// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump | FileCheck -check-prefix CHECK-X86_64 %s

sym_local_B:
.globl sym_globl_def_B
.globl sym_globl_undef_B
sym_local_A:
.globl sym_globl_def_A
.globl sym_globl_undef_A
sym_local_C:
.globl sym_globl_def_C
.globl sym_globl_undef_C
        
sym_globl_def_A: 
sym_globl_def_B: 
sym_globl_def_C: 
Lsym_asm_temp:
        .long 0
        
// CHECK-X86_32: ('cputype', 7)
// CHECK-X86_32: ('cpusubtype', 3)
// CHECK-X86_32: ('filetype', 1)
// CHECK-X86_32: ('num_load_commands', 3)
// CHECK-X86_32: ('load_commands_size', 228)
// CHECK-X86_32: ('flag', 0)
// CHECK-X86_32: ('load_commands', [
// CHECK-X86_32:   # Load Command 0
// CHECK-X86_32:  (('command', 1)
// CHECK-X86_32:   ('size', 124)
// CHECK-X86_32:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_32:   ('vm_addr', 0)
// CHECK-X86_32:   ('vm_size', 4)
// CHECK-X86_32:   ('file_offset', 256)
// CHECK-X86_32:   ('file_size', 4)
// CHECK-X86_32:   ('maxprot', 7)
// CHECK-X86_32:   ('initprot', 7)
// CHECK-X86_32:   ('num_sections', 1)
// CHECK-X86_32:   ('flags', 0)
// CHECK-X86_32:   ('sections', [
// CHECK-X86_32:     # Section 0
// CHECK-X86_32:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_32:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_32:     ('address', 0)
// CHECK-X86_32:     ('size', 4)
// CHECK-X86_32:     ('offset', 256)
// CHECK-X86_32:     ('alignment', 0)
// CHECK-X86_32:     ('reloc_offset', 0)
// CHECK-X86_32:     ('num_reloc', 0)
// CHECK-X86_32:     ('flags', 0x80000000)
// CHECK-X86_32:     ('reserved1', 0)
// CHECK-X86_32:     ('reserved2', 0)
// CHECK-X86_32:    ),
// CHECK-X86_32:   ])
// CHECK-X86_32:  ),
// CHECK-X86_32:   # Load Command 1
// CHECK-X86_32:  (('command', 2)
// CHECK-X86_32:   ('size', 24)
// CHECK-X86_32:   ('symoff', 260)
// CHECK-X86_32:   ('nsyms', 9)
// CHECK-X86_32:   ('stroff', 368)
// CHECK-X86_32:   ('strsize', 140)
// CHECK-X86_32:   ('_string_data', '\x00sym_globl_def_B\x00sym_globl_undef_B\x00sym_globl_def_A\x00sym_globl_undef_A\x00sym_globl_def_C\x00sym_globl_undef_C\x00sym_local_B\x00sym_local_A\x00sym_local_C\x00\x00')
// CHECK-X86_32:   ('_symbols', [
// CHECK-X86_32:     # Symbol 0
// CHECK-X86_32:    (('n_strx', 103)
// CHECK-X86_32:     ('n_type', 0xe)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_local_B')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 1
// CHECK-X86_32:    (('n_strx', 115)
// CHECK-X86_32:     ('n_type', 0xe)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_local_A')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 2
// CHECK-X86_32:    (('n_strx', 127)
// CHECK-X86_32:     ('n_type', 0xe)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_local_C')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 3
// CHECK-X86_32:    (('n_strx', 35)
// CHECK-X86_32:     ('n_type', 0xf)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_def_A')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 4
// CHECK-X86_32:    (('n_strx', 1)
// CHECK-X86_32:     ('n_type', 0xf)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_def_B')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 5
// CHECK-X86_32:    (('n_strx', 69)
// CHECK-X86_32:     ('n_type', 0xf)
// CHECK-X86_32:     ('n_sect', 1)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_def_C')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 6
// CHECK-X86_32:    (('n_strx', 51)
// CHECK-X86_32:     ('n_type', 0x1)
// CHECK-X86_32:     ('n_sect', 0)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_undef_A')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 7
// CHECK-X86_32:    (('n_strx', 17)
// CHECK-X86_32:     ('n_type', 0x1)
// CHECK-X86_32:     ('n_sect', 0)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_undef_B')
// CHECK-X86_32:    ),
// CHECK-X86_32:     # Symbol 8
// CHECK-X86_32:    (('n_strx', 85)
// CHECK-X86_32:     ('n_type', 0x1)
// CHECK-X86_32:     ('n_sect', 0)
// CHECK-X86_32:     ('n_desc', 0)
// CHECK-X86_32:     ('n_value', 0)
// CHECK-X86_32:     ('_string', 'sym_globl_undef_C')
// CHECK-X86_32:    ),
// CHECK-X86_32:   ])
// CHECK-X86_32:  ),
// CHECK-X86_32:   # Load Command 2
// CHECK-X86_32:  (('command', 11)
// CHECK-X86_32:   ('size', 80)
// CHECK-X86_32:   ('ilocalsym', 0)
// CHECK-X86_32:   ('nlocalsym', 3)
// CHECK-X86_32:   ('iextdefsym', 3)
// CHECK-X86_32:   ('nextdefsym', 3)
// CHECK-X86_32:   ('iundefsym', 6)
// CHECK-X86_32:   ('nundefsym', 3)
// CHECK-X86_32:   ('tocoff', 0)
// CHECK-X86_32:   ('ntoc', 0)
// CHECK-X86_32:   ('modtaboff', 0)
// CHECK-X86_32:   ('nmodtab', 0)
// CHECK-X86_32:   ('extrefsymoff', 0)
// CHECK-X86_32:   ('nextrefsyms', 0)
// CHECK-X86_32:   ('indirectsymoff', 0)
// CHECK-X86_32:   ('nindirectsyms', 0)
// CHECK-X86_32:   ('extreloff', 0)
// CHECK-X86_32:   ('nextrel', 0)
// CHECK-X86_32:   ('locreloff', 0)
// CHECK-X86_32:   ('nlocrel', 0)
// CHECK-X86_32:   ('_indirect_symbols', [
// CHECK-X86_32:   ])
// CHECK-X86_32:  ),
// CHECK-X86_32: ])

// CHECK-X86_64: ('cputype', 16777223)
// CHECK-X86_64: ('cpusubtype', 3)
// CHECK-X86_64: ('filetype', 1)
// CHECK-X86_64: ('num_load_commands', 3)
// CHECK-X86_64: ('load_commands_size', 256)
// CHECK-X86_64: ('flag', 0)
// CHECK-X86_64: ('reserved', 0)
// CHECK-X86_64: ('load_commands', [
// CHECK-X86_64:   # Load Command 0
// CHECK-X86_64:  (('command', 25)
// CHECK-X86_64:   ('size', 152)
// CHECK-X86_64:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:   ('vm_addr', 0)
// CHECK-X86_64:   ('vm_size', 4)
// CHECK-X86_64:   ('file_offset', 288)
// CHECK-X86_64:   ('file_size', 4)
// CHECK-X86_64:   ('maxprot', 7)
// CHECK-X86_64:   ('initprot', 7)
// CHECK-X86_64:   ('num_sections', 1)
// CHECK-X86_64:   ('flags', 0)
// CHECK-X86_64:   ('sections', [
// CHECK-X86_64:     # Section 0
// CHECK-X86_64:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('address', 0)
// CHECK-X86_64:     ('size', 4)
// CHECK-X86_64:     ('offset', 288)
// CHECK-X86_64:     ('alignment', 0)
// CHECK-X86_64:     ('reloc_offset', 0)
// CHECK-X86_64:     ('num_reloc', 0)
// CHECK-X86_64:     ('flags', 0x80000000)
// CHECK-X86_64:     ('reserved1', 0)
// CHECK-X86_64:     ('reserved2', 0)
// CHECK-X86_64:     ('reserved3', 0)
// CHECK-X86_64:    ),
// CHECK-X86_64:   ('_relocations', [
// CHECK-X86_64:   ])
// CHECK-X86_64:   ])
// CHECK-X86_64:  ),
// CHECK-X86_64:   # Load Command 1
// CHECK-X86_64:  (('command', 2)
// CHECK-X86_64:   ('size', 24)
// CHECK-X86_64:   ('symoff', 292)
// CHECK-X86_64:   ('nsyms', 9)
// CHECK-X86_64:   ('stroff', 436)
// CHECK-X86_64:   ('strsize', 140)
// CHECK-X86_64:   ('_string_data', '\x00sym_globl_def_B\x00sym_globl_undef_B\x00sym_globl_def_A\x00sym_globl_undef_A\x00sym_globl_def_C\x00sym_globl_undef_C\x00sym_local_B\x00sym_local_A\x00sym_local_C\x00\x00')
// CHECK-X86_64:   ('_symbols', [
// CHECK-X86_64:     # Symbol 0
// CHECK-X86_64:    (('n_strx', 103)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_local_B')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 1
// CHECK-X86_64:    (('n_strx', 115)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_local_A')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 2
// CHECK-X86_64:    (('n_strx', 127)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_local_C')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 3
// CHECK-X86_64:    (('n_strx', 35)
// CHECK-X86_64:     ('n_type', 0xf)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_def_A')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 4
// CHECK-X86_64:    (('n_strx', 1)
// CHECK-X86_64:     ('n_type', 0xf)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_def_B')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 5
// CHECK-X86_64:    (('n_strx', 69)
// CHECK-X86_64:     ('n_type', 0xf)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_def_C')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 6
// CHECK-X86_64:    (('n_strx', 51)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_undef_A')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 7
// CHECK-X86_64:    (('n_strx', 17)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_undef_B')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 8
// CHECK-X86_64:    (('n_strx', 85)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'sym_globl_undef_C')
// CHECK-X86_64:    ),
// CHECK-X86_64:   ])
// CHECK-X86_64:  ),
// CHECK-X86_64:   # Load Command 2
// CHECK-X86_64:  (('command', 11)
// CHECK-X86_64:   ('size', 80)
// CHECK-X86_64:   ('ilocalsym', 0)
// CHECK-X86_64:   ('nlocalsym', 3)
// CHECK-X86_64:   ('iextdefsym', 3)
// CHECK-X86_64:   ('nextdefsym', 3)
// CHECK-X86_64:   ('iundefsym', 6)
// CHECK-X86_64:   ('nundefsym', 3)
// CHECK-X86_64:   ('tocoff', 0)
// CHECK-X86_64:   ('ntoc', 0)
// CHECK-X86_64:   ('modtaboff', 0)
// CHECK-X86_64:   ('nmodtab', 0)
// CHECK-X86_64:   ('extrefsymoff', 0)
// CHECK-X86_64:   ('nextrefsyms', 0)
// CHECK-X86_64:   ('indirectsymoff', 0)
// CHECK-X86_64:   ('nindirectsyms', 0)
// CHECK-X86_64:   ('extreloff', 0)
// CHECK-X86_64:   ('nextrel', 0)
// CHECK-X86_64:   ('locreloff', 0)
// CHECK-X86_64:   ('nlocrel', 0)
// CHECK-X86_64:   ('_indirect_symbols', [
// CHECK-X86_64:   ])
// CHECK-X86_64:  ),
// CHECK-X86_64: ])
