// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .section        __TEXT,__text,regular,pure_instructions
        .section        __DATA,__thread_data,thread_local_regular
        .globl  _c$tlv$init
        .align  2
_c$tlv$init:
        .long   4

        .section        __DATA,__thread_vars,thread_local_variables
        .globl  _c
_c:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _c$tlv$init

        .section        __DATA,__thread_data,thread_local_regular
        .globl  _d$tlv$init
        .align  2
_d$tlv$init:
        .long   5

        .section        __DATA,__thread_vars,thread_local_variables
        .globl  _d
_d:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _d$tlv$init

.tbss _a$tlv$init, 4, 2

        .globl  _a
_a:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _a$tlv$init

.tbss _b$tlv$init, 4, 2

        .globl  _b
_b:
        .quad   ___tlv_bootstrap
        .quad   0
        .quad   _b$tlv$init

.subsections_via_symbols

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 496)
// CHECK: ('flag', 8192)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 392)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 112)
// CHECK:   ('file_offset', 528)
// CHECK:   ('file_size', 104)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 4)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 0)
// CHECK:     ('offset', 528)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__thread_data\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 8)
// CHECK:     ('offset', 528)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x11)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '04000000 05000000')
// CHECK:     # Section 2
// CHECK:    (('section_name', '__thread_vars\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 8)
// CHECK:     ('size', 96)
// CHECK:     ('offset', 536)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 632)
// CHECK:     ('num_reloc', 8)
// CHECK:     ('flags', 0x13)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x58),
// CHECK:      ('word-1', 0xe000001)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x48),
// CHECK:      ('word-1', 0xe000008)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x40),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x30),
// CHECK:      ('word-1', 0xe000008)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0x28),
// CHECK:      ('word-1', 0xe000007)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0x18),
// CHECK:      ('word-1', 0xe000008)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0x10),
// CHECK:      ('word-1', 0xe000005)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0xe000008)),
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000')
// CHECK:     # Section 3
// CHECK:    (('section_name', '__thread_bss\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 104)
// CHECK:     ('size', 8)
// CHECK:     ('offset', 0)
// CHECK:     ('alignment', 2)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x12)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', 'cffaedfe 07000001')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 696)
// CHECK:   ('nsyms', 9)
// CHECK:   ('stroff', 840)
// CHECK:   ('strsize', 80)
// CHECK:   ('_string_data', '\x00_c$tlv$init\x00_c\x00___tlv_bootstrap\x00_d$tlv$init\x00_d\x00_a\x00_b\x00_a$tlv$init\x00_b$tlv$init\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 54)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 104)
// CHECK:     ('_string', '_a$tlv$init')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 66)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 108)
// CHECK:     ('_string', '_b$tlv$init')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 48)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 56)
// CHECK:     ('_string', '_a')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 51)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 80)
// CHECK:     ('_string', '_b')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 13)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 8)
// CHECK:     ('_string', '_c')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_c$tlv$init')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 45)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 32)
// CHECK:     ('_string', '_d')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 33)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4)
// CHECK:     ('_string', '_d$tlv$init')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 16)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '___tlv_bootstrap')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 2)
// CHECK:   ('iextdefsym', 2)
// CHECK:   ('nextdefsym', 6)
// CHECK:   ('iundefsym', 8)
// CHECK:   ('nundefsym', 1)
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
