@ RUN: llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
        .text
_f0:
        bl _printf

_f1:
        bl _f0

        .data
_d0:
Ld0_0:
        .long Lsc0_0 - Ld0_0

	.section	__TEXT,__cstring,cstring_literals
Lsc0_0:
        .long 0

        .subsections_via_symbols

@ CHECK: ('cputype', 12)
@ CHECK: ('cpusubtype', 9)
@ CHECK: ('filetype', 1)
@ CHECK: ('num_load_commands', 3)
@ CHECK: ('load_commands_size', 364)
@ CHECK: ('flag', 8192)
@ CHECK: ('load_commands', [
@ CHECK:   # Load Command 0
@ CHECK:  (('command', 1)
@ CHECK:   ('size', 260)
@ CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:   ('vm_addr', 0)
@ CHECK:   ('vm_size', 16)
@ CHECK:   ('file_offset', 392)
@ CHECK:   ('file_size', 16)
@ CHECK:   ('maxprot', 7)
@ CHECK:   ('initprot', 7)
@ CHECK:   ('num_sections', 3)
@ CHECK:   ('flags', 0)
@ CHECK:   ('sections', [
@ CHECK:     # Section 0
@ CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 0)
@ CHECK:     ('size', 8)
@ CHECK:     ('offset', 392)
@ CHECK:     ('alignment', 0)
@ CHECK:     ('reloc_offset', 408)
@ CHECK:     ('num_reloc', 2)
@ CHECK:     ('flags', 0x80000400)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:     # Relocation 0
@ CHECK:     (('word-0', 0x4),
@ CHECK:      ('word-1', 0x55000001)),
@ CHECK:     # Relocation 1
@ CHECK:     (('word-0', 0x0),
@ CHECK:      ('word-1', 0x5d000003)),
@ CHECK:   ])
@ CHECK:   ('_section_data', 'feffffeb fdffffeb')
@ CHECK:     # Section 1
@ CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 8)
@ CHECK:     ('size', 4)
@ CHECK:     ('offset', 400)
@ CHECK:     ('alignment', 0)
@ CHECK:     ('reloc_offset', 424)
@ CHECK:     ('num_reloc', 2)
@ CHECK:     ('flags', 0x0)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:     # Relocation 0
@ CHECK:     (('word-0', 0xa2000000),
@ CHECK:      ('word-1', 0xc)),
@ CHECK:     # Relocation 1
@ CHECK:     (('word-0', 0xa1000000),
@ CHECK:      ('word-1', 0x8)),
@ CHECK:   ])
@ CHECK:   ('_section_data', '04000000')
@ CHECK:     # Section 2
@ CHECK:    (('section_name', '__cstring\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 12)
@ CHECK:     ('size', 4)
@ CHECK:     ('offset', 404)
@ CHECK:     ('alignment', 0)
@ CHECK:     ('reloc_offset', 0)
@ CHECK:     ('num_reloc', 0)
@ CHECK:     ('flags', 0x2)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:   ])
@ CHECK:   ('_section_data', '00000000')
@ CHECK:   ])
@ CHECK:  ),
@ CHECK:   # Load Command 1
@ CHECK:  (('command', 2)
@ CHECK:   ('size', 24)
@ CHECK:   ('symoff', 440)
@ CHECK:   ('nsyms', 4)
@ CHECK:   ('stroff', 488)
@ CHECK:   ('strsize', 24)
@ CHECK:   ('_string_data', '\x00_printf\x00_f0\x00_f1\x00_d0\x00\x00\x00\x00')
@ CHECK:   ('_symbols', [
@ CHECK:     # Symbol 0
@ CHECK:    (('n_strx', 9)
@ CHECK:     ('n_type', 0xe)
@ CHECK:     ('n_sect', 1)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 0)
@ CHECK:     ('_string', '_f0')
@ CHECK:    ),
@ CHECK:     # Symbol 1
@ CHECK:    (('n_strx', 13)
@ CHECK:     ('n_type', 0xe)
@ CHECK:     ('n_sect', 1)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 4)
@ CHECK:     ('_string', '_f1')
@ CHECK:    ),
@ CHECK:     # Symbol 2
@ CHECK:    (('n_strx', 17)
@ CHECK:     ('n_type', 0xe)
@ CHECK:     ('n_sect', 2)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 8)
@ CHECK:     ('_string', '_d0')
@ CHECK:    ),
@ CHECK:     # Symbol 3
@ CHECK:    (('n_strx', 1)
@ CHECK:     ('n_type', 0x1)
@ CHECK:     ('n_sect', 0)
@ CHECK:     ('n_desc', 0)
@ CHECK:     ('n_value', 0)
@ CHECK:     ('_string', '_printf')
@ CHECK:    ),
@ CHECK:   ])
@ CHECK:  ),
@ CHECK:   # Load Command 2
@ CHECK:  (('command', 11)
@ CHECK:   ('size', 80)
@ CHECK:   ('ilocalsym', 0)
@ CHECK:   ('nlocalsym', 3)
@ CHECK:   ('iextdefsym', 3)
@ CHECK:   ('nextdefsym', 0)
@ CHECK:   ('iundefsym', 3)
@ CHECK:   ('nundefsym', 1)
@ CHECK:   ('tocoff', 0)
@ CHECK:   ('ntoc', 0)
@ CHECK:   ('modtaboff', 0)
@ CHECK:   ('nmodtab', 0)
@ CHECK:   ('extrefsymoff', 0)
@ CHECK:   ('nextrefsyms', 0)
@ CHECK:   ('indirectsymoff', 0)
@ CHECK:   ('nindirectsyms', 0)
@ CHECK:   ('extreloff', 0)
@ CHECK:   ('nextrel', 0)
@ CHECK:   ('locreloff', 0)
@ CHECK:   ('nlocrel', 0)
@ CHECK:   ('_indirect_symbols', [
@ CHECK:   ])
@ CHECK:  ),
@ CHECK: ])
