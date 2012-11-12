// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .data
        .long undef
        .long (undef + 4)

        .globl local_a_ext
local_a_ext:
        .long local_a_ext

local_a:
        .long 0
local_a_elt:
        .long 0
local_b:
        .long local_b - local_c + 245
        .long 0
local_c:
        .long 0


        .long local_a_elt + 1
        .long local_a_elt + 10
        .short local_a_elt + 20
        .byte local_a_elt + 89

        .const

        .long
bar:
        .long local_a_elt - bar + 33

L0:
        .long L0
        .long L1

        .text
_f0:
L1:
        jmp	0xbabecafe
        jmp L0
        jmp L1
        ret

        .objc_class_name_A=0
	.globl .objc_class_name_A

        .text
        .globl _f1
        .weak_definition _f1
_f1:
        .data
        .long _f1
        .long _f1 + 4

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
// CHECK:   ('vm_size', 76)
// CHECK:   ('file_offset', 392)
// CHECK:   ('file_size', 76)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 3)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 13)
// CHECK:     ('offset', 392)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 468)
// CHECK:     ('num_reloc', 2)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x6),
// CHECK:      ('word-1', 0x5000003)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x1),
// CHECK:      ('word-1', 0x5000000)),
// CHECK:   ])
// CHECK:   ('_section_data', 'e9f9cabe bae93a00 0000ebf4 c3')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 13)
// CHECK:     ('size', 51)
// CHECK:     ('offset', 405)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 484)
// CHECK:     ('num_reloc', 11)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x2f),
// CHECK:      ('word-1', 0xc000007)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x2b),
// CHECK:      ('word-1', 0xc000007)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x8000002a),
// CHECK:      ('word-1', 0x1d)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x90000028),
// CHECK:      ('word-1', 0x1d)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0xa0000024),
// CHECK:      ('word-1', 0x1d)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0xa0000020),
// CHECK:      ('word-1', 0x1d)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0xa4000014),
// CHECK:      ('word-1', 0x21)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x29)),
// CHECK:     # Relocation 8
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0x4000002)),
// CHECK:     # Relocation 9
// CHECK:     (('word-0', 0x4),
// CHECK:      ('word-1', 0xc000009)),
// CHECK:     # Relocation 10
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0xc000009)),
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 04000000 15000000 00000000 00000000 ed000000 00000000 00000000 1e000000 27000000 31007600 00000004 000000')
// CHECK:     # Section 2
// CHECK:    (('section_name', '__const\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 64)
// CHECK:     ('size', 12)
// CHECK:     ('offset', 456)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 572)
// CHECK:     ('num_reloc', 4)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0x4000001)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x4),
// CHECK:      ('word-1', 0x4000003)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0xa4000000),
// CHECK:      ('word-1', 0x1d)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0xa1000000),
// CHECK:      ('word-1', 0x40)),
// CHECK:   ])
// CHECK:   ('_section_data', 'feffffff 44000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 604)
// CHECK:   ('nsyms', 10)
// CHECK:   ('stroff', 724)
// CHECK:   ('strsize', 88)
// CHECK:   ('_string_data', '\x00undef\x00local_a_ext\x00.objc_class_name_A\x00_f1\x00local_a\x00local_a_elt\x00local_b\x00local_c\x00bar\x00_f0\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 42)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 25)
// CHECK:     ('_string', 'local_a')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 50)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 29)
// CHECK:     ('_string', 'local_a_elt')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 62)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 33)
// CHECK:     ('_string', 'local_b')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 70)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 41)
// CHECK:     ('_string', 'local_c')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 78)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 64)
// CHECK:     ('_string', 'bar')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 82)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_f0')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 19)
// CHECK:     ('n_type', 0x3)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '.objc_class_name_A')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 38)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 128)
// CHECK:     ('n_value', 13)
// CHECK:     ('_string', '_f1')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 7)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 21)
// CHECK:     ('_string', 'local_a_ext')
// CHECK:    ),
// CHECK:     # Symbol 9
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', 'undef')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 6)
// CHECK:   ('iextdefsym', 6)
// CHECK:   ('nextdefsym', 3)
// CHECK:   ('iundefsym', 9)
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
