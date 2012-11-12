// RUN: llvm-mc -triple i386-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: macho-dump --dump-section-data < %t.o > %t.dump
// RUN: FileCheck --check-prefix=CHECK-I386 < %t.dump %s

// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o %t.o
// RUN: macho-dump --dump-section-data < %t.o > %t.dump
// RUN: FileCheck --check-prefix=CHECK-X86_64 < %t.dump %s

.data

        .long 0
a:
        .long 0
b = a

c:      .long b

d2 = d
.globl d2
d3 = d + 4
.globl d3

e = a + 4

g:
f = g
        .long 0
        
        .long b
        .long e
        .long a + 4
        .long d
        .long d2
        .long d3
        .long f
        .long g

///
        .text
t0:
Lt0_a:
        ret

	.data
Lt0_b:
Lt0_x = Lt0_a - Lt0_b
	.quad	Lt0_x

// CHECK-I386: ('cputype', 7)
// CHECK-I386: ('cpusubtype', 3)
// CHECK-I386: ('filetype', 1)
// CHECK-I386: ('num_load_commands', 3)
// CHECK-I386: ('load_commands_size', 296)
// CHECK-I386: ('flag', 0)
// CHECK-I386: ('load_commands', [
// CHECK-I386:   # Load Command 0
// CHECK-I386:  (('command', 1)
// CHECK-I386:   ('size', 192)
// CHECK-I386:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-I386:   ('vm_addr', 0)
// CHECK-I386:   ('vm_size', 57)
// CHECK-I386:   ('file_offset', 324)
// CHECK-I386:   ('file_size', 57)
// CHECK-I386:   ('maxprot', 7)
// CHECK-I386:   ('initprot', 7)
// CHECK-I386:   ('num_sections', 2)
// CHECK-I386:   ('flags', 0)
// CHECK-I386:   ('sections', [
// CHECK-I386:     # Section 0
// CHECK-I386:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-I386:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-I386:     ('address', 0)
// CHECK-I386:     ('size', 1)
// CHECK-I386:     ('offset', 324)
// CHECK-I386:     ('alignment', 0)
// CHECK-I386:     ('reloc_offset', 0)
// CHECK-I386:     ('num_reloc', 0)
// CHECK-I386:     ('flags', 0x80000400)
// CHECK-I386:     ('reserved1', 0)
// CHECK-I386:     ('reserved2', 0)
// CHECK-I386:    ),
// CHECK-I386:   ('_relocations', [
// CHECK-I386:   ])
// CHECK-I386:   ('_section_data', 'c3')
// CHECK-I386:     # Section 1
// CHECK-I386:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-I386:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-I386:     ('address', 1)
// CHECK-I386:     ('size', 56)
// CHECK-I386:     ('offset', 325)
// CHECK-I386:     ('alignment', 0)
// CHECK-I386:     ('reloc_offset', 384)
// CHECK-I386:     ('num_reloc', 9)
// CHECK-I386:     ('flags', 0x0)
// CHECK-I386:     ('reserved1', 0)
// CHECK-I386:     ('reserved2', 0)
// CHECK-I386:    ),
// CHECK-I386:   ('_relocations', [
// CHECK-I386:     # Relocation 0
// CHECK-I386:     (('word-0', 0x2c),
// CHECK-I386:      ('word-1', 0x4000002)),
// CHECK-I386:     # Relocation 1
// CHECK-I386:     (('word-0', 0x28),
// CHECK-I386:      ('word-1', 0x4000002)),
// CHECK-I386:     # Relocation 2
// CHECK-I386:     (('word-0', 0x24),
// CHECK-I386:      ('word-1', 0xc000009)),
// CHECK-I386:     # Relocation 3
// CHECK-I386:     (('word-0', 0x20),
// CHECK-I386:      ('word-1', 0xc000008)),
// CHECK-I386:     # Relocation 4
// CHECK-I386:     (('word-0', 0x1c),
// CHECK-I386:      ('word-1', 0xc000007)),
// CHECK-I386:     # Relocation 5
// CHECK-I386:     (('word-0', 0xa0000018),
// CHECK-I386:      ('word-1', 0x5)),
// CHECK-I386:     # Relocation 6
// CHECK-I386:     (('word-0', 0x14),
// CHECK-I386:      ('word-1', 0x4000002)),
// CHECK-I386:     # Relocation 7
// CHECK-I386:     (('word-0', 0x10),
// CHECK-I386:      ('word-1', 0x4000002)),
// CHECK-I386:     # Relocation 8
// CHECK-I386:     (('word-0', 0x8),
// CHECK-I386:      ('word-1', 0x4000002)),
// CHECK-I386:   ])
// CHECK-I386:   ('_section_data', '00000000 00000000 05000000 00000000 05000000 09000000 09000000 00000000 00000000 00000000 0d000000 0d000000 cfffffff ffffffff')
// CHECK-I386:   ])
// CHECK-I386:  ),
// CHECK-I386:   # Load Command 1
// CHECK-I386:  (('command', 2)
// CHECK-I386:   ('size', 24)
// CHECK-I386:   ('symoff', 456)
// CHECK-I386:   ('nsyms', 10)
// CHECK-I386:   ('stroff', 576)
// CHECK-I386:   ('strsize', 24)
// CHECK-I386:   ('_string_data', '\x00d2\x00d\x00d3\x00a\x00b\x00c\x00e\x00g\x00f\x00t0\x00')
// CHECK-I386:   ('_symbols', [
// CHECK-I386:     # Symbol 0
// CHECK-I386:    (('n_strx', 9)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 5)
// CHECK-I386:     ('_string', 'a')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 1
// CHECK-I386:    (('n_strx', 11)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 5)
// CHECK-I386:     ('_string', 'b')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 2
// CHECK-I386:    (('n_strx', 13)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 9)
// CHECK-I386:     ('_string', 'c')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 3
// CHECK-I386:    (('n_strx', 15)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 9)
// CHECK-I386:     ('_string', 'e')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 4
// CHECK-I386:    (('n_strx', 17)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 13)
// CHECK-I386:     ('_string', 'g')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 5
// CHECK-I386:    (('n_strx', 19)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 2)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 13)
// CHECK-I386:     ('_string', 'f')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 6
// CHECK-I386:    (('n_strx', 21)
// CHECK-I386:     ('n_type', 0xe)
// CHECK-I386:     ('n_sect', 1)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 0)
// CHECK-I386:     ('_string', 't0')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 7
// CHECK-I386:    (('n_strx', 4)
// CHECK-I386:     ('n_type', 0x1)
// CHECK-I386:     ('n_sect', 0)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 0)
// CHECK-I386:     ('_string', 'd')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 8
// CHECK-I386:    (('n_strx', 1)
// CHECK-I386:     ('n_type', 0x1)
// CHECK-I386:     ('n_sect', 0)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 0)
// CHECK-I386:     ('_string', 'd2')
// CHECK-I386:    ),
// CHECK-I386:     # Symbol 9
// CHECK-I386:    (('n_strx', 6)
// CHECK-I386:     ('n_type', 0x1)
// CHECK-I386:     ('n_sect', 0)
// CHECK-I386:     ('n_desc', 0)
// CHECK-I386:     ('n_value', 0)
// CHECK-I386:     ('_string', 'd3')
// CHECK-I386:    ),
// CHECK-I386:   ])
// CHECK-I386:  ),
// CHECK-I386:   # Load Command 2
// CHECK-I386:  (('command', 11)
// CHECK-I386:   ('size', 80)
// CHECK-I386:   ('ilocalsym', 0)
// CHECK-I386:   ('nlocalsym', 7)
// CHECK-I386:   ('iextdefsym', 7)
// CHECK-I386:   ('nextdefsym', 0)
// CHECK-I386:   ('iundefsym', 7)
// CHECK-I386:   ('nundefsym', 3)
// CHECK-I386:   ('tocoff', 0)
// CHECK-I386:   ('ntoc', 0)
// CHECK-I386:   ('modtaboff', 0)
// CHECK-I386:   ('nmodtab', 0)
// CHECK-I386:   ('extrefsymoff', 0)
// CHECK-I386:   ('nextrefsyms', 0)
// CHECK-I386:   ('indirectsymoff', 0)
// CHECK-I386:   ('nindirectsyms', 0)
// CHECK-I386:   ('extreloff', 0)
// CHECK-I386:   ('nextrel', 0)
// CHECK-I386:   ('locreloff', 0)
// CHECK-I386:   ('nlocrel', 0)
// CHECK-I386:   ('_indirect_symbols', [
// CHECK-I386:   ])
// CHECK-I386:  ),
// CHECK-I386: ])

// CHECK-X86_64: ('cputype', 16777223)
// CHECK-X86_64: ('cpusubtype', 3)
// CHECK-X86_64: ('filetype', 1)
// CHECK-X86_64: ('num_load_commands', 3)
// CHECK-X86_64: ('load_commands_size', 336)
// CHECK-X86_64: ('flag', 0)
// CHECK-X86_64: ('reserved', 0)
// CHECK-X86_64: ('load_commands', [
// CHECK-X86_64:   # Load Command 0
// CHECK-X86_64:  (('command', 25)
// CHECK-X86_64:   ('size', 232)
// CHECK-X86_64:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:   ('vm_addr', 0)
// CHECK-X86_64:   ('vm_size', 57)
// CHECK-X86_64:   ('file_offset', 368)
// CHECK-X86_64:   ('file_size', 57)
// CHECK-X86_64:   ('maxprot', 7)
// CHECK-X86_64:   ('initprot', 7)
// CHECK-X86_64:   ('num_sections', 2)
// CHECK-X86_64:   ('flags', 0)
// CHECK-X86_64:   ('sections', [
// CHECK-X86_64:     # Section 0
// CHECK-X86_64:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('address', 0)
// CHECK-X86_64:     ('size', 1)
// CHECK-X86_64:     ('offset', 368)
// CHECK-X86_64:     ('alignment', 0)
// CHECK-X86_64:     ('reloc_offset', 0)
// CHECK-X86_64:     ('num_reloc', 0)
// CHECK-X86_64:     ('flags', 0x80000400)
// CHECK-X86_64:     ('reserved1', 0)
// CHECK-X86_64:     ('reserved2', 0)
// CHECK-X86_64:     ('reserved3', 0)
// CHECK-X86_64:    ),
// CHECK-X86_64:   ('_relocations', [
// CHECK-X86_64:   ])
// CHECK-X86_64:   ('_section_data', 'c3')
// CHECK-X86_64:     # Section 1
// CHECK-X86_64:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-X86_64:     ('address', 1)
// CHECK-X86_64:     ('size', 56)
// CHECK-X86_64:     ('offset', 369)
// CHECK-X86_64:     ('alignment', 0)
// CHECK-X86_64:     ('reloc_offset', 428)
// CHECK-X86_64:     ('num_reloc', 9)
// CHECK-X86_64:     ('flags', 0x0)
// CHECK-X86_64:     ('reserved1', 0)
// CHECK-X86_64:     ('reserved2', 0)
// CHECK-X86_64:     ('reserved3', 0)
// CHECK-X86_64:    ),
// CHECK-X86_64:   ('_relocations', [
// CHECK-X86_64:     # Relocation 0
// CHECK-X86_64:     (('word-0', 0x2c),
// CHECK-X86_64:      ('word-1', 0xc000004)),
// CHECK-X86_64:     # Relocation 1
// CHECK-X86_64:     (('word-0', 0x28),
// CHECK-X86_64:      ('word-1', 0xc000005)),
// CHECK-X86_64:     # Relocation 2
// CHECK-X86_64:     (('word-0', 0x24),
// CHECK-X86_64:      ('word-1', 0xc000009)),
// CHECK-X86_64:     # Relocation 3
// CHECK-X86_64:     (('word-0', 0x20),
// CHECK-X86_64:      ('word-1', 0xc000008)),
// CHECK-X86_64:     # Relocation 4
// CHECK-X86_64:     (('word-0', 0x1c),
// CHECK-X86_64:      ('word-1', 0xc000007)),
// CHECK-X86_64:     # Relocation 5
// CHECK-X86_64:     (('word-0', 0x18),
// CHECK-X86_64:      ('word-1', 0xc000000)),
// CHECK-X86_64:     # Relocation 6
// CHECK-X86_64:     (('word-0', 0x14),
// CHECK-X86_64:      ('word-1', 0xc000003)),
// CHECK-X86_64:     # Relocation 7
// CHECK-X86_64:     (('word-0', 0x10),
// CHECK-X86_64:      ('word-1', 0xc000001)),
// CHECK-X86_64:     # Relocation 8
// CHECK-X86_64:     (('word-0', 0x8),
// CHECK-X86_64:      ('word-1', 0xc000001)),
// CHECK-X86_64:   ])
// CHECK-X86_64:   ('_section_data', '00000000 00000000 00000000 00000000 00000000 00000000 04000000 00000000 00000000 00000000 00000000 00000000 cfffffff ffffffff')
// CHECK-X86_64:   ])
// CHECK-X86_64:  ),
// CHECK-X86_64:   # Load Command 1
// CHECK-X86_64:  (('command', 2)
// CHECK-X86_64:   ('size', 24)
// CHECK-X86_64:   ('symoff', 500)
// CHECK-X86_64:   ('nsyms', 10)
// CHECK-X86_64:   ('stroff', 660)
// CHECK-X86_64:   ('strsize', 24)
// CHECK-X86_64:   ('_string_data', '\x00d2\x00d\x00d3\x00a\x00b\x00c\x00e\x00g\x00f\x00t0\x00')
// CHECK-X86_64:   ('_symbols', [
// CHECK-X86_64:     # Symbol 0
// CHECK-X86_64:    (('n_strx', 9)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 5)
// CHECK-X86_64:     ('_string', 'a')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 1
// CHECK-X86_64:    (('n_strx', 11)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 5)
// CHECK-X86_64:     ('_string', 'b')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 2
// CHECK-X86_64:    (('n_strx', 13)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 9)
// CHECK-X86_64:     ('_string', 'c')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 3
// CHECK-X86_64:    (('n_strx', 15)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 9)
// CHECK-X86_64:     ('_string', 'e')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 4
// CHECK-X86_64:    (('n_strx', 17)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 13)
// CHECK-X86_64:     ('_string', 'g')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 5
// CHECK-X86_64:    (('n_strx', 19)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 2)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 13)
// CHECK-X86_64:     ('_string', 'f')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 6
// CHECK-X86_64:    (('n_strx', 21)
// CHECK-X86_64:     ('n_type', 0xe)
// CHECK-X86_64:     ('n_sect', 1)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 't0')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 7
// CHECK-X86_64:    (('n_strx', 4)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'd')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 8
// CHECK-X86_64:    (('n_strx', 1)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'd2')
// CHECK-X86_64:    ),
// CHECK-X86_64:     # Symbol 9
// CHECK-X86_64:    (('n_strx', 6)
// CHECK-X86_64:     ('n_type', 0x1)
// CHECK-X86_64:     ('n_sect', 0)
// CHECK-X86_64:     ('n_desc', 0)
// CHECK-X86_64:     ('n_value', 0)
// CHECK-X86_64:     ('_string', 'd3')
// CHECK-X86_64:    ),
// CHECK-X86_64:   ])
// CHECK-X86_64:  ),
// CHECK-X86_64:   # Load Command 2
// CHECK-X86_64:  (('command', 11)
// CHECK-X86_64:   ('size', 80)
// CHECK-X86_64:   ('ilocalsym', 0)
// CHECK-X86_64:   ('nlocalsym', 7)
// CHECK-X86_64:   ('iextdefsym', 7)
// CHECK-X86_64:   ('nextdefsym', 0)
// CHECK-X86_64:   ('iundefsym', 7)
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
