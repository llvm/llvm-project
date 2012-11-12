// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .text

// FIXME: llvm-mc doesn't handle this in a way we can make compatible with 'as',
// currently, because of how we handle assembler variables.
//
// See <rdar://problem/7763719> improve handling of absolute symbols

// _baz = 4

_foo:
        xorl %eax,%eax
_g0:
        xorl %eax,%eax
L0:
        jmp 4
//        jmp _baz

// FIXME: Darwin 'as' for historical reasons widens this jump, but doesn't emit
// a relocation. It seems like 'as' widens any jump that is not to a temporary,
// which is inherited from the x86_32 behavior, even though x86_64 could do
// better.
//        jmp _g0

        jmp L0
        jmp _g1

// FIXME: Darwin 'as' gets this wrong as well, even though it could get it right
// given the other things we do on x86_64. It is using a short jump here. This
// is probably fallout of the hack that exists for x86_32.
//        jmp L1

// FIXME: We don't support this, and would currently get it wrong, it should be a jump to an absolute address.
//        jmp L0 - _g0

//        jmp _g1 - _g0
// FIXME: Darwin 'as' comes up with 'SIGNED' here instead of 'BRANCH'.
//        jmp _g1 - L1
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L1 - _g0

        jmp _g2
        jmp L2
        jmp _g3
        jmp L3
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L2 - _g3
//        jmp _g3 - _g2
// FIXME: Darwin 'as' comes up with 'SIGNED' here instead of 'BRANCH'.
//        jmp _g3 - L3
// FIXME: Darwin 'as' gets this completely wrong. It ends up with a single
// branch relocation. Fallout from the other delta hack?
//        jmp L3 - _g2

        movl %eax,4(%rip)
//        movl %eax,_baz(%rip)
        movl %eax,_g0(%rip)
        movl %eax,L0(%rip)
        movl %eax,_g1(%rip)
        movl %eax,L1(%rip)

// FIXME: Darwin 'as' gets most of these wrong, and there is an ambiguity in ATT
// syntax in what they should mean in the first place (absolute or
// rip-relative address).
//        movl %eax,L0 - _g0(%rip)
//        movl %eax,_g1 - _g0(%rip)
//        movl %eax,_g1 - L1(%rip)
//        movl %eax,L1 - _g0(%rip)

        movl %eax,_g2(%rip)
        movl %eax,L2(%rip)
        movl %eax,_g3(%rip)
        movl %eax,L3(%rip)

// FIXME: Darwin 'as' gets most of these wrong, and there is an ambiguity in ATT
// syntax in what they should mean in the first place (absolute or
// rip-relative address).
//        movl %eax,L2 - _g2(%rip)
//        movl %eax,_g3 - _g2(%rip)
//        movl %eax,_g3 - L3(%rip)
//        movl %eax,L3 - _g2(%rip)

_g1:
        xorl %eax,%eax
L1:
        xorl %eax,%eax

        .data
_g2:
        xorl %eax,%eax
L2:
        .quad 4
//        .quad _baz
        .quad _g2
        .quad L2
        .quad _g3
        .quad L3
        .quad L2 - _g2
        .quad _g3 - _g2
        .quad L3 - _g2
        .quad L3 - _g3

        .quad _g0
        .quad L0
        .quad _g1
        .quad L1
        .quad L0 - _g0
        .quad _g1 - _g0
        .quad L1 - _g0
        .quad L1 - _g1

_g3:
        xorl %eax,%eax
L3:
        xorl %eax,%eax

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 336)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 232)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 236)
// CHECK:   ('file_offset', 368)
// CHECK:   ('file_size', 236)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 94)
// CHECK:     ('offset', 368)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 604)
// CHECK:     ('num_reloc', 12)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [

// FIXME: Unfortunately, we do not get these relocations in exactly the same
// order as Darwin 'as'. It turns out that 'as' *usually* ends up emitting
// them in reverse address order, but sometimes it allocates some
// additional relocations late so these end up precede the other entries. I
// haven't figured out the exact criteria for this yet.
        
// CHECK:     (('word-0', 0x56),
// CHECK:      ('word-1', 0x1d000004)),
// CHECK:     (('word-0', 0x50),
// CHECK:      ('word-1', 0x1d000004)),
// CHECK:     (('word-0', 0x4a),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     (('word-0', 0x44),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     (('word-0', 0x3e),
// CHECK:      ('word-1', 0x1d000002)),
// CHECK:     (('word-0', 0x38),
// CHECK:      ('word-1', 0x1d000002)),
// CHECK:     (('word-0', 0x20),
// CHECK:      ('word-1', 0x2d000004)),
// CHECK:     (('word-0', 0x1b),
// CHECK:      ('word-1', 0x2d000004)),
// CHECK:     (('word-0', 0x16),
// CHECK:      ('word-1', 0x2d000003)),
// CHECK:     (('word-0', 0x11),
// CHECK:      ('word-1', 0x2d000003)),
// CHECK:     (('word-0', 0xc),
// CHECK:      ('word-1', 0x2d000002)),
// CHECK:     (('word-0', 0x5),
// CHECK:      ('word-1', 0x2d000000)),
// CHECK:   ])
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 94)
// CHECK:     ('size', 142)
// CHECK:     ('offset', 462)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 700)
// CHECK:     ('num_reloc', 16)
// CHECK:     ('flags', 0x400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x7a),
// CHECK:      ('word-1', 0x5e000001)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x7a),
// CHECK:      ('word-1', 0xe000002)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x72),
// CHECK:      ('word-1', 0x5e000001)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x72),
// CHECK:      ('word-1', 0xe000002)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0x62),
// CHECK:      ('word-1', 0xe000002)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0x5a),
// CHECK:      ('word-1', 0xe000002)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0x52),
// CHECK:      ('word-1', 0xe000001)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0x4a),
// CHECK:      ('word-1', 0xe000001)),
// CHECK:     # Relocation 8
// CHECK:     (('word-0', 0x3a),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 9
// CHECK:     (('word-0', 0x3a),
// CHECK:      ('word-1', 0xe000004)),
// CHECK:     # Relocation 10
// CHECK:     (('word-0', 0x32),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 11
// CHECK:     (('word-0', 0x32),
// CHECK:      ('word-1', 0xe000004)),
// CHECK:     # Relocation 12
// CHECK:     (('word-0', 0x22),
// CHECK:      ('word-1', 0xe000004)),
// CHECK:     # Relocation 13
// CHECK:     (('word-0', 0x1a),
// CHECK:      ('word-1', 0xe000004)),
// CHECK:     # Relocation 14
// CHECK:     (('word-0', 0x12),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:     # Relocation 15
// CHECK:     (('word-0', 0xa),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:   ])
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 828)
// CHECK:   ('nsyms', 5)
// CHECK:   ('stroff', 908)
// CHECK:   ('strsize', 24)
// CHECK:   ('_string_data', '\x00_foo\x00_g0\x00_g1\x00_g2\x00_g3\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_foo')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 6)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 2)
// CHECK:     ('_string', '_g0')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 10)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 90)
// CHECK:     ('_string', '_g1')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 14)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 94)
// CHECK:     ('_string', '_g2')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 18)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 232)
// CHECK:     ('_string', '_g3')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 5)
// CHECK:   ('iextdefsym', 5)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 5)
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
