// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

        .data

        .org 0x10
L0:
        .long 0
        .long 0
        .long 0
        .long 0

_d:
        .long 0
L1:
        .long 0

        .text

// These generate normal x86_64 (external) relocations. They could all use
// SIGNED, but don't for pedantic compatibility with Darwin 'as'.

        // SIGNED1
 	movb  $0x12, _d(%rip)

        // SIGNED
 	movb  $0x12, _d + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, _d(%rip)

        // SIGNED
 	movl  $0x12345678, _d + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, _d + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, _d + 3(%rip)

        // SIGNED
 	movl  $0x12345678, _d + 4(%rip)

	movb  %al, _d(%rip)
 	movb  %al, _d + 1(%rip)
 	movl  %eax, _d(%rip)
 	movl  %eax, _d + 1(%rip)
 	movl  %eax, _d + 2(%rip)
 	movl  %eax, _d + 3(%rip)
 	movl  %eax, _d + 4(%rip)

// These have to use local relocations. Since that uses an offset into the
// section in x86_64 (as opposed to a scattered relocation), and since the
// linker can only decode this to an atom + offset by scanning the section,
// it is not possible to correctly encode these without SIGNED<N>. This is
// ultimately due to a design flaw in the x86_64 relocation format, it is
// not possible to encode an address (L<foo> + <constant>) which is outside the
// atom containing L<foo>.

        // SIGNED1
 	movb  $0x12, L0(%rip)

        // SIGNED
 	movb  $0x12, L0 + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, L0(%rip)

        // SIGNED
 	movl  $0x12345678, L0 + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, L0 + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, L0 + 3(%rip)

        // SIGNED
 	movl  $0x12345678, L0 + 4(%rip)

 	movb  %al, L0(%rip)
 	movb  %al, L0 + 1(%rip)
 	movl  %eax, L0(%rip)
 	movl  %eax, L0 + 1(%rip)
 	movl  %eax, L0 + 2(%rip)
 	movl  %eax, L0 + 3(%rip)
 	movl  %eax, L0 + 4(%rip)

        // SIGNED1
 	movb  $0x12, L1(%rip)

        // SIGNED
 	movb  $0x12, L1 + 1(%rip)

        // SIGNED4
 	movl  $0x12345678, L1(%rip)

        // SIGNED
 	movl  $0x12345678, L1 + 1(%rip)

        // SIGNED2
 	movl  $0x12345678, L1 + 2(%rip)

        // SIGNED1
 	movl  $0x12345678, L1 + 3(%rip)

        // SIGNED
 	movl  $0x12345678, L1 + 4(%rip)

 	movb  %al, L1(%rip)
 	movb  %al, L1 + 1(%rip)
 	movl  %eax, L1(%rip)
 	movl  %eax, L1 + 1(%rip)
 	movl  %eax, L1 + 2(%rip)
 	movl  %eax, L1 + 3(%rip)
 	movl  %eax, L1 + 4(%rip)

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
// CHECK:   ('vm_size', 358)
// CHECK:   ('file_offset', 368)
// CHECK:   ('file_size', 358)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 318)
// CHECK:     ('offset', 368)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 728)
// CHECK:     ('num_reloc', 42)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x13a),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x134),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x12e),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x128),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0x122),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0x11c),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0x116),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0x10c),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 8
// CHECK:     (('word-0', 0x102),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:     # Relocation 9
// CHECK:     (('word-0', 0xf8),
// CHECK:      ('word-1', 0x7d000000)),
// CHECK:     # Relocation 10
// CHECK:     (('word-0', 0xee),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 11
// CHECK:     (('word-0', 0xe4),
// CHECK:      ('word-1', 0x8d000000)),
// CHECK:     # Relocation 12
// CHECK:     (('word-0', 0xdd),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 13
// CHECK:     (('word-0', 0xd6),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:     # Relocation 14
// CHECK:     (('word-0', 0xd0),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 15
// CHECK:     (('word-0', 0xca),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 16
// CHECK:     (('word-0', 0xc4),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 17
// CHECK:     (('word-0', 0xbe),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 18
// CHECK:     (('word-0', 0xb8),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 19
// CHECK:     (('word-0', 0xb2),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 20
// CHECK:     (('word-0', 0xac),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 21
// CHECK:     (('word-0', 0xa2),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 22
// CHECK:     (('word-0', 0x98),
// CHECK:      ('word-1', 0x65000002)),
// CHECK:     # Relocation 23
// CHECK:     (('word-0', 0x8e),
// CHECK:      ('word-1', 0x75000002)),
// CHECK:     # Relocation 24
// CHECK:     (('word-0', 0x84),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 25
// CHECK:     (('word-0', 0x7a),
// CHECK:      ('word-1', 0x85000002)),
// CHECK:     # Relocation 26
// CHECK:     (('word-0', 0x73),
// CHECK:      ('word-1', 0x15000002)),
// CHECK:     # Relocation 27
// CHECK:     (('word-0', 0x6c),
// CHECK:      ('word-1', 0x65000002)),
// CHECK:     # Relocation 28
// CHECK:     (('word-0', 0x66),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 29
// CHECK:     (('word-0', 0x60),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 30
// CHECK:     (('word-0', 0x5a),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 31
// CHECK:     (('word-0', 0x54),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 32
// CHECK:     (('word-0', 0x4e),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 33
// CHECK:     (('word-0', 0x48),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 34
// CHECK:     (('word-0', 0x42),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 35
// CHECK:     (('word-0', 0x38),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 36
// CHECK:     (('word-0', 0x2e),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:     # Relocation 37
// CHECK:     (('word-0', 0x24),
// CHECK:      ('word-1', 0x7d000000)),
// CHECK:     # Relocation 38
// CHECK:     (('word-0', 0x1a),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 39
// CHECK:     (('word-0', 0x10),
// CHECK:      ('word-1', 0x8d000000)),
// CHECK:     # Relocation 40
// CHECK:     (('word-0', 0x9),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 41
// CHECK:     (('word-0', 0x2),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:   ])
// CHECK:   ('_section_data', 'c605ffff ffff12c6 05000000 0012c705 fcffffff 78563412 c705fdff ffff7856 3412c705 feffffff 78563412 c705ffff ffff7856 3412c705 00000000 78563412 88050000 00008805 01000000 89050000 00008905 01000000 89050200 00008905 03000000 89050400 0000c605 dd000000 12c605d7 00000012 c705cc00 00007856 3412c705 c3000000 78563412 c705ba00 00007856 3412c705 b1000000 78563412 c705a800 00007856 34128805 9e000000 88059900 00008905 92000000 89058d00 00008905 88000000 89058300 00008905 7e000000 c6050300 000012c6 05040000 0012c705 00000000 78563412 c7050100 00007856 3412c705 02000000 78563412 c7050300 00007856 3412c705 04000000 78563412 88050400 00008805 05000000 89050400 00008905 05000000 89050600 00008905 07000000 89050800 0000')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 318)
// CHECK:     ('size', 40)
// CHECK:     ('offset', 686)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 1064)
// CHECK:   ('nsyms', 1)
// CHECK:   ('stroff', 1080)
// CHECK:   ('strsize', 4)
// CHECK:   ('_string_data', '\x00_d\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 350)
// CHECK:     ('_string', '_d')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 1)
// CHECK:   ('iextdefsym', 1)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 1)
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
