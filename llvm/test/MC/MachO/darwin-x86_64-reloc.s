// RUN: llvm-mc -n -triple x86_64-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// These examples are taken from <mach-o/x86_64/reloc.h>.

        .data
        .long 0

        .text
_foo:
        ret

_baz:
        call _foo
 	call _foo+4
 	movq _foo@GOTPCREL(%rip), %rax
 	pushq _foo@GOTPCREL(%rip)
 	movl _foo(%rip), %eax
 	movl _foo+4(%rip), %eax
 	movb  $0x12, _foo(%rip)
 	movl  $0x12345678, _foo(%rip)
 	.quad _foo
_bar:
 	.quad _foo+4
 	.quad _foo - _bar
 	.quad _foo - _bar + 4
 	.long _foo - _bar
 	leaq L1(%rip), %rax
 	leaq L0(%rip), %rax
        addl $6,L0(%rip)
        addw $500,L0(%rip)
        addl $500,L0(%rip)

_prev:
        .space 12,0x90
 	.quad L1
L0:
        .quad L0
L_pc:
 	.quad _foo - L_pc
 	.quad _foo - L1
L1:
 	.quad L1 - _prev

        .data
.long	_foobar@GOTPCREL+4
.long	_foo@GOTPCREL+4

        .section	__DWARF,__debug_frame,regular,debug
        .quad L1
        .quad _ext_foo

// Make sure local label which overlaps with non-local one is assigned to the
// right atom.
        .text
_f2:
L2_0:
        addl $0, %eax
L2_1:        
_f3:
        addl L2_1 - L2_0, %eax
        
        .data
L4:     
        .long 0
        .text
        movl L4(%rip), %eax

        .section __TEXT,__literal8,8byte_literals
	.quad 0
L5:     
	.quad 0
f6:
        .quad 0
L6:
        .quad 0
        
        .text
	movl L5(%rip), %eax
	movl f6(%rip), %eax
	movl L6(%rip), %eax
        
        .data
        .quad L5
        .quad f6
	.quad L6

        .text
        cmpq $0, _foo@GOTPCREL(%rip)
        
// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 496)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 392)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 311)
// CHECK:   ('file_offset', 528)
// CHECK:   ('file_size', 311)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 4)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__data\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 40)
// CHECK:     ('offset', 528)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 840)
// CHECK:     ('num_reloc', 5)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x20),
// CHECK:      ('word-1', 0x6000004)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x18),
// CHECK:      ('word-1', 0xe000006)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0x10),
// CHECK:      ('word-1', 0x6000004)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0x4d000000)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0x4),
// CHECK:      ('word-1', 0x4d000008)),
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 04000000 04000000 00000000 1f010000 00000000 00000000 00000000 2f010000 00000000')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 40)
// CHECK:     ('size', 223)
// CHECK:     ('offset', 568)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 880)
// CHECK:     ('num_reloc', 32)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0xda),
// CHECK:      ('word-1', 0x4d000000)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0xd3),
// CHECK:      ('word-1', 0x15000004)),
// CHECK:     # Relocation 2
// CHECK:     (('word-0', 0xcd),
// CHECK:      ('word-1', 0x1d000006)),
// CHECK:     # Relocation 3
// CHECK:     (('word-0', 0xc7),
// CHECK:      ('word-1', 0x15000004)),
// CHECK:     # Relocation 4
// CHECK:     (('word-0', 0xc1),
// CHECK:      ('word-1', 0x15000001)),
// CHECK:     # Relocation 5
// CHECK:     (('word-0', 0xa5),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 6
// CHECK:     (('word-0', 0xa5),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 7
// CHECK:     (('word-0', 0x9d),
// CHECK:      ('word-1', 0x5e000003)),
// CHECK:     # Relocation 8
// CHECK:     (('word-0', 0x9d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 9
// CHECK:     (('word-0', 0x95),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:     # Relocation 10
// CHECK:     (('word-0', 0x8d),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:     # Relocation 11
// CHECK:     (('word-0', 0x79),
// CHECK:      ('word-1', 0x8d000003)),
// CHECK:     # Relocation 12
// CHECK:     (('word-0', 0x71),
// CHECK:      ('word-1', 0x7d000003)),
// CHECK:     # Relocation 13
// CHECK:     (('word-0', 0x69),
// CHECK:      ('word-1', 0x6d000003)),
// CHECK:     # Relocation 14
// CHECK:     (('word-0', 0x63),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     # Relocation 15
// CHECK:     (('word-0', 0x5c),
// CHECK:      ('word-1', 0x1d000003)),
// CHECK:     # Relocation 16
// CHECK:     (('word-0', 0x55),
// CHECK:      ('word-1', 0x5c000002)),
// CHECK:     # Relocation 17
// CHECK:     (('word-0', 0x55),
// CHECK:      ('word-1', 0xc000000)),
// CHECK:     # Relocation 18
// CHECK:     (('word-0', 0x4d),
// CHECK:      ('word-1', 0x5e000002)),
// CHECK:     # Relocation 19
// CHECK:     (('word-0', 0x4d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 20
// CHECK:     (('word-0', 0x45),
// CHECK:      ('word-1', 0x5e000002)),
// CHECK:     # Relocation 21
// CHECK:     (('word-0', 0x45),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 22
// CHECK:     (('word-0', 0x3d),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 23
// CHECK:     (('word-0', 0x35),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 24
// CHECK:     (('word-0', 0x2d),
// CHECK:      ('word-1', 0x8d000000)),
// CHECK:     # Relocation 25
// CHECK:     (('word-0', 0x26),
// CHECK:      ('word-1', 0x6d000000)),
// CHECK:     # Relocation 26
// CHECK:     (('word-0', 0x20),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 27
// CHECK:     (('word-0', 0x1a),
// CHECK:      ('word-1', 0x1d000000)),
// CHECK:     # Relocation 28
// CHECK:     (('word-0', 0x14),
// CHECK:      ('word-1', 0x4d000000)),
// CHECK:     # Relocation 29
// CHECK:     (('word-0', 0xe),
// CHECK:      ('word-1', 0x3d000000)),
// CHECK:     # Relocation 30
// CHECK:     (('word-0', 0x7),
// CHECK:      ('word-1', 0x2d000000)),
// CHECK:     # Relocation 31
// CHECK:     (('word-0', 0x2),
// CHECK:      ('word-1', 0x2d000000)),
// CHECK:   ])
// CHECK:   ('_section_data', 'c3e80000 0000e804 00000048 8b050000 0000ff35 00000000 8b050000 00008b05 04000000 c605ffff ffff12c7 05fcffff ff785634 12000000 00000000 00040000 00000000 00000000 00000000 00040000 00000000 00000000 00488d05 2c000000 488d0514 00000083 05130000 00066681 05120000 00f40181 05100000 00f40100 00909090 90909090 90909090 902c0000 00000000 00140000 00000000 00e4ffff ffffffff ffd4ffff ffffffff ff2c0000 00000000 0083c000 03042503 0000008b 051fffff ff8b052c 0000008b 05000000 008b0530 00000048 833dffff ffff00')
// CHECK:     # Section 2
// CHECK:    (('section_name', '__debug_frame\x00\x00\x00')
// CHECK:     ('segment_name', '__DWARF\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 263)
// CHECK:     ('size', 16)
// CHECK:     ('offset', 791)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 1136)
// CHECK:     ('num_reloc', 2)
// CHECK:     ('flags', 0x2000000)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x8),
// CHECK:      ('word-1', 0xe000007)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0x6000002)),
// CHECK:   ])
// CHECK:   ('_section_data', 'd5000000 00000000 00000000 00000000')
// CHECK:     # Section 3
// CHECK:    (('section_name', '__literal8\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 279)
// CHECK:     ('size', 32)
// CHECK:     ('offset', 807)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x4)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 1152)
// CHECK:   ('nsyms', 9)
// CHECK:   ('stroff', 1296)
// CHECK:   ('strsize', 52)
// CHECK:   ('_string_data', '\x00_foobar\x00_ext_foo\x00_foo\x00_baz\x00_bar\x00_prev\x00_f2\x00_f3\x00f6\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 18)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 40)
// CHECK:     ('_string', '_foo')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 23)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 41)
// CHECK:     ('_string', '_baz')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 28)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 101)
// CHECK:     ('_string', '_bar')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 33)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 169)
// CHECK:     ('_string', '_prev')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 39)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 221)
// CHECK:     ('_string', '_f2')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 43)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 224)
// CHECK:     ('_string', '_f3')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 47)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 4)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 295)
// CHECK:     ('_string', 'f6')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 9)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_ext_foo')
// CHECK:    ),
// CHECK:     # Symbol 8
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_foobar')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 7)
// CHECK:   ('iextdefsym', 7)
// CHECK:   ('nextdefsym', 0)
// CHECK:   ('iundefsym', 7)
// CHECK:   ('nundefsym', 2)
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
