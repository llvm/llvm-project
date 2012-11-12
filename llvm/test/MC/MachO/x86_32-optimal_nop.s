// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

# 1 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nop
        # 0x90
        .align 1, 0x90
        ret
# 2 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # xchg %ax,%ax
        # 0x66, 0x90
        .align 2, 0x90
        ret
# 3 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl (%[re]ax)
        # 0x0f, 0x1f, 0x00
        .align 2, 0x90
        ret
# 4 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopl 0(%[re]ax)
        # 0x0f, 0x1f, 0x40, 0x00
        .align 3, 0x90
        ret
# 5 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopl 0(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 6 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 7 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 8 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 9 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw 0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 10 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 11 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 12 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 4, 0x90
        ret
# 13 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopl 0L(%[re]ax)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 14 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 15 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret

        # Only the .text sections gets optimal nops.
	.section	__TEXT,__const
f0:
        .byte 0
	.align	4, 0x90
        .long 0

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
// CHECK:   ('vm_size', 372)
// CHECK:   ('file_offset', 324)
// CHECK:   ('file_size', 372)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 2)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 337)
// CHECK:     ('offset', 324)
// CHECK:     ('alignment', 4)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', 'c390c300 00000000 00000000 00000000 c3c36690 c3000000 00000000 00000000 c30f1f00 c3000000 00000000 00000000 c3c3c3c3 0f1f4000 c3000000 00000000 c3c3c30f 1f440000 c3000000 00000000 c3c3660f 1f440000 c3000000 00000000 c30f1f80 00000000 c3000000 00000000 c3c3c3c3 c3c3c3c3 c3000000 00000000 c3c3c3c3 c3c3c366 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3c3c3c3 c3c3c366 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3c3c3c3 c366662e 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3c3c3c3 6666662e 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3c3c366 6666662e 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3c36666 6666662e 0f1f8400 00000000 c3000000 00000000 00000000 00000000 c3666666 6666662e 0f1f8400 00000000 c3')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__const\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 352)
// CHECK:     ('size', 20)
// CHECK:     ('offset', 676)
// CHECK:     ('alignment', 4)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x0)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '00909090 90909090 90909090 90909090 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 696)
// CHECK:   ('nsyms', 1)
// CHECK:   ('stroff', 708)
// CHECK:   ('strsize', 4)
// CHECK:   ('_string_data', '\x00f0\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 352)
// CHECK:     ('_string', 'f0')
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
