@ RUN: llvm-mc -n -triple thumbv7-apple-darwin10 %s -filetype=obj -o %t.obj
@ RUN: macho-dump --dump-section-data < %t.obj > %t.dump
@ RUN: FileCheck < %t.dump %s

	.syntax unified
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	2
	.code	16
	.thumb_func	_main
_main:
LPC0_0:
	blx	_printf
	.align	2
LCPI0_0:
	.long	L_.str-(LPC0_0+4)

	.section	__TEXT,__cstring,cstring_literals
	.align	2
L_.str:
	.asciz	 "s0"

.subsections_via_symbols

@ CHECK: ('cputype', 12)
@ CHECK: ('cpusubtype', 9)
@ CHECK: ('filetype', 1)
@ CHECK: ('num_load_commands', 3)
@ CHECK: ('load_commands_size', 296)
@ CHECK: ('flag', 8192)
@ CHECK: ('load_commands', [
@ CHECK:   # Load Command 0
@ CHECK:  (('command', 1)
@ CHECK:   ('size', 192)
@ CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:   ('vm_addr', 0)
@ CHECK:   ('vm_size', 11)
@ CHECK:   ('file_offset', 324)
@ CHECK:   ('file_size', 11)
@ CHECK:   ('maxprot', 7)
@ CHECK:   ('initprot', 7)
@ CHECK:   ('num_sections', 2)
@ CHECK:   ('flags', 0)
@ CHECK:   ('sections', [
@ CHECK:     # Section 0
@ CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 0)
@ CHECK:     ('size', 8)
@ CHECK:     ('offset', 324)
@ CHECK:     ('alignment', 2)
@ CHECK:     ('reloc_offset', 336)
@ CHECK:     ('num_reloc', 3)
@ CHECK:     ('flags', 0x80000400)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:     # Relocation 0
@ CHECK:     (('word-0', 0xa2000004),
@ CHECK:      ('word-1', 0x8)),
@ CHECK:     # Relocation 1
@ CHECK:     (('word-0', 0xa1000000),
@ CHECK:      ('word-1', 0x0)),
@ CHECK:     # Relocation 2
@ CHECK:     (('word-0', 0x0),
@ CHECK:      ('word-1', 0x6d000001)),
@ CHECK:   ])
@ CHECK-FIXME:   ('_section_data', 'fff7feef 04000000')
@ CHECK:     # Section 1
@ CHECK:    (('section_name', '__cstring\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
@ CHECK:     ('address', 8)
@ CHECK:     ('size', 3)
@ CHECK:     ('offset', 332)
@ CHECK:     ('alignment', 2)
@ CHECK:     ('reloc_offset', 0)
@ CHECK:     ('num_reloc', 0)
@ CHECK:     ('flags', 0x2)
@ CHECK:     ('reserved1', 0)
@ CHECK:     ('reserved2', 0)
@ CHECK:    ),
@ CHECK:   ('_relocations', [
@ CHECK:   ])
@ CHECK:   ('_section_data', '733000')
@ CHECK:   ])
@ CHECK:  ),
@ CHECK:   # Load Command 1
@ CHECK:  (('command', 2)
@ CHECK:   ('size', 24)
@ CHECK:   ('symoff', 360)
@ CHECK:   ('nsyms', 2)
@ CHECK:   ('stroff', 384)
@ CHECK:   ('strsize', 16)
@ CHECK:   ('_string_data', '\x00_main\x00_printf\x00\x00')
@ CHECK:   ('_symbols', [
@ CHECK:     # Symbol 0
@ CHECK:    (('n_strx', 1)
@ CHECK:     ('n_type', 0xf)
@ CHECK:     ('n_sect', 1)
@ CHECK:     ('n_desc', 8)
@ CHECK:     ('n_value', 0)
@ CHECK:     ('_string', '_main')
@ CHECK:    ),
@ CHECK:     # Symbol 1
@ CHECK:    (('n_strx', 7)
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
@ CHECK:   ('nlocalsym', 0)
@ CHECK:   ('iextdefsym', 0)
@ CHECK:   ('nextdefsym', 1)
@ CHECK:   ('iundefsym', 1)
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
