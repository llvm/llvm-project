// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

.tdata
_a$tlv$init:
	.long 4


.tlv
	.globl _a
_a:
	.quad __tlv_bootstrap
	.quad 0
	.quad _a$tlv$init

.text
	.globl _foo
	.align 4, 0x90

_foo:
	 movq   _a@TLVP(%rip), %rdi
	 call	*(%rdi) # returns &a in %rax
	 ret

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 416)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 312)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 38)
// CHECK:   ('file_offset', 448)
// CHECK:   ('file_size', 38)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 3)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 10)
// CHECK:     ('offset', 448)
// CHECK:     ('alignment', 4)
// CHECK:     ('reloc_offset', 488)
// CHECK:     ('num_reloc', 1)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x3),
// CHECK:      ('word-1', 0x9d000001)),
// CHECK:   ])
// CHECK:   ('_section_data', '488b3d00 000000ff 17c3')
// CHECK:     # Section 1
// CHECK:    (('section_name', '__thread_data\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 10)
// CHECK:     ('size', 4)
// CHECK:     ('offset', 458)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x11)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ('_section_data', '04000000')
// CHECK:     # Section 2
// CHECK:    (('section_name', '__thread_vars\x00\x00\x00')
// CHECK:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 14)
// CHECK:     ('size', 24)
// CHECK:     ('offset', 462)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 496)
// CHECK:     ('num_reloc', 2)
// CHECK:     ('flags', 0x13)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:     # Relocation 0
// CHECK:     (('word-0', 0x10),
// CHECK:      ('word-1', 0xe000000)),
// CHECK:     # Relocation 1
// CHECK:     (('word-0', 0x0),
// CHECK:      ('word-1', 0xe000003)),
// CHECK:   ])
// CHECK:   ('_section_data', '00000000 00000000 00000000 00000000 00000000 00000000')
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 512)
// CHECK:   ('nsyms', 4)
// CHECK:   ('stroff', 576)
// CHECK:   ('strsize', 40)
// CHECK:   ('_string_data', '\x00_a\x00__tlv_bootstrap\x00_foo\x00_a$tlv$init\x00\x00\x00\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 25)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 2)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 10)
// CHECK:     ('_string', '_a$tlv$init')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 3)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 14)
// CHECK:     ('_string', '_a')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 20)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_foo')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 4)
// CHECK:     ('n_type', 0x1)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '__tlv_bootstrap')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 1)
// CHECK:   ('iextdefsym', 1)
// CHECK:   ('nextdefsym', 2)
// CHECK:   ('iundefsym', 3)
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
