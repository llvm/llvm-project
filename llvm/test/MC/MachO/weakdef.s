// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

	.section	__DATA,__datacoal_nt,coalesced
	.section	__TEXT,__const_coal,coalesced
	.globl	__ZTS3optIbE            ## @_ZTS3optIbE
	.weak_definition	__ZTS3optIbE
__ZTS3optIbE:


	.section	__DATA,__datacoal_nt,coalesced
	.globl	__ZTI3optIbE            ## @_ZTI3optIbE
	.weak_definition	__ZTI3optIbE

__ZTI3optIbE:
	.long	__ZTS3optIbE

// CHECK:      ('cputype', 7)
// CHECK-NEXT: ('cpusubtype', 3)
// CHECK-NEXT: ('filetype', 1)
// CHECK-NEXT: ('num_load_commands', 3)
// CHECK-NEXT: ('load_commands_size', 364)
// CHECK-NEXT: ('flag', 0)
// CHECK-NEXT: ('load_commands', [
// CHECK-NEXT:   # Load Command 0
// CHECK-NEXT:  (('command', 1)
// CHECK-NEXT:   ('size', 260)
// CHECK-NEXT:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:   ('vm_addr', 0)
// CHECK-NEXT:   ('vm_size', 4)
// CHECK-NEXT:   ('file_offset', 392)
// CHECK-NEXT:   ('file_size', 4)
// CHECK-NEXT:   ('maxprot', 7)
// CHECK-NEXT:   ('initprot', 7)
// CHECK-NEXT:   ('num_sections', 3)
// CHECK-NEXT:   ('flags', 0)
// CHECK-NEXT:   ('sections', [
// CHECK-NEXT:     # Section 0
// CHECK-NEXT:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 0)
// CHECK-NEXT:     ('size', 0)
// CHECK-NEXT:     ('offset', 392)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 0)
// CHECK-NEXT:     ('num_reloc', 0)
// CHECK-NEXT:     ('flags', 0x80000000)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '')
// CHECK-NEXT:     # Section 1
// CHECK-NEXT:    (('section_name', '__datacoal_nt\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__DATA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 0)
// CHECK-NEXT:     ('size', 4)
// CHECK-NEXT:     ('offset', 392)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 396)
// CHECK-NEXT:     ('num_reloc', 1)
// CHECK-NEXT:     ('flags', 0xb)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:     # Relocation 0
// CHECK-NEXT:     (('word-0', 0x0),
// CHECK-NEXT:      ('word-1', 0xc000001)),
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '00000000')
// CHECK-NEXT:     # Section 2
// CHECK-NEXT:    (('section_name', '__const_coal\x00\x00\x00\x00')
// CHECK-NEXT:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK-NEXT:     ('address', 4)
// CHECK-NEXT:     ('size', 0)
// CHECK-NEXT:     ('offset', 396)
// CHECK-NEXT:     ('alignment', 0)
// CHECK-NEXT:     ('reloc_offset', 0)
// CHECK-NEXT:     ('num_reloc', 0)
// CHECK-NEXT:     ('flags', 0xb)
// CHECK-NEXT:     ('reserved1', 0)
// CHECK-NEXT:     ('reserved2', 0)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ('_relocations', [
// CHECK-NEXT:   ])
// CHECK-NEXT:   ('_section_data', '')
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT:   # Load Command 1
// CHECK-NEXT:  (('command', 2)
// CHECK-NEXT:   ('size', 24)
// CHECK-NEXT:   ('symoff', 404)
// CHECK-NEXT:   ('nsyms', 2)
// CHECK-NEXT:   ('stroff', 428)
// CHECK-NEXT:   ('strsize', 28)
// CHECK-NEXT:   ('_string_data', '\x00__ZTS3optIbE\x00__ZTI3optIbE\x00\x00')
// CHECK-NEXT:   ('_symbols', [
// CHECK-NEXT:     # Symbol 0
// CHECK-NEXT:    (('n_strx', 14)
// CHECK-NEXT:     ('n_type', 0xf)
// CHECK-NEXT:     ('n_sect', 2)
// CHECK-NEXT:     ('n_desc', 128)
// CHECK-NEXT:     ('n_value', 0)
// CHECK-NEXT:     ('_string', '__ZTI3optIbE')
// CHECK-NEXT:    ),
// CHECK-NEXT:     # Symbol 1
// CHECK-NEXT:    (('n_strx', 1)
// CHECK-NEXT:     ('n_type', 0xf)
// CHECK-NEXT:     ('n_sect', 3)
// CHECK-NEXT:     ('n_desc', 128)
// CHECK-NEXT:     ('n_value', 4)
// CHECK-NEXT:     ('_string', '__ZTS3optIbE')
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT:   # Load Command 2
// CHECK-NEXT:  (('command', 11)
// CHECK-NEXT:   ('size', 80)
// CHECK-NEXT:   ('ilocalsym', 0)
// CHECK-NEXT:   ('nlocalsym', 0)
// CHECK-NEXT:   ('iextdefsym', 0)
// CHECK-NEXT:   ('nextdefsym', 2)
// CHECK-NEXT:   ('iundefsym', 2)
// CHECK-NEXT:   ('nundefsym', 0)
// CHECK-NEXT:   ('tocoff', 0)
// CHECK-NEXT:   ('ntoc', 0)
// CHECK-NEXT:   ('modtaboff', 0)
// CHECK-NEXT:   ('nmodtab', 0)
// CHECK-NEXT:   ('extrefsymoff', 0)
// CHECK-NEXT:   ('nextrefsyms', 0)
// CHECK-NEXT:   ('indirectsymoff', 0)
// CHECK-NEXT:   ('nindirectsyms', 0)
// CHECK-NEXT:   ('extreloff', 0)
// CHECK-NEXT:   ('nextrel', 0)
// CHECK-NEXT:   ('locreloff', 0)
// CHECK-NEXT:   ('nlocrel', 0)
// CHECK-NEXT:   ('_indirect_symbols', [
// CHECK-NEXT:   ])
// CHECK-NEXT:  ),
// CHECK-NEXT: ])
