// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump | FileCheck %s

_bar:
  nop
_foo:
  nop

  .set foo_set1, (_foo + 0xffff0000)
  .set foo_set2, (_foo - _bar + 0xffff0000)

foo_equals = (_foo + 0xffff0000)
foo_equals2 = (_foo - _bar + 0xffff0000)

  .globl foo_set1_global;
  .set foo_set1_global, (_foo + 0xffff0000)

  .globl foo_set2_global;
  .set foo_set2_global, (_foo - _bar + 0xffff0000)

// CHECK: ('cputype', 16777223)
// CHECK: ('cpusubtype', 3)
// CHECK: ('filetype', 1)
// CHECK: ('num_load_commands', 3)
// CHECK: ('load_commands_size', 256)
// CHECK: ('flag', 0)
// CHECK: ('reserved', 0)
// CHECK: ('load_commands', [
// CHECK:   # Load Command 0
// CHECK:  (('command', 25)
// CHECK:   ('size', 152)
// CHECK:   ('segment_name', '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:   ('vm_addr', 0)
// CHECK:   ('vm_size', 2)
// CHECK:   ('file_offset', 288)
// CHECK:   ('file_size', 2)
// CHECK:   ('maxprot', 7)
// CHECK:   ('initprot', 7)
// CHECK:   ('num_sections', 1)
// CHECK:   ('flags', 0)
// CHECK:   ('sections', [
// CHECK:     # Section 0
// CHECK:    (('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('segment_name', '__TEXT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// CHECK:     ('address', 0)
// CHECK:     ('size', 2)
// CHECK:     ('offset', 288)
// CHECK:     ('alignment', 0)
// CHECK:     ('reloc_offset', 0)
// CHECK:     ('num_reloc', 0)
// CHECK:     ('flags', 0x80000400)
// CHECK:     ('reserved1', 0)
// CHECK:     ('reserved2', 0)
// CHECK:     ('reserved3', 0)
// CHECK:    ),
// CHECK:   ('_relocations', [
// CHECK:   ])
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 292)
// CHECK:   ('nsyms', 8)
// CHECK:   ('stroff', 420)
// CHECK:   ('strsize', 84)
// CHECK:   ('_string_data', '\x00foo_set1_global\x00foo_set2_global\x00_bar\x00_foo\x00foo_set1\x00foo_set2\x00foo_equals\x00foo_equals2\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 33)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_bar')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 38)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 1)
// CHECK:     ('_string', '_foo')
// CHECK:    ),
// CHECK:     # Symbol 2
// CHECK:    (('n_strx', 43)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_set1')
// CHECK:    ),
// CHECK:     # Symbol 3
// CHECK:    (('n_strx', 52)
// CHECK:     ('n_type', 0x2)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_set2')
// CHECK:    ),
// CHECK:     # Symbol 4
// CHECK:    (('n_strx', 61)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_equals')
// CHECK:    ),
// CHECK:     # Symbol 5
// CHECK:    (('n_strx', 72)
// CHECK:     ('n_type', 0x2)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_equals2')
// CHECK:    ),
// CHECK:     # Symbol 6
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xf)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_set1_global')
// CHECK:    ),
// CHECK:     # Symbol 7
// CHECK:    (('n_strx', 17)
// CHECK:     ('n_type', 0x3)
// CHECK:     ('n_sect', 0)
// CHECK:     ('n_desc', 32)
// CHECK:     ('n_value', 4294901761)
// CHECK:     ('_string', 'foo_set2_global')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
// CHECK:   # Load Command 2
// CHECK:  (('command', 11)
// CHECK:   ('size', 80)
// CHECK:   ('ilocalsym', 0)
// CHECK:   ('nlocalsym', 6)
// CHECK:   ('iextdefsym', 6)
// CHECK:   ('nextdefsym', 2)
// CHECK:   ('iundefsym', 8)
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
