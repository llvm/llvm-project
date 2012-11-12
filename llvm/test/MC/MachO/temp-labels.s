// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -L -o - | macho-dump --dump-section-data | FileCheck %s

// CHECK:   # Load Command 1
// CHECK:  (('command', 2)
// CHECK:   ('size', 24)
// CHECK:   ('symoff', 296)
// CHECK:   ('nsyms', 2)
// CHECK:   ('stroff', 328)
// CHECK:   ('strsize', 8)
// CHECK:   ('_string_data', '\x00_f0\x00L0\x00')
// CHECK:   ('_symbols', [
// CHECK:     # Symbol 0
// CHECK:    (('n_strx', 1)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 0)
// CHECK:     ('_string', '_f0')
// CHECK:    ),
// CHECK:     # Symbol 1
// CHECK:    (('n_strx', 5)
// CHECK:     ('n_type', 0xe)
// CHECK:     ('n_sect', 1)
// CHECK:     ('n_desc', 0)
// CHECK:     ('n_value', 4)
// CHECK:     ('_string', 'L0')
// CHECK:    ),
// CHECK:   ])
// CHECK:  ),
_f0:
        .long 0
L0:
        .long 0
