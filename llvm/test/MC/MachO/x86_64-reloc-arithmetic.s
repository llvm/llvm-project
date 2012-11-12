// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | macho-dump | FileCheck %s

// rdar://9906375
.org 0x100
_foo:
_bar = _foo + 2
_baz:
        leaq    _bar(%rip), %rcx

// CHECK:        ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('word-0', 0x103),
// CHECK-NEXT:     ('word-1', 0x1d000001))

// CHECK:         # Symbol 1
// CHECK-NEXT:   (('n_strx', 6)
// CHECK-NEXT:    ('n_type', 0xe)
// CHECK-NEXT:    ('n_sect', 1)
// CHECK-NEXT:    ('n_desc', 0)
// CHECK-NEXT:    ('n_value', 258)
// CHECK-NEXT:    ('_string', '_bar')
