// RUN: llvm-mc -triple i386-apple-darwin10 %s -filetype=obj -o - | macho-dump | FileCheck %s

.space 0x1ed280
       .section        __DATA,__const
       .align  4
.space 0x5181020
_foo:
       .long   _bar
       .long   0
       .long   _bar+8
       .long   _bar+24
       .long   0
       .long   _bar+16

.zerofill __DATA,__bss,__dummy,0x5d780
.zerofill __DATA,__bss,_bar,48,4

// Normally scattered relocations are used for sym+offset expressions. When
// the value exceeds 24-bits, however, it's outside what MachO can encode,
// so the assembler falls back to non-scattered relocations.
// rdar://12358909

// CHECK: ('_relocations', [
// CHECK:   # Relocation 0
// CHECK:   (('word-0', 0x5181034),
// CHECK:    ('word-1', 0x4000003)),
// CHECK:   # Relocation 1
// CHECK:   (('word-0', 0x518102c),
// CHECK:    ('word-1', 0x4000003)),
// CHECK:   # Relocation 2
// CHECK:   (('word-0', 0x5181028),
// CHECK:    ('word-1', 0x4000003)),
// CHECK:   # Relocation 3
// CHECK:   (('word-0', 0x5181020),
// CHECK:    ('word-1', 0x4000003)),
// CHECK: ])
