// RUN: llvm-mc -triple i386-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s

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

// CHECK: Relocations [
// CHECK:   Section __const {
// CHECK:     0x5181034 0 2 0 GENERIC_RELOC_VANILLA 0 __bss
// CHECK:     0x518102C 0 2 0 GENERIC_RELOC_VANILLA 0 __bss
// CHECK:     0x5181028 0 2 0 GENERIC_RELOC_VANILLA 0 __bss
// CHECK:     0x5181020 0 2 0 GENERIC_RELOC_VANILLA 0 __bss
// CHECK:   }
// CHECK: ]
