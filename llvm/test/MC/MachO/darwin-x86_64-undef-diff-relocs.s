// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -S --sd -r - | FileCheck %s

// Subtraction expressions where one or both symbols are undefined are
// representable as an X86_64_RELOC_SUBTRACTOR / X86_64_RELOC_UNSIGNED pair.

        .text
_foo:
        nop
        nop
        nop
        nop
_bar:
        .long _undef - _bar
        .long (_undef + 4) - _bar
        .quad _undef - _foo
        .quad (_undef - 8) - _bar
        .long _undef - L0
L0:
        .long _undef - .
        .long _bar - _undef
        .long (L0 + 4) - _undef
        .quad _undef - _undef2

        .data
_d:
        .quad _undef - _d
        .quad _undef - _bar


// CHECK:      Name: __text
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 90909090 00000000 04000000 00000000
// CHECK-NEXT:   0010: 00000000 F8FFFFFF FFFFFFFF E4FFFFFF
// CHECK-NEXT:   0020: E4FFFFFF 00000000 20000000 00000000
// CHECK-NEXT:   0030: 00000000
// CHECK-NEXT: )

// CHECK:      Name: __data
// CHECK:      SectionData (
// CHECK-NEXT:   0000: 00000000 00000000 00000000 00000000
// CHECK-NEXT: )

// CHECK:      Relocations [
// CHECK-NEXT:   Section __text {
// CHECK-NEXT:     0x2C 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _undef2
// CHECK-NEXT:     0x2C 0 3 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x28 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _undef
// CHECK-NEXT:     0x28 0 2 1 X86_64_RELOC_UNSIGNED 0 _bar
// CHECK-NEXT:     0x24 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _undef
// CHECK-NEXT:     0x24 0 2 1 X86_64_RELOC_UNSIGNED 0 _bar
// CHECK-NEXT:     0x20 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x20 0 2 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x1C 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x1C 0 2 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x14 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x14 0 3 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0xC 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _foo
// CHECK-NEXT:     0xC 0 3 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x8 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x8 0 2 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x4 0 2 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x4 0 2 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:   }
// CHECK-NEXT:   Section __data {
// CHECK-NEXT:     0x8 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _bar
// CHECK-NEXT:     0x8 0 3 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:     0x0 0 3 1 X86_64_RELOC_SUBTRACTOR 0 _d
// CHECK-NEXT:     0x0 0 3 1 X86_64_RELOC_UNSIGNED 0 _undef
// CHECK-NEXT:   }
// CHECK-NEXT: ]
