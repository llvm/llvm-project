// RUN: llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o - | llvm-readobj -r --symbols - | FileCheck %s

// rdar://9906375
.org 0x100
_foo:
_bar = _foo + 2
_baz:
        leaq    _bar(%rip), %rcx

// CHECK: File: <stdin>
// CHECK-NEXT: Format: Mach-O 64-bit x86-64
// CHECK-NEXT: Arch: x86_64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-NEXT: Relocations [
// CHECK-NEXT:   Section __text {
// CHECK-NEXT:     0x103 1 2 1 X86_64_RELOC_SIGNED 0 _bar
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: _foo (11)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x100
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: _bar (6)
// CHECK-NEXT:     Type: Section (0xE)
// CHECK-NEXT:     Section: __text (0x1)
// CHECK-NEXT:     RefType: UndefinedNonLazy (0x0)
// CHECK-NEXT:     Flags [ (0x200)
// CHECK-NEXT:       AltEntry (0x200)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Value: 0x102
// CHECK-NEXT:   }
