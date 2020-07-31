// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr --symbols - | FileCheck %s

// When doing a rename, all the checks for where the relocation should go
// should be performed with the original symbol. Only if we decide to relocate
// with the symbol we should then use the renamed one.

// This is a regression test for a bug where we used bar5@@@zed when deciding
// if we should relocate with the symbol or with the section and we would then
// not produce a relocation with .text.

defined1:
defined3:
        .symver defined3, bar5@@@zed
        .long defined3

        .global defined1

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .rela.text
// CHECK-NEXT:     Type: SHT_RELA (0x4)
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x0 R_X86_64_32 .text 0x0
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
