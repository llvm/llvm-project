// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck  %s

        .2byte 42, 1, 2, 3
        .4byte 42, 1, 2, 3
        .8byte 42, 1, 2, 3
        .int 42, 1, 2, 3

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 72
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 2A000100 02000300 2A000000 01000000
// CHECK-NEXT:       0010: 02000000 03000000 2A000000 00000000
// CHECK-NEXT:       0020: 01000000 00000000 02000000 00000000
// CHECK-NEXT:       0030: 03000000 00000000 2A000000 01000000
// CHECK-NEXT:       0040: 02000000 03000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
