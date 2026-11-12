// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sd - | FileCheck %s

        .sleb128 .Lfoo - .Lbar
.Lfoo:
        .uleb128 .Lbar - .Lfoo
        .fill 126, 1, 0x90
.Lbar:

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_EXECINSTR
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 129
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 817F7F90 90909090 90909090 90909090
// CHECK-NEXT:       0010: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0020: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0030: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0040: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0050: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0060: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0070: 90909090 90909090 90909090 90909090
// CHECK-NEXT:       0080: 90
// CHECK-NEXT:     )
// CHECK-NEXT:   }
