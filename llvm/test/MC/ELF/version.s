// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | llvm-readobj -S --sd - | FileCheck  %s

.version "1234"
.version "123"

// CHECK:        Section {
// CHECK:          Name: .note
// CHECK-NEXT:     Type: SHT_NOTE
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x34
// CHECK-NEXT:     Size: 36
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 05000000 00000000 01000000 31323334
// CHECK-NEXT:       0010: 00000000 04000000 00000000 01000000
// CHECK-NEXT:       0020: 31323300
// CHECK-NEXT:     )
// CHECK-NEXT:   }
