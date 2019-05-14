// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck %s

// CHECK:        Section {
// CHECK:          Name: .comment
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_MERGE
// CHECK-NEXT:       SHF_STRINGS
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 13
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 1
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 00666F6F 00626172 007A6564 00
// CHECK-NEXT:     )
// CHECK-NEXT:   }

        .ident "foo"
        .ident "bar"
        .ident "zed"
