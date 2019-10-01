// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --symbols --sd | FileCheck %s

// Test that we produce the group sections and that they are before the members

// CHECK:        Section {
// CHECK:          Index: 3
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 12
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info: 1
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000:    01000000 04000000 05000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Index: 6
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 8
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info: 2
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000:    01000000 07000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Index: 8
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Type: SHT_GROUP
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 12
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info: 3
// CHECK-NEXT:     AddressAlignment: 4
// CHECK-NEXT:     EntrySize: 4
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000:    01000000 09000000 0A000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }

// Test that g1 and g2 are local, but g3 is an undefined global.

// CHECK:        Symbol {
// CHECK:          Name: g1
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .foo
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: g2
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .group
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: g3
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }


	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"ax?",@progbits
        nop

        .section	.zed,"axG",@progbits,g2,comdat
        nop

        .section	.baz,"axG",@progbits,g3,comdat
        .long g3
