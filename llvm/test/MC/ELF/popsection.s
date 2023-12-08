// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-readobj -S - | FileCheck %s

// This used to crash. Test that it create an empty section instead.

        .pushsection foo
        .popsection

// CHECK:       Section {
// CHECK:         Index:
// CHECK:         Name: foo
// CHECK-NEXT:    Type: SHT_PROGBITS
// CHECK-NEXT:    Flags [ (0x0)
// CHECK-NEXT:    ]
// CHECK-NEXT:    Address: 0x0
// CHECK-NEXT:    Offset:
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Link: 0
// CHECK-NEXT:    Info: 0
// CHECK-NEXT:    AddressAlignment: 1
// CHECK-NEXT:    EntrySize: 0
// CHECK-NEXT:  }
