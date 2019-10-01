// Regression test for PR23914.
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r --symbols | FileCheck %s

defined:
        .symver defined, aaaaaaaaaaaaaaaaaa@@@AAAAAAAAAAAAA

// CHECK:      Symbol {
// CHECK:        Name: aaaaaaaaaaaaaaaaaa@@AAAAAAAAAAAAA
// CHECK-NEXT:   Value: 0x0
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .text
// CHECK-NEXT: }

