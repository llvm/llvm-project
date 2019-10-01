// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck %s

        .zero 4
foo:
        .zero 4
        .org foo+16

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK:          ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 20
