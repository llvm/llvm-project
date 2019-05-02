// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r --symbols | FileCheck %s

// Test that this produces a R_X86_64_PLT32 with bar.

        .globl foo
foo:
bar = foo
        .section zed, "", @progbits
        call bar@PLT


// Test that this produres a relocation with bar2

    .weak    foo2
foo2:
    .weak    bar2
    .set    bar2,foo2
    .quad    bar2

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .relazed {
// CHECK-NEXT:     0x1 R_X86_64_PLT32 bar 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x5 R_X86_64_64 bar2 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:      Symbols [
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name:  (0)
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local (0x0)
// CHECK-NEXT:     Type: None (0x0)
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Undefined (0x0)
// CHECK-NEXT:   }
// CHECK-NEXT:   Symbol {
// CHECK-NEXT:     Name: bar
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .text
// CHECK-NEXT:   }

// CHECK:        Symbol {
// CHECK:          Name: bar2
// CHECK-NEXT:     Value: 0x5
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Weak
// CHECK-NEXT:     Type: None
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: zed
// CHECK-NEXT:   }
