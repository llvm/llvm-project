// RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -filetype=obj -o %t.o 
// RUN: llvm-readobj -elf-section-groups %t.o | FileCheck %s

// Test that we can handle numeric COMDAT names.

.section .foo,"G",@progbits,123,comdat
.section .bar,"G",@progbits,abc,comdat

// CHECK:      Groups {
// CHECK-NEXT:   Group {
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     Type: COMDAT
// CHECK-NEXT:     Signature: 123
// CHECK-NEXT:     Section(s) in group [
// CHECK-NEXT:       .foo
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   Group {
// CHECK-NEXT:     Name: .group
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     Type: COMDAT
// CHECK-NEXT:     Signature: abc
// CHECK-NEXT:     Section(s) in group [
// CHECK-NEXT:       .bar
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: }
