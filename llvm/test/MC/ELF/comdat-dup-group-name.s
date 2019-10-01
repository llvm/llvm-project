// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --symbols | FileCheck %s

// Test that we produce two foo sections, each in separate groups

// CHECK: Index: 3
// CHECK-NEXT: Name: .group

// CHECK: Index: 4
// CHECK-NEXT: Name: .foo

// CHECK: Index: 5
// CHECK-NEXT: Name: .group

// CHECK: Index: 6
// CHECK-NEXT: Name: .foo

// CHECK: Symbols [

// CHECK: Name: f1
// CHECK-NOT: }
// CHECK: Section: .group (0x3)

// CHECK: Name: f2
// CHECK-NOT: }
// CHECK: Section: .group (0x5)

	.section	.foo,"axG",@progbits,f1,comdat
        nop

	.section	.foo,"axG",@progbits,f2,comdat
        nop

