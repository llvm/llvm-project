// RUN: llvm-mc -filetype=obj %s -o - -triple x86_64-pc-linux | llvm-readobj -S | FileCheck %s

// CHECK:      Name: .text
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 4092

        .globl  foo
foo:
        .space 4
bar:
        .space  4092 - (bar - foo)
