// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin %s -o - | llvm-readobj -r - | FileCheck %s

// CHECK:      Relocations [
// CHECK-NEXT: ]

  .section foo,bar
La:
Lb:
 .long   La-Lb
