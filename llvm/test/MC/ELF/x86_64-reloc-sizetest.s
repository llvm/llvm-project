// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | llvm-readobj -r - | FileCheck %s

// Tests that relocation value fits in the provided size
// Original bug http://llvm.org/bugs/show_bug.cgi?id=10568

L: movq $(L + 2147483648),%rax


// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{[0-9]+}}) .rela.text {
// CHECK-NEXT:     0x3 R_X86_64_32S {{[^ ]+}} 0x80000000
// CHECK-NEXT:   }
// CHECK-NEXT: ]
