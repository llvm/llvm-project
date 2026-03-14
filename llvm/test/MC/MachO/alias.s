// RUN: llvm-mc -triple x86_64-apple-macosx10.12.0 %s -filetype=obj | llvm-readobj -r - | FileCheck %s

l_a:
l_b = l_a
l_c = l_b
        .long l_c

// CHECK:      Relocations [
// CHECK-NEXT:   Section __text {
// CHECK-NEXT:     0x0 0 2 1 X86_64_RELOC_UNSIGNED 0 l_c
// CHECK-NEXT:   }
// CHECK-NEXT: ]
