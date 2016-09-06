// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o - | llvm-readobj -r | FileCheck %s

// these should not produce relaxable relocations

        movq foo@GOT, %rax
        mulq foo@GOTPCREL(%rip)
        .long foo@GOTPCREL

// CHECK:      Relocations [
// CHECK:        Section ({{.*}}) .rela.text {
// CHECK-NEXT:     R_X86_64_GOT32 foo
// CHECK-NEXT:     R_X86_64_GOTPCREL foo
// CHECK-NEXT:     R_X86_64_GOTPCREL foo
// CHECK-NEXT:   }
// CHECK-NEXT: ]
